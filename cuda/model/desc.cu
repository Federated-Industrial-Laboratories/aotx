/* Purpose: Hold the state of the model module and bind each descriptor to its tensors.
 * Owns: The descriptor, the call block, the buffer block and the cache position table.
 * Launch shape: One thread for each tensor name of one model.
 * Lifetime: From model load to the end of the run. */
#include "mem/mem.cuh"
#include "model/forward.cuh"
#include "model/names.h"
#include "text/text.cuh"

/* The state of the module. The host glue writes each block once, at the model load and at
 * the graph capture, and the kernels of a pass read them. */
__device__ aotx_model_desc aotx_model[AOTX_MODEL_ROLES];
__device__ aotx_model_run aotx_model_call[AOTX_MODEL_ROLES];
__device__ aotx_model_work aotx_model_space[AOTX_MODEL_ROLES];
__device__ unsigned int aotx_model_seen[AOTX_SLOTS];
__device__ unsigned int aotx_model_draw[AOTX_SLOTS];
__device__ unsigned int aotx_model_faults;
__device__ unsigned int aotx_model_head_type[AOTX_MODEL_ROLES][2];

/* The names the device forms. The host reads the same two lists from names.h. */
__device__ const char aotx_desc_whole[AOTX_DESC_WHOLE][AOTX_DESC_NAME] =
    AOTX_DESC_WHOLE_LIST;
__device__ const char aotx_desc_layer[AOTX_DESC_PER_LAYER][AOTX_DESC_NAME] =
    AOTX_DESC_LAYER_LIST;

/* Bytes of the longest name. The name holds "blk.", two digits, a point, the longest layer
 * name of eleven bytes, and ".weight", with room to spare. */
#define AOTX_DESC_BUFFER  48u

/* Copy a name that ends with a zero byte and give the bytes copied. */
__device__ __forceinline__ static unsigned int aotx_desc_copy(const char *from, char *out,
                                                              unsigned int at,
                                                              unsigned int max)
{
    for (unsigned int i = 0u; from[i] != '\0' && at < max; ++i) {
        out[at++] = from[i];
    }
    return at;
}

/* Write the name of one place of the name list. The layer number comes from the number
 * writer of the text module, so no name is formed on the host. */
__device__ __forceinline__ static unsigned int aotx_desc_write(unsigned int at, char *out,
                                                               unsigned int max)
{
    if (at < AOTX_DESC_WHOLE) {
        unsigned int held = aotx_desc_copy(aotx_desc_whole[at], out, 0u, max);
        out[held] = '\0';
        return held;
    }
    unsigned int layer = (at - AOTX_DESC_WHOLE) / AOTX_DESC_PER_LAYER;
    unsigned int which = (at - AOTX_DESC_WHOLE) % AOTX_DESC_PER_LAYER;
    unsigned int held = aotx_desc_copy("blk.", out, 0u, max);
    held += aotx_text_utoa((unsigned long long)layer, out + held, max - held);
    held = aotx_desc_copy(".", out, held, max);
    held = aotx_desc_copy(aotx_desc_layer[which], out, held, max);
    held = aotx_desc_copy(".weight", out, held, max);
    out[held] = '\0';
    return held;
}

/* The tensors of a layer that every model of this family must have. The output tensor is
 * absent when the model reads the logits from the token embedding. The class tensor is
 * absent in a model which does not give a rank. */
__device__ __forceinline__ static int aotx_desc_needed(unsigned int role, unsigned int at)
{
    if (at == 2u) {
        return 0;
    }
    if (at == 3u) {
        return role == AOTX_MODEL_RERANKER;
    }
    return 1;
}

__global__ void aotx_model_bind(unsigned int role, unsigned int model, unsigned int count,
                                unsigned int *missing)
{
    char name[AOTX_DESC_BUFFER];
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (role >= AOTX_MODEL_ROLES || at >= count) {
        return;
    }
    aotx_model_desc *desc = &aotx_model[role];

    /* Every offset of the descriptor is an unsigned long long, and the first four stand for
     * the tensors of the whole model. The layer offsets follow in the layer structure. The
     * bind therefore writes an offset by its number. */
    unsigned long long *slot;
    if (at < AOTX_DESC_WHOLE) {
        slot = &(&desc->token_embd)[at];
    } else {
        unsigned int layer = (at - AOTX_DESC_WHOLE) / AOTX_DESC_PER_LAYER;
        unsigned int which = (at - AOTX_DESC_WHOLE) % AOTX_DESC_PER_LAYER;
        if (layer >= AOTX_MODEL_MAX_LAYERS) {
            return;
        }
        slot = &(&desc->layer[layer].attn_norm)[which];
    }

    unsigned int bytes = aotx_desc_write(at, name, AOTX_DESC_BUFFER - 1u);
    const aotx_mem_tensor *tensor = aotx_mem_tensor_find(aotx_mem_name(name, bytes), model);
    if (tensor == 0) {
        *slot = AOTX_MODEL_ABSENT;
        if (aotx_desc_needed(role, at) != 0) {
            atomicAdd(&missing[0], 1u);
            atomicMin(&missing[1], at);
        }
        return;
    }
    *slot = tensor->offset;
    if (at == 0u) {
        /* The token embedding gives the vocabulary and the block type of the embedding. The
         * second dimension of the tensor is its row count. */
        desc->embd_type = tensor->type;
        desc->vocab = (unsigned int)tensor->dims[1];
    }
    if (at == 2u) {
        desc->tied_output = 0u;
        aotx_model_head_type[role][0] = tensor->type;
    }
    if (at == 3u) {
        aotx_model_head_type[role][1] = tensor->type;
    }
    if (at == AOTX_DESC_WHOLE + 1u) {
        /* The query projection of the first layer gives the block type of every projection.
         * One model file holds one block type for its projections. */
        desc->weight_type = tensor->type;
    }
}
