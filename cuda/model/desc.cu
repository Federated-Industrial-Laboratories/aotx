/* Purpose: Hold the state of the model module and bind each descriptor to its tensors.
 * Owns: The descriptor, the call block, the buffer block and the cache position table.
 * Launch shape: One thread for each tensor name of one model.
 * Lifetime: From model load to the end of the run. */
#include "mem/mem.cuh"
#include "model/forward.cuh"

/* The state of the module. The host glue writes each block once, at the model load and at
 * the graph capture, and the kernels of a pass read them. */
__device__ aotx_model_desc aotx_model[AOTX_MODEL_ROLES];
__device__ aotx_model_run aotx_model_call[AOTX_MODEL_ROLES];
__device__ aotx_model_work aotx_model_space[AOTX_MODEL_ROLES];
__device__ unsigned int aotx_model_seen[AOTX_SLOTS];
__device__ unsigned int aotx_model_draw[AOTX_SLOTS];
__device__ unsigned int aotx_model_faults;
__device__ unsigned int aotx_model_head_type[AOTX_MODEL_ROLES][2];

__global__ void aotx_model_bind(unsigned int role, unsigned int model,
                                const aotx_model_binding *binding, unsigned int count,
                                unsigned int *missing)
{
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (role >= AOTX_MODEL_ROLES || at >= count) {
        return;
    }
    aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_binding *one = &binding[at];
    unsigned int slots = AOTX_DESC_WHOLE
                       + AOTX_MODEL_MAX_LAYERS * AOTX_LAYER_TENSOR_SLOTS;
    if (one->slot >= slots) {
        if (one->needed != 0u) {
            atomicAdd(&missing[0], 1u);
            atomicMin(&missing[1], at);
        }
        return;
    }
    unsigned long long *slot = &desc->token_embd + one->slot;
    unsigned int bytes = 0u;
    while (bytes < AOTX_DESC_BUFFER && one->name[bytes] != '\0') {
        bytes += 1u;
    }
    const aotx_mem_tensor *tensor = aotx_mem_tensor_find(aotx_mem_name(one->name, bytes),
                                                         model);
    if (tensor == 0) {
        *slot = AOTX_MODEL_ABSENT;
        if (one->needed != 0u) {
            atomicAdd(&missing[0], 1u);
            atomicMin(&missing[1], at);
        }
        return;
    }
    *slot = tensor->offset;
    if (one->slot == 0u) {
        /* The token embedding gives the vocabulary and the block type of the embedding. The
         * second dimension of the tensor is its row count. */
        desc->embd_type = tensor->type;
        desc->vocab = (unsigned int)tensor->dims[1];
    }
    if (one->slot == 2u) {
        desc->tied_output = 0u;
        aotx_model_head_type[role][0] = tensor->type;
    }
    if (one->slot == 3u) {
        aotx_model_head_type[role][1] = tensor->type;
    }
    if (one->slot >= AOTX_DESC_WHOLE) {
        unsigned int at = one->slot - AOTX_DESC_WHOLE;
        desc->layer_type[at / AOTX_LAYER_TENSOR_SLOTS][at % AOTX_LAYER_TENSOR_SLOTS]
            = (unsigned char)tensor->type;
    }
    if (one->slot == AOTX_DESC_WHOLE + 1u) {
        /* The query projection of the first layer names the block type of the file. A
         * product takes the type of its own tensor; this one names the file on the panel. */
        desc->weight_type = tensor->type;
    }
}
