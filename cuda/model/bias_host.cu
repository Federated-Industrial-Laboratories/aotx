/* Purpose: Check attention bias tensors and capture their layer nodes.
 * Owns: Nothing; the model hold owns the stream and workspace.
 * Launch shape: Host glue; each device node takes the token batch.
 * Lifetime: One model load or graph capture. */
#include "model/graph_host.h"
#include "model/kinds.h"

extern "C" {
#include "disk/modelfile/modelfile.h"
}

int aotx_model_check_bias(const aotx_modelfile *file, const aotx_model_desc *desc,
                          char *reason, size_t reason_size)
{
    if (desc->hidden == 0u || desc->hidden % AOTX_MATRIX_BLOCK != 0u
        || desc->ffn == 0u || desc->ffn % AOTX_MATRIX_BLOCK != 0u) {
        snprintf(reason, reason_size, "the attention bias widths must be positive multiples of 32");
        return 1;
    }
    uint64_t h = desc->hidden, f = desc->ffn;
    uint64_t q = (uint64_t)desc->heads * desc->head_dim;
    uint64_t k = (uint64_t)desc->kv_heads * desc->head_dim;
    const uint64_t shape[][2] = {
        {h, 0}, {h, q}, {h, k}, {h, k}, {q, h}, {h, 0},
        {h, f}, {h, f}, {f, h}, {q, 0}, {k, 0}, {k, 0}
    };
    const aotx_layer_kind *kind = aotx_layer_kind_of(AOTX_LAYER_KIND_ATTENTION_BIAS);
    for (unsigned int layer = 0u; layer < desc->layers; ++layer) {
        if (desc->kind[layer] != AOTX_LAYER_KIND_ATTENTION_BIAS) continue;
        for (unsigned int i = 0u; i < kind->tensors; ++i) {
            char name[AOTX_DESC_BUFFER];
            aotx_tensor_info tensor;
            const aotx_layer_tensor *entry = &kind->tensor[i];
            aotx_layer_name(name, sizeof name, layer, entry);
            unsigned int dimensions = shape[i][1] ? 2u : 1u;
            if (aotx_modelfile_find(file, name, &tensor) != 0) {
                snprintf(reason, reason_size, "the attention bias layer has no tensor %s", name);
                return 1;
            }
            if (tensor.dim_count != dimensions
                || memcmp(tensor.dims, shape[i], dimensions * sizeof(uint64_t)) != 0) {
                snprintf(reason, reason_size, "the attention bias tensor %s has an incompatible shape", name);
                return 1;
            }
            unsigned int scalar = entry->slot == AOTX_SLOT_ATTN_NORM
                || entry->slot == AOTX_SLOT_FFN_NORM || entry->slot == AOTX_BIAS_Q
                || entry->slot == AOTX_BIAS_K || entry->slot == AOTX_BIAS_V;
            if ((scalar && tensor.type != AOTX_TENSOR_F32) || !aotx_matrix_known(tensor.type)) {
                snprintf(reason, reason_size, "the attention bias tensor %s has an incompatible type", name);
                return 1;
            }
        }
        for (unsigned int i = 0u; i < sizeof aotx_layer_attention_tensor
                                      / sizeof aotx_layer_attention_tensor[0]; ++i) {
            const aotx_layer_tensor *entry = &aotx_layer_attention_tensor[i];
            if (entry->slot != AOTX_ATTENTION_Q_NORM && entry->slot != AOTX_ATTENTION_K_NORM)
                continue;
            char name[AOTX_DESC_BUFFER];
            aotx_tensor_info tensor;
            aotx_layer_name(name, sizeof name, layer, entry);
            if (aotx_modelfile_find(file, name, &tensor) == 0) {
                snprintf(reason, reason_size, "the attention bias layer must not hold %s", name);
                return 1;
            }
        }
    }
    return 0;
}

void aotx_model_capture_attention_bias(aotx_model_hold *hold, unsigned int role,
                                       unsigned int layer)
{
    const aotx_model_desc *desc = &hold->desc;
    const aotx_model_layer *weights = &desc->layer[layer];
    const unsigned char *type = desc->layer_type[layer];
    aotx_model_work *work = &hold->work;
    cudaStream_t stream = hold->stream;
    unsigned int tokens = hold->max_tokens;
    unsigned int wave = (hold->wave == 0u) ? tokens : hold->wave;
    unsigned int wide = desc->heads * desc->head_dim;
    unsigned int narrow = desc->kv_heads * desc->head_dim;
    aotx_model_norm<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(
        role, layer, AOTX_MODEL_NORM_ATTN);
    aotx_model_matrix(hold, aotx_model_tensor(weights->offset[AOTX_SLOT_ATTN_Q]), type[AOTX_SLOT_ATTN_Q], wide,
                      desc->hidden, work->x, tokens, work->q, AOTX_MODEL_BATCH_TOKENS);
    aotx_model_matrix(hold, aotx_model_tensor(weights->offset[AOTX_SLOT_ATTN_K]), type[AOTX_SLOT_ATTN_K], narrow,
                      desc->hidden, work->x, tokens, work->k, AOTX_MODEL_BATCH_TOKENS);
    aotx_model_matrix(hold, aotx_model_tensor(weights->offset[AOTX_SLOT_ATTN_V]), type[AOTX_SLOT_ATTN_V], narrow,
                      desc->hidden, work->x, tokens, work->v, AOTX_MODEL_BATCH_TOKENS);
    aotx_model_qkv_bias<<<dim3(wave, desc->heads + desc->kv_heads),
                           desc->head_dim / 2u, 0, stream>>>(role, layer);
    dim3 attention((wave + AOTX_MODEL_ATTN_TOKENS - 1u) / AOTX_MODEL_ATTN_TOKENS,
                   desc->heads);
    aotx_model_attend<<<attention, AOTX_MODEL_ATTN_THREADS, 0, stream>>>(role, layer);
    aotx_model_matrix(hold, aotx_model_tensor(weights->offset[AOTX_SLOT_ATTN_O]), type[AOTX_SLOT_ATTN_O], desc->hidden,
                      wide, work->att, tokens, work->proj, AOTX_MODEL_BATCH_TOKENS);
    aotx_model_residual<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role);
    aotx_model_norm<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(
        role, layer, AOTX_MODEL_NORM_FFN);
    aotx_model_matrix(hold, aotx_model_tensor(weights->offset[AOTX_SLOT_FFN_GATE]), type[AOTX_SLOT_FFN_GATE], desc->ffn,
                      desc->hidden, work->x, tokens, work->gate, AOTX_MODEL_BATCH_TOKENS);
    aotx_model_matrix(hold, aotx_model_tensor(weights->offset[AOTX_SLOT_FFN_UP]), type[AOTX_SLOT_FFN_UP], desc->ffn,
                      desc->hidden, work->x, tokens, work->up, AOTX_MODEL_BATCH_TOKENS);
    aotx_model_swiglu<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role);
    aotx_model_matrix(hold, aotx_model_tensor(weights->offset[AOTX_SLOT_FFN_DOWN]), type[AOTX_SLOT_FFN_DOWN],
                      desc->hidden, desc->ffn, work->act, tokens, work->proj,
                      AOTX_MODEL_BATCH_TOKENS);
    aotx_model_residual<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role);
    aotx_model_conduct<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role, layer);
}
