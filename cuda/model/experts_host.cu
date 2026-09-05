/* Purpose: Check expert model files and capture their layer nodes.
 * Owns: Nothing; the model hold owns the stream and workspace.
 * Launch shape: Host glue; each device node takes the token batch.
 * Lifetime: One model load or graph capture. */
#include "model/experts.cuh"
#include "model/graph_host.h"
#include "model/kinds.h"

extern "C" {
#include "disk/modelfile/modelfile.h"
}

int aotx_model_check_experts(const aotx_modelfile *file, const aotx_model_desc *desc,
                             char *reason, size_t reason_size)
{
    const char *arch = NULL;
    size_t length = 0u;
    if (aotx_modelfile_string(file, "general.architecture", &arch, &length) != 0
        || length != 5u || memcmp(arch, "olmoe", 5u) != 0) {
        snprintf(reason, reason_size, "the expert layer requires the olmoe routing rule");
        return 1;
    }
    if (desc->hidden % AOTX_MATRIX_BLOCK != 0u || desc->ffn == 0u
        || desc->ffn % AOTX_MATRIX_BLOCK != 0u) {
        snprintf(reason, reason_size, "the expert widths must be positive multiples of 32");
        return 1;
    }
    uint64_t h = desc->hidden, f = desc->ffn, e = desc->expert_count;
    uint64_t q = (uint64_t)desc->heads * desc->head_dim;
    uint64_t k = (uint64_t)desc->kv_heads * desc->head_dim;
    const uint64_t shape[12][3] = {
        {h, 0, 0}, {h, q, 0}, {h, k, 0}, {h, k, 0}, {q, h, 0},
        {q, 0, 0}, {k, 0, 0}, {h, 0, 0}, {h, f, e}, {h, f, e},
        {f, h, e}, {h, e, 0}
    };
    for (unsigned int layer = 0u; layer < desc->layers; ++layer) {
        if (desc->kind[layer] != AOTX_LAYER_KIND_FFN_EXPERTS) continue;
        for (unsigned int slot = 0u; slot < 12u; ++slot) {
            char name[AOTX_DESC_BUFFER];
            aotx_tensor_info tensor;
            aotx_layer_name(name, sizeof name, layer, &aotx_layer_experts_tensor[slot]);
            unsigned int dimensions = shape[slot][2] ? 3u : (shape[slot][1] ? 2u : 1u);
            if (aotx_modelfile_find(file, name, &tensor) != 0) {
                snprintf(reason, reason_size, "the expert layer has no tensor %s", name);
                return 1;
            }
            if (tensor.dim_count != dimensions
                || memcmp(tensor.dims, shape[slot], dimensions * sizeof(uint64_t)) != 0) {
                snprintf(reason, reason_size, "the expert tensor %s has an incompatible shape", name);
                return 1;
            }
            unsigned int at = aotx_layer_experts_tensor[slot].slot;
            unsigned int scalar = at == AOTX_SLOT_ATTN_NORM || at == AOTX_ATTENTION_Q_NORM
                || at == AOTX_ATTENTION_K_NORM || at == AOTX_SLOT_FFN_NORM
                || at == AOTX_EXPERT_ROUTER;
            if ((scalar && tensor.type != AOTX_TENSOR_F32) || !aotx_matrix_known(tensor.type)) {
                snprintf(reason, reason_size, "the expert tensor %s has an incompatible type", name);
                return 1;
            }
        }
    }
    return 0;
}

void aotx_model_capture_experts(aotx_model_hold *hold, unsigned int role, unsigned int layer)
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
    aotx_model_qk_norm<<<dim3(wave, 2u), AOTX_MODEL_ROW_THREADS, 0, stream>>>(role, layer);
    aotx_model_qkv_turn<<<dim3(wave, desc->heads + desc->kv_heads),
                           desc->head_dim / 2u, 0, stream>>>(role, layer);
    dim3 attention((wave + AOTX_MODEL_ATTN_TOKENS - 1u) / AOTX_MODEL_ATTN_TOKENS,
                   desc->heads);
    aotx_model_attend<<<attention, AOTX_MODEL_ATTN_THREADS, 0, stream>>>(role, layer);
    aotx_model_matrix(hold, aotx_model_tensor(weights->offset[AOTX_SLOT_ATTN_O]), type[AOTX_SLOT_ATTN_O], desc->hidden,
                      wide, work->att, tokens, work->proj, AOTX_MODEL_BATCH_TOKENS);
    aotx_model_residual<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role);
    aotx_model_norm<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(
        role, layer, AOTX_MODEL_NORM_FFN);
    aotx_model_expert_route<<<wave, AOTX_EXPERT_ROUTE_THREADS, 0, stream>>>(role, layer);
    dim3 feed((desc->ffn + AOTX_EXPERT_MATRIX_ROWS - 1u) / AOTX_EXPERT_MATRIX_ROWS, wave);
    dim3 down((desc->hidden + AOTX_EXPERT_MATRIX_ROWS - 1u) / AOTX_EXPERT_MATRIX_ROWS, wave);
    /* Each rank has independent token choices. The graph never reads those choices on the host. */
    for (unsigned int rank = 0u; rank < desc->expert_used_count; ++rank) {
        aotx_model_expert_matrix<<<feed, AOTX_EXPERT_MATRIX_THREADS, 0, stream>>>(
            role, rank, aotx_model_tensor(weights->offset[AOTX_SLOT_FFN_GATE]), type[AOTX_SLOT_FFN_GATE], desc->ffn,
            desc->hidden, work->x, work->gate);
        aotx_model_expert_matrix<<<feed, AOTX_EXPERT_MATRIX_THREADS, 0, stream>>>(
            role, rank, aotx_model_tensor(weights->offset[AOTX_SLOT_FFN_UP]), type[AOTX_SLOT_FFN_UP], desc->ffn,
            desc->hidden, work->x, work->up);
        aotx_model_swiglu<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role);
        aotx_model_expert_matrix<<<down, AOTX_EXPERT_MATRIX_THREADS, 0, stream>>>(
            role, rank, aotx_model_tensor(weights->offset[AOTX_SLOT_FFN_DOWN]), type[AOTX_SLOT_FFN_DOWN], desc->hidden,
            desc->ffn, work->act, work->proj);
        aotx_model_expert_add<<<wave, AOTX_EXPERT_ADD_THREADS, 0, stream>>>(role, rank);
    }
    aotx_model_residual<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role);
    aotx_model_conduct<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role, layer);
}
