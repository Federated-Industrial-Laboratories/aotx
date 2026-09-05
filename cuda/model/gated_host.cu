/* Purpose: Capture one gated attention layer and its dense feed forward operations.
 * Owns: Nothing; the model hold owns the buffers and stream.
 * Launch shape: Host glue; each node reads the token batch from the model call.
 * Lifetime: One graph capture. */
#include "model/graph_host.h"
#include "model/hybrid.cuh"

void aotx_model_capture_gated(aotx_model_hold *hold, unsigned int role, unsigned int layer)
{
    const aotx_model_desc *desc = &hold->desc;
    aotx_model_work *work = &hold->work;
    cudaStream_t stream = hold->stream;
    unsigned int wave = hold->wave ? hold->wave : hold->max_tokens;
    unsigned int wide = desc->heads * desc->head_dim;
    unsigned int narrow = desc->kv_heads * desc->head_dim;
    aotx_model_norm<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role, layer, AOTX_MODEL_NORM_ATTN);
    aotx_model_hybrid_matrix(hold, layer, AOTX_SLOT_ATTN_Q, 2u * wide, desc->hidden,
                             work->x, work->qgate);
    aotx_model_hybrid_matrix(hold, layer, AOTX_SLOT_ATTN_K, narrow, desc->hidden,
                             work->x, work->k);
    aotx_model_hybrid_matrix(hold, layer, AOTX_SLOT_ATTN_V, narrow, desc->hidden,
                             work->x, work->v);
    dim3 heads(wave, desc->heads + desc->kv_heads);
    aotx_model_gated_qkv<<<heads, 32u, 0, stream>>>(role, layer);
    dim3 tiles((wave + AOTX_MODEL_ATTN_TOKENS - 1u) / AOTX_MODEL_ATTN_TOKENS, desc->heads);
    aotx_model_gated_attend<<<tiles, AOTX_MODEL_ATTN_THREADS, 0, stream>>>(role, layer);
    aotx_model_hybrid_matrix(hold, layer, AOTX_SLOT_ATTN_O, desc->hidden, wide,
                             work->att, work->proj);
    aotx_model_residual<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role);
    aotx_model_hybrid_ffn(hold, role, layer);
}
