/* Purpose: Capture the fixed-state delta layer and its feed-forward nodes.
 * Owns: Nothing; the model hold owns the weights, stream, and storage.
 * Launch shape: Host glue; nodes read the token and sequence counts from the call block.
 * Lifetime: One graph capture at model open. */
#include "model/graph_host.h"
#include "model/hybrid.cuh"
#include "model/kinds.h"
#include "model/delta.cuh"

void aotx_model_capture_delta(aotx_model_hold *hold, unsigned int role, unsigned int layer)
{
    const aotx_model_desc *desc = &hold->desc;
    aotx_model_work *work = &hold->work;
    aotx_delta_work *delta = &work->delta;
    unsigned int tokens = hold->max_tokens;
    unsigned int wave = hold->wave == 0u ? tokens : hold->wave;
    unsigned int channels = 2u * desc->delta_key_heads * desc->delta_dim + desc->delta_inner;
    cudaStream_t stream = hold->stream;
    aotx_model_norm<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(
        role, layer, AOTX_MODEL_NORM_ATTN);
    aotx_model_hybrid_matrix(hold, layer, AOTX_DELTA_QKV, channels,
                              desc->hidden, work->x, delta->qkv);
    aotx_model_hybrid_matrix(hold, layer, AOTX_DELTA_GATE, desc->delta_inner,
                              desc->hidden, work->x, delta->z);
    aotx_model_hybrid_matrix(hold, layer, AOTX_DELTA_ALPHA, desc->delta_heads,
                              desc->hidden, work->x, delta->alpha);
    aotx_model_hybrid_matrix(hold, layer, AOTX_DELTA_BETA, desc->delta_heads,
                              desc->hidden, work->x, delta->beta);
    aotx_model_delta_conv<<<dim3(AOTX_SLOTS, (channels + AOTX_DELTA_THREADS - 1u) /
                                 AOTX_DELTA_THREADS), AOTX_DELTA_THREADS, 0, stream>>>(role, layer);
    aotx_model_delta_qk<<<dim3(tokens, 2u * desc->delta_key_heads), 32u, 0, stream>>>(role, layer);
    aotx_model_delta_scan<<<dim3(AOTX_SLOTS, desc->delta_heads,
                                 (desc->delta_dim + AOTX_DELTA_VALUES - 1u) / AOTX_DELTA_VALUES),
                            AOTX_DELTA_THREADS, 0, stream>>>(role, layer);
    aotx_model_delta_gate<<<dim3(tokens, desc->delta_heads), 32u, 0, stream>>>(role, layer);
    aotx_model_hybrid_matrix(hold, layer, AOTX_DELTA_OUT, desc->hidden,
                              desc->delta_inner, delta->act, work->proj);
    aotx_model_residual<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role);
    aotx_model_hybrid_ffn(hold, role, layer);
}
