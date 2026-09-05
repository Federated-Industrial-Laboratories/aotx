/* Purpose: Allocate fixed recurrent state and capture batch-stable matrix operations.
 * Owns: One contiguous device allocation in the model hold.
 * Launch shape: Host glue; allocation precedes graph capture.
 * Lifetime: From model open to model close. */
#include <stdio.h>
#include <string.h>

#include "model/graph_host.h"
#include "model/hybrid.cuh"

static void *aotx_hybrid_take(char *base, unsigned long long *at,
                              unsigned long long bytes)
{
    *at = (*at + 255ull) & ~255ull;
    void *result = base ? base + *at : NULL;
    *at += bytes;
    return result;
}

static unsigned long long aotx_hybrid_layout(const aotx_model_desc *desc,
                                              unsigned int tokens, char *base,
                                              aotx_model_work *work)
{
    unsigned int layers = aotx_layer_state_count(desc, AOTX_STATE_KIND_DELTA_STATE);
    unsigned int gated = 0u;
    for (unsigned int i = 0u; i < desc->layers; ++i)
        gated += desc->kind[i] == AOTX_LAYER_KIND_ATTENTION_GATED;
    if (layers == 0u && gated == 0u) return 0ull;
    unsigned long long at = 0ull;
    unsigned long long m = tokens;
    unsigned long long inner = desc->delta_inner;
    unsigned long long channels = 2ull * desc->delta_key_heads * desc->delta_dim + inner;
    aotx_delta_work *delta = &work->delta;
    memset(delta->layer, 0xff, sizeof delta->layer);
    delta->layers = layers;
    unsigned int compact = 0u;
    for (unsigned int i = 0u; i < desc->layers; ++i) {
        const aotx_layer_kind *kind = aotx_layer_kind_of(desc->kind[i]);
        if (kind->state == AOTX_STATE_KIND_DELTA_STATE)
            delta->layer[i] = (unsigned char)compact++;
    }
    delta->state_elements = (unsigned long long)AOTX_SLOTS * layers
                         * desc->delta_heads * desc->delta_dim * desc->delta_dim;
    delta->history_elements = (unsigned long long)AOTX_SLOTS * layers
                           * channels * (desc->delta_conv - 1u);
    if (layers != 0u) {
        delta->state = (float *)aotx_hybrid_take(base, &at, delta->state_elements * sizeof(float));
        delta->history = (float *)aotx_hybrid_take(base, &at, delta->history_elements * sizeof(float));
        delta->qkv = (float *)aotx_hybrid_take(base, &at, m * channels * sizeof(float));
        delta->z = (float *)aotx_hybrid_take(base, &at, m * inner * sizeof(float));
        delta->alpha = (float *)aotx_hybrid_take(base, &at, m * desc->delta_heads * sizeof(float));
        delta->beta = (float *)aotx_hybrid_take(base, &at, m * desc->delta_heads * sizeof(float));
        delta->conv = (float *)aotx_hybrid_take(base, &at, m * channels * sizeof(float));
        delta->out = (float *)aotx_hybrid_take(base, &at, m * inner * sizeof(float));
        delta->act = (half *)aotx_hybrid_take(base, &at, m * inner * sizeof(half));
    }
    if (gated != 0u)
        work->qgate = (float *)aotx_hybrid_take(base, &at,
            m * 2ull * desc->heads * desc->head_dim * sizeof(float));
    return at;
}

unsigned long long aotx_model_hybrid_bytes(const aotx_model_desc *desc,
                                           unsigned int tokens)
{
    aotx_model_work work = {};
    return aotx_hybrid_layout(desc, tokens, NULL, &work);
}

int aotx_model_hybrid_open(aotx_model_hold *hold)
{
    unsigned long long bytes = aotx_model_hybrid_bytes(&hold->desc, hold->max_tokens);
    if (bytes == 0ull) return 0;
    cudaError_t status = cudaMalloc(&hold->hybrid_piece, (size_t)bytes);
    if (status == cudaSuccess)
        status = cudaMemset(hold->hybrid_piece, 0, (size_t)bytes);
    if (status != cudaSuccess) {
        fprintf(stderr, "the fixed state and buffers need %llu bytes: %s\n",
                bytes, cudaGetErrorString(status));
        if (hold->hybrid_piece) cudaFree(hold->hybrid_piece);
        hold->hybrid_piece = NULL;
        return 1;
    }
    aotx_hybrid_layout(&hold->desc, hold->max_tokens, (char *)hold->hybrid_piece, &hold->work);
    return 0;
}

void aotx_model_hybrid_matrix(aotx_model_hold *hold, unsigned int layer,
                               unsigned int slot, unsigned int n, unsigned int k,
                               const half *x, float *y)
{
    const aotx_model_desc *desc = &hold->desc;
    const void *weight = aotx_model_tensor(desc->layer[layer].offset[slot]);
    unsigned int type = desc->layer_type[layer][slot];
    unsigned int blocks = (n + AOTX_GEMV_ROWS_CTA - 1u) / AOTX_GEMV_ROWS_CTA;
    aotx_model_hybrid_product<<<blocks, AOTX_GEMV_THREADS, 0, hold->stream>>>(
        hold->role, weight, type, n, k, x, y);
}

void aotx_model_hybrid_ffn(aotx_model_hold *hold, unsigned int role, unsigned int layer)
{
    const aotx_model_desc *desc = &hold->desc;
    aotx_model_work *work = &hold->work;
    cudaStream_t stream = hold->stream;
    unsigned int wave = hold->wave ? hold->wave : hold->max_tokens;
    aotx_model_norm<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role, layer, AOTX_MODEL_NORM_FFN);
    aotx_model_hybrid_matrix(hold, layer, AOTX_SLOT_FFN_GATE, desc->ffn, desc->hidden,
                             work->x, work->gate);
    aotx_model_hybrid_matrix(hold, layer, AOTX_SLOT_FFN_UP, desc->ffn, desc->hidden,
                             work->x, work->up);
    aotx_model_swiglu<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role);
    aotx_model_hybrid_matrix(hold, layer, AOTX_SLOT_FFN_DOWN, desc->hidden, desc->ffn,
                             work->act, work->proj);
    aotx_model_residual<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role);
    aotx_model_conduct<<<wave, AOTX_MODEL_ROW_THREADS, 0, stream>>>(role, layer);
}
