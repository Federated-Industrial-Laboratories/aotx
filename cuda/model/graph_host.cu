/* Purpose: Put the nodes of the forward pass in the capture, layer by layer.
 * Owns: Nothing; the state of the role belongs to model_host.cu.
 * Launch shape: Host glue only; the graph holds the kernels.
 * Lifetime: One graph capture. */
#include "embed/embed.cuh"
#include "mem/mem.cuh"
#include "model/graph_host.h"
#include "rerank/rerank.cuh"

const void *aotx_model_tensor(unsigned long long at)
{
    if (at == AOTX_MODEL_ABSENT) {
        return 0;
    }
    return (const void *)(aotx_mem_weights_base() + at);
}

void aotx_model_matrix(aotx_model_hold *hold, const void *w, unsigned int type,
                       unsigned int n, unsigned int k, const half *x, unsigned int m,
                       float *y, unsigned int which)
{
    cudaStream_t stream = hold->stream;
    if (hold->decode == 0u) {
        dim3 grid(aotx_model_tiles_n(n), aotx_model_tiles_m(m), 1);
        aotx_model_gemm<<<grid, AOTX_GEMM_THREADS, 0, stream>>>(w, type, n, k, x, m, y);
        return;
    }

    /* The batch of a tick is not known at the capture, so each grid holds the largest
     * batch that its product takes. A node whose batch is not its own exits at once. */
    dim3 tile(aotx_model_tiles_n(n), aotx_model_tiles_m(m), 1);
    dim3 line((n + AOTX_GEMV_ROWS_CTA - 1u) / AOTX_GEMV_ROWS_CTA, 1, 1);
    const unsigned int *batch = aotx_decode_batch_word(hold->role, which);
    unsigned int module = aotx_model_module_node(hold, w, type, n, k, x, y, line, which);
    aotx_model_line<<<line, AOTX_GEMV_THREADS, 0, stream>>>(batch, w, type, n, k, x, y,
                                                            module);
    aotx_model_product<<<tile, AOTX_GEMM_THREADS, 0, stream>>>(batch, w, type, n, k, x, y);
}

void aotx_model_capture_layer(aotx_model_hold *hold, unsigned int role, unsigned int l)
{
    const aotx_model_desc *desc = &hold->desc;
    aotx_model_work *work = &hold->work;
    cudaStream_t s = hold->stream;
    unsigned int m = hold->max_tokens;
    unsigned int wave = (hold->wave == 0u) ? hold->max_tokens : hold->wave;
    unsigned int wide = desc->heads * desc->head_dim;
    unsigned int narrow = desc->kv_heads * desc->head_dim;
    unsigned int type = desc->weight_type;

    aotx_model_norm<<<wave, AOTX_MODEL_ROW_THREADS, 0, s>>>(role, l, AOTX_MODEL_NORM_ATTN);
    aotx_model_matrix(hold, aotx_model_tensor(desc->layer[l].attn_q), type, wide,
                      desc->hidden, work->x, m, work->q, AOTX_MODEL_BATCH_TOKENS);
    aotx_model_matrix(hold, aotx_model_tensor(desc->layer[l].attn_k), type, narrow,
                      desc->hidden, work->x, m, work->k, AOTX_MODEL_BATCH_TOKENS);
    aotx_model_matrix(hold, aotx_model_tensor(desc->layer[l].attn_v), type, narrow,
                      desc->hidden, work->x, m, work->v, AOTX_MODEL_BATCH_TOKENS);
    dim3 heads(wave, desc->heads + desc->kv_heads, 1);

    /* The head norm adds over the block with a warp exchange. The block therefore holds a
     * whole warp, even when a head has fewer pairs than a warp has lanes. */
    unsigned int pairs = desc->head_dim / 2u;
    aotx_model_qkv<<<heads, (pairs < 32u) ? 32u : pairs, 0, s>>>(role, l);
    dim3 tiles((wave + AOTX_MODEL_ATTN_TOKENS - 1u) / AOTX_MODEL_ATTN_TOKENS,
               desc->heads, 1);
    aotx_model_attend<<<tiles, AOTX_MODEL_ATTN_THREADS, 0, s>>>(role, l);
    aotx_model_matrix(hold, aotx_model_tensor(desc->layer[l].attn_o), type, desc->hidden,
                      wide, work->att, m, work->proj, AOTX_MODEL_BATCH_TOKENS);
    aotx_model_residual<<<wave, AOTX_MODEL_ROW_THREADS, 0, s>>>(role);
    aotx_model_norm<<<wave, AOTX_MODEL_ROW_THREADS, 0, s>>>(role, l, AOTX_MODEL_NORM_FFN);
    aotx_model_matrix(hold, aotx_model_tensor(desc->layer[l].ffn_gate), type, desc->ffn,
                      desc->hidden, work->x, m, work->gate, AOTX_MODEL_BATCH_TOKENS);
    aotx_model_matrix(hold, aotx_model_tensor(desc->layer[l].ffn_up), type, desc->ffn,
                      desc->hidden, work->x, m, work->up, AOTX_MODEL_BATCH_TOKENS);
    aotx_model_swiglu<<<wave, AOTX_MODEL_ROW_THREADS, 0, s>>>(role);
    aotx_model_matrix(hold, aotx_model_tensor(desc->layer[l].ffn_down), type,
                      desc->hidden, desc->ffn, work->act, m, work->proj,
                      AOTX_MODEL_BATCH_TOKENS);
    aotx_model_residual<<<wave, AOTX_MODEL_ROW_THREADS, 0, s>>>(role);
}

/* The embedding head reads the residual stream, so that role needs no last norm and no row
 * list. The rank head reads the same rows, because its two class rows are short enough to
 * run in single precision. The language role takes the last norm, the row list and the
 * output head. */
void aotx_model_capture_head(aotx_model_hold *hold, unsigned int role)
{
    const aotx_model_desc *desc = &hold->desc;
    cudaStream_t s = hold->stream;
    if (role == AOTX_MODEL_EMBEDDING) {
        aotx_embed_pool<<<AOTX_SLOTS, AOTX_MODEL_ROW_THREADS, 0, s>>>(role);
        return;
    }
    if (role == AOTX_MODEL_RERANKER) {
        aotx_rerank_score<<<AOTX_SLOTS, AOTX_MODEL_ROW_THREADS, 0, s>>>(role);
        return;
    }
    unsigned int wave = (hold->wave == 0u) ? hold->max_tokens : hold->wave;

    /* The decode takes the last row of each sequence. Its head therefore holds one row
     * for each slot of the page cache and never one row for each token. */
    unsigned int rows = (hold->decode != 0u) ? AOTX_SLOTS : hold->max_rows;
    aotx_model_norm<<<wave, AOTX_MODEL_ROW_THREADS, 0, s>>>(role, 0u, AOTX_MODEL_NORM_OUT);
    aotx_model_select<<<rows, AOTX_MODEL_ROW_THREADS, 0, s>>>(role);
    hold->head_k = desc->hidden;
    hold->head_x = hold->work.sel;
    hold->head_m = rows;
    hold->head_y = hold->work.head;
    aotx_model_matrix(hold, hold->head_w, hold->head_type, hold->head_n, hold->head_k,
                      hold->head_x, hold->head_m, hold->head_y, AOTX_MODEL_BATCH_ROWS);

    /* The node that was launched last is the one dependency the capture holds. The handle
     * of the head node therefore comes from the capture and not from a search. The graph
     * of the decode takes its row count from the call block, so it keeps no such node. */
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    const cudaGraphNode_t *depends = 0;
    size_t held = 0;
    if (hold->decode == 0u
        && cudaStreamGetCaptureInfo(s, &status, 0, 0, &depends, 0, &held) == cudaSuccess
        && held == 1u) {
        hold->head_node = depends[0];
    }
    aotx_model_pick<<<rows, AOTX_MODEL_ROW_THREADS, 0, s>>>(role);
}
