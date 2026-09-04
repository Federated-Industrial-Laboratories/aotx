/* Purpose: Hold the buffers of a forward pass and capture the pass as one graph.
 * Owns: The buffers, the pinned call block, the stream, the graph and its instance.
 * Launch shape: Host glue only; the graph holds the kernels.
 * Lifetime: From the capture at model load to the close at the end of the run. */
#include <stdio.h>
#include <string.h>

#include "boot/check.h"
#include "mem/mem.cuh"
#include "model/graph_host.h"
#include "model/kinds.h"
#include "rerank/rerank.cuh"

static aotx_model_hold aotx_model_state[AOTX_MODEL_ROLES];

aotx_model_hold *aotx_model_hold_of(unsigned int role)
{
    return (role < AOTX_MODEL_ROLES) ? &aotx_model_state[role] : 0;
}

/* Take one buffer of the pass and keep the pointer, so the close gives every piece back. */
static void *aotx_model_take(aotx_model_hold *hold, unsigned long long bytes)
{
    void *piece = 0;
    aotx_check_runtime(cudaMalloc(&piece, (size_t)bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemset(piece, 0, (size_t)bytes), "cudaMemset");
    hold->piece[hold->pieces++] = piece;
    return piece;
}

/* Size every buffer of one role from its shape and give the block to the device. */
static void aotx_model_buffers(aotx_model_hold *hold, unsigned int role)
{
    const aotx_model_desc *desc = &hold->desc;
    aotx_model_work *work = &hold->work;
    unsigned long long m = hold->max_tokens;
    unsigned long long rows = hold->max_rows;
    unsigned int wide = desc->heads * desc->head_dim;
    unsigned int narrow = desc->kv_heads * desc->head_dim;
    unsigned long long width = aotx_model_is_language(role) ? desc->vocab
                                                             : AOTX_RERANK_CLASSES;
    work->resid = (float *)aotx_model_take(hold, m * desc->hidden * sizeof(float));
    work->x = (half *)aotx_model_take(hold, m * desc->hidden * sizeof(half));
    work->q = (float *)aotx_model_take(hold, m * wide * sizeof(float));
    work->qh = (half *)aotx_model_take(hold, m * wide * sizeof(half));
    work->k = (float *)aotx_model_take(hold, m * narrow * sizeof(float));
    work->v = (float *)aotx_model_take(hold, m * narrow * sizeof(float));
    work->att = (half *)aotx_model_take(hold, m * wide * sizeof(half));
    work->proj = (float *)aotx_model_take(hold, m * desc->hidden * sizeof(float));
    work->gate = (float *)aotx_model_take(hold, m * desc->ffn * sizeof(float));
    work->up = (float *)aotx_model_take(hold, m * desc->ffn * sizeof(float));
    work->act = (half *)aotx_model_take(hold, m * desc->ffn * sizeof(half));
    work->xnorm = (half *)aotx_model_take(hold, m * desc->hidden * sizeof(half));
    work->sel = (half *)aotx_model_take(hold, rows * desc->hidden * sizeof(half));
    work->head = (float *)aotx_model_take(hold,
        (unsigned long long)AOTX_SLOTS * width * sizeof(float));
    work->row = (unsigned int *)aotx_model_take(hold, m * sizeof(unsigned int));
    work->base = (unsigned int *)aotx_model_take(hold,
        AOTX_SLOTS * sizeof(unsigned int));
    work->weights = aotx_mem_weights_base();
    work->max_tokens = hold->max_tokens;
    work->max_rows = hold->max_rows;
    aotx_kvl_make_desc(&work->shape, desc);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_space, work, sizeof *work,
                                          (size_t)role * sizeof *work),
                       "cudaMemcpyToSymbol");
}

/* Name the weight, the block type and the width of the output head of one role. */
static void aotx_model_head_of(aotx_model_hold *hold, unsigned int role)
{
    const aotx_model_desc *desc = &hold->desc;
    unsigned int type[2] = { 0u, 0u };
    aotx_check_runtime(cudaMemcpyFromSymbol(type, aotx_model_head_type, sizeof type,
                                            (size_t)role * sizeof type),
                       "cudaMemcpyFromSymbol");
    (void)type;
    if (aotx_model_is_language(role) == 0) {
        return;
    }
    hold->head_w = aotx_model_tensor(desc->tied_output ? desc->token_embd : desc->output);
    hold->head_type = desc->tied_output ? desc->embd_type : type[0];
    hold->head_n = desc->vocab;
}

int aotx_model_open(unsigned int role, unsigned int max_tokens)
{
    if (role >= AOTX_MODEL_ROLES || max_tokens == 0u || max_tokens > AOTX_MODEL_MAX_TOKENS) {
        return 1;
    }
    aotx_model_hold *hold = &aotx_model_state[role];
    memset(hold, 0, sizeof *hold);
    aotx_check_runtime(cudaMemcpyFromSymbol(&hold->desc, aotx_model, sizeof hold->desc,
                                            (size_t)role * sizeof hold->desc),
                       "cudaMemcpyFromSymbol");
    if (hold->desc.layers == 0u || hold->desc.hidden == 0u) {
        fprintf(stderr, "the model of role %u has no shape\n", role);
        return 1;
    }
    if (aotx_layer_desc_valid(&hold->desc) == 0) {
        fprintf(stderr, "the model of role %u has an invalid layer kind\n", role);
        return 1;
    }
    hold->max_tokens = max_tokens;
    hold->role = role;
    hold->max_rows = aotx_model_is_language(role) ? max_tokens : AOTX_SLOTS;
    aotx_model_buffers(hold, role);
    aotx_model_head_of(hold, role);

    aotx_check_runtime(cudaMallocHost((void **)&hold->pinned, sizeof *hold->pinned),
                       "cudaMallocHost");
    memset(hold->pinned, 0, sizeof *hold->pinned);
    aotx_check_runtime(cudaStreamCreateWithFlags(&hold->stream, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags");
    aotx_check_runtime(cudaEventCreateWithFlags(&hold->event, cudaEventDisableTiming),
                       "cudaEventCreateWithFlags");

    void *call = 0;
    aotx_check_runtime(cudaGetSymbolAddress(&call, aotx_model_call), "cudaGetSymbolAddress");
    call = (void *)((char *)call + (size_t)role * sizeof *hold->pinned);
    aotx_check_runtime(cudaStreamBeginCapture(hold->stream, cudaStreamCaptureModeThreadLocal),
                       "cudaStreamBeginCapture");
    aotx_check_runtime(cudaMemcpyAsync(call, hold->pinned, sizeof *hold->pinned,
                                       cudaMemcpyHostToDevice, hold->stream),
                       "cudaMemcpyAsync");
    aotx_model_open_rows<<<1, AOTX_SLOTS, 0, hold->stream>>>(role);
    aotx_model_gather<<<max_tokens, AOTX_MODEL_ROW_THREADS, 0, hold->stream>>>(role);
    for (unsigned int l = 0u; l < hold->desc.layers; ++l) {
        aotx_model_capture_layer(hold, role, l);
    }
    aotx_model_capture_head(hold, role);
    aotx_model_shut_rows<<<1, AOTX_SLOTS, 0, hold->stream>>>(role);
    aotx_check_runtime(cudaStreamEndCapture(hold->stream, &hold->graph),
                       "cudaStreamEndCapture");
    aotx_check_runtime(cudaGraphInstantiate(&hold->exec, hold->graph, 0),
                       "cudaGraphInstantiate");
    if (aotx_model_is_language(role) && hold->head_node == 0) {
        fprintf(stderr, "the graph of role %u has no head node\n", role);
        return 1;
    }
    hold->ready = 1u;
    return 0;
}

int aotx_model_launch(unsigned int role, const aotx_model_run *set)
{
    aotx_model_hold *hold = aotx_model_hold_of(role);
    if (hold == 0 || hold->ready == 0u || set->tokens > hold->max_tokens
        || set->rows > hold->max_rows || set->seqs > AOTX_SLOTS) {
        return 1;
    }
    if (set->tokens == 0u || set->seqs == 0u || set->rows == 0u) {
        /* An empty batch gives a grid of no block, which no launch takes. */
        return 1;
    }
    *hold->pinned = *set;
    if (hold->head_node != 0) {
        /* The head takes every row of the batch or one row of each sequence. The row count
         * moves the grid of that one node, and the result goes where the caller asks. */
        hold->head_m = set->rows;
        hold->head_y = (set->logits != 0) ? set->logits : hold->work.head;
        if (hold->head_y == hold->work.head && set->rows > AOTX_SLOTS) {
            /* The buffer of the module holds one row for each sequence. A caller which
             * wants a row for every token gives a buffer of its own. */
            return 1;
        }
        void *args[7];
        args[0] = &hold->head_w;
        args[1] = &hold->head_type;
        args[2] = &hold->head_n;
        args[3] = &hold->head_k;
        args[4] = &hold->head_x;
        args[5] = &hold->head_m;
        args[6] = &hold->head_y;
        cudaKernelNodeParams params;
        memset(&params, 0, sizeof params);
        params.func = (void *)aotx_model_gemm;
        params.gridDim = dim3(aotx_model_tiles_n(hold->head_n),
                              aotx_model_tiles_m(hold->head_m), 1);
        params.blockDim = dim3(AOTX_GEMM_THREADS, 1, 1);
        params.kernelParams = args;
        aotx_check_runtime(cudaGraphExecKernelNodeSetParams(hold->exec, hold->head_node,
                                                            &params),
                           "cudaGraphExecKernelNodeSetParams");
    }
    aotx_check_runtime(cudaGraphLaunch(hold->exec, hold->stream), "cudaGraphLaunch");
    aotx_check_runtime(cudaEventRecord(hold->event, hold->stream), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(hold->event), "cudaEventSynchronize");
    return 0;
}

int aotx_model_pages(unsigned int role, const unsigned int *offset, unsigned int seqs,
                     const unsigned int *agent)
{
    aotx_model_hold *hold = aotx_model_hold_of(role);
    if (hold == 0 || hold->ready == 0u || seqs > AOTX_SLOTS) {
        return 1;
    }
    memset(hold->pinned, 0, sizeof *hold->pinned);
    hold->pinned->offset = offset;
    hold->pinned->agent = agent;
    hold->pinned->seqs = seqs;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, hold->pinned,
                                          sizeof *hold->pinned,
                                          (size_t)role * sizeof *hold->pinned),
                       "cudaMemcpyToSymbol");
    aotx_model_request<<<1, AOTX_SLOTS, 0, hold->stream>>>(role);
    aotx_check_runtime(cudaStreamSynchronize(hold->stream), "cudaStreamSynchronize");
    return 0;
}

void aotx_model_forget(void)
{
    unsigned int seen[AOTX_SLOTS];
    unsigned int none = 0u;
    memset(seen, 0, sizeof seen);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_seen, seen, sizeof seen),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_faults, &none, sizeof none),
                       "cudaMemcpyToSymbol");
}

void aotx_model_restream(void)
{
    unsigned int draw[AOTX_SLOTS];
    memset(draw, 0, sizeof draw);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_draw, draw, sizeof draw),
                       "cudaMemcpyToSymbol");
}

unsigned int aotx_model_nodes(unsigned int role)
{
    aotx_model_hold *hold = aotx_model_hold_of(role);
    size_t count = 0;
    if (hold == 0 || hold->graph == 0) {
        return 0u;
    }
    aotx_check_runtime(cudaGraphGetNodes(hold->graph, 0, &count), "cudaGraphGetNodes");
    return (unsigned int)count;
}

unsigned int aotx_model_faulted(void)
{
    unsigned int count = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&count, aotx_model_faults, sizeof count),
                       "cudaMemcpyFromSymbol");
    return count;
}

void aotx_model_shut(unsigned int role)
{
    aotx_model_hold *hold = aotx_model_hold_of(role);
    if (hold == 0) {
        return;
    }
    if (hold->exec != 0) {
        cudaGraphExecDestroy(hold->exec);
    }
    if (hold->graph != 0) {
        cudaGraphDestroy(hold->graph);
    }
    if (hold->event != 0) {
        cudaEventDestroy(hold->event);
    }
    if (hold->stream != 0) {
        cudaStreamDestroy(hold->stream);
    }
    if (hold->pinned != 0) {
        cudaFreeHost(hold->pinned);
    }
    for (unsigned int i = 0u; i < hold->pieces; ++i) {
        cudaFree(hold->piece[i]);
    }
    memset(hold, 0, sizeof *hold);
}
