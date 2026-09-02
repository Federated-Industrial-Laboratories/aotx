/* Purpose: Capture the embedding pass and the tool nodes into the tick stream.
 * Owns: The child graph of the embedding pass and the addresses of the tool memory.
 * Launch shape: Host glue only; the graphs hold the kernels.
 * Lifetime: From the first capture to the close at the end of the run. */
#include <cuda_runtime.h>
#include <stddef.h>

#include "boot/check.h"
#include "model/graph_host.h"
#include "tool/module.cuh"
#include "tool/tool_state.cuh"
#ifdef AOTX_AFFECT
#include "quality/quality.cuh"
#endif

/* Blocks of a launch of the embedding pass that takes a run of rows. */
#define AOTX_TOOL_WAVE  64u

static cudaGraph_t aotx_tool_pass;

/* The parts of the tool memory, read once. The addresses of device state do not move, so
 * the search runs one time and every capture takes the same pointers. */
typedef struct aotx_tool_parts {
    unsigned char *gear;    /* the block of the tool memory */
    unsigned char *batch;   /* the batch of the embedding pass */
    int ready;
} aotx_tool_parts;

static aotx_tool_parts aotx_tool_where;

static int aotx_tool_find(void)
{
    if (aotx_tool_where.ready != 0) {
        return 0;
    }
    void *gear = 0;
    void *batch = 0;
    if (cudaGetSymbolAddress(&gear, aotx_tool_gear) != cudaSuccess
        || cudaGetSymbolAddress(&batch, aotx_tool_embed) != cudaSuccess) {
        return 1;
    }
    aotx_tool_where.gear = (unsigned char *)gear;
    aotx_tool_where.batch = (unsigned char *)batch;
    aotx_tool_where.ready = 1;
    return 0;
}

static void *aotx_tool_part(size_t at)
{
    return (void *)(aotx_tool_where.gear + at);
}

static void *aotx_tool_field(size_t at)
{
    return (void *)(aotx_tool_where.batch + at);
}

static aotx_text_batch aotx_tool_raw(void)
{
    aotx_text_batch batch;
    batch.bytes = (const unsigned char *)aotx_tool_part(offsetof(aotx_tool_work, text));
    batch.start = (const unsigned int *)aotx_tool_part(offsetof(aotx_tool_work, start));
    batch.length = (const unsigned int *)aotx_tool_part(offsetof(aotx_tool_work, length));
    batch.count = AOTX_TOOL_BATCH_ROWS;
    return batch;
}

static aotx_text_batch aotx_tool_clean_batch(void)
{
    aotx_text_batch batch;
    batch.bytes = (const unsigned char *)aotx_tool_part(offsetof(aotx_tool_work, clean));
    batch.start = (const unsigned int *)aotx_tool_part(offsetof(aotx_tool_work,
                                                                clean_start));
    batch.length = (const unsigned int *)aotx_tool_part(offsetof(aotx_tool_work,
                                                                 clean_length));
    batch.count = AOTX_TOOL_BATCH_ROWS;
    return batch;
}

static aotx_text_pieces aotx_tool_pieces(void)
{
    aotx_text_pieces pieces;
    pieces.start = (unsigned int *)aotx_tool_part(offsetof(aotx_tool_work, piece_start));
    pieces.length = (unsigned int *)aotx_tool_part(offsetof(aotx_tool_work, piece_length));
    pieces.token = (unsigned int *)aotx_tool_part(offsetof(aotx_tool_work, piece_token));
    pieces.count = (unsigned int *)aotx_tool_part(offsetof(aotx_tool_work, piece_count));
    pieces.work = (unsigned int *)aotx_tool_part(offsetof(aotx_tool_work, work));
    pieces.works = (unsigned int *)aotx_tool_part(offsetof(aotx_tool_work, works));
    pieces.stride = AOTX_TOOL_TOKEN_STRIDE;
    return pieces;
}

static aotx_text_tokens aotx_tool_tokens(void)
{
    aotx_text_tokens tokens;
    tokens.id = (unsigned int *)aotx_tool_part(offsetof(aotx_tool_work, id));
    tokens.count = (unsigned int *)aotx_tool_part(offsetof(aotx_tool_work, count));
    tokens.chunk = (unsigned int *)aotx_tool_part(offsetof(aotx_tool_work, chunk));
    tokens.scratch = (unsigned int *)aotx_tool_part(offsetof(aotx_tool_work, scratch));
    tokens.merge = (unsigned char *)aotx_tool_part(offsetof(aotx_tool_work, merge));
    tokens.warps = AOTX_TOOL_WARPS;
    tokens.stride = AOTX_TOOL_TOKEN_STRIDE;
    return tokens;
}

static aotx_embed_query aotx_tool_query(unsigned int width)
{
    aotx_embed_query set;
    set.vector = (const float *)aotx_tool_field(offsetof(aotx_tool_embed_batch, vector));
    set.live = (const unsigned int *)aotx_tool_field(offsetof(aotx_tool_embed_batch, live));
    set.hit = (unsigned int *)aotx_tool_field(offsetof(aotx_tool_embed_batch, hit));
    set.score = (float *)aotx_tool_field(offsetof(aotx_tool_embed_batch, score));
    set.count = (const unsigned int *)aotx_tool_field(offsetof(aotx_tool_embed_batch, seqs));
    set.width = width;
    return set;
}

int aotx_tool_open(void)
{
    unsigned int role = AOTX_MODEL_EMBEDDING;
    aotx_model_hold *hold = aotx_model_hold_of(role);
    if (hold == 0) {
        return 1;
    }
    if (hold->ready == 0u) {
        /* A run with no embedding file takes the tool step alone. The descriptor says so
         * before the open runs, so a run with no model writes no message about it. */
        unsigned int layers = 0u;
        aotx_check_runtime(cudaMemcpyFromSymbol(&layers, aotx_model, sizeof layers,
                                                (size_t)role * sizeof(aotx_model_desc)
                                                + offsetof(aotx_model_desc, layers)),
                           "cudaMemcpyFromSymbol");
        if (layers == 0u || aotx_model_open(role, AOTX_MODEL_MAX_TOKENS) != 0) {
            return 1;
        }
    }
    hold = aotx_model_hold_of(role);
    if (hold == 0 || hold->ready == 0u || hold->desc.hidden > AOTX_EMBED_WIDTH) {
        return 1;
    }
    aotx_model_batch_of(role);
    if (aotx_tool_pass != 0) {
        cudaGraphDestroy(aotx_tool_pass);
        aotx_tool_pass = 0;
    }

    /* The pass is captured as a graph of its own and it holds no copy node. The plan of
     * the tool path writes the call block on the device. Every matrix node takes its batch
     * from that block, so a tick with no text runs the pass at no rows. */
    hold->decode = 1u;
    hold->wave = AOTX_TOOL_WAVE;
    aotx_check_runtime(cudaStreamBeginCapture(hold->stream,
                                              cudaStreamCaptureModeThreadLocal),
                       "cudaStreamBeginCapture");
    aotx_model_open_rows<<<1, AOTX_SLOTS, 0, hold->stream>>>(role);
    aotx_model_gather<<<AOTX_TOOL_WAVE, AOTX_MODEL_ROW_THREADS, 0, hold->stream>>>(role);
    for (unsigned int l = 0u; l < hold->desc.layers; ++l) {
        aotx_model_capture_layer(hold, role, l);
    }
    aotx_model_capture_head(hold, role);
    aotx_model_shut_rows<<<1, AOTX_SLOTS, 0, hold->stream>>>(role);
    aotx_check_runtime(cudaStreamEndCapture(hold->stream, &aotx_tool_pass),
                       "cudaStreamEndCapture");
    hold->decode = 0u;
    hold->wave = 0u;
    if (aotx_tool_pass == 0) {
        return 1;
    }

    unsigned int width = hold->desc.hidden;
    unsigned int ready = 1u;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_tool_embed, &role, sizeof role,
                                          offsetof(aotx_tool_embed_batch, role)),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_tool_embed, &width, sizeof width,
                                          offsetof(aotx_tool_embed_batch, width)),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_tool_embed, &ready, sizeof ready,
                                          offsetof(aotx_tool_embed_batch, ready)),
                       "cudaMemcpyToSymbol");
    return 0;
}

unsigned int aotx_tool_pass_nodes(void)
{
    size_t count = 0;
    if (aotx_tool_pass == 0) {
        return 0u;
    }
    aotx_check_runtime(cudaGraphGetNodes(aotx_tool_pass, 0, &count), "cudaGraphGetNodes");
    return (unsigned int)count;
}

void aotx_tool_close(void)
{
    aotx_tool_module_close();
    if (aotx_tool_pass != 0) {
        cudaGraphDestroy(aotx_tool_pass);
        aotx_tool_pass = 0;
    }
    aotx_tool_where.ready = 0;
}

/* The nodes of the tool path. The fill step writes the batch table of the tokenizer and
 * the rows of every module node. The four tokenizer steps give the tokens and the plan
 * writes the call block. The pass runs as one child node and the search reads the note
 * store. One node for each device tool module follows, and the step gives every result. A
 * run with no embedding role holds the fill, the module nodes and the step. */
int aotx_tool_capture(void *stream)
{
    cudaStream_t on = (cudaStream_t)stream;
    unsigned int width = 0u;
    if (aotx_tool_find() != 0 || aotx_tool_pass == 0) {
#ifdef AOTX_AFFECT
        aotx_tool_fill<<<AOTX_SLOTS, AOTX_QUALITY_FILL_THREADS, 0, on>>>();
#else
        aotx_tool_fill<<<AOTX_TOOL_SLOT_BLOCKS, AOTX_TOOL_SLOT_THREADS, 0, on>>>();
#endif
        aotx_tool_module_capture(on);
        aotx_tool_step<<<AOTX_TOOL_SLOT_BLOCKS, AOTX_TOOL_SLOT_THREADS, 0, on>>>(0ull);
        return 1;
    }
    aotx_model_hold *hold = aotx_model_hold_of(AOTX_MODEL_EMBEDDING);
    width = (hold != 0) ? hold->desc.hidden : 0u;

    aotx_text_batch raw = aotx_tool_raw();
    aotx_text_batch batch = aotx_tool_clean_batch();
    aotx_text_pieces pieces = aotx_tool_pieces();
    aotx_text_tokens tokens = aotx_tool_tokens();

#ifdef AOTX_AFFECT
    aotx_tool_fill<<<AOTX_SLOTS, AOTX_QUALITY_FILL_THREADS, 0, on>>>();
#else
    aotx_tool_fill<<<AOTX_TOOL_SLOT_BLOCKS, AOTX_TOOL_SLOT_THREADS, 0, on>>>();
#endif
    aotx_text_clean<<<AOTX_TOOL_TEXT_BLOCKS, AOTX_TOOL_TEXT_THREADS, 0, on>>>(
        raw, (unsigned char *)aotx_tool_part(offsetof(aotx_tool_work, clean)),
        (unsigned int *)aotx_tool_part(offsetof(aotx_tool_work, clean_start)),
        (unsigned int *)aotx_tool_part(offsetof(aotx_tool_work, clean_length)),
        AOTX_TOOL_CLEAN_STRIDE);
    aotx_text_pretok<<<AOTX_TOOL_TEXT_BLOCKS, AOTX_TOOL_TEXT_THREADS, 0, on>>>(batch,
                                                                               pieces);
    aotx_text_merge<<<AOTX_TOOL_BLOCKS, 32u * AOTX_TEXT_WARPS, 0, on>>>(batch, pieces,
                                                                        tokens);
    aotx_text_gather<<<AOTX_TOOL_TEXT_BLOCKS, AOTX_TOOL_TEXT_THREADS, 0, on>>>(batch,
                                                                               pieces,
                                                                               tokens);
    aotx_tool_plan<<<1, AOTX_SLOTS, 0, on>>>(0ull);

    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaGraph_t graph = 0;
    const cudaGraphNode_t *depends = 0;
    size_t held = 0;
    aotx_check_runtime(cudaStreamGetCaptureInfo(on, &status, 0, &graph, &depends, 0, &held),
                       "cudaStreamGetCaptureInfo");
    if (status != cudaStreamCaptureStatusActive) {
        return 1;
    }
    cudaGraphNode_t child = 0;
    aotx_check_runtime(cudaGraphAddChildGraphNode(&child, graph, depends, held,
                                                  aotx_tool_pass),
                       "cudaGraphAddChildGraphNode");
    aotx_check_runtime(cudaStreamUpdateCaptureDependencies(on, &child, 0, 1u,
                                                           cudaStreamSetCaptureDependencies),
                       "cudaStreamUpdateCaptureDependencies");

    aotx_embed_search<<<AOTX_SLOTS, AOTX_MODEL_ROW_THREADS, 0, on>>>(
        aotx_tool_query(width));
    /* One node for each device tool module of the catalog. The node stands between the
     * fill, which wrote its rows, and the step, which reads its output. */
    aotx_tool_module_capture(on);
    aotx_tool_step<<<AOTX_TOOL_SLOT_BLOCKS, AOTX_TOOL_SLOT_THREADS, 0, on>>>(0ull);
    return 0;
}
