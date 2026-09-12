/* Purpose: Put the nodes of the say path in the stream that captures the tick graph.
 * Owns: The addresses of the tokenizer memory of this path.
 * Launch shape: Host glue only; the graph holds the kernels.
 * Lifetime: From the first capture to the end of the run. */
#include <cuda_runtime.h>
#include <stddef.h>

#include "cli/prompt.cuh"
#include "model/forward.cuh"
#ifdef AOTX_AFFECT
#include "affect/affect.cuh"
#endif

/* Blocks that cover the sequence slots with one thread for each slot. */
#define AOTX_SAY_SLOT_THREADS 64u
#define AOTX_SAY_SLOT_BLOCKS  ((AOTX_SLOTS + AOTX_SAY_SLOT_THREADS - 1u) \
                               / AOTX_SAY_SLOT_THREADS)

/* The parts of the tokenizer memory, read once. The addresses of device state do not move,
 * so the search runs one time and every capture takes the same pointers. */
typedef struct aotx_say_parts {
    unsigned char *base;    /* the block of the tokenizer memory */
    unsigned char *prompt;  /* the prompt table, which is the byte run of the batch */
    unsigned int *id;
    unsigned int *count;
    int ready;
} aotx_say_parts;

static aotx_say_parts aotx_say_where;

/* Read the address of each device symbol this path launches with. The return is 0 when
 * every address came back. */
static int aotx_say_find(void)
{
    if (aotx_say_where.ready != 0) {
        return 0;
    }
    void *base = 0;
    void *prompt = 0;
    void *id = 0;
    void *count = 0;
    if (cudaGetSymbolAddress(&base, aotx_say_gear) != cudaSuccess
        || cudaGetSymbolAddress(&prompt, aotx_say) != cudaSuccess
        || cudaGetSymbolAddress(&id, aotx_say_id) != cudaSuccess
        || cudaGetSymbolAddress(&count, aotx_say_count) != cudaSuccess) {
        return 1;
    }
    aotx_say_where.base = (unsigned char *)base;
    /* The prompt table sits inside the state of this path, so its address comes from the
     * state and the place of the table in it. */
    aotx_say_where.prompt = (unsigned char *)prompt + offsetof(aotx_say_state, prompt);
    aotx_say_where.id = (unsigned int *)id;
    aotx_say_where.count = (unsigned int *)count;
    aotx_say_where.ready = 1;
    return 0;
}

/* Give a part of the tokenizer memory from its place in the block. */
static void *aotx_say_part(size_t at)
{
    return (void *)(aotx_say_where.base + at);
}

static aotx_text_batch aotx_say_raw(void)
{
    aotx_text_batch batch;
    batch.bytes = aotx_say_where.prompt;
    batch.start = (const unsigned int *)aotx_say_part(offsetof(aotx_say_work, start));
    batch.length = (const unsigned int *)aotx_say_part(offsetof(aotx_say_work, length));
    batch.count = AOTX_SLOTS;
    return batch;
}

static aotx_text_batch aotx_say_clean_batch(void)
{
    aotx_text_batch batch;
    batch.bytes = (const unsigned char *)aotx_say_part(offsetof(aotx_say_work, clean));
    batch.start = (const unsigned int *)aotx_say_part(offsetof(aotx_say_work, clean_start));
    batch.length = (const unsigned int *)aotx_say_part(offsetof(aotx_say_work,
                                                                clean_length));
    batch.count = AOTX_SLOTS;
    return batch;
}

static aotx_text_pieces aotx_say_pieces(void)
{
    aotx_text_pieces pieces;
    pieces.start = (unsigned int *)aotx_say_part(offsetof(aotx_say_work, piece_start));
    pieces.length = (unsigned int *)aotx_say_part(offsetof(aotx_say_work, piece_length));
    pieces.token = (unsigned int *)aotx_say_part(offsetof(aotx_say_work, piece_token));
    pieces.count = (unsigned int *)aotx_say_part(offsetof(aotx_say_work, piece_count));
    pieces.work = (unsigned int *)aotx_say_part(offsetof(aotx_say_work, work));
    pieces.works = (unsigned int *)aotx_say_part(offsetof(aotx_say_work, works));
    pieces.stride = AOTX_SAY_PIECES;
    return pieces;
}

static aotx_text_tokens aotx_say_tokens(void)
{
    aotx_text_tokens tokens;
    tokens.id = aotx_say_where.id;
    tokens.count = aotx_say_where.count;
    tokens.chunk = (unsigned int *)aotx_say_part(offsetof(aotx_say_work, chunk));
    tokens.scratch = (unsigned int *)aotx_say_part(offsetof(aotx_say_work, scratch));
    tokens.merge = (unsigned char *)aotx_say_part(offsetof(aotx_say_work, merge));
    tokens.warps = AOTX_SAY_WARPS;
    tokens.stride = AOTX_SAY_TOKENS;
    return tokens;
}

/* The nodes that turn the prompts of the tick into sequences. The fill step writes the
 * batch table, the four tokenizer steps give the tokens, and the start step opens each
 * sequence. Every step takes the whole batch of slots, so the shape never changes. */
int aotx_cli_say_capture(void *stream)
{
    if (aotx_say_find() != 0) {
        return 1;
    }
    cudaStream_t on = (cudaStream_t)stream;
    aotx_text_batch raw = aotx_say_raw();
    aotx_text_batch batch = aotx_say_clean_batch();
    aotx_text_pieces pieces = aotx_say_pieces();
    aotx_text_tokens tokens = aotx_say_tokens();

    aotx_media_prepare<<<AOTX_SAY_SLOT_BLOCKS, AOTX_SAY_SLOT_THREADS, 0, on>>>();
    for (unsigned pass = 0; pass < 2u; ++pass) {
        unsigned role = AOTX_MODEL_ROLES + 1u + pass;
        aotx_text_vocab_select<<<1,1,0,on>>>(pass ? 2u : 0u);
    aotx_say_fill<<<AOTX_SAY_SLOT_BLOCKS, AOTX_SAY_SLOT_THREADS, 0, on>>>(role);
    aotx_text_clean<<<AOTX_SAY_SLOT_BLOCKS, AOTX_SAY_SLOT_THREADS, 0, on>>>(
        raw, (unsigned char *)aotx_say_part(offsetof(aotx_say_work, clean)),
        (unsigned int *)aotx_say_part(offsetof(aotx_say_work, clean_start)),
        (unsigned int *)aotx_say_part(offsetof(aotx_say_work, clean_length)),
        AOTX_SAY_CLEAN);
    aotx_text_pretok<<<AOTX_SAY_SLOT_BLOCKS, AOTX_SAY_SLOT_THREADS, 0, on>>>(batch, pieces);
    aotx_text_merge<<<AOTX_SAY_BLOCKS, 32u * AOTX_TEXT_WARPS, 0, on>>>(batch, pieces,
                                                                       tokens);
    aotx_text_gather<<<AOTX_SAY_SLOT_BLOCKS, AOTX_SAY_SLOT_THREADS, 0, on>>>(batch, pieces,
                                                                             tokens);
#ifdef AOTX_AFFECT
    aotx_affect_build<<<AOTX_SLOTS, 256u, 0, on>>>();
#endif
    aotx_say_start<<<AOTX_SAY_SLOT_BLOCKS, AOTX_SAY_SLOT_THREADS, 0, on>>>(role);
    }
    aotx_text_vocab_select<<<1,1,0,on>>>(0);
    return 0;
}

/* The node that puts the new bytes of every live sequence on the console. */
int aotx_cli_reply_capture(void *stream)
{
    cudaStream_t on = (cudaStream_t)stream;
    aotx_say_reply<<<AOTX_SAY_SLOT_BLOCKS, AOTX_SAY_SLOT_THREADS, 0, on>>>();
    return 0;
}
