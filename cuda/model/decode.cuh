/* Purpose: Run the language model for every live sequence in each tick.
 * Owns: The sequence table and the plan of the tick.
 * Launch shape: One thread for each sequence in the plan; the forward pass batches every token.
 * Lifetime: The whole run. */
#ifndef DECODE_CUH
#define DECODE_CUH

#include "profile/profile.cuh"
#include "model/forward.cuh"
#include "seam/wire.h"

/* The sequence slots, the tokens of one sequence and the token budget of one tick come
 * from the profile of the build. The reply limit and the budget are settings; the values
 * below are the bounds the tables hold. */
#define AOTX_SEQ_TICK_BUDGET   512u   /* prompt tokens the plan admits in one tick, all slots */

/* Sequence states. A slot in the DONE state keeps its pages until the release the next tick
 * makes. The SEQUENCE record events in wire.h are a different list. */
#define AOTX_SEQ_STATE_FREE    0u
#define AOTX_SEQ_STATE_PREFILL 1u
#define AOTX_SEQ_STATE_DECODE  2u
#define AOTX_SEQ_STATE_DONE    3u

typedef struct aotx_seq {
    unsigned int state;         /* AOTX_SEQ_STATE_* */
    unsigned int role;          /* the model role */
    unsigned int prompt;        /* prompt tokens */
    unsigned int held;          /* tokens in the key value cache, prompt and reply */
    unsigned int sampled;       /* reply tokens made so far */
    unsigned int limit;         /* reply tokens allowed */
    unsigned int page_limit;    /* pages this sequence may hold */
    unsigned int stop;          /* the token that ends the reply */
    aotx_model_how sample;      /* sampling row fixed for this sequence */
    unsigned int thinking;      /* 1 while reply tokens are in a thinking span */
    unsigned int think_tokens;  /* tokens inside the span, without its markers */
    unsigned long long seed;
    unsigned long long draw;    /* draws made on this slot's stream */
    unsigned long long opened;  /* the tick the sequence opened */
    unsigned int last;          /* the last token made or applied */
    unsigned int flags;         /* AOTX_TOKEN_LAST when the last token ended the reply */
} aotx_seq;

typedef struct aotx_seq_table {
    aotx_seq slot[AOTX_SLOTS];
    int tokens[AOTX_SLOTS][AOTX_SEQ_MAX_TOKENS];
    unsigned int live;          /* slots not FREE */
    unsigned int refused;       /* refused sequence opens and token applications */
} aotx_seq_table;

extern __device__ aotx_seq_table aotx_seqs;

/* Open a sequence on a slot with its prompt tokens. The caller is the command layer's serial
 * thread or the apply of a replayed line. Returns 0, or 1 when the slot is not free or the
 * prompt does not fit. */
__device__ int aotx_seq_open(unsigned int slot, unsigned int role, const int *ids,
                             unsigned int count, unsigned int limit, unsigned int page_limit,
                             const aotx_model_how *sample,
                             unsigned long long tick);

/* Stop a sequence at the next tick; its pages are released after the DONE event. */
__device__ void aotx_seq_stop(unsigned int slot);

/* Apply one replayed token record to its slot: the token joins the sequence at its position
 * and no sample is drawn. The restore path calls this from the apply's serial thread. */
__device__ int aotx_seq_apply(const aotx_token_body *body);

/* The tick's decode has three parts, captured into the tick graph in this order. The plan
 * gathers every live sequence's next tokens into the batch within the tick budget: a prompt
 * chunk, or the last sampled token. The forward pass runs as a child graph. The commit
 * writes one TOKEN record for each sequence that advanced and appends the sampled token. It
 * ends a sequence at its stop token or its limit, and writes the SEQUENCE events. */
__global__ void aotx_decode_plan(unsigned long long tick);
__global__ void aotx_decode_commit(unsigned long long tick);

/* Host glue: capture the decode into the stream that is capturing the tick graph. Returns 0,
 * or 1 when no language role is loaded (the tick then runs with no decode nodes). */
int aotx_decode_capture(void *stream);

/* The slot a console reply reads: the bytes of the reply tokens made since the last read,
 * detokenized, at most max bytes; returns the count. The console panel and the CONSOLE
 * records take it from the commit. */
__device__ unsigned int aotx_seq_take_text(unsigned int slot, unsigned char *out,
                                           unsigned int max);

/* Give the detokenized bytes of one reply token. */
__device__ unsigned int aotx_seq_token_text(unsigned int token, unsigned char *out,
                                            unsigned int room);

#endif
