/* Purpose: Hold the private state of the decode: the batch of a tick and the marks a slot keeps.
 * Owns: The batch tables of the tick and the two marks of each sequence slot.
 * Launch shape: Device state; the plan writes it and the commit reads it.
 * Lifetime: The whole run.
 *
 * The tables are device state and no host address goes in them. The plan writes the call
 * block of the forward pass from these tables. The pass therefore takes the batch of the
 * tick with no copy across the seam. */
#ifndef AOTX_MODEL_DECODE_STATE_CUH
#define AOTX_MODEL_DECODE_STATE_CUH

#include "model/decode.cuh"
#include "model/forward.cuh"
#include "seam/seam.cuh"

/* The sample of a sequence that a replay opens. A TOKEN record carries the seed and the
 * draw of a token. It does not carry the three values that shape the set the draw takes.
 * A replayed sequence therefore continues with the values of the model card. */
#define AOTX_DECODE_TOP_K       20u
#define AOTX_DECODE_TOP_P       0.95f
#define AOTX_DECODE_TEMPERATURE 0.7f

/* The two tokens that end a reply. The slot holds the first of them; the second ends every
 * reply, because it ends the text of this model family. */
#define AOTX_DECODE_STOP_END    151645u
#define AOTX_DECODE_STOP_TEXT   151643u

/* The mark that the stop call puts on a slot. The flags field of a slot holds the flags of
 * the last token below bit 8, so this mark stands above them. */
#define AOTX_DECODE_MARK_STOP   0x0100u

/* Blocks of a launch of the decode graph that takes a run of rows. The batch of a tick is
 * from one row to the whole budget, and the grid of a captured node cannot change. A grid
 * of this many blocks holds the machine at the largest batch, and starts and stops few
 * blocks at the smallest. */
#define AOTX_DECODE_WAVE          64u

/* The batch of one tick, and what each slot gave it. Every array here is device memory of
 * this structure, so the plan gives the call block the addresses without a host call. */
typedef struct aotx_decode_state {
    int ids[AOTX_SEQ_TICK_BUDGET];              /* the token of every row of the batch */
    unsigned int offset[AOTX_SEQ_SLOTS + 1u];   /* the first row of each sequence */
    unsigned int agent[AOTX_SEQ_SLOTS];         /* the page cache slot of each sequence */
    int token[AOTX_SEQ_SLOTS];                  /* the token the sample gave each sequence */
    unsigned int draw[AOTX_SEQ_SLOTS];          /* the stream position each draw took */
    aotx_model_how how[AOTX_SEQ_SLOTS];         /* the sample of each sequence */
    unsigned int rows[AOTX_SEQ_SLOTS];          /* rows the slot gave the batch this tick */
    unsigned int first[AOTX_SEQ_SLOTS];         /* the position of the first of those rows */
    unsigned int place[AOTX_SEQ_SLOTS];         /* the sequence of the slot, or the slot count */
    unsigned int role;         /* the language role the tick runs */
    unsigned int ready;        /* 1 after the host glue captured the pass */
    unsigned int seqs;         /* sequences of the batch of this tick */
    unsigned int tokens;       /* rows of the batch of this tick */
    unsigned int waited;       /* slots that got no room in the budget of a tick */
    unsigned int short_of;     /* slots that waited for a page */
    unsigned long long steps;  /* ticks the plan gave a batch of one row or more */
} aotx_decode_state;

extern __device__ aotx_decode_state aotx_decode;

/* Tokens of each slot that the journal holds. A token goes in the journal one time. A
 * prompt token goes in when it enters the page cache. A sampled token goes in when the
 * draw makes it. A replay fills this mark, so a restored sequence rebuilds its pages
 * without a second entry in the journal. */
extern __device__ unsigned int aotx_seq_kept[AOTX_SEQ_SLOTS];

/* Reply tokens of each slot that a take of the text has read. */
extern __device__ unsigned int aotx_seq_shown[AOTX_SEQ_SLOTS];

/* Pages each slot has asked the page cache for. A request that waits in the queue counts
 * here. A run of records over one context therefore makes one request and not one request
 * for every record. */
extern __device__ unsigned int aotx_seq_asked[AOTX_SEQ_SLOTS];

/* The tokens of a slot that no page holds yet. */
__device__ __forceinline__ unsigned int aotx_seq_pending(const aotx_seq *seq,
                                                         unsigned int held)
{
    unsigned int list = seq->prompt + seq->sampled;
    return (list > held) ? (list - held) : 0u;
}

/* Ask the page cache for the pages that a context of the given tokens needs. The return is
 * 1 when the slot holds them already. A request the queue still holds is counted, so a run
 * of records over one context makes one request. */
__device__ __forceinline__ int aotx_seq_pages(unsigned int slot, unsigned int role,
                                              unsigned int context)
{
    unsigned int need = aotx_kvl_pages(&aotx_model_space[role].shape, context);
    unsigned int held = aotx_kv.count[slot];
    if (need <= held) {
        aotx_seq_asked[slot] = held;
        return 1;
    }
    unsigned int asked = (aotx_seq_asked[slot] > held) ? aotx_seq_asked[slot] : held;
    if (need > asked) {
        aotx_seq_asked[slot] = (aotx_kv_request(slot, need - asked) != 0) ? need : held;
    }
    return 0;
}

#endif
