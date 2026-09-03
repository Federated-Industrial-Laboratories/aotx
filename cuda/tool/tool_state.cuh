/* Purpose: Hold the private state of the tool module: the texts, the embed batch, the hits.
 * Owns: The text of each request, the tokenizer memory of this path and the embed batch.
 * Launch shape: Device state; the tool nodes of the tick write it and read it.
 * Lifetime: The whole run.
 *
 * A device tool needs the vector of its text. The texts of one tick therefore go through
 * the tokenizer of this path and then through the pass of the embedding role as one batch.
 * Every table here is device state and holds no host address. */
#ifndef AOTX_TOOL_STATE_CUH
#define AOTX_TOOL_STATE_CUH

#include "embed/embed.cuh"
#include "text/text.cuh"
#include "tool/tool.cuh"
#ifdef AOTX_AFFECT
#include "quality/quality.cuh"
#endif

/* Bytes of one text of this path. The argument of a tool call is the longest of them. */
#define AOTX_TOOL_TEXT_BYTES   192u

/* Bytes of one text after the clean step. That step gives at most three bytes for one byte
 * which is not part of a character. */
#define AOTX_TOOL_CLEAN        (3u * AOTX_TOOL_TEXT_BYTES)

/* Piece slots and token slots of one text. A piece holds one byte at the least, and a token
 * holds one byte at the least, so each count is under the byte count. */
#define AOTX_TOOL_PIECES       AOTX_TOOL_TEXT_BYTES
#define AOTX_TOOL_TOKENS       AOTX_TOOL_TEXT_BYTES

/* Blocks of the merge step of this path, and the warps they hold. */
#define AOTX_TOOL_BLOCKS       4u
#define AOTX_TOOL_WARPS        (AOTX_TOOL_BLOCKS * AOTX_TEXT_WARPS)

#ifdef AOTX_AFFECT
#define AOTX_TOOL_BATCH_ROWS   (AOTX_SLOTS + AOTX_QUALITY_ROWS)
#define AOTX_TOOL_TEXT_SPACE   (AOTX_SLOTS * AOTX_TOOL_TEXT_BYTES \
                                + AOTX_QUALITY_ROWS * AOTX_QUALITY_BYTES)
#define AOTX_TOOL_CLEAN_STRIDE (3u * AOTX_QUALITY_BYTES)
#define AOTX_TOOL_TOKEN_STRIDE AOTX_QUALITY_BYTES
#define AOTX_TOOL_TEXT_THREADS AOTX_TOOL_BATCH_ROWS
#else
#define AOTX_TOOL_BATCH_ROWS   AOTX_SLOTS
#define AOTX_TOOL_TEXT_SPACE   (AOTX_SLOTS * AOTX_TOOL_TEXT_BYTES)
#define AOTX_TOOL_CLEAN_STRIDE AOTX_TOOL_CLEAN
#define AOTX_TOOL_TOKEN_STRIDE AOTX_TOOL_TOKENS
#define AOTX_TOOL_TEXT_THREADS AOTX_TOOL_SLOT_THREADS
#endif
#define AOTX_TOOL_TEXT_BLOCKS ((AOTX_TOOL_BATCH_ROWS + AOTX_TOOL_TEXT_THREADS - 1u) \
                               / AOTX_TOOL_TEXT_THREADS)

/* Threads of a launch that takes one thread for each request slot. The step of the tool
 * path claims its late records with a scan over the whole block. The block therefore holds
 * every slot and the launch holds one block. A block holds 1,024 threads at the most,
 * which the profile check of AOTX_SLOTS keeps. */
#define AOTX_TOOL_SLOT_THREADS AOTX_SLOTS
#define AOTX_TOOL_SLOT_BLOCKS  ((AOTX_SLOTS + AOTX_TOOL_SLOT_THREADS - 1u) \
                                / AOTX_TOOL_SLOT_THREADS)

/* Bytes of content that a reply may leave in a result. The tail of the result is kept for
 * the part that carries a reason, so a reason always lands whatever the content did. */
#define AOTX_TOOL_CONTENT_BYTES (AOTX_TOOL_RESULT_BYTES - AOTX_TOOL_REPLY_BYTES)

/* Asks for pages a text or a quality row makes before it gives up. A pool with no free
 * page leaves a slot short at every service; the bound ends the wait. */
#define AOTX_TOOL_ASK_LIMIT    32u

/* Where a device tool stands on the way to its vector. */
#define AOTX_TOOL_EMBED_NONE   0u   /* the request is a host tool, or it is done */
#define AOTX_TOOL_EMBED_WAIT   1u   /* the text waits for a place in the batch */
#define AOTX_TOOL_EMBED_RUN    2u   /* the text is in the batch of this tick */
#define AOTX_TOOL_EMBED_DONE   3u   /* the vector is in hand */

/* The store keeps the argument of a call, so the two bounds are one bound. */
typedef char aotx_tool_text_check[(AOTX_EMBED_TEXT == AOTX_TOOL_ARG_BYTES) ? 1 : -1];

/* The list of a recall is the list a search gives. */
typedef char aotx_tool_hits_check[(AOTX_EMBED_HITS == AOTX_RECALL_COUNT) ? 1 : -1];

/* The memory the tokenizer of this path holds. One block holds every array, so the host
 * glue reads one address and gives the parts to the kernels of the tokenizer. */
typedef struct aotx_tool_work {
    unsigned char text[AOTX_TOOL_TEXT_SPACE];
    unsigned char clean[AOTX_TOOL_BATCH_ROWS * AOTX_TOOL_CLEAN_STRIDE];
    unsigned int start[AOTX_TOOL_BATCH_ROWS];
    unsigned int length[AOTX_TOOL_BATCH_ROWS];
    unsigned int clean_start[AOTX_TOOL_BATCH_ROWS];
    unsigned int clean_length[AOTX_TOOL_BATCH_ROWS];
    unsigned int piece_start[AOTX_TOOL_BATCH_ROWS * AOTX_TOOL_TOKEN_STRIDE];
    unsigned int piece_length[AOTX_TOOL_BATCH_ROWS * AOTX_TOOL_TOKEN_STRIDE];
    unsigned int piece_token[AOTX_TOOL_BATCH_ROWS * AOTX_TOOL_TOKEN_STRIDE];
    unsigned int piece_count[AOTX_TOOL_BATCH_ROWS];
    unsigned int work[AOTX_TOOL_BATCH_ROWS * AOTX_TOOL_TOKEN_STRIDE];
    unsigned int works;
    unsigned int chunk[AOTX_TOOL_BATCH_ROWS * AOTX_TOOL_TOKEN_STRIDE];
    unsigned int scratch[AOTX_TOOL_BATCH_ROWS * AOTX_TOOL_CLEAN_STRIDE];
    unsigned char merge[AOTX_TOOL_WARPS * AOTX_TEXT_WARP_BYTES];
    unsigned int id[AOTX_TOOL_BATCH_ROWS * AOTX_TOOL_TOKEN_STRIDE];
    unsigned int count[AOTX_TOOL_BATCH_ROWS];
    unsigned int bytes[AOTX_SLOTS];  /* the text of a slot, which the fill step
                                              * gives the tokenizer when the slot asks */
} aotx_tool_work;

extern __device__ aotx_tool_work aotx_tool_gear;

/* The batch of one tick of the embedding pass, and what each request slot gave it. */
typedef struct aotx_tool_embed_batch {
    int ids[AOTX_MODEL_MAX_TOKENS];                 /* the token of every row */
    unsigned int offset[AOTX_TOOL_BATCH_ROWS + 1u]; /* the first row of each sequence */
    unsigned int agent[AOTX_TOOL_BATCH_ROWS]; /* the page cache slot of each sequence */
    unsigned int who[AOTX_TOOL_BATCH_ROWS];  /* the source row of each sequence */
#ifdef AOTX_AFFECT
    unsigned int kind[AOTX_TOOL_BATCH_ROWS]; /* zero for a tool, else quality row plus one */
#endif
    unsigned int place[AOTX_SLOTS];         /* the sequence of a slot, or the count */
    unsigned int live[AOTX_TOOL_BATCH_ROWS]; /* 1 when the slot asks for a search */
    unsigned int hit[AOTX_TOOL_BATCH_ROWS * AOTX_EMBED_HITS];
    float score[AOTX_TOOL_BATCH_ROWS * AOTX_EMBED_HITS];
    float vector[AOTX_TOOL_BATCH_ROWS * AOTX_EMBED_WIDTH];
    unsigned int state[AOTX_SLOTS];         /* AOTX_TOOL_EMBED_* of each slot */
    unsigned int prov[AOTX_SLOTS];          /* the provenance of a memory_write */
    unsigned int asked[AOTX_SLOTS];         /* pages the slot has asked for */
    unsigned int starved[AOTX_SLOTS];       /* asks of the text with no page served */
    unsigned int width;     /* floats of one vector of the embedding role */
    unsigned int replayed;  /* 1 when the tick before this one replayed the journal */
    char note[AOTX_BUS_TEXT_BYTES];  /* the text of the note a refused reply writes; the
                                      * apply of the tick writes it from one thread */
    unsigned int made[AOTX_SLOTS];  /* requests each slot has opened */
    unsigned int outcome[AOTX_SLOTS]; /* the armed result of the next tool of a slot: zero
                                       * for none, else a tool status plus one */
    unsigned int role;      /* the embedding role, or the role count */
    unsigned int ready;     /* 1 after the host glue captured the pass */
    unsigned int seqs;      /* sequences of the batch of this tick */
    unsigned int tokens;    /* rows of the batch of this tick */
    unsigned int waited;    /* slots that got no room in the budget of a tick */
    unsigned int short_of;  /* slots that waited for a page */
} aotx_tool_embed_batch;

extern __device__ aotx_tool_embed_batch aotx_tool_embed;

/* The counts the tool module keeps for the panel and the tests. */
typedef struct aotx_tool_counts {
    unsigned int opened;      /* requests that opened */
    unsigned int device_done; /* device tool calls that gave a result */
    unsigned int host_open;   /* host tool requests written */
    unsigned int replies;     /* reply parts applied */
    unsigned int late;        /* requests that reached their deadline */
    unsigned int refused;     /* replies for a request that no slot holds */
    unsigned int written;     /* findings that went in the note store */
    unsigned int recalled;    /* recalls that gave a result */
    unsigned int parsed;      /* replies the parser took a call from */
    unsigned int rejected;    /* replies the parser refused */
    unsigned int dropped;     /* reply parts that no room in the result would hold */
    unsigned int starved;     /* texts that left the batch with no page over the bound */
} aotx_tool_counts;

extern __device__ aotx_tool_counts aotx_tool_count;

/* The result of a request is in hand. The status field of a request holds a tool status.
 * The value of a good result is zero, so a mark of its own says that a result came. */
extern __device__ unsigned int aotx_tool_done[AOTX_SLOTS];

/* The nodes of the tool path, in the order the tick graph holds them. */
__global__ void aotx_tool_fill(void);
__global__ void aotx_tool_plan(unsigned long long tick);

/* Give the name of a provenance value, or an empty name. */
__device__ __forceinline__ const char *aotx_tool_provenance_name(unsigned int provenance)
{
    switch (provenance) {
    case AOTX_PROV_COMPUTED:  return "computed";
    case AOTX_PROV_FETCHED:   return "fetched";
    case AOTX_PROV_RECALLED:  return "recalled";
    case AOTX_PROV_TESTIMONY: return "testimony";
    default:                  return "";
    }
}

/* Add a text that ends with a zero byte to a result and give the position after it. */
__device__ __forceinline__ unsigned int aotx_tool_put(char *out, unsigned int at,
                                                      const char *text)
{
    for (unsigned int i = 0u; text[i] != '\0' && at < AOTX_TOOL_RESULT_BYTES; ++i) {
        out[at] = text[i];
        at += 1u;
    }
    return at;
}

/* Add a run of bytes to a result and give the position after it. */
__device__ __forceinline__ unsigned int aotx_tool_put_run(char *out, unsigned int at,
                                                          const unsigned char *text,
                                                          unsigned int length)
{
    for (unsigned int i = 0u; i < length && at < AOTX_TOOL_RESULT_BYTES; ++i) {
        out[at] = (char)text[i];
        at += 1u;
    }
    return at;
}

/* Write the record that names a request. The drain gives it to the feeder. A second
 * record for the same request carries the answer of the operator. */
__device__ unsigned long long aotx_tool_note_request(const aotx_request *slot,
                                                      unsigned int turn);

/* Host glue: open the pass of the embedding role before the tick capture starts. The
 * return is zero when the pass is ready. */
int aotx_tool_open(void);

/* Give the child graph of the embedding pass back. */
void aotx_tool_close(void);

/* Nodes of the child graph of the embedding pass, or zero when it is not captured. */
unsigned int aotx_tool_pass_nodes(void);

#endif
