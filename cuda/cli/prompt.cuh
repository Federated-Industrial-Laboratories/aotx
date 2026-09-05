/* Purpose: Hold the prompt of each slot and turn it into a sequence in the tick graph.
 * Owns: The prompt table, the tokenizer memory of that path and the reply state of a slot.
 * Launch shape: One thread for each sequence slot.
 * Lifetime: The whole run.
 *
 * The command layer and the agent module both put a prompt here. A caller that runs inside
 * a node of the tick cannot launch a kernel. The tokenizer of the device is a run of
 * kernels. A caller therefore leaves the wrapped bytes in this table. The nodes below then
 * tokenize the whole batch of slots and open the sequences in the same tick. */
#ifndef AOTX_CLI_PROMPT_CUH
#define AOTX_CLI_PROMPT_CUH

#include "cli/agents.cuh"
#include "cli/cli.cuh"
#include "model/wrap.cuh"
#include "settings/settings.cuh"

/* The slot of the conductor. The console follows one sequence in this version. */
#define AOTX_SAY_SLOT      0u

/* A worker waits one second at the default tick period for its token count. */
#define AOTX_SAY_TOKEN_WAIT_TICKS 100ull

/* Bytes of one wrapped prompt come from the profile. The line the editor takes is at most
 * AOTX_BODY_BYTES. The model's bounded wrap spans surround that text. An agent uses the
 * same table for its role overlay, turn text and tool result. */

/* Bytes of a wrapped prompt after the clean step. That step gives at most three bytes for
 * one byte which is not part of a character. */
#define AOTX_SAY_CLEAN     (3u * AOTX_SAY_BYTES)

/* Piece slots and token slots of one prompt. A piece holds one byte at the least. A token
 * holds one byte at the least. The count of each one is therefore under the byte count. */
#define AOTX_SAY_PIECES    AOTX_SAY_BYTES
#define AOTX_SAY_TOKENS    AOTX_SAY_BYTES

/* Blocks of the merge step of this path, and the warps they hold. Each warp holds one piece
 * at a time and takes AOTX_TEXT_WARP_BYTES of the merge memory. */
#define AOTX_SAY_BLOCKS    8u
#define AOTX_SAY_WARPS     (AOTX_SAY_BLOCKS * AOTX_TEXT_WARPS)

/* Bytes of reply that one sequence gives the console in one tick. The bound is the width of
 * a console line, and it is under the body of a record. */
#define AOTX_SAY_TAKE      AOTX_CONSOLE_COLS

/* Sample values and the reply limit come from the settings table. A generation closes an
 * empty thinking block only when the model defines thinking spans. Empty spans add no
 * thinking block. */

/* What one slot of the say path holds. The command layer fills the prompt fields; the nodes
 * of the tick graph read them, open the sequence, and then show the reply. */
typedef struct aotx_say_slot {
    unsigned int wanted;          /* 1 when a prompt waits for the tokenize step */
    unsigned int length;          /* bytes of the wrapped prompt */
    unsigned int live;            /* 1 while the console shows the reply of this slot */
    unsigned int tokens;          /* reply tokens the commit made, as the last take saw them */
    unsigned int page_limit;      /* pages this turn may take; zero takes the profile limit */
    unsigned int turn_at;         /* first prompt byte of this turn */
    unsigned int turn_tokens;     /* tokens owned by this turn */
    unsigned long long reply_first; /* first console record of this reply, or zero */
    unsigned int reply_records;   /* console records that hold this reply */
    unsigned int prompt;          /* prompt tokens the tokenize step gave */
    unsigned int ready;           /* 1 when the last open of this slot gave a sequence */
    unsigned int column;          /* 1 when the line that grows is open */
    unsigned int console_mode;    /* 0 prefix, 1 reply text, 2 hidden tool call */
    unsigned int console_prefix;  /* bytes held while the tool-call prefix is tested */
    unsigned char prefix[12];
    unsigned long long at;        /* the console line the reply grows into, or zero */
    unsigned long long opened;    /* the tick the sequence opened */
    unsigned long long token_deadline; /* last tick that waits for an agent token count */
    unsigned char text[AOTX_SAY_TAKE];  /* the bytes of one take, and then the end message */
} aotx_say_slot;

typedef struct aotx_say_state {
    aotx_say_slot slot[AOTX_SLOTS];
    unsigned char prompt[AOTX_SLOTS][AOTX_SAY_BYTES];
    unsigned int said;            /* say commands the parser took */
    unsigned int refused;         /* say commands the parser refused */
    unsigned int stopped;         /* stop commands that ended a reply */
    unsigned int opened;          /* sequences the say path opened */
    unsigned int shown;           /* takes that put bytes on the console */
} aotx_say_state;

extern __device__ aotx_say_state aotx_say;

/* The memory the tokenizer of this path holds. One block holds every array, so the host
 * glue reads one address and gives the parts to the kernels of the tokenizer. The block is
 * device state of the run and never crosses the seam. */
typedef struct aotx_say_work {
    unsigned char clean[AOTX_SLOTS * AOTX_SAY_CLEAN];
    unsigned int start[AOTX_SLOTS];
    unsigned int length[AOTX_SLOTS];
    unsigned int clean_start[AOTX_SLOTS];
    unsigned int clean_length[AOTX_SLOTS];
    unsigned int piece_start[AOTX_SLOTS * AOTX_SAY_PIECES];
    unsigned int piece_length[AOTX_SLOTS * AOTX_SAY_PIECES];
    unsigned int piece_token[AOTX_SLOTS * AOTX_SAY_PIECES];
    unsigned int piece_count[AOTX_SLOTS];
    unsigned int work[AOTX_SLOTS * AOTX_SAY_PIECES];
    unsigned int works;
    unsigned int chunk[AOTX_SLOTS * AOTX_SAY_PIECES];
    unsigned int scratch[AOTX_SLOTS * AOTX_SAY_CLEAN];
    unsigned char merge[AOTX_SAY_WARPS * AOTX_TEXT_WARP_BYTES];
} aotx_say_work;

extern __device__ aotx_say_work aotx_say_gear;

/* The tokens of each slot and their counts. The check reads them against the golden list. */
extern __device__ unsigned int aotx_say_id[AOTX_SLOTS * AOTX_SAY_TOKENS];
extern __device__ unsigned int aotx_say_count[AOTX_SLOTS];


/* Copy a text that ends with a zero byte into a prompt and give the position after it. */
__device__ __forceinline__ unsigned int aotx_say_put(unsigned char *out, unsigned int at,
                                                     const char *text)
{
    for (unsigned int i = 0u; text[i] != '\0' && at < AOTX_SAY_BYTES; ++i) {
        out[at] = (unsigned char)text[i];
        at += 1u;
    }
    return at;
}

/* Put the chat wrap of a text in the prompt table of a slot. The return is 0, or 1 when the
 * slot is busy or the text does not fit. The parser calls this from its serial thread.
 *
 * The function is in the header because the parser runs inside the apply node of the tick.
 * A call from that node into another translation unit makes the node keep a frame. That
 * frame holds the registers of the call, and the spill gate binds it. */
__device__ __forceinline__ int aotx_say_ask(unsigned int slot, const unsigned char *text,
                                            unsigned int length)
{
    if (slot >= AOTX_SLOTS) {
        return 1;
    }
    aotx_say_slot *state = &aotx_say.slot[slot];
    if (state->wanted != 0u || state->live != 0u) {
        return 1;
    }
    const aotx_wrap *wrap = aotx_wrap_active();
    if (wrap->usable == 0u) return 1;
    unsigned char *out = aotx_say.prompt[slot];
    unsigned int at = aotx_wrap_prefix(out, 0u, AOTX_SAY_BYTES, wrap);
    at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_USER_HEAD);
    if (at > AOTX_SAY_BYTES || length > AOTX_SAY_BYTES - at) return 1;
    for (unsigned int i = 0u; i < length; ++i) {
        out[at] = text[i];
        at += 1u;
    }
    at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_USER_TAIL);
    at = aotx_wrap_generation(out, at, AOTX_SAY_BYTES, wrap);
    if (at > AOTX_SAY_BYTES) return 1;
    state->length = at;
    state->at = 0ull;
    state->column = 0u;
    state->console_mode = 0u;
    state->console_prefix = 0u;
    state->tokens = 0u;
    state->prompt = 0u;
    state->page_limit = 0u;
    state->turn_at = 0u;
    state->turn_tokens = 0u;
    state->token_deadline = 0ull;
    state->wanted = 1u;
    return 0;
}

/* Give the name of a sequence state, or a dash when the value is not one of the four. */
__device__ __forceinline__ const char *aotx_say_state_name(unsigned int state)
{
    switch (state) {
    case AOTX_SEQ_STATE_FREE:    return "free";
    case AOTX_SEQ_STATE_PREFILL: return "prefill";
    case AOTX_SEQ_STATE_DECODE:  return "decode";
    case AOTX_SEQ_STATE_DONE:    return "done";
    default:                     return "-";
    }
}

/* Give the name of a model role, or a dash when the value is not one of the four. */
__device__ __forceinline__ const char *aotx_say_role_name(unsigned int role)
{
    switch (role) {
    case AOTX_MODEL_EMBEDDING:   return "embedding";
    case AOTX_MODEL_RERANKER:    return "reranker";
    case AOTX_MODEL_LANGUAGE:    return "language";
    case AOTX_MODEL_LANGUAGE_Q4: return "language-q4";
    default:                     return "-";
    }
}

/* Put a run of reply bytes on the console line of a slot. A newline byte ends that line and
 * the bytes after it start a new one. A line which is full also starts a new one. */
__device__ void aotx_say_show(unsigned int slot, const unsigned char *text,
                              unsigned int length);


/* Ticks that one rate sample covers. The window holds one sample of each slot for each of
 * these ticks. */
#define AOTX_SAY_WINDOW    16u

/* One sample of one slot, written once each tick. The device clock gives the time, so a
 * rate from two samples is a measurement of this run and not the pace of the pump. */
typedef struct aotx_say_sample {
    unsigned long long tick;      /* the tick the sample was taken; zero when empty */
    unsigned long long ns;        /* the device clock at the start of that tick */
    unsigned long long opened;    /* the tick the sequence opened, which keeps two apart */
    unsigned int sampled;         /* reply tokens the sequence had made */
    unsigned int reserved;
} aotx_say_sample;

extern __device__ aotx_say_sample aotx_say_window[AOTX_SLOTS][AOTX_SAY_WINDOW];

/* Give the reply tokens each second of a slot, from the oldest and the newest sample of one
 * sequence in the window. Both the tokens and the time come from those two samples. The
 * return is zero when the window holds fewer than two samples of the sequence that runs. */
__device__ __forceinline__ unsigned long long aotx_say_rate(unsigned int slot)
{
    if (slot >= AOTX_SLOTS) {
        return 0ull;
    }
    const aotx_say_sample *last = 0;
    for (unsigned int i = 0u; i < AOTX_SAY_WINDOW; ++i) {
        const aotx_say_sample *at = &aotx_say_window[slot][i];
        if (at->tick != 0ull && (last == 0 || at->tick > last->tick)) {
            last = at;
        }
    }
    if (last == 0) {
        return 0ull;
    }
    const aotx_say_sample *first = 0;
    for (unsigned int i = 0u; i < AOTX_SAY_WINDOW; ++i) {
        const aotx_say_sample *at = &aotx_say_window[slot][i];
        if (at->tick == 0ull || at->opened != last->opened) {
            continue;
        }
        if (first == 0 || at->tick < first->tick) {
            first = at;
        }
    }
    if (first == 0 || first == last || last->ns <= first->ns
        || last->sampled < first->sampled) {
        return 0ull;
    }
    return ((unsigned long long)(last->sampled - first->sampled) * 1000000000ull)
           / (last->ns - first->ns);
}

/* The nodes of the say path, in the order the tick graph holds them. The fill step writes
 * the batch table of the tokenizer and clears the work count. The start step opens a
 * sequence for each prompt the tokenize step read. The reply step takes the new bytes of
 * each live sequence and puts them on the console. */
__global__ void aotx_say_fill(void);
__global__ void aotx_say_start(void);
__global__ void aotx_say_reply(void);

/* Host glue: capture the say nodes into the stream that is capturing the tick graph. The
 * say nodes go after the apply node and before the plan of the decode. The reply node goes
 * after the commit of the decode and before the flush. Each one returns 0 when the nodes
 * are in the stream. */
int aotx_cli_say_capture(void *stream);
int aotx_cli_reply_capture(void *stream);

#endif
