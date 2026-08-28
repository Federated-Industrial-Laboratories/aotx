/* Purpose: Wrap a text in the chat template, open a sequence, and stream its reply.
 * Owns: The prompt table, the tokenizer memory of this path and the reply state of a slot.
 * Launch shape: One thread for each sequence slot.
 * Lifetime: The whole run. */
#include "bus/bus.cuh"
#include "cli/prompt.cuh"
#include "model/model.cuh"
#include "rng/rng.cuh"

__device__ aotx_say_state aotx_say;
__device__ unsigned int aotx_say_id[AOTX_SEQ_SLOTS * AOTX_SAY_TOKENS];
__device__ unsigned int aotx_say_count[AOTX_SEQ_SLOTS];

/* The memory of the tokenizer of this path. The bytes of the prompts are the table in
 * aotx_say, so no copy makes a second run of them. The host glue reads the address of this
 * block once and gives the parts to the kernels of the tokenizer. */
__device__ aotx_say_work aotx_say_gear;

/* The rate samples of every slot. The reply node fills one entry of each slot each tick. */
__device__ aotx_say_sample aotx_say_window[AOTX_SEQ_SLOTS][AOTX_SAY_WINDOW];

__device__ void aotx_say_show(unsigned int slot, const unsigned char *text,
                              unsigned int length)
{
    if (slot >= AOTX_SEQ_SLOTS || length == 0u) {
        return;
    }
    aotx_say_slot *state = &aotx_say.slot[slot];
    /* One console record for each take holds the bytes the new tokens made. A take that
     * goes at the end of an open line carries the fragment flag, so a reader joins it to
     * the record before it. A take that starts a line carries no flag. The record stream
     * and the buffer line then hold the same bytes in the same order. */
    unsigned int flags = (state->column != 0u) ? (unsigned int)AOTX_FLAG_FRAGMENT : 0u;
    aotx_seam_write(AOTX_WRITER_CONSOLE, AOTX_CLASS_B, AOTX_REC_CONSOLE, flags, text,
                    length);
    unsigned int at = 0u;
    while (at < length) {
        unsigned int end = at;
        while (end < length && text[end] != (unsigned char)'\n') {
            end += 1u;
        }
        while (at < end) {
            unsigned int took = 0u;
            if (state->column != 0u) {
                took = aotx_console_grow(state->at, text + at, end - at);
            }
            if (took == 0u) {
                /* The line is full, or a newer line took its place in the buffer. The
                 * bytes that are left start a line of their own. */
                unsigned int room = end - at;
                if (room > AOTX_CONSOLE_COLS) {
                    room = AOTX_CONSOLE_COLS;
                }
                state->at = aotx_console_put(text + at, room);
                state->column = 1u;
                took = room;
            }
            at += took;
        }
        if (end < length) {
            /* A newline byte ends the line. The next byte starts a new one. */
            state->column = 0u;
            state->at = 0ull;
            at = end + 1u;
        }
    }
}



__global__ void aotx_say_fill(void)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= AOTX_SEQ_SLOTS) {
        return;
    }
    /* Every slot is in the batch of every tick, so the shape of the graph never changes. A
     * slot with no prompt gives a byte run of no length and no piece. */
    aotx_say_gear.start[slot] = slot * AOTX_SAY_BYTES;
    aotx_say_gear.length[slot] = (aotx_say.slot[slot].wanted != 0u)
                               ? aotx_say.slot[slot].length : 0u;
    if (slot == 0u) {
        aotx_say_gear.works = 0u;
    }
}

/* The language role of the run: the eight bit file when it is loaded, and the four bit
 * file when it is not. The decode captures its pass for the same role in the same order. */
__device__ __forceinline__ static unsigned int aotx_say_language(void)
{
    return (aotx_model[AOTX_MODEL_LANGUAGE].layers != 0u) ? AOTX_MODEL_LANGUAGE
                                                          : AOTX_MODEL_LANGUAGE_Q4;
}

/* Give the seed of a slot from the Philox stream of that slot at this tick. Two slots that
 * open in the same tick take different numbers, and the run identity keeps two runs apart. */
static __device__ __forceinline__ unsigned long long aotx_say_seed(unsigned int slot,
                                                                   unsigned long long tick)
{
    uint4 draw = aotx_rng_lane(aotx_seam.boot_id, slot, 0u, tick);
    return ((unsigned long long)draw.y << 32) | (unsigned long long)draw.x;
}

__global__ void aotx_say_start(void)
{
    const unsigned long long tick = aotx_time_tick;
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= AOTX_SEQ_SLOTS) {
        return;
    }
    aotx_say_slot *state = &aotx_say.slot[slot];
    if (state->wanted == 0u) {
        return;
    }
    state->wanted = 0u;
    unsigned int count = aotx_say_count[slot];
    state->prompt = count;
    int bad = 1;
    if (count != 0u) {
        bad = aotx_seq_open(slot, aotx_say_language(),
                            (const int *)(aotx_say_id + slot * AOTX_SAY_TOKENS), count,
                            AOTX_SEQ_REPLY_DEFAULT, aotx_say_seed(slot, tick),
                            AOTX_SAY_TOP_K, AOTX_SAY_TOP_P, AOTX_SAY_TEMPERATURE, tick);
    }
    if (bad != 0) {
        state->live = 0u;
        state->column = 0u;
        state->at = 0ull;
        aotx_console_write("say: the sequence did not open", 30u);
        return;
    }
    state->live = 1u;
    state->opened = tick;
    state->tokens = 0u;
    atomicAdd(&aotx_say.opened, 1u);
}

/* Write the message that ends a reply into the take buffer of a slot and give its length. */
static __device__ __forceinline__ unsigned int aotx_say_end_text(aotx_say_slot *state,
                                                                 unsigned long long ticks)
{
    unsigned int at = 0u;
    at = aotx_say_put(state->text, at, "reply of ");
    at += aotx_text_utoa((unsigned long long)state->tokens, (char *)state->text + at,
                         AOTX_SAY_TAKE - at);
    at = aotx_say_put(state->text, at, " tokens in ");
    at += aotx_text_utoa(ticks, (char *)state->text + at, AOTX_SAY_TAKE - at);
    return aotx_say_put(state->text, at, " ticks");
}

__global__ void aotx_say_reply(void)
{
    const unsigned long long tick = aotx_time_tick;
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= AOTX_SEQ_SLOTS) {
        return;
    }
    aotx_say_slot *state = &aotx_say.slot[slot];
    /* Every slot leaves one sample each tick, whether it runs or not. The rate of a panel
     * row then comes from two samples of the same sequence, which carry the tokens and the
     * device clock together. No figure of the row comes from the pace of the pump. */
    aotx_say_sample *sample = &aotx_say_window[slot][tick % (unsigned long long)AOTX_SAY_WINDOW];
    sample->tick = tick;
    sample->ns = aotx_sched.start_ns;
    sample->opened = aotx_seqs.slot[slot].opened;
    sample->sampled = aotx_seqs.slot[slot].sampled;

    /* Only a sequence the say command opened has a console line. A sequence that a restore
     * gave back from the token records has none, and the console does not show it again. */
    if (state->live == 0u) {
        return;
    }
    unsigned int got = aotx_seq_take_text(slot, state->text, AOTX_SAY_TAKE);
    if (got != 0u) {
        aotx_say_show(slot, state->text, got);
        atomicAdd(&aotx_say.shown, 1u);
    }
    state->tokens = aotx_seqs.slot[slot].sampled;
    unsigned int live = aotx_seqs.slot[slot].state;
    if (live == AOTX_SEQ_STATE_PREFILL || live == AOTX_SEQ_STATE_DECODE) {
        return;
    }
    /* The sequence ended. The line closes and one bus message states what the reply cost. */
    state->live = 0u;
    state->column = 0u;
    state->at = 0ull;
    unsigned long long ticks = (tick > state->opened) ? (tick - state->opened) : 0ull;
    unsigned int at = aotx_say_end_text(state, ticks);
    aotx_bus_append(AOTX_WRITER_CONSOLE, AOTX_BUS_NOTE, 0u, (const char *)state->text, at,
                    0ull, 0ull, 0.0f, tick);
}
