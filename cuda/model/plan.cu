/* Purpose: Build the batch of one tick from every live sequence.
 * Owns: Nothing; the batch tables and the call block hold the result.
 * Launch shape: One block of one thread for each sequence slot.
 * Lifetime: One node of every tick.
 *
 * A slot gives the tokens that no page holds yet. A decode slot gives one row, which is
 * the token the tick before sampled. A prefill slot gives the next piece of its list. The
 * decode rows come first, so a reply never waits for a long prompt. The prompt pieces take
 * the rest of the token budget of the tick, and a slot that gets no room waits.
 *
 * The batch of the tick is at most AOTX_SEQ_TICK_BUDGET rows, which is the token count the
 * host glue captured the forward pass for. */
#include "model/decode_state.cuh"
#include "sched/sched.cuh"

/* An inclusive add over the slots of the block. Every thread of the block takes part. */
static __device__ __forceinline__ unsigned int aotx_plan_scan(unsigned int *cell,
                                                              unsigned int value)
{
    unsigned int at = threadIdx.x;
    __syncthreads();
    cell[at] = value;
    __syncthreads();
    for (unsigned int step = 1u; step < AOTX_SEQ_SLOTS; step <<= 1) {
        unsigned int add = (at >= step) ? cell[at - step] : 0u;
        __syncthreads();
        cell[at] += add;
        __syncthreads();
    }
    return cell[at];
}

__global__ void aotx_decode_plan(unsigned long long tick)
{
    __shared__ unsigned int cell[AOTX_SEQ_SLOTS];
    __shared__ unsigned int decode_rows;
    __shared__ unsigned int total_rows;
    __shared__ unsigned int total_seqs;

    /* The records of the commit carry the tick, so the plan needs no tick of its own. */
    (void)tick;
    unsigned int slot = threadIdx.x;
    unsigned int role = aotx_decode.role;
    if (slot >= AOTX_SEQ_SLOTS || role >= AOTX_MODEL_ROLES) {
        return;
    }
    aotx_seq *seq = &aotx_seqs.slot[slot];
    unsigned int state = seq->state;
    unsigned int held = aotx_model_seen[slot];
    unsigned int want = 0u;

    /* A held tick advances no sequence, and the batch of that tick holds no row. A replay
     * of the journal gives every slot its tokens again. The decode therefore waits for the
     * end of that replay and takes each sequence up from the state its records leave. */
    int runs = (aotx_decode.ready != 0u) && (aotx_sched.held == 0ull)
             && (aotx_seam.replaying == 0ull);
    if (runs && seq->role == role
        && (state == AOTX_SEQ_STATE_PREFILL || state == AOTX_SEQ_STATE_DECODE)) {
        unsigned int pending = aotx_seq_pending(seq, held);
        if (pending > AOTX_SEQ_TICK_BUDGET) {
            pending = AOTX_SEQ_TICK_BUDGET;
        }
        if (pending > 0u) {
            if (aotx_seq_pages(slot, role, held + pending) != 0) {
                want = pending;
            } else {
                atomicAdd(&aotx_decode.short_of, 1u);
            }
        }
    }

    /* The decode rows first: one row for each slot that holds its whole list in pages. */
    unsigned int step = (state == AOTX_SEQ_STATE_DECODE) ? want : 0u;
    unsigned int step_scan = aotx_plan_scan(cell, step);
    if (slot == AOTX_SEQ_SLOTS - 1u) {
        decode_rows = step_scan;
    }
    __syncthreads();
    unsigned int made = decode_rows;

    /* The prompt pieces take the room that is left. A piece that crosses the end of the
     * budget takes the room to the end, and every piece after it waits. */
    unsigned int piece = (state == AOTX_SEQ_STATE_PREFILL) ? want : 0u;
    unsigned int piece_scan = aotx_plan_scan(cell, piece);
    unsigned int before = piece_scan - piece;
    unsigned int room = AOTX_SEQ_TICK_BUDGET - made;
    unsigned int give = 0u;
    if (piece > 0u) {
        if (before < room) {
            give = (piece < room - before) ? piece : (room - before);
        } else {
            atomicAdd(&aotx_decode.waited, 1u);
        }
    }
    unsigned int mark = (give > 0u) ? 1u : 0u;
    unsigned int mark_scan = aotx_plan_scan(cell, mark);
    if (slot == AOTX_SEQ_SLOTS - 1u) {
        total_rows = made + ((piece_scan < room) ? piece_scan : room);
        total_seqs = made + mark_scan;
    }
    __syncthreads();

    unsigned int rows = 0u;
    unsigned int start = 0u;
    unsigned int place = AOTX_SEQ_SLOTS;
    if (step > 0u) {
        rows = step;
        start = step_scan - step;
        place = start;
    } else if (give > 0u) {
        rows = give;
        start = made + ((before < room) ? before : room);
        place = made + (mark_scan - mark);
    }
    aotx_decode.rows[slot] = rows;
    aotx_decode.first[slot] = held;
    aotx_decode.place[slot] = place;
    if (rows > 0u) {
        aotx_decode.agent[place] = slot;
        aotx_decode.offset[place] = start;
        aotx_decode.how[place].top_k = seq->top_k;
        aotx_decode.how[place].top_p = seq->top_p;
        aotx_decode.how[place].temperature = seq->temperature;
        aotx_decode.how[place].seed = seq->seed;
        for (unsigned int i = 0u; i < rows; ++i) {
            aotx_decode.ids[start + i] = aotx_seqs.tokens[slot][held + i];
        }
    }

    /* The call block of the forward pass. The pass takes its batch from the device, so no
     * node of the graph copies it and no node of the graph changes its parameters. */
    if (slot == 0u) {
        aotx_model_run *run = &aotx_model_call[role];
        aotx_decode.offset[total_seqs] = total_rows;
        aotx_decode.seqs = total_seqs;
        aotx_decode.tokens = total_rows;
        if (total_rows > 0u) {
            aotx_decode.steps += 1ull;
        }
        run->ids = aotx_decode.ids;
        run->offset = aotx_decode.offset;
        run->agent = aotx_decode.agent;
        run->logits = 0;
        run->pooled = 0;
        run->score = 0;
        run->token = aotx_decode.token;
        run->draw = aotx_decode.draw;
        run->how = aotx_decode.how;
        run->seed = 0ull;
        run->seqs = total_seqs;
        run->tokens = total_rows;
        run->rows = total_seqs;
        run->select = AOTX_MODEL_ROWS_LAST;
        run->top_k = AOTX_DECODE_TOP_K;
        run->top_p = AOTX_DECODE_TOP_P;
        run->temperature = AOTX_DECODE_TEMPERATURE;
    }
}
