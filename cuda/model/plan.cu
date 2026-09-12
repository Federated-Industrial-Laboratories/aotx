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
#include "settings/settings.cuh"

/* One ready role owns each total tick budget. Ready roles take turns. */
__global__ void aotx_decode_begin(void)
{
    aotx_decode.selected=aotx_decode.default_role;
    for(unsigned step=0;step<AOTX_MODEL_ROLES;++step){
        unsigned role=(aotx_decode.next_role+step)%AOTX_MODEL_ROLES;
        if(!(aotx_decode.roles&(1u<<role)))continue;
        for(unsigned slot=0;slot<AOTX_SLOTS;++slot){
            const aotx_seq &s=aotx_seqs.slot[slot];
            if(s.role==role && (s.state==AOTX_SEQ_STATE_PREFILL || s.state==AOTX_SEQ_STATE_DECODE)){
                aotx_decode.selected=role;aotx_decode.next_role=(role+1u)%AOTX_MODEL_ROLES;return;
            }
        }
    }
}
__global__ void aotx_decode_select(unsigned role)
{
    aotx_decode.role=role;
}

/* An inclusive add over the slots of the block. Every thread of the block takes part. */
static __device__ __forceinline__ unsigned int aotx_plan_scan(unsigned int *cell,
                                                              unsigned int value)
{
    unsigned int at = threadIdx.x;
    __syncthreads();
    cell[at] = value;
    __syncthreads();
    for (unsigned int step = 1u; step < AOTX_SLOTS; step <<= 1) {
        unsigned int add = (at >= step) ? cell[at - step] : 0u;
        __syncthreads();
        cell[at] += add;
        __syncthreads();
    }
    return cell[at];
}

__global__ void aotx_decode_plan(unsigned long long tick)
{
    __shared__ unsigned int cell[AOTX_SLOTS];
    __shared__ unsigned int decode_rows;
    __shared__ unsigned int total_rows;
    __shared__ unsigned int total_seqs;

    /* The records of the commit carry the tick, so the plan needs no tick of its own. */
    (void)tick;
    unsigned int lane = threadIdx.x;
    unsigned int role = aotx_decode.role;
    unsigned int slot = role < AOTX_MODEL_ROLES ? (lane + aotx_decode.cursor[role]) % AOTX_SLOTS : lane;
    if (slot >= AOTX_SLOTS || role >= AOTX_MODEL_ROLES) {
        return;
    }
    /* The prompt tokens the plan admits in one tick. The setting names the count, and the
     * batch table of the tick holds AOTX_SEQ_TICK_BUDGET rows, so the table bounds it. */
    unsigned int budget = aotx_setting_count(AOTX_SET_PREFILL_TOKENS);
    if (budget > AOTX_SEQ_TICK_BUDGET) {
        budget = AOTX_SEQ_TICK_BUDGET;
    }
    aotx_seq *seq = &aotx_seqs.slot[slot];
    unsigned int state = seq->state;
    unsigned int held = aotx_model_seen[slot];
    unsigned int want = 0u;

    /* A held tick advances no sequence, and the batch of that tick holds no row. A replay
     * of the journal gives every slot its tokens again. The decode therefore waits for the
     * end of that replay and takes each sequence up from the state its records leave. */
    int runs = (aotx_decode.ready != 0u) && (aotx_sched.held == 0ull)
             && (aotx_seam.replaying == 0ull)
             && (!aotx_decode.roles || aotx_decode.selected == role);
    if (runs && seq->role == role
        && (state == AOTX_SEQ_STATE_PREFILL || state == AOTX_SEQ_STATE_DECODE)) {
        unsigned int pending = aotx_seq_pending(seq, held);
        if (pending > budget) {
            pending = budget;
        }
        if (pending > 0u) {
            if (aotx_seq_pages(slot, role, seq->prompt + seq->limit) != 0) {
                want = pending;
            } else {
                atomicAdd(&aotx_decode.short_of, 1u);
            }
        }
    }

    /* The decode rows first: one row for each slot that holds its whole list in pages. */
    unsigned int step = (state == AOTX_SEQ_STATE_DECODE && want) ? 1u : 0u;
    unsigned int step_scan = aotx_plan_scan(cell, step);
    if (lane == AOTX_SLOTS - 1u) {
        decode_rows = min(step_scan, budget);
    }
    __syncthreads();
    if (step_scan > budget) step=0u;
    unsigned int made = decode_rows;

    /* The prompt pieces take the room that is left. A piece that crosses the end of the
     * budget takes the room to the end, and every piece after it waits. */
    unsigned int piece = (state == AOTX_SEQ_STATE_PREFILL) ? want : 0u;
    unsigned int piece_scan = aotx_plan_scan(cell, piece);
    unsigned int before = piece_scan - piece;
    unsigned int room = (budget > made) ? (budget - made) : 0u;
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
    if (lane == AOTX_SLOTS - 1u) {
        total_rows = made + ((piece_scan < room) ? piece_scan : room);
        total_seqs = made + mark_scan;
    }
    __syncthreads();

    unsigned int rows = 0u;
    unsigned int start = 0u;
    unsigned int place = AOTX_SLOTS;
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
        aotx_decode.how[place] = seq->sample;
        aotx_decode.how[place].seed = seq->seed;
        for (unsigned int i = 0u; i < rows; ++i) {
            aotx_decode.ids[start + i] = aotx_seqs.tokens[slot][held + i];
            aotx_model_input *input = aotx_decode.input + start + i;
            if (seq->input_count && held + i < seq->input_count) {
                *input = aotx_seq_input[slot][held + i];
            } else {
                input->feature = 0; input->width = 0;
                unsigned position = seq->input_count ? seq->rotary_next + held + i - seq->input_count : ~0u;
                for (unsigned axis = 0; axis < 3; ++axis) input->position[axis] = position;
            }
        }
    }

    /* The call block of the forward pass. The pass takes its batch from the device, so no
     * node of the graph copies it and no node of the graph changes its parameters. */
    if (lane == 0u) {
        aotx_model_run *run = &aotx_model_call[role];
        aotx_decode.offset[total_seqs] = total_rows;
        aotx_decode.seqs = total_seqs;
        aotx_decode.tokens = total_rows;
        if (total_rows > 0u) {
            aotx_decode.steps += 1ull;
            aotx_decode.cursor[role]=(aotx_decode.cursor[role]+1u)%AOTX_SLOTS;
        }
        run->ids = aotx_decode.ids;
        run->input = aotx_decode.input;
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
        /* Every row of the batch carries its own sample values in how, so these three
         * stand for a call that gives none. */
        run->top_k = aotx_setting_count(AOTX_SET_TOP_K);
        run->top_p = aotx_setting_fraction(AOTX_SET_TOP_P);
        run->temperature = aotx_setting_fraction(AOTX_SET_TEMPERATURE);
        run->telemetry = 1u;
    }
}
