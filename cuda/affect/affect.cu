/* Purpose: Update the state of each turn that ended and write its trace and state records.
 * Owns: The state and the accumulator of each agent, the probe row table and the probe
 *   matrix pointer.
 * Launch shape: The turn node runs one block of one thread for each agent after the agent
 *   step. The apply runs on the thread of the restore.
 * Lifetime: The whole run. */
#include "affect/affect.cuh"

#include "agent/agent_state.cuh"
#include "model/decode_state.cuh"
#include "seam/seam.cuh"

__device__ aotx_affect_table aotx_affect_rows;
__device__ const float *aotx_affect_probe;
__device__ aotx_affect_sums aotx_affect_acc[AOTX_SLOTS];
#define AOTX_AFFECT_NEUTRAL {{0, 0, 0, 0}, {0, 0, 0, 0}, AOTX_AFFECT_SCALE_ONE, \
                             AOTX_AFFECT_DATA_AXES, 0u}
#define AOTX_AFFECT_NEUTRAL_2  AOTX_AFFECT_NEUTRAL, AOTX_AFFECT_NEUTRAL
#define AOTX_AFFECT_NEUTRAL_4  AOTX_AFFECT_NEUTRAL_2, AOTX_AFFECT_NEUTRAL_2
#define AOTX_AFFECT_NEUTRAL_8  AOTX_AFFECT_NEUTRAL_4, AOTX_AFFECT_NEUTRAL_4
#define AOTX_AFFECT_NEUTRAL_16 AOTX_AFFECT_NEUTRAL_8, AOTX_AFFECT_NEUTRAL_8
#define AOTX_AFFECT_NEUTRAL_32 AOTX_AFFECT_NEUTRAL_16, AOTX_AFFECT_NEUTRAL_16
#define AOTX_AFFECT_NEUTRAL_64 AOTX_AFFECT_NEUTRAL_32, AOTX_AFFECT_NEUTRAL_32
#if AOTX_SLOTS == 64u
__device__ aotx_affect_agent_state aotx_affect_state[AOTX_SLOTS] = {
    AOTX_AFFECT_NEUTRAL_64
};
#else
__device__ aotx_affect_agent_state aotx_affect_state[AOTX_SLOTS] = {
    AOTX_AFFECT_NEUTRAL_32
};
#endif
#undef AOTX_AFFECT_NEUTRAL_64
#undef AOTX_AFFECT_NEUTRAL_32
#undef AOTX_AFFECT_NEUTRAL_16
#undef AOTX_AFFECT_NEUTRAL_8
#undef AOTX_AFFECT_NEUTRAL_4
#undef AOTX_AFFECT_NEUTRAL_2
#undef AOTX_AFFECT_NEUTRAL
__device__ aotx_affect_law aotx_affect_laws[AOTX_SLOTS];

/* The mean of a sum over a count, or zero for no count. */
static __device__ __forceinline__ float aotx_affect_mean(float sum, unsigned int count)
{
    return (count != 0u) ? sum / (float)count : 0.0f;
}

/* A value the drain takes: a finite float. */
static __device__ __forceinline__ float aotx_affect_finite(float value)
{
    return isfinite(value) ? value : 0.0f;
}

/* The events the node reads from the state at the end of the turn. The agent step marks
 * the others where they happen, because their fields are gone when the node runs. */
static __device__ __forceinline__ unsigned int aotx_affect_events(unsigned int agent,
                                                                  const aotx_affect_sums *acc,
                                                                  unsigned int think)
{
    const aotx_agent *me = &aotx_agents.agent[agent];
    const aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int mask = acc->events;
    mask |= (gear->last_token != 0u) ? (1u << AOTX_AFFECT_EVENT_STOP) : 0u;
    mask |= (gear->limit_end != 0u) ? (1u << AOTX_AFFECT_EVENT_LIMIT) : 0u;
    mask |= (gear->stopped != 0u) ? (1u << AOTX_AFFECT_EVENT_OPERATOR_STOP) : 0u;
    mask |= (me->budget_left == 0u) ? (1u << AOTX_AFFECT_EVENT_BUDGET) : 0u;
    mask |= (think * 2u > gear->out_tokens) ? (1u << AOTX_AFFECT_EVENT_THINK_RATIO) : 0u;
    mask |= (acc->sampled != 0u
             && aotx_affect_mean(acc->logprob_sum, acc->sampled) < AOTX_AFFECT_LOGPROB_BOUND)
          ? (1u << AOTX_AFFECT_EVENT_LOW_LOGPROB) : 0u;
    return mask & AOTX_AFFECT_EVENT_MASK;
}

/* Fill the trace of one agent whose turn ended in this tick, up to the state fields. The
 * think count comes from the sequence of the turn, and a turn with no reply states zero. */
static __device__ __forceinline__ void aotx_affect_trace(unsigned int agent,
                                                         const aotx_affect_sums *acc,
                                                         aotx_affect_trace_body *body)
{
    const aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int think = (gear->out_tokens != 0u) ? aotx_seqs.slot[agent].think_tokens : 0u;
    body->agent = agent;
    body->turn = aotx_agents.agent[agent].turn;
    for (unsigned int i = 0u; i < AOTX_AFFECT_TRACE_AXES; ++i) {
        body->prompt[i] = aotx_affect_finite(acc->prompt[i]);
        body->reply[i] = aotx_affect_finite(aotx_affect_mean(acc->reply_sum[i],
                                                             acc->reply_rows));
        body->effective[i] = 0;
    }
    for (unsigned int i = 0u; i < AOTX_AFFECT_AXES - AOTX_AFFECT_GUARD_AXIS; ++i) {
        body->guard[i] = aotx_affect_finite(
            aotx_affect_mean(acc->reply_sum[AOTX_AFFECT_GUARD_AXIS + i], acc->reply_rows));
    }
    body->logprob = aotx_affect_finite(aotx_affect_mean(acc->logprob_sum, acc->sampled));
    body->entropy = aotx_affect_finite(aotx_affect_mean(acc->entropy_sum, acc->sampled));
    body->entropy = (body->entropy < 0.0f) ? 0.0f : body->entropy;
    body->rows = acc->reply_rows;
    body->think = think;
    body->reason = aotx_affect_events(agent, acc, think);
    body->flags = (aotx_affect_rows.count != 0u) ? AOTX_AFFECT_FLAG_PROBES : 0u;
}

/* A fraction as a Q1.15 value: rounded, and bound to the range of the store. */
static __device__ __forceinline__ short aotx_affect_q15(float value)
{
    float scaled = fminf(fmaxf(rintf(value * 32768.0f), -32768.0f), 32767.0f);
    return (short)scaled;
}

/* The readout of one axis for the drive, or zero for an axis with no row or with a
 * monitor row. A monitor row reaches the trace and never the drive. */
static __device__ __forceinline__ float aotx_affect_probe_drive(unsigned int axis,
                                                                float readout)
{
    for (unsigned int p = 0u; p < aotx_affect_rows.count; ++p) {
        if (aotx_affect_rows.row[p].axis == axis) {
            return (aotx_affect_rows.row[p].monitor != 0u) ? 0.0f : readout;
        }
    }
    return 0.0f;
}

/* The update law of one agent at the end of its turn. The drive is the event drive and
 * the probe drive at the probe gain: the reply readout for valence, the prompt readout
 * for arousal. Each part computes in float from the quantized value and is quantized
 * before the store, so the table holds what the record holds. The effective state is the
 * sum of the two parts, bound by the cap of the axis; the return flags a cap that bound
 * it. The scale of the next turn is one. */
static __device__ __forceinline__ unsigned int aotx_affect_update(unsigned int agent,
                                                                  unsigned int mask,
                                                                  const aotx_affect_sums *acc,
                                                                  short *effective)
{
    aotx_affect_agent_state *state = &aotx_affect_state[agent];
    const aotx_affect_law *law = &aotx_affect_laws[agent];
    float drive[AOTX_AFFECT_STATE_AXES];
    aotx_affect_event_drive(mask, drive);
    if (law->probe_gain != 0.0f) {
        float reply = aotx_affect_finite(aotx_affect_mean(acc->reply_sum[0], acc->reply_rows));
        drive[0] += law->probe_gain * aotx_affect_probe_drive(0u, reply);
        drive[1] += law->probe_gain
                  * aotx_affect_probe_drive(1u, aotx_affect_finite(acc->prompt[1]));
    }
    unsigned int capped = 0u;
    for (unsigned int j = 0u; j < AOTX_AFFECT_STATE_AXES; ++j) {
        float fast = (float)state->fast[j] / 32768.0f;
        float slow = (float)state->slow[j] / 32768.0f;
        short fast_q = aotx_affect_q15(tanhf(law->decay_fast * fast
                                              + law->gain_fast * drive[j]));
        short slow_q = aotx_affect_q15(tanhf(law->decay_slow * slow
                                              + law->gain_slow * drive[j]));
        state->fast[j] = fast_q;
        state->slow[j] = slow_q;
        float cap = (j < AOTX_AFFECT_DATA_AXES) ? law->cap[j] : 1.0f;
        int bound = (int)rintf(cap * 32768.0f);
        int sum = (int)fast_q + (int)slow_q;
        if (sum > bound) {
            sum = bound;
            capped = 1u;
        } else if (sum < -bound) {
            sum = -bound;
            capped = 1u;
        }
        effective[j] = (short)((sum > 32767) ? 32767 : sum);
    }
    state->axes = (unsigned short)AOTX_AFFECT_DATA_AXES;
    return (capped != 0u) ? AOTX_AFFECT_FLAG_CAP : 0u;
}

/* The state record of one agent, from the table. */
static __device__ __forceinline__ void aotx_affect_record(unsigned int agent,
                                                          const aotx_affect_trace_body *trace,
                                                          aotx_affect_body *body)
{
    const aotx_affect_agent_state *state = &aotx_affect_state[agent];
    body->agent = agent;
    body->turn = trace->turn;
    for (unsigned int j = 0u; j < AOTX_AFFECT_STATE_AXES; ++j) {
        body->fast[j] = state->fast[j];
        body->slow[j] = state->slow[j];
    }
    body->scale = state->scale;
    body->axes = state->axes;
    body->reason = trace->reason;
    body->flags = trace->flags;
}

/* An inclusive add over the agents of the block. Every thread of the block takes part. */
static __device__ __forceinline__ unsigned int aotx_affect_scan(unsigned int *cell,
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

__global__ void aotx_affect_turn(void)
{
    __shared__ unsigned int cell[AOTX_SLOTS];
    __shared__ unsigned long long claimed;
    unsigned int agent = threadIdx.x;
    aotx_affect_sums *acc = &aotx_affect_acc[(agent < AOTX_SLOTS) ? agent : 0u];
    unsigned int taken = (agent < AOTX_SLOTS && acc->ended != 0u) ? 1u : 0u;
    /* A replay runs no update and writes no record. The trace is derived and the journal
     * holds the turn. The apply of the restore sets the state from the record. */
    unsigned int writes = (taken != 0u && acc->flag != 0u && aotx_seam.replaying == 0ull)
                        ? 1u : 0u;
    aotx_affect_body body;
    if (taken != 0u) {
        unsigned int guards = 0u;
        for (unsigned int i = 0u; i < aotx_affect_rows.count; ++i) {
            unsigned int axis = aotx_affect_rows.row[i].axis;
            if (axis == 4u || axis == 5u) guards |= 1u << (axis - 4u);
        }
        float first = aotx_affect_mean(acc->reply_sum[4], acc->reply_rows);
        float second = aotx_affect_mean(acc->reply_sum[5], acc->reply_rows);
        aotx_quality_guard(agent, first, second,
                           (acc->flag != 0u && guards == 3u) ? 1u : 0u);
    }
    if (writes != 0u) {
        aotx_affect_trace_body trace;
        aotx_affect_trace(agent, acc, &trace);
        trace.flags |= aotx_affect_state[agent].actuator_flags;
        trace.flags |= aotx_affect_update(agent, trace.reason, acc, trace.effective);
        aotx_seam_write(AOTX_WRITER_AGENT_BASE + agent, AOTX_CLASS_B, AOTX_REC_AFFECT_TRACE,
                        0u, &trace, (unsigned int)sizeof trace);
        aotx_affect_record(agent, &trace, &body);
    }
    /* The state records of the tick take one run of sequences in agent order. One thread
     * claims the run, and one thread folds the bodies into the state hash in that order.
     * That is the order the journal holds and the order the apply of a restore folds. */
    unsigned int scan = aotx_affect_scan(cell, writes);
    unsigned int total = cell[AOTX_SLOTS - 1u];
    if (agent == AOTX_SLOTS - 1u) {
        claimed = (total != 0u) ? aotx_seam_claim(total) : 0ull;
    }
    __syncthreads();
    if (writes != 0u) {
        unsigned long long seq = claimed + (unsigned long long)(scan - 1u);
        aotx_record_header *header = aotx_seam_slot(seq);
        unsigned char *to = aotx_seam_body(header);
        const unsigned char *from = (const unsigned char *)&body;
        for (unsigned int i = 0u; i < (unsigned int)sizeof body; ++i) {
            to[i] = from[i];
        }
        aotx_seam_publish(header, seq, AOTX_WRITER_AGENT_BASE + agent, AOTX_CLASS_A,
                          AOTX_REC_AFFECT, 0u, (unsigned int)sizeof body);
    }
    __syncthreads();
    if (agent == 0u && total != 0u) {
        unsigned long long hash = aotx_seam.apply.state_hash;
        for (unsigned int r = 0u; r < total; ++r) {
            hash = aotx_seam_fnv1a(hash, aotx_seam_body_of(claimed + r),
                                   (unsigned int)sizeof(aotx_affect_body));
        }
        aotx_seam.apply.state_hash = hash;
        aotx_seam.apply.applied_count += (unsigned long long)total;
    }
    /* The turn is taken. The marks of its events go with it, whatever the flag, so no
     * event of a turn with no trace reaches a later one. */
    if (taken != 0u) {
        aotx_affect_sums clear = {};
        *acc = clear;
    }
}

__device__ void aotx_affect_apply(const aotx_affect_body *body)
{
    /* Every class A body folds and counts before semantic apply. A body past the table
     * stays folded and changes no state. */
    if (body->agent >= AOTX_SLOTS) {
        return;
    }
    aotx_affect_agent_state *state = &aotx_affect_state[body->agent];
    for (unsigned int j = 0u; j < AOTX_AFFECT_DATA_AXES; ++j) {
        state->fast[j] = body->fast[j];
        state->slow[j] = body->slow[j];
    }
    for (unsigned int j = AOTX_AFFECT_DATA_AXES; j < AOTX_AFFECT_STATE_AXES; ++j) {
        state->fast[j] = 0;
        state->slow[j] = 0;
    }
    state->scale = body->scale;
    state->axes = (unsigned short)AOTX_AFFECT_DATA_AXES;
    state->actuator_flags = body->flags
                          & (AOTX_AFFECT_FLAG_COMPOSITE | AOTX_AFFECT_FLAG_BUDGET);
}
