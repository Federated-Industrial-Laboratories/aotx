/* Purpose: Read the probe rows of a residual row and write the trace of each turn.
 * Owns: The accumulator of each agent, the probe row table and the probe matrix pointer.
 * Launch shape: The readout runs in the block of one residual row; the turn node runs one
 *   thread for each agent after the agent step.
 * Lifetime: The whole run. */
#include "affect/affect.cuh"

#include "agent/agent_state.cuh"
#include "model/decode_state.cuh"
#include "seam/seam.cuh"

__device__ aotx_affect_table aotx_affect_rows;
__device__ const float *aotx_affect_probe;
__device__ aotx_affect_sums aotx_affect_acc[AOTX_SLOTS];

/* The block reduces the dot product of the row with each probe row of this layer. Every
 * warp adds with a shuffle, then one thread adds over the warps. The last prompt row gives
 * the prompt readout of the axis, and a reply row goes in the running sum. The first row
 * of the table alone counts the reply rows. Every row of the batch passes the layer of
 * that row one time, so the count is exact. The whole block makes the call, because the
 * block synchronizes here. */
__device__ void aotx_affect_readout(const aotx_model_run *run, unsigned int hidden,
                                    const float *resid, unsigned int row,
                                    unsigned int seq, unsigned int layer)
{
    __shared__ float part[AOTX_MODEL_ROW_THREADS / 32u];
    unsigned int agent = run->agent[seq];
    /* The shuffle takes whole warps, so a block that is not a multiple of 32 reads none. */
    if (agent >= AOTX_SLOTS || aotx_affect_rows.hidden != hidden
        || blockDim.x > AOTX_MODEL_ROW_THREADS || (blockDim.x & 31u) != 0u) {
        return;
    }
    unsigned int position = aotx_decode.first[agent] + (row - run->offset[seq]);
    unsigned int prompt = aotx_seqs.slot[agent].prompt;
    for (unsigned int p = 0u; p < aotx_affect_rows.count; ++p) {
        const aotx_affect_row *probe = &aotx_affect_rows.row[p];
        if (probe->layer != layer) {
            continue;
        }
        const float *direction = aotx_affect_probe + (unsigned long long)p * hidden;
        float sum = 0.0f;
        for (unsigned int x = threadIdx.x; x < hidden; x += blockDim.x) {
            sum += resid[x] * direction[x];
        }
        for (unsigned int lane = 16u; lane > 0u; lane >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, lane);
        }
        if ((threadIdx.x & 31u) == 0u) {
            part[threadIdx.x >> 5] = sum;
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            float total = 0.0f;
            for (unsigned int w = 0u; w < (blockDim.x + 31u) / 32u; ++w) {
                total += part[w];
            }
            float value = (total - probe->mean) / probe->scale;
            aotx_affect_sums *acc = &aotx_affect_acc[agent];
            if (position + 1u == prompt) {
                acc->prompt[probe->axis] = value;
            } else if (position >= prompt) {
                atomicAdd(&acc->reply_sum[probe->axis], value);
                if (p == 0u) {
                    atomicAdd(&acc->reply_rows, 1u);
                }
            }
        }
        __syncthreads();
    }
}

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

/* Write the trace of one agent whose turn ended in this tick. The think count comes from
 * the sequence of the turn, and a turn with no reply states zero. */
static __device__ __forceinline__ void aotx_affect_trace(unsigned int agent,
                                                         const aotx_affect_sums *acc)
{
    const aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int think = (gear->out_tokens != 0u) ? aotx_seqs.slot[agent].think_tokens : 0u;
    aotx_affect_trace_body body;
    body.agent = agent;
    body.turn = aotx_agents.agent[agent].turn;
    for (unsigned int i = 0u; i < AOTX_AFFECT_TRACE_AXES; ++i) {
        body.prompt[i] = aotx_affect_finite(acc->prompt[i]);
        body.reply[i] = aotx_affect_finite(aotx_affect_mean(acc->reply_sum[i],
                                                            acc->reply_rows));
        body.effective[i] = 0;
    }
    for (unsigned int i = 0u; i < AOTX_AFFECT_AXES - AOTX_AFFECT_GUARD_AXIS; ++i) {
        body.guard[i] = aotx_affect_finite(
            aotx_affect_mean(acc->reply_sum[AOTX_AFFECT_GUARD_AXIS + i], acc->reply_rows));
    }
    body.logprob = aotx_affect_finite(aotx_affect_mean(acc->logprob_sum, acc->sampled));
    body.entropy = aotx_affect_finite(aotx_affect_mean(acc->entropy_sum, acc->sampled));
    body.entropy = (body.entropy < 0.0f) ? 0.0f : body.entropy;
    body.rows = acc->reply_rows;
    body.think = think;
    body.reason = aotx_affect_events(agent, acc, think);
    body.flags = (aotx_affect_rows.count != 0u) ? AOTX_AFFECT_FLAG_PROBES : 0u;
    aotx_seam_write(AOTX_WRITER_AGENT_BASE + agent, AOTX_CLASS_B, AOTX_REC_AFFECT_TRACE, 0u,
                    &body, (unsigned int)sizeof body);
}

__global__ void aotx_affect_turn(void)
{
    unsigned int agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent >= AOTX_SLOTS) {
        return;
    }
    aotx_affect_sums *acc = &aotx_affect_acc[agent];
    if (acc->ended == 0u) {
        return;
    }
    unsigned int guards = 0u;
    for (unsigned int i = 0u; i < aotx_affect_rows.count; ++i) {
        unsigned int axis = aotx_affect_rows.row[i].axis;
        if (axis == 4u || axis == 5u) guards |= 1u << (axis - 4u);
    }
    float first = aotx_affect_mean(acc->reply_sum[4], acc->reply_rows);
    float second = aotx_affect_mean(acc->reply_sum[5], acc->reply_rows);
    aotx_quality_guard(agent, first, second,
                       (acc->flag != 0u && guards == 3u) ? 1u : 0u);
    /* A replay writes no trace. The trace is derived, and the journal holds the turn. */
    if (acc->flag != 0u && aotx_seam.replaying == 0ull) {
        aotx_affect_trace(agent, acc);
    }
    /* The turn is taken. The marks of its events go with it, whatever the flag, so no
     * event of a turn with no trace reaches a later one. */
    aotx_affect_sums clear = {};
    *acc = clear;
}
