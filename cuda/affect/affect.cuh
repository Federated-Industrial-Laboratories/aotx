/* Purpose: Hold the affect sums of each agent, the probe rows and the event marks.
 * Owns: The accumulator of each agent slot, the probe row table and the probe matrix.
 * Launch shape: The conduct kernel and the sample kernel add; one thread for each agent
 *   writes the trace of a turn.
 * Lifetime: The whole run; the probe rows from model load to model release. */
#ifndef AOTX_AFFECT_CUH
#define AOTX_AFFECT_CUH

#include "model/forward.cuh"
#include "profile/profile.cuh"
#include "settings/settings.cuh"

/* The axes a probe file can name. The trace record carries the first four and the two
 * guard axes after them. */
#define AOTX_AFFECT_AXES        6u
#define AOTX_AFFECT_TRACE_AXES  4u
#define AOTX_AFFECT_GUARD_AXIS  4u

/* A row whose held-out accuracy is under this value reaches the trace only. */
#define AOTX_AFFECT_MONITOR_ACCURACY 0.8f

/* The events of one turn, as bits of the reason mask of the trace record. */
#define AOTX_AFFECT_EVENT_STOP           0u   /* the stop token ended the reply */
#define AOTX_AFFECT_EVENT_LIMIT          1u   /* the reply limit ended the reply */
#define AOTX_AFFECT_EVENT_OPERATOR_STOP  2u   /* the operator stopped the reply */
#define AOTX_AFFECT_EVENT_ROLE_REFUSED   3u   /* the role does not hold the tool called */
#define AOTX_AFFECT_EVENT_TOOL_OK        4u   /* the turn carried a good tool result */
#define AOTX_AFFECT_EVENT_TOOL_ERROR     5u   /* the turn carried an error or a late result */
#define AOTX_AFFECT_EVENT_TOOL_REFUSED   6u   /* the operator refused the tool call */
#define AOTX_AFFECT_EVENT_DEADLINE       7u   /* the deadline passed with no result */
#define AOTX_AFFECT_EVENT_TASK_DONE      8u   /* the task ended done */
#define AOTX_AFFECT_EVENT_TASK_FAILED    9u   /* the task ended failed */
#define AOTX_AFFECT_EVENT_BUDGET         10u  /* the turn budget is used up */
#define AOTX_AFFECT_EVENT_ROOM_CUT       11u  /* a tool result was cut to the room */
#define AOTX_AFFECT_EVENT_THINK_RATIO    12u  /* more than half of the reply was thought */
#define AOTX_AFFECT_EVENT_LOW_LOGPROB    13u  /* the mean logprob is under the bound */
#define AOTX_AFFECT_EVENT_VERDICT_REFUTE 14u  /* the verdict of the turn is refute */
#define AOTX_AFFECT_EVENT_MASK           0x7fffu

/* The mean logprob, in nats, under which the low logprob event fires. */
#define AOTX_AFFECT_LOGPROB_BOUND        (-1.5f)

/* The flags of the trace record. */
#define AOTX_AFFECT_FLAG_PROBES  1u   /* probe rows are loaded */

/* One probe row: the layer it reads, the axis it names and its standardization. */
typedef struct aotx_affect_row {
    unsigned int layer;
    unsigned int axis;
    float mean;
    float scale;
    unsigned int monitor;       /* 1 when the row reaches the trace only */
} aotx_affect_row;

typedef struct aotx_affect_table {
    aotx_affect_row row[AOTX_AFFECT_AXES];
    unsigned int count;
    unsigned int hidden;        /* the residual width of every row */
    unsigned long long layers;  /* one bit for each layer a row reads */
} aotx_affect_table;

extern __device__ aotx_affect_table aotx_affect_rows;
extern __device__ const float *aotx_affect_probe;  /* count rows of hidden floats */

/* The sums of one agent over the turn in hand. The open of a sequence clears the sums and
 * takes the affect mark of its how row. The marks of the events stay across the open,
 * because the tool events of a turn are marked before its sequence opens. */
typedef struct aotx_affect_sums {
    float prompt[AOTX_AFFECT_AXES];
    float reply_sum[AOTX_AFFECT_AXES];
    unsigned int reply_rows;
    float logprob_sum;
    float entropy_sum;
    unsigned int sampled;
    unsigned int flag;          /* the affect mark of the sequence of the turn */
    unsigned int events;        /* event bits the agent step marks */
    unsigned int ended;         /* 1 from the end of the turn to the turn node */
} aotx_affect_sums;

extern __device__ aotx_affect_sums aotx_affect_acc[AOTX_SLOTS];

/* Take the affect mark into the how row of a sequence that opens and clear the sums. The
 * say path calls this for its slot before the open. */
__device__ __forceinline__ void aotx_affect_open(unsigned int agent, aotx_model_how *how)
{
    if (agent >= AOTX_SLOTS) {
        return;
    }
    aotx_affect_sums *acc = &aotx_affect_acc[agent];
    how->affect = (aotx_setting_count(AOTX_SET_AFFECT_ON) != 0u) ? 1u : 0u;
    for (unsigned int i = 0u; i < AOTX_AFFECT_AXES; ++i) {
        acc->prompt[i] = 0.0f;
        acc->reply_sum[i] = 0.0f;
    }
    acc->reply_rows = 0u;
    acc->logprob_sum = 0.0f;
    acc->entropy_sum = 0.0f;
    acc->sampled = 0u;
    acc->flag = how->affect;
}

/* Mark one event of the turn of an agent. The thread of the agent calls this. */
__device__ __forceinline__ void aotx_affect_mark(unsigned int agent, unsigned int bit)
{
    if (agent < AOTX_SLOTS) {
        aotx_affect_acc[agent].events |= 1u << bit;
    }
}

/* Mark the end of the turn of an agent. The turn node of the same tick reads it. */
__device__ __forceinline__ void aotx_affect_end(unsigned int agent)
{
    if (agent < AOTX_SLOTS) {
        aotx_affect_acc[agent].ended = 1u;
    }
}

/* Add the figures of one emitted token. The sample kernel calls this from one thread for
 * each sequence, so the adds take no atomic. */
__device__ __forceinline__ void aotx_affect_pick(unsigned int agent,
                                                 const aotx_model_how *how,
                                                 float logprob, float entropy)
{
    if (how == 0 || how->affect == 0u || agent >= AOTX_SLOTS) {
        return;
    }
    aotx_affect_sums *acc = &aotx_affect_acc[agent];
    acc->logprob_sum += logprob;
    acc->entropy_sum += entropy;
    acc->sampled += 1u;
}

/* Read one residual row along the probe rows of a layer and add to the sums of its agent.
 * The whole block of the conduct kernel makes the call, because the block reduces. */
__device__ void aotx_affect_readout(const aotx_model_run *run, unsigned int hidden,
                                    const float *resid, unsigned int row,
                                    unsigned int seq, unsigned int layer);

/* Write the trace of every turn that ended in this tick. One thread for each agent. */
__global__ void aotx_affect_turn(void);

/* Host glue: load the probe rows of a model store, release them, and capture the node. */
int aotx_affect_load_store(const char *dir);
void aotx_affect_release(void);
int aotx_affect_capture(void *stream);

#endif
