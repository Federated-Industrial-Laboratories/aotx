/* Purpose: Hold the affect state and sums of each agent, the probe rows and the event marks.
 * Owns: The state and the accumulator of each agent slot, the event weight table, the probe
 *   row table and the probe matrix.
 * Launch shape: The conduct kernel and the sample kernel add. One block of one thread for
 *   each agent updates the state and writes the records of a turn.
 * Lifetime: The whole run; the probe rows from model load to model release. */
#ifndef AOTX_AFFECT_CUH
#define AOTX_AFFECT_CUH

#include "model/forward.cuh"
#include "profile/profile.cuh"
#include "settings/settings.cuh"
#include "quality/quality.cuh"

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

/* The flags of the trace record and of the state record. */
#define AOTX_AFFECT_FLAG_PROBES  1u   /* probe rows are loaded */
#define AOTX_AFFECT_FLAG_COMPOSITE 2u /* the composite row is applied */
#define AOTX_AFFECT_FLAG_BUDGET  4u   /* the budget reduced the composite dose */
#define AOTX_AFFECT_FLAG_CAP     8u   /* a cap bound the effective state */

/* The weight of each event on the valence axis and on the arousal axis, in that order.
 * The weights are constants and not settings. A restore applies the recorded state and
 * never computes it again, so a changed table cannot make a journal diverge.
 * X(bit, valence, arousal). */
#define AOTX_AFFECT_EVENT_TABLE(X) \
    X(AOTX_AFFECT_EVENT_STOP,            0.10f,  0.00f) \
    X(AOTX_AFFECT_EVENT_LIMIT,          -0.25f,  0.25f) \
    X(AOTX_AFFECT_EVENT_OPERATOR_STOP,  -0.50f,  0.50f) \
    X(AOTX_AFFECT_EVENT_ROLE_REFUSED,   -0.50f,  0.25f) \
    X(AOTX_AFFECT_EVENT_TOOL_OK,         0.50f,  0.00f) \
    X(AOTX_AFFECT_EVENT_TOOL_ERROR,     -0.50f,  0.25f) \
    X(AOTX_AFFECT_EVENT_TOOL_REFUSED,   -0.30f,  0.00f) \
    X(AOTX_AFFECT_EVENT_DEADLINE,       -0.50f,  0.50f) \
    X(AOTX_AFFECT_EVENT_TASK_DONE,       0.50f,  0.00f) \
    X(AOTX_AFFECT_EVENT_TASK_FAILED,    -0.50f,  0.25f) \
    X(AOTX_AFFECT_EVENT_BUDGET,         -0.25f,  0.25f) \
    X(AOTX_AFFECT_EVENT_ROOM_CUT,       -0.10f,  0.00f) \
    X(AOTX_AFFECT_EVENT_THINK_RATIO,     0.00f,  0.25f) \
    X(AOTX_AFFECT_EVENT_LOW_LOGPROB,    -0.10f,  0.25f) \
    X(AOTX_AFFECT_EVENT_VERDICT_REFUTE, -0.25f,  0.00f)

/* The state of one agent. It holds the fast part and the slow part of each axis in Q1.15.
 * It holds the budget scale of the next turn in Q0.16 and the count of the axes that carry
 * data. Two axes carry data; the other two stay zero. The zero state is the neutral state.
 * The table holds the quantized values, so it holds exactly what the state record holds. */
#define AOTX_AFFECT_STATE_AXES  4u
#define AOTX_AFFECT_DATA_AXES   2u
#define AOTX_AFFECT_ONE         32768    /* 1.0 in Q1.15; the store bound is 32767 */
#define AOTX_AFFECT_SCALE_ONE   65535u   /* 1.0 in Q0.16 */

typedef struct aotx_affect_agent_state {
    short fast[AOTX_AFFECT_STATE_AXES];
    short slow[AOTX_AFFECT_STATE_AXES];
    unsigned short scale;
    unsigned short axes;
    unsigned int actuator_flags;
    float budget_spent;         /* the divergence the composite applied at the open, in nats */
} aotx_affect_agent_state;

extern __device__ aotx_affect_agent_state aotx_affect_state[AOTX_SLOTS];

/* The law settings of one open sequence. The turn uses this copy, so a setting change
 * during the reply applies only to the next sequence. */
typedef struct aotx_affect_law {
    float probe_gain;
    float decay_fast;
    float decay_slow;
    float gain_fast;
    float gain_slow;
    float cap[AOTX_AFFECT_DATA_AXES];
    float temperature_gain;
    float voice_gain;
    float steer_gain;
    float budget;
    float temperature_base;     /* the temperature of the row before the coupling */
    unsigned int on;
} aotx_affect_law;

extern __device__ aotx_affect_law aotx_affect_laws[AOTX_SLOTS];

/* The two marked composite directions and their quadratic budget matrix. The direction
 * rows and the built agent rows use the compact layer order of the layer mask. */
typedef struct aotx_affect_composite_desc {
    float K[AOTX_AFFECT_DATA_AXES][AOTX_AFFECT_DATA_AXES];
    unsigned long long layers;
    unsigned int layer_count;
    unsigned int hidden;
    unsigned int trusted;
} aotx_affect_composite_desc;

extern __device__ aotx_affect_composite_desc aotx_affect_composite_table;
extern __device__ const float *aotx_affect_composite;
extern __device__ float *aotx_affect_steer;

/* Copy the sequence-bound settings into the law table of one agent. */
__device__ __forceinline__ void aotx_affect_snapshot(unsigned int agent)
{
    aotx_affect_law *law = &aotx_affect_laws[agent];
    law->probe_gain = aotx_setting_fraction(AOTX_SET_AFFECT_PROBE_GAIN);
    law->decay_fast = aotx_setting_fraction(AOTX_SET_AFFECT_DECAY_FAST);
    law->decay_slow = aotx_setting_fraction(AOTX_SET_AFFECT_DECAY_SLOW);
    law->gain_fast = aotx_setting_fraction(AOTX_SET_AFFECT_GAIN_FAST);
    law->gain_slow = aotx_setting_fraction(AOTX_SET_AFFECT_GAIN_SLOW);
    law->cap[0] = aotx_setting_fraction(AOTX_SET_AFFECT_CAP_VALENCE);
    law->cap[1] = aotx_setting_fraction(AOTX_SET_AFFECT_CAP_AROUSAL);
    law->temperature_gain = aotx_setting_fraction(AOTX_SET_AFFECT_TEMPERATURE_GAIN);
    law->voice_gain = aotx_setting_fraction(AOTX_SET_AFFECT_VOICE_GAIN);
    law->steer_gain = (aotx_affect_composite_table.trusted != 0u)
                    ? aotx_setting_fraction(AOTX_SET_AFFECT_STEER_GAIN) : 0.0f;
    law->budget = aotx_setting_fraction(AOTX_SET_AFFECT_BUDGET);
    law->temperature_base = 0.0f;
    law->on = (aotx_setting_count(AOTX_SET_AFFECT_ON) != 0u) ? 1u : 0u;
}

/* The drive of the events of one turn on the four axes: the sum of the weights of the
 * events that fired. Both sides compute it, so a check states the same table. */
__host__ __device__ __forceinline__ void aotx_affect_event_drive(unsigned int mask,
                                                                  float *drive)
{
    for (unsigned int j = 0u; j < AOTX_AFFECT_STATE_AXES; ++j) {
        drive[j] = 0.0f;
    }
#define AOTX_AFFECT_ONE_EVENT(bit, valence, arousal) \
    if ((mask & (1u << (bit))) != 0u) { drive[0] += (valence); drive[1] += (arousal); }
    AOTX_AFFECT_EVENT_TABLE(AOTX_AFFECT_ONE_EVENT)
#undef AOTX_AFFECT_ONE_EVENT
}

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
    float entropy_base_sum;     /* the same picks at the base temperature */
    float class_sum;            /* the class probability the voice bias moved */
    unsigned int sampled;
    unsigned int flag;          /* the affect mark of the sequence of the turn */
    unsigned int events;        /* event bits the agent step marks */
    unsigned int ended;         /* 1 from the end of the turn to the turn node */
} aotx_affect_sums;

extern __device__ aotx_affect_sums aotx_affect_acc[AOTX_SLOTS];

/* Take the affect mark into the how row of a sequence that opens and clear the sums. The
 * say path calls this for its slot before the open. An open with the setting off sets the
 * state of the agent to zero, so a later turn with the setting on starts neutral. */
__device__ __forceinline__ void aotx_affect_open(unsigned int agent, aotx_model_how *how)
{
    if (agent >= AOTX_SLOTS) {
        return;
    }
    aotx_affect_sums *acc = &aotx_affect_acc[agent];
    aotx_affect_snapshot(agent);
    how->affect = aotx_affect_laws[agent].on;
    if (how->affect == 0u) {
        aotx_affect_agent_state neutral = {};
        neutral.scale = (unsigned short)AOTX_AFFECT_SCALE_ONE;
        neutral.axes = (unsigned short)AOTX_AFFECT_DATA_AXES;
        aotx_affect_state[agent] = neutral;
    }
    for (unsigned int i = 0u; i < AOTX_AFFECT_AXES; ++i) {
        acc->prompt[i] = 0.0f;
        acc->reply_sum[i] = 0.0f;
    }
    acc->reply_rows = 0u;
    acc->logprob_sum = 0.0f;
    acc->entropy_sum = 0.0f;
    acc->entropy_base_sum = 0.0f;
    acc->class_sum = 0.0f;
    acc->sampled = 0u;
    acc->flag = how->affect;
    aotx_quality_open(agent, how);
}

/* Apply the coupling figures of the open sequence to its sampler row. */
__device__ void aotx_affect_apply_how(unsigned int agent, aotx_model_how *how);

/* Build the composite row for every agent whose sequence opens in this tick. */
__global__ void aotx_affect_build(void);

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
                                                 float logprob, float entropy,
                                                 float entropy_base, float class_shift)
{
    if (how == 0 || how->affect == 0u || agent >= AOTX_SLOTS) {
        return;
    }
    aotx_affect_sums *acc = &aotx_affect_acc[agent];
    acc->logprob_sum += logprob;
    acc->entropy_sum += entropy;
    acc->entropy_base_sum += entropy_base;
    acc->class_sum += class_shift;
    acc->sampled += 1u;
}

/* Read one residual row along the probe rows of a layer and add to the sums of its agent.
 * The whole block of the conduct kernel makes the call, because the block reduces. */
__device__ void aotx_affect_readout(const aotx_model_run *run, unsigned int hidden,
                                    const float *resid, unsigned int row,
                                    unsigned int seq, unsigned int layer);

/* Update the state and write the trace and the state record of every turn that ended in
 * this tick. One block of AOTX_SLOTS threads, one for each agent. The block claims one run
 * of sequences for the state records and folds them into the state hash in order. */
__global__ void aotx_affect_turn(void);

/* Set the state of one agent from a state record. The apply of a restore calls this, so
 * the state after a restore is the state the recorded run had. */
__device__ void aotx_affect_apply(const aotx_affect_body *body);

/* Host glue: load the probe rows of a model store, release them, and capture the node. */
int aotx_affect_load_store(const char *dir);
int aotx_affect_load_calibration(const char *dir);
void aotx_affect_release_calibration(void);
void aotx_affect_release(void);
int aotx_affect_capture(void *stream);

#endif
