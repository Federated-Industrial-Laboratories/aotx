/* Purpose: Run one tick: read the cursor, hold agents, set parameters.
 * Owns: The work queues and the tick statistics.
 * Launch shape: One block for each queue.
 * Lifetime: The whole run. */
#ifndef AOTX_SCHED_CUH
#define AOTX_SCHED_CUH

#include "agent/agent.cuh"
#include "kvcache/kvcache.cuh"
#include "model/decode.cuh"
#include "seam/seam.cuh"

typedef struct aotx_sched_state {
    unsigned long long held;         /* 1 when the tick is held, 0 when it runs */
    unsigned long long holding;      /* 1 while a hold lasts; a hold writes one stall record */
    unsigned long long held_count;   /* ticks held since start */
    unsigned long long free_bytes;   /* host ring bytes free at tick start */
    unsigned long long overrun_seen; /* the drop count that the last stall record reported */
    unsigned long long blocks;       /* blocks published since start */
    unsigned long long records;      /* the sequence of the last commit record */
    unsigned long long start_ns;     /* device clock at the start of the tick that runs */
    unsigned long long commit_ns;    /* device clock after the commit record, or zero */
    unsigned long long flush_ns;     /* device time the two flush nodes of the tick before
                                      * took; a commit record cannot wait for its own */
} aotx_sched_state;

extern __device__ aotx_sched_state aotx_sched;

__global__ void aotx_sched_tick_start(unsigned long long workload);
__global__ void aotx_sched_workload(unsigned long long workload);
__global__ void aotx_sched_commit(void);

/* Threads in one block of the tick load. */
#define AOTX_WORKLOAD_THREADS 256u

/* Records that the decode of one tick writes at the most. The set is one record for each
 * prompt token of the token budget. It also holds one token record, one event record and
 * one console record and one token statistics record for each sequence slot. A cadence
 * flush can write one page record for every page of every slot. */
#define AOTX_DECODE_RECORDS_MAX ((unsigned long long)AOTX_SEQ_TICK_BUDGET \
                                 + 4ull * (unsigned long long)AOTX_SLOTS \
                                 + (unsigned long long)AOTX_SLOTS * AOTX_KV_PAGES_EACH)

/* Records the agents and the tools of one tick write at the most. Each agent may write a
 * manifest, a task, an agent, a tool request and a bus record. An affect build adds one
 * trace and one quality record. Each tool may write a finding beside its result. */
#ifdef AOTX_AFFECT
#define AOTX_AGENT_RECORDS_MAX (9ull * (unsigned long long)AOTX_SLOTS)
#else
#define AOTX_AGENT_RECORDS_MAX (8ull * (unsigned long long)AOTX_SLOTS)
#endif

/* Nodes of the tick itself: the tick start, the apply, the tick load, the tick commit, the
 * record flush and the bulk flush. The say path and the decode add their own. */
#define AOTX_TICK_NODES_TICK  6u

/* Nodes of the decode: the plan, the forward pass as one child node, and the commit. */
#define AOTX_TICK_NODES_DECODE 3u

/* Nodes of the say path of the command layer, and of the reply that follows the decode.
 * The say path fills the batch table, cuts the text, merges the pairs, gathers the tokens
 * and opens the sequence. An affect build adds one node when that option is present.
 * The reply takes the new bytes of every live sequence. */
#ifdef AOTX_AFFECT
#define AOTX_TICK_NODES_SAY    7u
#else
#define AOTX_TICK_NODES_SAY    6u
#endif
#define AOTX_TICK_NODES_REPLY  1u

/* Nodes of the tool path. The fill step writes the batch table of the tokenizer and the
 * rows of every module node. Four steps give the tokens and the plan writes the call
 * block. The pass of the embedding role is one child node and the search reads the note
 * store. A run with no embedding role holds the fill and the step alone. One node for each
 * device tool module of the catalog stands beside these. */
#define AOTX_TICK_NODES_TOOL      9u
#define AOTX_TICK_NODES_TOOL_BARE 2u

/* Nodes of the agent path: the step, the affect turn and the quality turn. */
#ifdef AOTX_AFFECT
#define AOTX_TICK_NODES_AGENT  3u
#else
#define AOTX_TICK_NODES_AGENT  1u
#endif

/* Nodes of the tick graph at the most. The graph holds the nodes of the tick, of the say
 * path and of the decode. The forward pass of the decode is one child node. */
#define AOTX_TICK_NODES_MAX   64u

/* What the host glue keeps to launch one tick. The graph holds one node for each kernel and
 * the shape of the graph never changes. */
typedef struct aotx_pump {
    cudaStream_t stream;
    cudaEvent_t event;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaGraphNode_t start_node;   /* the tick start node takes the same parameter */
    cudaGraphNode_t work_node;    /* the tick load node takes the count and the grid */
    aotx_kv_map kv;               /* the page range; the pump answers page requests */
    unsigned long long workload;  /* records the tick load writes */
    unsigned int blocks;          /* blocks of the tick load */
    unsigned int decode;          /* 1 when the graph holds the nodes of the decode */
    unsigned int nodes;           /* nodes of the tick graph */
    unsigned int say_nodes;       /* nodes the say path put in the capture */
    unsigned int decode_nodes;    /* nodes the decode put in the capture */
    unsigned int reply_nodes;     /* nodes the reply of the console put in the capture */
    unsigned int tool_nodes;      /* nodes the tool path put in the capture */
    unsigned int agent_nodes;     /* nodes the agent step put in the capture */
    unsigned int embed;           /* 1 when the graph holds the pass of the embedding role */
    unsigned int modules;         /* device tool modules the graph holds a node for */
    unsigned int gen;             /* the catalog device number the graph was built with */
    unsigned int recaptures;      /* captures the pump made after the first one */
    unsigned int recapture_us;    /* microseconds the last capture took */
    unsigned int console_agent;   /* 1 after the agent of the console took slot 0 */
    unsigned int model_refused;   /* 1 when a replayed model cannot be placed */
    long long next_ns;            /* the time the next tick starts, for the pace */
} aotx_pump;

/* What the pump reports after a tick. */
typedef struct aotx_pump_report {
    unsigned long long records;    /* records written since start */
    unsigned long long blocks;     /* blocks published since start */
    unsigned long long held;       /* ticks held since start */
    unsigned long long tick;       /* the tick that ended last */
    unsigned long long state_hash; /* the state hash after the last apply */
    unsigned long long applied;    /* class A records applied since start */
    unsigned long long rejected;   /* inbound slots the length check refused */
    unsigned long long tail;       /* the last claimed record sequence */
    unsigned long long flushed;    /* the last record sequence in the host ring */
    unsigned long long consumed;   /* inbound slots consumed */
    unsigned long long overrun;    /* runs of records that the flush dropped */
    unsigned long long paced;      /* ticks of a replay that took no record of the journal */
    unsigned int refused;          /* sequence calls the decode refused */
    unsigned int pages;            /* key value cache pages the slots hold */
    unsigned int live;             /* slots that are not free */
    unsigned int console_agent;    /* 1 after the agent of the console took slot 0 */
    unsigned long long model_bytes; /* model bytes placed after boot */
} aotx_pump_report;

/* A sample of the monotonic clock in nanoseconds. The pace of the pump reads it. */
static inline long long aotx_pump_now_ns(void)
{
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (long long)at.tv_sec * 1000000000ll + (long long)at.tv_nsec;
}

/* Capture the tick graph once and instantiate it once. */
int aotx_pump_build(aotx_pump *pump, unsigned long long workload, unsigned int blocks);

/* Capture the tick graph and instantiate it. The instance that stood is given back, so a
 * caller of this function holds no launch of it. */
int aotx_pump_capture(aotx_pump *pump);

/* Capture the tick graph again with the device tools the catalog holds now. The record of
 * the capture names the tick, the node count before and after, and the microseconds. */
int aotx_pump_recapture(aotx_pump *pump);

/* Report whether the catalog holds a set of device tools the graph was not built with. */
int aotx_pump_stale(const aotx_pump *pump);

/* Set the tick load for the next tick. The shape of the graph does not change. */
int aotx_pump_set(aotx_pump *pump, unsigned long long workload, unsigned int blocks);

/* Launch one tick and wait for it on an event. */
void aotx_pump_tick(aotx_pump *pump);

/* Run the flush alone, with no tick. The tick count does not change. */
void aotx_pump_flush(aotx_pump *pump);

/* Sleep the rest of the tick period. */
void aotx_pump_pace(aotx_pump *pump);

/* Read the counters that the tick keeps. */
void aotx_pump_read(aotx_pump_report *report);

/* Give back the graph, the stream and the event. */
void aotx_pump_close(aotx_pump *pump);

#endif
