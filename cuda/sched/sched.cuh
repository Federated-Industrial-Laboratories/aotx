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
 * one console record for each sequence slot. */
#define AOTX_DECODE_RECORDS_MAX ((unsigned long long)AOTX_SEQ_TICK_BUDGET \
                                 + 3ull * (unsigned long long)AOTX_SLOTS)

/* Records the agents and the tools of one tick write at the most. Each agent may write a
 * manifest record, a task record, an agent record, a tool request and a bus message. Each
 * tool may write a finding beside its result. */
#define AOTX_AGENT_RECORDS_MAX (8ull * (unsigned long long)AOTX_SLOTS)

/* Nodes of the tick itself: the tick start, the apply, the tick load, the tick commit, the
 * record flush and the bulk flush. The say path and the decode add their own. */
#define AOTX_TICK_NODES_TICK  6u

/* Nodes of the decode: the plan, the forward pass as one child node, and the commit. */
#define AOTX_TICK_NODES_DECODE 3u

/* Nodes of the say path of the command layer, and of the reply that follows the decode.
 * The say path fills the batch table, cuts the text, merges the pairs, gathers the tokens
 * and opens the sequence. The reply takes the new bytes of every live sequence. */
#define AOTX_TICK_NODES_SAY    6u
#define AOTX_TICK_NODES_REPLY  1u

/* Nodes of the tool path. The fill step writes the batch table of the tokenizer. Four
 * steps give the tokens and the plan writes the call block. The pass of the embedding role
 * is one child node and the search reads the note store. The step gives every result. A
 * run with no embedding role holds the step alone. */
#define AOTX_TICK_NODES_TOOL      9u
#define AOTX_TICK_NODES_TOOL_BARE 1u

/* Nodes of the agent path: the agent step. */
#define AOTX_TICK_NODES_AGENT  1u

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
} aotx_pump_report;

/* Capture the tick graph once and instantiate it once. */
int aotx_pump_build(aotx_pump *pump, unsigned long long workload, unsigned int blocks);

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
