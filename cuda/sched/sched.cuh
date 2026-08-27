/* Purpose: Run one tick: read the cursor, hold agents, set parameters.
 * Owns: The work queues and the tick statistics.
 * Launch shape: One block for each queue.
 * Lifetime: The whole run. */
#ifndef AOTX_SCHED_CUH
#define AOTX_SCHED_CUH

#include "seam/seam.cuh"

typedef struct aotx_sched_state {
    unsigned long long held;         /* 1 when the tick is held, 0 when it runs */
    unsigned long long holding;      /* 1 while a hold lasts; a hold writes one stall record */
    unsigned long long held_count;   /* ticks held since start */
    unsigned long long free_bytes;   /* host ring bytes free at tick start */
    unsigned long long overrun_seen; /* the drop count that the last stall record reported */
    unsigned long long blocks;       /* blocks published since start */
    unsigned long long records;      /* the sequence of the last commit record */
} aotx_sched_state;

extern __device__ aotx_sched_state aotx_sched;

__global__ void aotx_sched_tick_start(unsigned long long workload);
__global__ void aotx_sched_workload(unsigned long long workload);
__global__ void aotx_sched_commit(void);

/* Threads in one block of the tick load. */
#define AOTX_WORKLOAD_THREADS 256u

/* The tick period. The pump makes at most 100 ticks in one second. */
#define AOTX_TICK_PERIOD_NS   10000000ll

/* What the host glue keeps to launch one tick. The graph holds one node for each kernel and
 * the shape of the graph never changes. */
typedef struct aotx_pump {
    cudaStream_t stream;
    cudaEvent_t event;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaGraphNode_t start_node;   /* the tick start node takes the same parameter */
    cudaGraphNode_t work_node;    /* the tick load node takes the count and the grid */
    unsigned long long workload;  /* records the tick load writes */
    unsigned int blocks;          /* blocks of the tick load */
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
