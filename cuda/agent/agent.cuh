/* Purpose: Step each agent through its states.
 * Owns: The agent records and the task table; the catalog holds the roles.
 * Launch shape: One thread for each agent in the control step.
 * Lifetime: From agent creation to agent release. */
#ifndef AGENT_CUH
#define AGENT_CUH

#include "catalog/catalog.cuh"
#include "profile/profile.cuh"
#include "seam/wire.h"

/* Agent i owns sequence slot i, so the agent count is AOTX_SLOTS of the profile. */
#define AOTX_TASK_SLOTS        256u

/* A role is a module of the catalog. The role of an agent is the entry of that module.
 * The row of the entry names the tools, the skills, the model and the budget. A value of
 * AOTX_MODULE_SLOTS stands for no role. */
#define AOTX_ROLE_NONE         AOTX_MODULE_SLOTS

/* Agent states. The sequence states of decode.cuh sit inside PREFILL and DECODE. */
#define AOTX_AGENT_STATE_FREE   0u
#define AOTX_AGENT_STATE_IDLE   1u    /* spawned, no prompt in hand */
#define AOTX_AGENT_STATE_PROMPT 2u    /* a prompt is being built and tokenized */
#define AOTX_AGENT_STATE_RUN    3u    /* the sequence runs: prefill or decode */
#define AOTX_AGENT_STATE_TOOL   4u    /* a tool runs, or a request waits for its reply */
#define AOTX_AGENT_STATE_POST   5u    /* the turn is recorded and the next is decided */

typedef struct aotx_agent {
    unsigned int state;         /* AOTX_AGENT_STATE_* */
    unsigned int role;          /* the catalog entry of the role, or AOTX_ROLE_NONE */
    unsigned int parent;        /* the agent that made it; itself for a root */
    unsigned int turn;          /* turns taken on the current task */
    unsigned int task;          /* the task in hand, or ~0u */
    unsigned int request;       /* the pending tool request, or 0 */
    unsigned int tool;          /* the catalog entry of the pending tool, or none */
    unsigned int budget_left;
    unsigned long long deadline;  /* the tick the pending request fails */
    unsigned long long spawned;   /* the tick the agent spawned */
    unsigned long long mailbox;   /* the bus record seq the agent has read to */
    unsigned int verdict;         /* a verifier's last verdict: 0 none, 1 uphold, 2 refute, 3 uncertain */
    unsigned int reserved;
} aotx_agent;

typedef struct aotx_task {
    unsigned int state;         /* AOTX_TASK_* */
    unsigned int agent;         /* the assignee, or ~0u */
    unsigned int giver;         /* the agent or operator that opened it; ~0u for the operator */
    unsigned int verify;        /* AOTX_VERIFY_* */
    unsigned int attempts;
    unsigned int verifier;      /* the verifier agent, or ~0u */
    unsigned int text_len;
    unsigned int result_len;
    unsigned long long opened;  /* the tick the task opened */
    char text[AOTX_TASK_TEXT_BYTES];
    char result[AOTX_TASK_TEXT_BYTES];
} aotx_task;

typedef struct aotx_agent_table {
    aotx_agent agent[AOTX_SLOTS];
    aotx_task task[AOTX_TASK_SLOTS];
    unsigned int live;
    unsigned int tasks;
    unsigned int refused;
    unsigned int next_request;  /* request ids, from 1 */
} aotx_agent_table;

extern __device__ aotx_agent_table aotx_agents;

/* Spawn an agent of a role on a free slot. The role is a catalog entry. Returns the slot,
 * or ~0u when none is free. The caller is the command layer's serial thread. Agent 0 is
 * the agent of the role the console speaks to, spawned when that role is installed. */
__device__ unsigned int aotx_agent_spawn(unsigned int role, unsigned int parent,
                                         unsigned long long tick);

/* Give an agent a message from the operator: the conductor's `say`. The agent's next prompt
 * carries it. Returns 0, or 1 when the agent is not idle. */
__device__ int aotx_agent_message(unsigned int agent, const unsigned char *text,
                                  unsigned int length, unsigned long long tick);

/* Open a task for an agent, or for the first idle agent of a role when agent is ~0u.
 * Returns the task index, or ~0u when the table is full. */
__device__ unsigned int aotx_task_open(unsigned int agent, unsigned int role,
                                       const unsigned char *text, unsigned int length,
                                       unsigned int verify, unsigned long long tick);

/* Answer a pending authorization. Returns 0, or 1 when no such request waits. */
__device__ int aotx_agent_authorize(unsigned int request, unsigned int granted,
                                    unsigned long long tick);

/* The agent step of the tick, after the decode commit and the tool step. An agent whose
 * sequence ended takes the reply and records the turn as a MANIFEST record. It then parses a
 * tool call or ends the task. An idle agent with a message or a task builds its next prompt.
 *
 * The engine gives pending tasks to idle agents by role. It sends a done result to a verifier
 * when the task asks for it. One thread for each agent. */
__global__ void aotx_agent_step(unsigned long long tick);

/* Host glue: capture the agent step and the prompt path into the tick stream. */
int aotx_agent_capture(void *stream);

#endif
