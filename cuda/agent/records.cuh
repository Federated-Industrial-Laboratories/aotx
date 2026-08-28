/* Purpose: Write the agent, task and manifest records of the tick.
 * Owns: Nothing; the record ring holds the records.
 * Launch shape: Device functions; one call for each event.
 * Lifetime: Each call.
 *
 * The three records are derived. The tokens and the operator lines that made them are the
 * class A records. A restore therefore rebuilds these three from those. The writer of each
 * one is the agent, so a reader of the seam knows whose event it is. */
#ifndef AOTX_AGENT_RECORDS_CUH
#define AOTX_AGENT_RECORDS_CUH

#include "agent/agent_state.cuh"
#include "seam/seam.cuh"

/* One event of one agent. */
__device__ __forceinline__ void aotx_agent_note(unsigned int agent, unsigned int event,
                                                unsigned long long tick)
{
    if (agent >= AOTX_AGENT_SLOTS) {
        return;
    }
    const aotx_agent *me = &aotx_agents.agent[agent];
    aotx_agent_body body;
    body.agent = agent;
    body.role = me->role;
    body.parent = me->parent;
    body.state = me->state;
    body.event = event;
    body.turn = me->turn;
    body.ticks = (tick > me->spawned) ? (tick - me->spawned) : 0ull;
    aotx_seam_write(AOTX_WRITER_AGENT_BASE + agent, AOTX_CLASS_B, AOTX_REC_AGENT, 0u,
                    &body, (unsigned int)sizeof body);
}

/* One event of one task. The text of the record is the task text, or the first bytes of
 * the result when the task ends. */
__device__ __forceinline__ void aotx_task_note(unsigned int task, unsigned int writer,
                                               const char *text, unsigned int length,
                                               unsigned long long tick)
{
    if (task >= AOTX_TASK_SLOTS) {
        return;
    }
    const aotx_task *hold = &aotx_agents.task[task];
    aotx_task_body body;
    body.task = task;
    body.agent = hold->agent;
    body.state = hold->state;
    body.verify = hold->verify;
    body.attempts = hold->attempts;
    unsigned int bytes = (length > AOTX_TASK_TEXT_BYTES) ? AOTX_TASK_TEXT_BYTES : length;
    body.text_len = bytes;
    body.ticks = (tick > hold->opened) ? (tick - hold->opened) : 0ull;
    for (unsigned int i = 0u; i < AOTX_TASK_TEXT_BYTES; ++i) {
        body.text[i] = (i < bytes) ? text[i] : '\0';
    }
    aotx_seam_write(writer, AOTX_CLASS_B, AOTX_REC_TASK, 0u, &body,
                    (unsigned int)sizeof body);
}

/* One completed turn of one agent. The prompt is hashed at the build; the reply is hashed
 * here. The record names the tool the reply called and the request it made. */
__device__ __forceinline__ void aotx_agent_manifest(unsigned int agent, unsigned int finish,
                                                    unsigned int tool, unsigned int request)
{
    if (agent >= AOTX_AGENT_SLOTS) {
        return;
    }
    const aotx_agent_work *gear = &aotx_agent_gear[agent];
    aotx_manifest_body body;
    body.agent = agent;
    body.turn = aotx_agents.agent[agent].turn;
    body.input_hash = gear->input_hash;
    body.output_hash = aotx_agent_hash(gear->reply, gear->reply_len);
    body.output_tokens = aotx_seqs.slot[agent].sampled;
    body.finish = finish;
    body.tool = tool;
    body.request = request;
    aotx_seam_write(AOTX_WRITER_AGENT_BASE + agent, AOTX_CLASS_B, AOTX_REC_MANIFEST, 0u,
                    &body, (unsigned int)sizeof body);
}

#endif
