/* Purpose: Open a tool request and apply the reply of a host tool to it.
 * Owns: The request table of the run.
 * Launch shape: Device functions; one call for each request and for each reply part.
 * Lifetime: The whole run.
 *
 * A device tool is queued for the tool step of the tick. A host tool writes one record,
 * which the drain gives the feeder. The reply comes back over the inbound ring in parts,
 * and the apply of the tick puts each part in the result of its request. */
#include "agent/agent.cuh"
#include "bus/bus.cuh"
#include "catalog/catalog.cuh"
#include "seam/seam.cuh"
#include "tool/tool_state.cuh"

__device__ aotx_request_table aotx_requests;
__device__ unsigned int aotx_tool_done[AOTX_SLOTS];
__device__ aotx_tool_counts aotx_tool_count;

/* Write the record that names a request. The drain gives it to the feeder. The record is
 * derived from the reply of the agent, so it is class B. */
__device__ void aotx_tool_note_request(const aotx_request *slot, unsigned int turn)
{
    aotx_tool_request_body body;
    body.agent = slot->agent;
    body.turn = turn;
    body.tool = slot->tool;
    body.request = slot->request;
    body.deadline = slot->deadline;
    body.auth = slot->auth;
    body.arg_len = slot->result_len;
    for (unsigned int i = 0u; i < AOTX_TOOL_ARG_BYTES; ++i) {
        body.arg[i] = (i < slot->result_len) ? slot->result[i] : '\0';
    }
    aotx_seam_write(AOTX_WRITER_AGENT_BASE + slot->agent, AOTX_CLASS_B,
                    AOTX_REC_TOOL_REQUEST, 0u, &body, (unsigned int)sizeof body);
}

__device__ unsigned int aotx_tool_request(unsigned int agent, const aotx_tool_call *call,
                                          unsigned int needs_auth, unsigned long long tick)
{
    if (agent >= AOTX_SLOTS || call == 0
        || aotx_catalog_is(call->entry, AOTX_MODULE_TOOL) == 0) {
        return 0u;
    }
    aotx_request *slot = &aotx_requests.slot[agent];
    if (slot->request != 0u) {
        return 0u;
    }
    /* The number of a request is a value of the slot and the count of the requests that
     * slot made. Two runs of the same inputs give the same number to the same request.
     * A replay therefore finds the request that a recorded reply names. A number from one
     * counter over 64 threads would depend on which thread arrived first. */
    unsigned int made = aotx_tool_embed.made[agent];
    unsigned int id = made * AOTX_SLOTS + agent + 1u;
    aotx_tool_embed.made[agent] = made + 1u;
    atomicAdd(&aotx_agents.next_request, 1u);
    slot->agent = agent;
    slot->entry = call->entry;
    slot->tool = call->tool;
    aotx_tool_embed.prov[agent] = call->provenance;
    slot->status = AOTX_TOOL_OK;
    slot->parts_in = 0u;
    slot->parts = 0u;
    aotx_tool_done[agent] = 0u;

    /* The argument goes in the result field until the reply takes its place. The record
     * of a host tool therefore carries the argument. The text of a device tool stands
     * where the tokenizer of the tool path reads it. */
    unsigned int bytes = (call->arg_len > AOTX_TOOL_ARG_BYTES) ? AOTX_TOOL_ARG_BYTES
                                                               : call->arg_len;
    for (unsigned int i = 0u; i < bytes; ++i) {
        slot->result[i] = call->arg[i];
    }
    slot->result_len = bytes;

    /* A tool of the memory pair needs the vector of its text. That text goes through the
     * tokenizer of the tool path and the pass of the embedding role. Every other tool ends
     * in the tool step of a tick with no pass of its own. */
    unsigned int embeds = (call->tool == AOTX_TOOL_MEMORY_RECALL
                           || call->tool == AOTX_TOOL_MEMORY_WRITE) ? 1u : 0u;
    if (embeds == 0u) {
        slot->auth = (needs_auth != 0u) ? AOTX_AUTH_PENDING : AOTX_AUTH_NONE;
        aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_NONE;
        /* A request that waits for the operator takes no deadline. The operator answers in
         * human time, and the answer of the operator starts the deadline. */
        slot->deadline = (slot->auth == AOTX_AUTH_PENDING)
                       ? AOTX_TOOL_NO_DEADLINE
                       : tick + aotx_setting_deadline();
        if (slot->auth == AOTX_AUTH_PENDING) {
            atomicAdd(&aotx_requests.pending_auth, 1u);
        }
        slot->request = id;
        /* Only a built-in host tool writes a request record for the feeder. The program of
         * a host tool that came in as a module comes with the tool module contract. */
        if (call->tool == AOTX_TOOL_FS_READ) {
            aotx_tool_note_request(slot, aotx_agents.agent[agent].turn);
            atomicAdd(&aotx_tool_count.host_open, 1u);
        }
    } else {
        slot->auth = AOTX_AUTH_NONE;
        slot->deadline = tick + aotx_setting_deadline();
        unsigned char *text = aotx_tool_gear.text + (unsigned long long)agent
                                                    * AOTX_TOOL_TEXT_BYTES;
        for (unsigned int i = 0u; i < bytes; ++i) {
            text[i] = (unsigned char)call->arg[i];
        }
        aotx_tool_gear.bytes[agent] = bytes;
        /* The pages a slot asked for belong to the request before this one. A slot that
         * gave its pages back would never ask again while that count stood. */
        aotx_tool_embed.asked[agent] = 0u;
        aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_WAIT;
        slot->request = id;
    }
    atomicAdd(&aotx_tool_count.opened, 1u);
    return id;
}

/* Write a note on the bus that names a reply the device refused, and the reason. The apply
 * of the tick calls this from one thread, so the text buffer has one writer. A reader of
 * the bus therefore sees every reply that found no request. */
__device__ static int aotx_tool_refuse(unsigned int request, const char *why)
{
    char *out = aotx_tool_embed.note;
    const char *head = "tool reply refused: request ";
    unsigned int at = 0u;
    for (unsigned int i = 0u; head[i] != '\0' && at < AOTX_BUS_TEXT_BYTES; ++i) {
        out[at++] = head[i];
    }
    at += aotx_text_utoa((unsigned long long)request, out + at, AOTX_BUS_TEXT_BYTES - at);
    if (at + 2u < AOTX_BUS_TEXT_BYTES) {
        out[at++] = ',';
        out[at++] = ' ';
    }
    for (unsigned int i = 0u; why[i] != '\0' && at < AOTX_BUS_TEXT_BYTES; ++i) {
        out[at++] = why[i];
    }
    atomicAdd(&aotx_requests.refused, 1u);
    atomicAdd(&aotx_tool_count.refused, 1u);
    aotx_bus_append(AOTX_WRITER_SYSTEM, AOTX_BUS_NOTE, 0u, out, at, 0ull, 0ull, 0.0f,
                    aotx_time_tick);
    return 1;
}

__device__ int aotx_tool_reply_apply(const aotx_tool_reply_body *body)
{
    if (body == 0) {
        return aotx_tool_refuse(0u, "the record carries no body");
    }
    /* The request of a number is found by a walk of the table. */
    unsigned int found = AOTX_SLOTS;
    for (unsigned int i = 0u; i < AOTX_SLOTS; ++i) {
        if (aotx_requests.slot[i].request == body->request
            && aotx_requests.slot[i].request != 0u) {
            found = i;
            break;
        }
    }
    if (found >= AOTX_SLOTS) {
        return aotx_tool_refuse(body->request, "no request holds that number");
    }
    aotx_request *slot = &aotx_requests.slot[found];

    /* A part that carries a reason ends the reply, and the count of the parts may not
     * hold it. Such a part is taken once, even after the content is complete. Every other
     * part of a reply whose result is in hand is refused and counted. */
    int reason = (body->status != AOTX_TOOL_OK) ? 1 : 0;
    if (reason == 0 && aotx_tool_done[found] != 0u) {
        return aotx_tool_refuse(body->request, "the result of that request is in hand");
    }
    if (reason != 0 && slot->status != AOTX_TOOL_OK) {
        return aotx_tool_refuse(body->request, "that request holds a reason already");
    }
    if (slot->agent != body->agent) {
        return aotx_tool_refuse(body->request, "the reply names another agent");
    }
    /* Only a host tool takes content from a reply. A part that carries a reason ends any
     * request, because the device writes the late verdict of a request as such a part. */
    if (reason == 0 && slot->tool != AOTX_TOOL_FS_READ) {
        return aotx_tool_refuse(body->request, "that request is not a host tool");
    }
    if (body->parts == 0u || (reason == 0 && body->part >= body->parts)) {
        return aotx_tool_refuse(body->request, "the part is outside the count of parts");
    }

    unsigned int len = (body->len > AOTX_TOOL_REPLY_BYTES) ? AOTX_TOOL_REPLY_BYTES
                                                           : body->len;
    if (slot->parts_in == 0u) {
        slot->result_len = 0u;
    }

    /* The content of the ok parts stands at the place of each part, inside the content
     * bound. A part that does not fit whole is dropped and counted, so no byte of the
     * result goes past the buffer. The tail of the buffer holds the reason. */
    unsigned long long at = 0ull;
    if (reason == 0) {
        at = (unsigned long long)body->part * AOTX_TOOL_REPLY_BYTES;
        if (at + (unsigned long long)len > (unsigned long long)AOTX_TOOL_CONTENT_BYTES) {
            slot->parts = body->parts;
            slot->parts_in += 1u;
            atomicAdd(&aotx_tool_count.dropped, 1u);
            atomicAdd(&aotx_tool_count.replies, 1u);
            if (slot->parts_in >= slot->parts) {
                aotx_tool_done[found] = 1u;
            }
            return 0;
        }
    } else {
        at = (unsigned long long)slot->result_len;
        if (at > (unsigned long long)AOTX_TOOL_CONTENT_BYTES) {
            at = (unsigned long long)AOTX_TOOL_CONTENT_BYTES;
        }
        if (at + (unsigned long long)len > (unsigned long long)AOTX_TOOL_RESULT_BYTES) {
            len = (unsigned int)((unsigned long long)AOTX_TOOL_RESULT_BYTES - at);
        }
    }
    for (unsigned int i = 0u; i < len; ++i) {
        slot->result[at + i] = body->bytes[i];
    }
    unsigned int end = (unsigned int)at + len;
    if (end > slot->result_len) {
        slot->result_len = end;
    }
    slot->parts = body->parts;
    slot->parts_in += 1u;
    slot->status = body->status;
    atomicAdd(&aotx_tool_count.replies, 1u);
    if (reason != 0 || slot->parts_in >= slot->parts) {
        aotx_tool_done[found] = 1u;
        /* A device tool that ends here gives its place in the batch back. The text of the
         * slot then joins no pass of a later tick. */
        aotx_tool_embed.state[found] = AOTX_TOOL_EMBED_NONE;
    }
    return 0;
}
