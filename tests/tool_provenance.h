/* Purpose: Check semantic errors through turn completion and published tool results.
 * Owns: Bounded reply fixtures and temporary agent states.
 * Launch shape: One thread for each agent at one and 64 slots.
 * Lifetime: One case of tool_test.cu. */
#ifndef AOTX_TEST_TOOL_PROVENANCE_H
#define AOTX_TEST_TOOL_PROVENANCE_H

#include "agent/transcript.cuh"

#define AOTX_TOOL_PROVENANCE_REASON \
    "provenance must be computed, fetched, recalled, or testimony; no note was saved"

__global__ void aotx_tool_provenance_turn(const aotx_tool_test_batch *batch,
    unsigned int count, unsigned int role, unsigned int mode, unsigned int *wrong)
{
    unsigned int slot = threadIdx.x;
    if (slot >= AOTX_SLOTS) return;
    aotx_agent *me = &aotx_agents.agent[slot];
    if (slot >= count) {
        me->state = AOTX_AGENT_STATE_FREE;
        return;
    }
    aotx_agent_work *gear = &aotx_agent_gear[slot];
    memset(gear, 0, sizeof *gear);
    memset(&aotx_transcript[slot], 0, sizeof aotx_transcript[slot]);
    me->state = AOTX_AGENT_STATE_POST;
    me->role = mode == 2u ? AOTX_MODULE_SLOTS : role;
    me->task = ~0u;
    me->budget_left = 0u;
    gear->kind = AOTX_AGENT_TURN_MESSAGE;
    gear->reply_len = batch->length[slot];
    for (unsigned int b = 0u; b < gear->reply_len; ++b) {
        gear->reply[b] = (unsigned char)batch->text[batch->start[slot] + b];
    }
    wrong[slot] = aotx_tool_parse(gear->reply, gear->reply_len, &gear->call) != 3;
    aotx_say.slot[slot].wanted = 0u;
    aotx_say.slot[slot].turn_tokens = 0u;
    aotx_say.slot[slot].reply_records = 0u;
    aotx_tool_embed.outcome[slot] = 0u;
    if (mode == 1u) aotx_tool_outcome_arm(slot, AOTX_TOOL_CALL_RESULT);
}

__global__ void aotx_tool_provenance_check(unsigned int count, unsigned int mode,
    unsigned int taken, unsigned int *wrong)
{
    unsigned int slot = threadIdx.x;
    if (slot >= count) return;
    const aotx_request *request = &aotx_requests.slot[slot];
    const aotx_tool_call *call = &aotx_agent_gear[slot].call;
    wrong[slot] += call->error != AOTX_TOOL_CALL_PROVENANCE || call->pack_len == 0u
        || aotx_tool_embed.state[slot] != AOTX_TOOL_EMBED_NONE
        || aotx_tool_embed.outcome[slot] != 0u;
    if (taken != 0u || mode == 2u) {
        wrong[slot] += request->request != 0u
            || aotx_agents.agent[slot].state != AOTX_AGENT_STATE_IDLE;
        return;
    }
    wrong[slot] += request->request == 0u || request->arg_len == 0u
        || request->call_seq != 0ull || request->auth != AOTX_AUTH_NONE
        || aotx_tool_done[slot] != 1u
        || aotx_agents.agent[slot].state != AOTX_AGENT_STATE_TOOL
        || request->status != (mode == 1u ? AOTX_TOOL_OK : AOTX_TOOL_ERROR);
    unsigned int at = 0u, span = 0u;
    if (aotx_tool_argument_of(request->arg, request->arg_len, "provenance", 10u,
                              &at, &span) == 0) {
        wrong[slot] += 1u;
    } else {
        unsigned int key = 0u;
        for (; key < AOTX_CATALOG_ARGS; ++key) {
            if (key != call->key && call->length[key] == span && span != 0u) break;
        }
        if (key == AOTX_CATALOG_ARGS) wrong[slot] += 1u;
        else for (unsigned int b = 0u; b < span; ++b) {
            wrong[slot] += request->arg[at + b] != call->pack[call->at[key] + b];
        }
    }
    if (mode == 0u) {
        const char *reason = AOTX_TOOL_PROVENANCE_REASON;
        unsigned int b = 0u;
        for (; reason[b] != '\0'; ++b) {
            wrong[slot] += b >= request->result_len || request->result[b] != reason[b];
        }
        wrong[slot] += b != request->result_len;
    }
}

static void aotx_tool_provenance_cases(unsigned int count, unsigned int *applied,
                                        unsigned int *failed)
{
    unsigned int role = aotx_test_catalog_entry("worker", AOTX_MODULE_ROLE);
    aotx_agent_table *saved = (aotx_agent_table *)calloc(1, sizeof *saved);
    aotx_agent_work *saved_work = (aotx_agent_work *)calloc(AOTX_SLOTS, sizeof *saved_work);
    aotx_transcript_agent *saved_history =
        (aotx_transcript_agent *)calloc(AOTX_SLOTS, sizeof *saved_history);
    unsigned char *saved_text =
        (unsigned char *)calloc(AOTX_SLOTS, AOTX_TRANSCRIPT_TEXT_BYTES);
    aotx_say_state *saved_say = (aotx_say_state *)calloc(1, sizeof *saved_say);
    aotx_tool_test_batch *batch = (aotx_tool_test_batch *)calloc(1, sizeof *batch);
    aotx_tool_test_batch *on = (aotx_tool_test_batch *)aotx_tool_test_take(sizeof *on);
    unsigned int *wrong = (unsigned int *)aotx_tool_test_take(count * sizeof(unsigned int));
    unsigned int marks[AOTX_SLOTS];
    unsigned int bad = 0u;
    char value[128], source[64];
    aotx_check_runtime(cudaMemcpyFromSymbol(saved, aotx_agents, sizeof *saved), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(saved_work, aotx_agent_gear,
        AOTX_SLOTS * sizeof *saved_work), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(saved_history, aotx_transcript,
        AOTX_SLOTS * sizeof *saved_history), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(saved_text, aotx_transcript_text,
        (size_t)AOTX_SLOTS * AOTX_TRANSCRIPT_TEXT_BYTES), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(saved_say, aotx_say, sizeof *saved_say),
        "cudaMemcpyFromSymbol");
    for (unsigned int kind = AOTX_CALL_HERMES; kind <= AOTX_CALL_QWEN_XML; ++kind) {
        aotx_test_call_upload(kind);
        for (unsigned int i = 0u; i < count; ++i) {
            batch->start[i] = i * AOTX_TOOL_CASE_BYTES;
            batch->length[i] = aotx_tool_format_text(kind, i, 4u,
                batch->text + batch->start[i], AOTX_TOOL_CASE_BYTES, value, source);
        }
        aotx_check_runtime(cudaMemcpy(on, batch, sizeof *on, cudaMemcpyHostToDevice), "cudaMemcpy");
        for (unsigned int mode = 0u; mode < 3u; ++mode) {
            aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(1u);
            aotx_tool_provenance_turn<<<1, AOTX_SLOTS>>>(on, count, role, mode, wrong);
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            aotx_seam_state seam;
            aotx_check_runtime(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam), "cudaMemcpyFromSymbol");
            unsigned long long first = seam.dev.tail;
            aotx_agent_step<<<1, AOTX_SLOTS>>>(70ull);
            aotx_tool_provenance_check<<<1, AOTX_SLOTS>>>(count, mode, 0u, wrong);
            aotx_tool_step<<<1, AOTX_SLOTS>>>(71ull);
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            if (mode != 2u) bad += aotx_tool_service_records(count, AOTX_CLASS_B);
            aotx_agent_step<<<1, AOTX_SLOTS>>>(72ull);
            aotx_tool_provenance_check<<<1, AOTX_SLOTS>>>(count, mode, 1u, wrong);
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            aotx_check_runtime(cudaMemcpy(marks, wrong, count * sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy");
            for (unsigned int i = 0u; i < count; ++i) bad += marks[i];
            aotx_check_runtime(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam), "cudaMemcpyFromSymbol");
            unsigned int visible = 0u;
            for (unsigned long long seq = first + 1ull; seq <= seam.dev.tail; ++seq) {
                unsigned char record[AOTX_SLOT_BYTES];
                const unsigned char *at = seam.dev.base + ((seq - 1ull) & seam.dev.mask) * AOTX_SLOT_BYTES;
                aotx_check_runtime(cudaMemcpy(record, at, sizeof record, cudaMemcpyDeviceToHost), "cudaMemcpy");
                const aotx_record_header *header = (const aotx_record_header *)record;
                bad += header->seq != seq || header->type == AOTX_REC_TOOL_REQUEST;
                const unsigned char *body = record + AOTX_HEADER_BYTES;
                if (header->type == AOTX_REC_BUS) {
                    const aotx_bus_body *message = (const aotx_bus_body *)body;
                    bad += header->body_len < offsetof(aotx_bus_body, text)
                        || message->kind == AOTX_BUS_FINDING;
                }
                unsigned int length = (unsigned int)strlen(AOTX_TOOL_PROVENANCE_REASON);
                if (header->type == AOTX_REC_CONSOLE) {
                    for (unsigned int b = 0u; b + length <= header->body_len; ++b) {
                        if (memcmp(body + b, AOTX_TOOL_PROVENANCE_REASON, length) == 0) visible++;
                    }
                }
            }
            if (mode == 0u) bad += visible != count;
        }
    }
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(1u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    /* Restore history before live agents. A leftover turn can claim the next tool vector
     * through transcript maintenance, even when its request slot is empty. */
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_agent_gear, saved_work,
        AOTX_SLOTS * sizeof *saved_work), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_transcript, saved_history,
        AOTX_SLOTS * sizeof *saved_history), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_transcript_text, saved_text,
        (size_t)AOTX_SLOTS * AOTX_TRANSCRIPT_TEXT_BYTES), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_say, saved_say, sizeof *saved_say),
        "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_agents, saved, sizeof *saved), "cudaMemcpyToSymbol");
    aotx_test_call_upload(AOTX_CALL_HERMES);
    aotx_tool_service_check(bad, "invalid provenance completion", count, applied, failed);
    cudaFree(on); cudaFree(wrong); free(batch); free(saved);
    free(saved_work); free(saved_history); free(saved_text); free(saved_say);
}

#endif
