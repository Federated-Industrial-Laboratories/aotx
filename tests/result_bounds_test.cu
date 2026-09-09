/* Purpose: Check encoded result bounds and completed tool continuations.
 * Owns: Distinct request content, circular sources, and guarded output buffers.
 * Launch shape: One thread per agent, at one and the profile slot count.
 * Lifetime: One test process. */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agent/prompt.cuh"
#include "seam/seam.cuh"
#include "settings/settings.cuh"
#include "wrap_fixture.h"

typedef struct aotx_result_bounds_row {
    char call[256];
    char query[64];
    unsigned char source[AOTX_TOOL_RESULT_BYTES];
    unsigned int length;
    aotx_tool_reply_body part;
    unsigned int parsed, initial, applied, received, room;
    unsigned int state, turn, wanted, request, done, status, result_len, prompt_len;
    unsigned int encoded, prefix, exact, below, invalid, cut_ok, small_ok;
    unsigned int guard_before;
    unsigned char rendered[AOTX_SAY_BYTES];
    unsigned int guard_after;
    unsigned int rendered_len, refused, guards;
    unsigned char result[AOTX_TOOL_RESULT_BYTES];
    unsigned char prompt[AOTX_SAY_BYTES + 1u];
    aotx_request cut;
} aotx_result_bounds_row;

__global__ void aotx_result_bounds_reset(void)
{
    memset(&aotx_agents, 0, sizeof aotx_agents);
    memset(aotx_agent_gear, 0, sizeof aotx_agent_gear);
    memset(aotx_transcript, 0, sizeof aotx_transcript);
    memset(&aotx_say, 0, sizeof aotx_say);
    memset(&aotx_requests, 0, sizeof aotx_requests);
    memset(aotx_tool_done, 0, sizeof aotx_tool_done);
    memset(&aotx_tool_embed, 0, sizeof aotx_tool_embed);
    memset(&aotx_agent_count, 0, sizeof aotx_agent_count);
    memset(&aotx_console, 0, sizeof aotx_console);
    aotx_settings_reset();
#ifdef AOTX_AFFECT
    aotx_setting_table.row[AOTX_SET_QUALITY_ON].value = 0ll;
#endif
    aotx_model[AOTX_MODEL_LANGUAGE].layers = 1u;
    aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, 1u, 1u, 32u);
    unsigned int role = AOTX_MODULE_SLOTS - 1u;
    memset(&aotx_catalog.entry[role], 0, sizeof aotx_catalog.entry[role]);
    aotx_catalog.entry[role].kind = AOTX_MODULE_ROLE;
    aotx_catalog.entry[role].state = AOTX_CATALOG_INSTALLED;
    aotx_catalog_mask_set(aotx_catalog.entry[role].role.tools,
                          aotx_catalog_find("fs_read", 7u, AOTX_MODULE_TOOL));
    aotx_catalog.count.room_cut = 0u;
}

/* Open a real request after rendering the input and storing its completed call. */
__global__ void aotx_result_bounds_open(aotx_result_bounds_row *rows, unsigned int count)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count) return;
    aotx_result_bounds_row *row = &rows[agent];
    aotx_agent *me = &aotx_agents.agent[agent];
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    me->role = AOTX_MODULE_SLOTS - 1u;
    me->task = ~0u;
    me->state = AOTX_AGENT_STATE_IDLE;
    me->budget_left = 3u;
    gear->kind = AOTX_AGENT_TURN_MESSAGE;
    while (row->query[gear->message_len]) {
        gear->message[gear->message_len] = row->query[gear->message_len];
        ++gear->message_len;
    }
    aotx_transcript[agent].pages = AOTX_KV_PAGES_EACH;
    row->initial = aotx_agent_prompt(agent, 0, gear->message, gear->message_len,
                                     0, 0, 0u, 0, 0u);
    aotx_say.slot[agent].wanted = 0u;
    unsigned int bytes = 0u;
    while (row->call[bytes]) ++bytes;
    row->parsed = aotx_tool_parse((const unsigned char *)row->call, bytes, &gear->call) == 1;
    memcpy(gear->reply, row->call, bytes);
    gear->reply_len = bytes;
    me->turn = 1u;
    aotx_transcript[agent].text_head = AOTX_TRANSCRIPT_TEXT_BYTES - 23u - agent;
    aotx_transcript_finish(agent, gear->message, gear->message_len, gear->reply,
                           gear->reply_len, 32u, 1000ull + agent);
    me->request = aotx_tool_request(agent, &gear->call, 0u, 2ull);
    me->tool = gear->call.entry;
    me->deadline = aotx_requests.slot[agent].deadline;
    me->state = AOTX_AGENT_STATE_TOOL;
}

/* Apply the same bounded parts that the host result producer submits. */
__global__ void aotx_result_bounds_reply(aotx_result_bounds_row *rows, unsigned int count,
                                         unsigned int mode)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count) return;
    aotx_result_bounds_row *row = &rows[agent];
    row->applied = 1u;
    unsigned int parts = (row->length + AOTX_TOOL_REPLY_BYTES - 1u) / AOTX_TOOL_REPLY_BYTES;
    for (unsigned int part = 0u; part < parts; ++part) {
        memset(&row->part, 0, sizeof row->part);
        row->part.agent = agent;
        row->part.request = aotx_agents.agent[agent].request;
        row->part.parts = parts;
        row->part.part = part;
        row->part.status = AOTX_TOOL_OK;
        unsigned int at = part * AOTX_TOOL_REPLY_BYTES;
        row->part.len = row->length - at;
        if (row->part.len > AOTX_TOOL_REPLY_BYTES) row->part.len = AOTX_TOOL_REPLY_BYTES;
        memcpy(row->part.bytes, row->source + at, row->part.len);
        row->applied &= aotx_tool_reply_apply(&row->part, 2000ull + agent * parts + part) == 0;
    }
    row->received = aotx_tool_done[agent] && aotx_requests.slot[agent].result_len == row->length;
    if (mode == 4u) {
        memset(aotx_agent_gear[agent].message, 'x', AOTX_SAY_BYTES);
        aotx_agent_gear[agent].message_len = AOTX_SAY_BYTES;
    }
    row->room = aotx_agent_result_room(agent);
}

__global__ void aotx_result_bounds_snapshot(aotx_result_bounds_row *rows, unsigned int count)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count) return;
    aotx_result_bounds_row *row = &rows[agent];
    const aotx_request *request = &aotx_requests.slot[agent];
    row->state = aotx_agents.agent[agent].state;
    row->turn = aotx_agents.agent[agent].turn;
    row->wanted = aotx_say.slot[agent].wanted;
    row->request = aotx_agents.agent[agent].request | request->request;
    row->done = aotx_tool_done[agent];
    row->status = request->status;
    row->result_len = request->result_len;
    memcpy(row->result, request->result, AOTX_TOOL_RESULT_BYTES);
    row->prompt_len = aotx_say.slot[agent].length;
    if (row->prompt_len <= AOTX_SAY_BYTES) {
        memcpy(row->prompt, aotx_say.prompt[agent], row->prompt_len);
        row->prompt[row->prompt_len] = 0u;
    }
}

/* Exact prefixes include complete JSON escapes, including across the circular boundary. */
__global__ void aotx_result_bounds_helpers(aotx_result_bounds_row *rows, unsigned int count)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count) return;
    aotx_result_bounds_row *row = &rows[agent];
    int json = aotx_call_format_active()->result_json != 0u;
    row->encoded = aotx_result_bytes(row->source, 0u, row->length, AOTX_TOOL_RESULT_BYTES, json);
    row->exact = aotx_result_prefix(row->source, 0u, row->length,
                                     AOTX_TOOL_RESULT_BYTES, json, row->encoded);
    row->below = aotx_result_prefix(row->source, 0u, row->length,
                                     AOTX_TOOL_RESULT_BYTES, json, row->encoded - 1u);
    row->invalid = aotx_result_bytes(row->source, AOTX_TOOL_RESULT_BYTES, 1u,
                                     AOTX_TOOL_RESULT_BYTES, json) == ~0u
        && aotx_result_prefix(row->source, 0u, AOTX_TOOL_RESULT_BYTES + 1u,
                               AOTX_TOOL_RESULT_BYTES, json, ~0u) == 0u;
    row->cut.agent = agent;
    memset(row->cut.result, '\n', sizeof row->cut.result);
    row->cut.result[0] = (char)('A' + agent % 26u);
    row->cut.result_len = AOTX_TOOL_RESULT_BYTES;
    unsigned int suffix = (unsigned int)sizeof(AOTX_RESULT_CUT_TEXT) - 1u;
    row->small_ok = aotx_agent_cut_result(&row->cut, suffix - 1u) == 0
                   && row->cut.result_len == AOTX_TOOL_RESULT_BYTES;
    row->cut_ok = aotx_agent_cut_result(&row->cut, suffix + 1u) != 0
        && row->cut.result_len == suffix + 1u
        && row->cut.result[0] == (char)('A' + agent % 26u);
    for (unsigned int i = 0u; i < suffix; ++i)
        row->cut_ok &= row->cut.result[i + 1u] == AOTX_RESULT_CUT_TEXT[i];
    row->guard_before = 0x12563478u + agent;
    row->guard_after = 0x87654321u + agent;
    memset(row->rendered, 0x5a, sizeof row->rendered);
    row->rendered_len = aotx_call_result(row->rendered, 0u, row->source,
                                         0u, row->length, AOTX_TOOL_RESULT_BYTES);
    row->refused = aotx_call_result(row->rendered, AOTX_SAY_BYTES, row->source,
                                   0u, row->length, AOTX_TOOL_RESULT_BYTES);
    row->guards = row->guard_before == 0x12563478u + agent
               && row->guard_after == 0x87654321u + agent;
    if (row->rendered_len < AOTX_SAY_BYTES)
        row->guards &= row->rendered[row->rendered_len] == 0x5au;
    unsigned int base = AOTX_TOOL_RESULT_BYTES - 2u;
    unsigned char *circular = (unsigned char *)row->cut.result;
    circular[base] = '"'; circular[base + 1u] = '\\'; circular[0] = '\n';
    row->prefix = aotx_result_bytes(circular, base, 3u, AOTX_TOOL_RESULT_BYTES, 1) == 10u
        && aotx_result_prefix(circular, base, 3u, AOTX_TOOL_RESULT_BYTES, 1, 9u) == 2u
        && aotx_result_prefix(circular, base, 3u, AOTX_TOOL_RESULT_BYTES, 1, 10u) == 3u;
}

static unsigned int checked, failed;
static void check(int pass, const char *text, unsigned int kind, unsigned int mode, unsigned int agent)
{
    ++checked;
    if (!pass) { ++failed; printf("result bounds: kind %u mode %u agent %u: %s\n", kind, mode, agent, text); }
}

static int contains(const unsigned char *bytes, unsigned int length, const char *text)
{
    size_t span = strlen(text);
    for (unsigned int i = 0u; i + span <= length; ++i)
        if (!memcmp(bytes + i, text, span)) return 1;
    return 0;
}

static void run(unsigned int kind, unsigned int mode, unsigned int count)
{
    aotx_result_bounds_row *rows = (aotx_result_bounds_row *)calloc(count, sizeof *rows), *device = NULL;
    if (!rows) exit(2);
    for (unsigned int agent = 0u; agent < count; ++agent) {
        aotx_result_bounds_row *row = &rows[agent];
        snprintf(row->query, sizeof row->query, "read result-bounds-%u", agent);
        snprintf(row->call, sizeof row->call,
            "<tool_call>{\"name\":\"fs_read\",\"arguments\":{\"path\":\"result-bounds-%u\"}}</tool_call>", agent);
        unsigned int at = (unsigned int)snprintf((char *)row->source, sizeof row->source, "result-bounds-%u:", agent);
        unsigned int ordinary = AOTX_TOOL_CONTENT_BYTES < 4000u ? AOTX_TOOL_CONTENT_BYTES : 4000u;
        unsigned int payload = mode == 0u ? ordinary - at
                             : mode == 2u ? 1800u : mode == 6u ? 3500u : 1200u;
        for (unsigned int i = 0u; i < payload; ++i) {
            row->source[at + i] = mode == 0u ? (unsigned char)('a' + agent % 26u)
                : (mode == 2u || mode == 6u) ? (i % 2u ? '"' : '\\')
                : mode == 3u ? (unsigned char)(i % 32u) : '\n';
        }
        row->length = at + payload;
    }
    aotx_check_runtime(cudaMalloc(&device, count * sizeof *device), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device, rows, count * sizeof *rows, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_test_wrap_open();
    aotx_result_bounds_reset<<<1, 1>>>();
    aotx_result_bounds_open<<<1, AOTX_SLOTS>>>(device, count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_call_upload(kind);
    aotx_result_bounds_reply<<<1, AOTX_SLOTS>>>(device, count, mode);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    if (mode == 5u) {
        aotx_wrap wrap = aotx_test_wrap_table();
        wrap.usable = 0u;
        aotx_test_wrap_upload(&wrap);
    }
    aotx_agent_step<<<1, AOTX_SLOTS>>>(3ull);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(4ull);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(5ull);
    aotx_result_bounds_snapshot<<<1, AOTX_SLOTS>>>(device, count);
    unsigned int cuts_before = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&cuts_before, aotx_catalog, sizeof cuts_before,
        offsetof(aotx_catalog_state, count.room_cut)), "cudaMemcpyFromSymbol");
    aotx_result_bounds_helpers<<<1, AOTX_SLOTS>>>(device, count);
    aotx_check_runtime(cudaMemcpy(rows, device, count * sizeof *rows, cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int cuts_after = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&cuts_after, aotx_catalog, sizeof cuts_after,
        offsetof(aotx_catalog_state, count.room_cut)), "cudaMemcpyFromSymbol");
    check(cuts_after >= cuts_before + count,
          "explicit cuts count each bounded result", kind, mode, 0u);
    for (unsigned int agent = 0u; agent < count; ++agent) {
        aotx_result_bounds_row *row = &rows[agent];
        int hard = mode == 4u || mode == 5u;
        int cut = row->encoded > row->room;
        unsigned int encoded = 0u;
        for (unsigned int i = 0u; i < row->length; ++i) {
            unsigned char byte = row->source[i];
            encoded += kind != AOTX_CALL_LLAMA_JSON ? 1u : byte < 0x20u ? 6u
                     : byte == '"' || byte == '\\' ? 2u : 1u;
        }
        check(row->encoded == encoded, "encoded length counts controls, quotes, and backslashes", kind, mode, agent);
        check(row->initial && row->parsed && row->applied && row->received,
              "real request and reply parts complete", kind, mode, agent);
        check(row->request == 0u && row->done == 0u && row->status == AOTX_TOOL_OK,
              "completion clears identity without changing tool outcome", kind, mode, agent);
        check(row->state == (hard ? AOTX_AGENT_STATE_IDLE : AOTX_AGENT_STATE_PROMPT)
                && row->turn == (hard ? 1u : 2u) && row->wanted == (hard ? 0u : 1u),
              "completed request cannot stay in TOOL or open repeated turns", kind, mode, agent);
        if (!hard) {
            check(row->prompt_len && row->prompt_len <= AOTX_SAY_BYTES, "continuation fits", kind, mode, agent);
            if (cut) {
                unsigned int suffix = (unsigned int)sizeof(AOTX_RESULT_CUT_TEXT) - 1u;
                check(row->result_len >= suffix && row->result_len < row->length
                    && !memcmp(row->result + row->result_len - suffix, AOTX_RESULT_CUT_TEXT, suffix),
                    "cut keeps full explicit suffix", kind, mode, agent);
                check(cuts_before >= count && contains(row->prompt, row->prompt_len, AOTX_RESULT_CUT_TEXT),
                      "continuation includes and counts the cutoff", kind, mode, agent);
            } else {
                check(row->result_len == row->length && !memcmp(row->result, row->source, row->length),
                      "fitting ordinary and escaped results retain bytes", kind, mode, agent);
            }
            const char *tail = kind == AOTX_CALL_LLAMA_JSON ? "\"<|eot_id|>" : "</tool_response>";
            check(contains(row->prompt, row->prompt_len, tail), "result keeps closing native frame", kind, mode, agent);
            if (kind == AOTX_CALL_LLAMA_JSON && (mode == 1u || mode == 3u))
                check(contains(row->prompt, row->prompt_len, "\\u000a"), "control bytes use full JSON escapes", kind, mode, agent);
        }
        check(row->exact == row->length && row->below + 1u == row->length && row->invalid && row->prefix,
              "encoded prefix boundaries and circular sources", kind, mode, agent);
        check(row->cut_ok && row->small_ok, "suffix is complete or cutting is refused", kind, mode, agent);
        check(row->rendered_len <= AOTX_SAY_BYTES && row->refused == AOTX_SAY_BYTES + 1u && row->guards,
              "result renderer respects accepted buffer", kind, mode, agent);
    }
    if (mode == 4u || mode == 5u) {
        aotx_console_state console;
        aotx_check_runtime(cudaMemcpyFromSymbol(&console, aotx_console, sizeof console), "cudaMemcpyFromSymbol");
        int notice = 0;
        for (unsigned int i = 0u; i < AOTX_CONSOLE_LINES; ++i) {
            const aotx_console_line *line = &console.line[i];
            const char *text = "agent: the tool continuation prompt was refused; give new input to resume";
            notice |= line->length == strlen(text) && !memcmp(line->text, text, strlen(text));
        }
        check(notice, "hard refusal emits an explicit console outcome", kind, mode, 0u);
    }
    cudaFree(device);
    free(rows);
}

int main(void)
{
    aotx_check_runtime(cudaSetDevice(0), "cudaSetDevice");
    aotx_catalog_open();
    unsigned char *ring = NULL;
    aotx_check_runtime(cudaMallocManaged(&ring, 4096ull * AOTX_SLOT_BYTES), "cudaMallocManaged");
    memset(ring, 0, 4096ull * AOTX_SLOT_BYTES);
    aotx_seam_state seam = {};
    seam.dev.base = ring;
    seam.dev.slot_count = 4096ull;
    seam.dev.mask = 4095ull;
    seam.apply.state_hash = AOTX_FNV_BASIS;
    seam.boot_id = 31ull;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof seam), "cudaMemcpyToSymbol");
    for (unsigned int kind = AOTX_CALL_HERMES; kind < AOTX_CALL_FORMAT_KINDS; ++kind) {
        for (unsigned int mode = 0u; mode < 7u; ++mode) {
            run(kind, mode, 1u);
            run(kind, mode, AOTX_SLOTS);
        }
    }
    cudaFree(ring);
    printf("result bounds: %u checks, %u failed\n", checked, failed);
    return failed != 0u;
}
