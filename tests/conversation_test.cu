/* Purpose: Check long input lines and the three conversation memory tiers.
 * Owns: One device ring, one inbound ring and the counters of this check.
 * Launch shape: The apply grid and one-thread state probes.
 * Lifetime: One run of the check program. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>

#include "agent/prompt.cuh"
#include "agent/transcript.cuh"
#include "wrap_fixture.h"
#include "bus/bus.cuh"
#include "cli/cli.cuh"
#include "seam/seam.cuh"
#include "settings/settings.cuh"
#define AOTX_CONVERSATION_LINE       4000u
#define AOTX_CONVERSATION_RING       32768ull
#define AOTX_CONVERSATION_INPUT      4096ull

typedef struct aotx_conversation_fixture {
    unsigned char *ring;
    unsigned char *inbound;
    aotx_inbound_preamble *preamble;
    unsigned char *slots;
} aotx_conversation_fixture;

typedef struct aotx_conversation_result {
    unsigned int wrong;
    unsigned int worker_hot;
    unsigned int worker_warm;
    unsigned int conductor_hot;
    unsigned int auto_free;
    unsigned int auto_full;
    unsigned int prompt_has_first;
    unsigned int prompt_has_tools;
    unsigned int prompt_tail;
    unsigned int prompt_hash;
    unsigned int prompt_refused;
    unsigned int prompt_reason;
    unsigned int index_records;
    unsigned int recall_count;
    unsigned int recall_turn;
    unsigned int recall_prompt;
    unsigned int replay_searches;
    unsigned int replay_choices;
    unsigned int same_hash;
    unsigned int automatic_compact;
    unsigned int folded;
    unsigned int warm;
    unsigned int summary_prompt;
    unsigned int manual_compact;
    unsigned int corrected_summary;
    unsigned int range_finding;
    unsigned int stress_newest;
    unsigned int stress_pool;
    unsigned int invalid_selection_refused;
    unsigned int command_auto;
    unsigned int command_pages;
    unsigned int command_compact;
    unsigned int command_refused;
} aotx_conversation_result;
static unsigned int aotx_conversation_applied;
static unsigned int aotx_conversation_failed;

static void aotx_conversation_check(int pass, const char *text)
{
    aotx_conversation_applied += 1u;
    if (!pass) {
        aotx_conversation_failed += 1u;
        printf("conversation: FAILED %s\n", text);
    }
}
#define AOTX_CONVERSATION_CLEAR(symbol, bytes) do {                              \
    void *address = NULL;                                                        \
    aotx_check_runtime(cudaGetSymbolAddress(&address, symbol),                   \
                       "cudaGetSymbolAddress");                                 \
    aotx_check_runtime(cudaMemset(address, 0, (bytes)), "cudaMemset");           \
} while (0)

static void aotx_conversation_clear(void)
{
    AOTX_CONVERSATION_CLEAR(aotx_transcript,
                            sizeof(aotx_transcript_agent) * AOTX_SLOTS);
    AOTX_CONVERSATION_CLEAR(aotx_transcript_text,
                            (size_t)AOTX_SLOTS * AOTX_TRANSCRIPT_TEXT_BYTES);
    AOTX_CONVERSATION_CLEAR(aotx_transcript_count, sizeof(aotx_transcript_counts));
    AOTX_CONVERSATION_CLEAR(aotx_agents, sizeof(aotx_agent_table));
    AOTX_CONVERSATION_CLEAR(aotx_agent_gear,
                            sizeof(aotx_agent_work) * AOTX_SLOTS);
    AOTX_CONVERSATION_CLEAR(aotx_say, sizeof(aotx_say_state));
    AOTX_CONVERSATION_CLEAR(aotx_setting_table, sizeof(aotx_settings_state));
    AOTX_CONVERSATION_CLEAR(aotx_kv, sizeof(aotx_kv_table));
    AOTX_CONVERSATION_CLEAR(aotx_tool_embed, sizeof(aotx_tool_embed_batch));
    AOTX_CONVERSATION_CLEAR(aotx_requests, sizeof(aotx_request_table));
    AOTX_CONVERSATION_CLEAR(aotx_cli, sizeof(aotx_cli_state));
    AOTX_CONVERSATION_CLEAR(aotx_cli_count, sizeof(aotx_cli_counts));
    AOTX_CONVERSATION_CLEAR(aotx_console, sizeof(aotx_console_state));
    AOTX_CONVERSATION_CLEAR(aotx_seam_line_holds, sizeof(unsigned long long));
    AOTX_CONVERSATION_CLEAR(aotx_seam_line_orphans, sizeof(unsigned long long));
}
static void aotx_conversation_fixture_open(aotx_conversation_fixture *fixture)
{
    size_t ring_bytes = (size_t)AOTX_CONVERSATION_RING * AOTX_SLOT_BYTES;
    size_t inbound_bytes = sizeof(aotx_inbound_preamble)
                         + (size_t)AOTX_CONVERSATION_INPUT * AOTX_SLOT_BYTES;
    memset(fixture, 0, sizeof *fixture);
    aotx_check_runtime(cudaMallocManaged(&fixture->ring, ring_bytes), "cudaMallocManaged");
    aotx_check_runtime(cudaMallocManaged(&fixture->inbound, inbound_bytes),
                       "cudaMallocManaged");
    memset(fixture->ring, 0, ring_bytes);
    memset(fixture->inbound, 0, inbound_bytes);
    fixture->preamble = (aotx_inbound_preamble *)fixture->inbound;
    fixture->slots = fixture->inbound + sizeof(aotx_inbound_preamble);
    fixture->preamble->magic = AOTX_WIRE_MAGIC;
    fixture->preamble->layout = AOTX_WIRE_LAYOUT;
    fixture->preamble->preamble_bytes = sizeof(aotx_inbound_preamble);
    fixture->preamble->slot_count = AOTX_CONVERSATION_INPUT;

    aotx_seam_state seam;
    memset(&seam, 0, sizeof seam);
    seam.dev.base = fixture->ring;
    seam.dev.slot_count = AOTX_CONVERSATION_RING;
    seam.dev.mask = AOTX_CONVERSATION_RING - 1ull;
    seam.in.preamble = fixture->inbound;
    seam.in.slots = fixture->slots;
    seam.in.slot_count = AOTX_CONVERSATION_INPUT;
    seam.in.mask = AOTX_CONVERSATION_INPUT - 1ull;
    seam.apply.state_hash = AOTX_FNV_BASIS;
    seam.boot_id = 12ull;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof seam),
                       "cudaMemcpyToSymbol");
    unsigned long long tick = 1ull;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_time_tick, &tick, sizeof tick),
                       "cudaMemcpyToSymbol");
}
static void aotx_conversation_fixture_close(aotx_conversation_fixture *fixture)
{
    cudaFree(fixture->inbound);
    cudaFree(fixture->ring);
}
static void aotx_conversation_put_type(aotx_conversation_fixture *fixture,
                                       unsigned long long index, unsigned int type,
                                       unsigned int flags, const unsigned char *body,
                                       unsigned int length)
{
    aotx_record_header *header = (aotx_record_header *)(fixture->slots
        + (index & (AOTX_CONVERSATION_INPUT - 1ull)) * AOTX_SLOT_BYTES);
    memset(header, 0, AOTX_SLOT_BYTES);
    header->magic = AOTX_WIRE_MAGIC;
    header->layout = AOTX_WIRE_LAYOUT;
    header->header_bytes = AOTX_HEADER_BYTES;
    header->boot_id = 12ull;
    header->tick = 1ull;
    header->writer = AOTX_WRITER_FEEDER;
    header->cls = AOTX_CLASS_A;
    header->type = (unsigned char)type;
    header->flags = (unsigned short)flags;
    header->body_len = length;
    if (length != 0u) {
        memcpy((unsigned char *)header + AOTX_HEADER_BYTES, body, length);
    }
    __atomic_store_n(&header->seq, index + 1ull, __ATOMIC_RELEASE);
}
static void aotx_conversation_put(aotx_conversation_fixture *fixture,
                                  unsigned long long index, unsigned int flags,
                                  const unsigned char *body, unsigned int length)
{
    aotx_conversation_put_type(fixture, index, AOTX_REC_INPUT_LINE, flags, body, length);
}
static void aotx_conversation_apply(unsigned long long count,
                                    unsigned long long available)
{
    aotx_seam_state seam;
    aotx_check_runtime(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam),
                       "cudaMemcpyFromSymbol");
    seam.apply.this_tick = count;
    seam.apply.available = available;
    seam.apply.first_seq = seam.dev.tail + 1ull;
    seam.dev.tail += AOTX_APPLY_RECORDS_EACH * count;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof seam),
                       "cudaMemcpyToSymbol");
    aotx_seam_apply_inbound<<<AOTX_APPLY_BLOCKS, AOTX_APPLY_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}
static unsigned long long aotx_conversation_hash(unsigned long long hash,
                                                 const unsigned char *text,
                                                 unsigned int length)
{
    for (unsigned int i = 0u; i < length; ++i) {
        hash ^= (unsigned long long)text[i];
        hash *= AOTX_FNV_PRIME;
    }
    return hash;
}
static void aotx_conversation_line_case(unsigned int lines)
{
    aotx_conversation_fixture fixture;
    unsigned char *text = (unsigned char *)malloc((size_t)lines * AOTX_CONVERSATION_LINE);
    unsigned long long records = 0ull;
    unsigned long long want_hash = AOTX_FNV_BASIS;
    unsigned int parts = (AOTX_CONVERSATION_LINE + AOTX_BODY_BYTES - 1u)
                       / AOTX_BODY_BYTES;
    aotx_conversation_clear();
    aotx_conversation_fixture_open(&fixture);

    for (unsigned int line = 0u; line < lines; ++line) {
        unsigned char *one = text + (size_t)line * AOTX_CONVERSATION_LINE;
        int head = snprintf((char *)one, AOTX_CONVERSATION_LINE, "line %u ", line);
        memset(one + head, (int)('a' + line % 26u), AOTX_CONVERSATION_LINE - (unsigned)head);
        for (unsigned int part = 0u; part < parts; ++part) {
            unsigned int at = part * AOTX_BODY_BYTES;
            unsigned int length = AOTX_CONVERSATION_LINE - at;
            if (length > AOTX_BODY_BYTES) {
                length = AOTX_BODY_BYTES;
            }
            aotx_conversation_put(&fixture, records,
                                  (part == 0u) ? 0u : AOTX_FLAG_FRAGMENT,
                                  one + at, length);
            want_hash = aotx_conversation_hash(want_hash, one + at, length);
            records += 1ull;
        }
    }
    __atomic_store_n(&fixture.preamble->head, records, __ATOMIC_RELEASE);
    unsigned long long left = records;
    while (left != 0ull) {
        unsigned long long take = (left > AOTX_INBOUND_MAX_TICK)
                                ? AOTX_INBOUND_MAX_TICK : left;
        aotx_conversation_apply(take, left);
        left -= take;
    }

    aotx_seam_state seam;
    aotx_cli_counts cli;
    aotx_check_runtime(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&cli, aotx_cli_count, sizeof cli),
                       "cudaMemcpyFromSymbol");
    aotx_conversation_check(seam.apply.state_hash == want_hash,
                            "the hash covers every input part in order");
    aotx_conversation_check(seam.apply.applied_count == records,
                            "every input part is applied");
    aotx_conversation_check(cli.commands == lines,
                            "every assembled line reaches the parser once");

    unsigned int input_records = 0u;
    unsigned int echoes = 0u;
    for (unsigned long long seq = 1ull; seq <= seam.dev.tail; ++seq) {
        const aotx_record_header *record = (const aotx_record_header *)(fixture.ring
            + ((seq - 1ull) & (AOTX_CONVERSATION_RING - 1ull)) * AOTX_SLOT_BYTES);
        if (record->seq != seq || record->magic != AOTX_WIRE_MAGIC) {
            continue;
        }
        if (record->type == AOTX_REC_INPUT_LINE) {
            input_records += 1u;
            aotx_conversation_check(record->cls == AOTX_CLASS_A,
                                    "an input part stays class A");
        }
        if (record->type != AOTX_REC_CONSOLE || record->body_len < 2u
            || record->flags != 0u) {
            continue;
        }
        const unsigned char *body = (const unsigned char *)record + AOTX_HEADER_BYTES;
        if (body[0] != '>' || body[1] != ' ') {
            continue;
        }
        unsigned int made = record->body_len - 2u;
        unsigned char joined[AOTX_CONVERSATION_LINE];
        memcpy(joined, body + 2u, made);
        while (seq < seam.dev.tail) {
            const aotx_record_header *next = (const aotx_record_header *)(fixture.ring
                + (seq & (AOTX_CONVERSATION_RING - 1ull)) * AOTX_SLOT_BYTES);
            if (next->seq != seq + 1ull || next->type != AOTX_REC_CONSOLE
                || (next->flags & AOTX_FLAG_FRAGMENT) == 0u) {
                break;
            }
            memcpy(joined + made, (const unsigned char *)next + AOTX_HEADER_BYTES,
                   next->body_len);
            made += next->body_len;
            seq += 1ull;
        }
        aotx_conversation_check(made == AOTX_CONVERSATION_LINE,
                                "the echo has the whole line");
        aotx_conversation_check(echoes < lines
            && memcmp(joined, text + (size_t)echoes * AOTX_CONVERSATION_LINE,
                      AOTX_CONVERSATION_LINE) == 0,
            "the echo parts join to the input line");
        echoes += 1u;
    }
    aotx_conversation_check(input_records == records,
                            "every input part is written to the device ring");
    aotx_conversation_check(echoes == lines, "every line has one joined echo");
    printf("conversation: %u lines of %u bytes used %llu input parts\n", lines,
           AOTX_CONVERSATION_LINE, records);
    free(text);
    aotx_conversation_fixture_close(&fixture);
}
static void aotx_conversation_part_guards(void)
{
    aotx_conversation_fixture fixture;
    unsigned char body[AOTX_BODY_BYTES];
    memset(body, 'x', sizeof body);
    aotx_conversation_clear();
    aotx_conversation_fixture_open(&fixture);

    aotx_conversation_put(&fixture, 0ull, 0u, body, AOTX_BODY_BYTES);
    aotx_conversation_put(&fixture, 1ull, AOTX_FLAG_FRAGMENT, body, 17u);
    __atomic_store_n(&fixture.preamble->head, 2ull, __ATOMIC_RELEASE);
    aotx_conversation_apply(1ull, 2ull);
    unsigned long long held = 0ull;
    aotx_cli_counts cli;
    aotx_check_runtime(cudaMemcpyFromSymbol(&held, aotx_seam_line_holds, sizeof held),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&cli, aotx_cli_count, sizeof cli),
                       "cudaMemcpyFromSymbol");
    aotx_conversation_check(held == 1ull && cli.commands == 0u,
                            "a line at an apply boundary is held");
    aotx_conversation_apply(1ull, 1ull);
    aotx_check_runtime(cudaMemcpyFromSymbol(&cli, aotx_cli_count, sizeof cli),
                       "cudaMemcpyFromSymbol");
    aotx_conversation_check(cli.commands == 1u, "the held line runs after its last part");

    aotx_conversation_put(&fixture, 2ull, AOTX_FLAG_FRAGMENT, body, 1u);
    __atomic_store_n(&fixture.preamble->head, 3ull, __ATOMIC_RELEASE);
    aotx_conversation_apply(1ull, 1ull);
    unsigned long long orphan = 0ull;
    aotx_seam_state seam;
    aotx_check_runtime(cudaMemcpyFromSymbol(&orphan, aotx_seam_line_orphans,
                                            sizeof orphan), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam),
                       "cudaMemcpyFromSymbol");
    aotx_conversation_check(orphan == 1ull && seam.apply.rejected == 1ull,
                            "a part without a head is refused and counted");

    unsigned long long first = 3ull;
    for (unsigned int part = 0u; part < AOTX_LINE_PARTS_MAX + 1u; ++part) {
        aotx_conversation_put(&fixture, first + part,
                              (part == 0u) ? 0u : AOTX_FLAG_FRAGMENT,
                              body, (part < AOTX_LINE_PARTS_MAX) ? AOTX_BODY_BYTES : 1u);
    }
    __atomic_store_n(&fixture.preamble->head,
                     first + AOTX_LINE_PARTS_MAX + 1ull, __ATOMIC_RELEASE);
    aotx_conversation_apply(AOTX_LINE_PARTS_MAX + 1ull,
                            AOTX_LINE_PARTS_MAX + 1ull);
    aotx_check_runtime(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam),
                       "cudaMemcpyFromSymbol");
    aotx_conversation_check(seam.apply.rejected > 1ull,
                            "a line beyond the part bound is refused");

    first += AOTX_LINE_PARTS_MAX + 1ull;
    const unsigned char help[] = "help";
    const unsigned char stats[] = "stats";
    aotx_conversation_put(&fixture, first, 0u, help, (unsigned int)sizeof help - 1u);
    aotx_conversation_put(&fixture, first + 1ull, AOTX_FLAG_FRAGMENT, body, 1u);
    aotx_record_header *bad = (aotx_record_header *)(fixture.slots
        + ((first + 1ull) & (AOTX_CONVERSATION_INPUT - 1ull)) * AOTX_SLOT_BYTES);
    bad->body_len = AOTX_BODY_BYTES + 1u;
    aotx_conversation_put(&fixture, first + 2ull, 0u, stats,
                          (unsigned int)sizeof stats - 1u);
    __atomic_store_n(&fixture.preamble->head, first + 3ull, __ATOMIC_RELEASE);
    unsigned int commands_before = cli.commands;
    aotx_conversation_apply(3ull, 3ull);
    aotx_check_runtime(cudaMemcpyFromSymbol(&cli, aotx_cli_count, sizeof cli),
                       "cudaMemcpyFromSymbol");
    aotx_conversation_check(cli.commands == commands_before + 1u,
                            "a bad part discards its head and the next line runs");
    aotx_conversation_fixture_close(&fixture);
}
static __device__ unsigned int aotx_conversation_find(const unsigned char *text,
                                                       unsigned int length,
                                                       const char *word)
{
    unsigned int span = 0u;
    while (word[span] != '\0') {
        span += 1u;
    }
    for (unsigned int at = 0u; at + span <= length; ++at) {
        unsigned int same = 1u;
        for (unsigned int i = 0u; i < span; ++i) {
            same &= (text[at + i] == (unsigned char)word[i]) ? 1u : 0u;
        }
        if (same != 0u) {
            return 1u;
        }
    }
    return 0u;
}
static __device__ unsigned long long aotx_conversation_number(
    const unsigned char *text, unsigned int length, const char *key)
{
    unsigned int key_len = 0u;
    while (key[key_len] != '\0') key_len++;
    for (unsigned int at = 0u; at + key_len < length; ++at) {
        unsigned int same = 1u;
        for (unsigned int i = 0u; i < key_len; ++i) same &= text[at + i] == key[i];
        if (same == 0u) continue;
        unsigned long long value = 0ull;
        for (unsigned int i = at + key_len; i < length && text[i] >= '0'
             && text[i] <= '9'; ++i) value = value * 10ull + (text[i] - '0');
        return value;
    }
    return ~0ull;
}

static __device__ void aotx_conversation_turn(unsigned int agent, unsigned int number,
                                               unsigned int tokens)
{
    unsigned char text[48];
    unsigned char reply[32];
    unsigned int at = 0u;
    const char *head = (number == 2u) ? "turn two fact amber " : "turn ";
    while (head[at] != '\0') {
        text[at] = (unsigned char)head[at];
        at += 1u;
    }
    at += aotx_text_utoa(number, (char *)text + at, (unsigned int)sizeof text - at);
    unsigned int reply_len = 0u;
    const char *answer = "answer";
    while (answer[reply_len] != '\0') {
        reply[reply_len] = (unsigned char)answer[reply_len];
        reply_len += 1u;
    }
    aotx_agents.agent[agent].turn = number;
    aotx_agent_gear[agent].source_seq = 1000ull
                                         + (unsigned long long)agent * 100ull + number;
    aotx_transcript_finish(agent, text, at, reply, reply_len, tokens,
                           5000ull + (unsigned long long)agent * 100ull + number);
    if (number == 2u) {
        aotx_request *request = &aotx_requests.slot[agent];
        const char result[] = "turn two tool result indigo";
        request->auth = AOTX_AUTH_GRANTED;
        request->result_seq = 900ull + agent;
        request->answer_seq = 850ull + agent;
        request->result_len = (unsigned int)sizeof result - 1u;
        for (unsigned int i = 0u; i < request->result_len; ++i) request->result[i] = result[i];
        aotx_transcript_result(agent, request);
    }
}

__global__ void aotx_conversation_hot(unsigned int count, aotx_conversation_result *out)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_agents.live = count;
    aotx_model[AOTX_MODEL_LANGUAGE].layers = 36u;
    aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, 36u, 8u, 128u);
    for (unsigned int agent = 0u; agent < count; ++agent) {
        aotx_agents.agent[agent].state = AOTX_AGENT_STATE_IDLE;
        aotx_agents.agent[agent].role = AOTX_ROLE_NONE;
        aotx_transcript[agent].pages = 32u;
        for (unsigned int turn = 1u; turn <= 4u; ++turn) {
            aotx_conversation_turn(agent, turn, 112u);
        }
        aotx_transcript_maintain(agent);
        if (aotx_transcript[agent].hot != 3u || aotx_transcript[agent].warm != 1u) {
            out->wrong += 1u;
        }
    }
    out->worker_hot = aotx_transcript[0].hot;
    out->worker_warm = aotx_transcript[0].warm;
    aotx_transcript[0].pages = 160u;
    aotx_transcript_maintain(0u);
    out->conductor_hot = aotx_transcript[0].hot;
    aotx_transcript[0].pages = AOTX_TRANSCRIPT_AUTO;
    aotx_agents.live = 1u;
    aotx_kv.mapped_pages = 0u;
    out->auto_free = aotx_transcript_page_limit(0u);
    aotx_agents.live = AOTX_SLOTS;
    out->auto_full = aotx_transcript_page_limit(0u);
}

__global__ void aotx_conversation_prompt(aotx_conversation_result *out)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_agents.live = 1u;
    aotx_agents.agent[0].state = AOTX_AGENT_STATE_IDLE;
    aotx_agents.agent[0].role = AOTX_ROLE_NONE;
    aotx_agents.agent[0].task = ~0u;
    aotx_transcript[0].pages = AOTX_KV_PAGES_EACH;
    aotx_say.slot[0].reply_first = 700ull;
    aotx_say.slot[0].reply_records = 2u;
    aotx_say.slot[0].prompt = 100u;
    aotx_agent_gear[0].out_tokens = 7u;
    aotx_requests.slot[0].call_seq = 800ull;
    aotx_agent_gear[0].call.entry = 0u;
    aotx_agent_gear[0].call.arg_len = 10u;
    const char call[] = "call amber";
    for (unsigned int i = 0u; i < 10u; ++i) aotx_agent_gear[0].call.arg[i] = call[i];
    aotx_conversation_turn(0u, 1u, 112u);
    aotx_request request = {};
    request.result_seq = 900ull;
    request.answer_seq = 850ull;
    request.auth = AOTX_AUTH_GRANTED;
    request.result_len = 11u;
    const char result[] = "tool result";
    for (unsigned int i = 0u; i < request.result_len; ++i) request.result[i] = result[i];
    aotx_transcript_result(0u, &request);
    const aotx_transcript_turn *first = &aotx_transcript[0].turn[0];
    out->index_records = (first->seq == 1001ull && first->reply_seq == 700ull
                          && first->reply_records == 2u && first->token_first == 100u
                          && first->token_count == 7u && first->call_seq == 800ull
                          && first->answer_seq == 850ull && first->result_seq == 900ull)
                       ? 1u : 0u;
    aotx_say.slot[0].reply_first = 0ull;
    aotx_say.slot[0].reply_records = 0u;
    aotx_requests.slot[0].call_seq = 0ull;
    aotx_conversation_turn(0u, 2u, 112u);
    request.auth = AOTX_AUTH_REFUSED;
    request.result_len = 0u;
    aotx_transcript_result(0u, &request);
    const unsigned char question[] = "what was the fact";
    unsigned int length = aotx_agent_prompt(0u, 0, question,
        (unsigned int)sizeof question - 1u, 0, 0, 0u, 0, 0u);
    out->prompt_has_first = aotx_conversation_find(aotx_say.prompt[0], length,
                                                    "turn two fact amber");
    out->prompt_has_tools = aotx_conversation_find(aotx_say.prompt[0], length, "[tool call]")
                          && aotx_conversation_find(aotx_say.prompt[0], length, "[tool grant]")
                          && aotx_conversation_find(aotx_say.prompt[0], length, "[tool result]")
                          && aotx_conversation_find(aotx_say.prompt[0], length, "[tool refusal]");
    aotx_say.slot[0].wanted = 0u;
    for (unsigned int i = 0u; i < 4000u; ++i) aotx_agent_gear[0].message[i] = 'p';
    const char tail[] = "accepted-tail";
    for (unsigned int i = 0u; i < sizeof tail - 1u; ++i) {
        aotx_agent_gear[0].message[4000u - (sizeof tail - 1u) + i] = tail[i];
    }
    length = aotx_agent_prompt(0u, 0, aotx_agent_gear[0].message, 4000u,
                               0, 0, 0u, 0, 0u);
    out->prompt_tail = aotx_conversation_find(aotx_say.prompt[0], length, tail);
    out->prompt_hash = (length == aotx_agent_gear[0].prompt_len
        && aotx_agent_gear[0].input_hash == aotx_agent_hash(aotx_say.prompt[0], length));
    aotx_say.slot[0].wanted = 0u;
    unsigned long long refused = aotx_transcript_count.prompt_refused;
    length = aotx_agent_prompt(0u, 0, aotx_agent_gear[0].message,
                               AOTX_SAY_BYTES - 1u, 0, 0, 0u, 0, 0u);
    out->prompt_refused = (length == 0u
        && aotx_transcript_count.prompt_refused == refused + 1ull);
    const aotx_console_line *line = &aotx_console.line[(aotx_console.count - 1ull)
        & (AOTX_CONSOLE_LINES - 1u)];
    out->prompt_reason = aotx_conversation_find(line->text, line->length,
                                                 "prompt does not fit");
}

__global__ void aotx_conversation_recall(aotx_conversation_result *out)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_agents.live = 1u;
    aotx_agents.agent[0].state = AOTX_AGENT_STATE_IDLE;
    aotx_agents.agent[0].role = AOTX_ROLE_NONE;
    aotx_agents.agent[0].task = ~0u;
    aotx_transcript[0].pages = 32u;
    aotx_setting_table.row[AOTX_SET_RECALL_K].value = 1;
    for (unsigned int turn = 1u; turn <= 40u; ++turn) {
        aotx_conversation_turn(0u, turn, 112u);
    }
    aotx_transcript_maintain(0u);
    for (unsigned int i = 0u; i < aotx_transcript[0].count; ++i) {
        unsigned int which = (aotx_transcript[0].first + i) % AOTX_MEMORY_TURNS;
        if (aotx_transcript[0].turn[which].tier == AOTX_MEMORY_HOT) {
            continue;
        }
        aotx_transcript_vector[0][which][0] =
            (aotx_transcript[0].turn[which].number == 2u) ? 1.0f : 0.0f;
        aotx_transcript_vector[0][which][1] =
            (aotx_transcript[0].turn[which].number == 2u) ? 0.0f : 1.0f;
        aotx_transcript[0].turn[which].vector_ready = 1u;
    }
    float query[2] = { 1.0f, 0.0f };
    aotx_transcript[0].embed_kind = AOTX_MEMORY_EMBED_QUERY;
    aotx_transcript_embed_done(0u, query, 2u);
    const unsigned char question[] = "what was amber";
    unsigned int length = aotx_agent_prompt(0u, 0, question,
        (unsigned int)sizeof question - 1u, 0, 0, 0u, 0, 0u);
    aotx_selection_body saved = aotx_transcript[0].choice;
    unsigned long long live_hash = aotx_agent_gear[0].input_hash;
    out->recall_count = saved.count;
    out->recall_turn = (saved.count == 1u && saved.seq[0] == 1002ull) ? 2u : 0u;
    out->recall_prompt = aotx_conversation_find(aotx_say.prompt[0], length,
                                                 "[memory turn 2]")
                      && aotx_conversation_find(aotx_say.prompt[0], length,
                                                "turn two tool result indigo");
    aotx_transcript_commit(aotx_time_tick);

    aotx_transcript_agent fresh = {};
    aotx_transcript[0] = fresh;
    aotx_agent_work fresh_work = {};
    aotx_agent_gear[0] = fresh_work;
    aotx_say_slot fresh_say = {};
    aotx_say.slot[0] = fresh_say;
    aotx_agents.agent[0].turn = 0u;
    aotx_transcript[0].pages = 32u;
    for (unsigned int turn = 1u; turn <= 40u; ++turn) {
        aotx_conversation_turn(0u, turn, 112u);
    }
    aotx_transcript_count.searches = 0ull;
    aotx_seam.replaying = 1ull;
    aotx_transcript_selection_apply(&saved);
    length = aotx_agent_prompt(0u, 0, question, (unsigned int)sizeof question - 1u,
                               0, 0, 0u, 0, 0u);
    out->replay_searches = (unsigned int)aotx_transcript_count.searches;
    out->replay_choices = (unsigned int)aotx_transcript_count.replay_selections;
    out->same_hash = (live_hash == aotx_agent_gear[0].input_hash) ? 1u : 0u;
    aotx_seam.replaying = 0ull;

    aotx_selection_body invalid = saved;
    invalid.count = AOTX_SELECTION_MAX + 1u;
    unsigned long long before = aotx_transcript_count.replay_selections;
    aotx_transcript_selection_apply(&invalid);
    out->invalid_selection_refused =
        (before == aotx_transcript_count.replay_selections) ? 1u : 0u;
}

__global__ void aotx_conversation_compact(aotx_conversation_result *out)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_agents.live = 1u;
    aotx_agents.agent[0].state = AOTX_AGENT_STATE_IDLE;
    aotx_agents.agent[0].role = AOTX_ROLE_NONE;
    aotx_agents.agent[0].task = ~0u;
    aotx_transcript[0].pages = 1u;
    aotx_setting_table.row[AOTX_SET_COMPACT_AT].value = 128;
    for (unsigned int turn = 1u; turn <= 129u; ++turn) {
        aotx_conversation_turn(0u, turn, 112u);
    }
    aotx_transcript_maintain(0u);
    out->automatic_compact = aotx_transcript_needs_compact(0u);
    const unsigned char instruction[] = "write a summary";
    const unsigned char summary[] = "summary amber";
    unsigned int total = 0u;
    unsigned int ranges = 1u;
    unsigned long long prior = 0ull;
    do {
        unsigned int before = aotx_transcript[0].warm;
        aotx_agent_gear[0].kind = AOTX_AGENT_TURN_COMPACT;
        aotx_transcript_prepare(0u, instruction, (unsigned int)sizeof instruction - 1u,
                                130u + total, 0u);
        aotx_transcript_summary(0u, summary, (unsigned int)sizeof summary - 1u,
                                aotx_time_tick);
        unsigned int count = before - aotx_transcript[0].warm;
        const aotx_bus_body *range = aotx_bus_body_of(aotx_transcript[0].summary_seq);
        ranges &= range != 0 && range->re_seq == 1001ull + total
            && range->corrects_seq == prior
            && aotx_conversation_number((const unsigned char *)range->text,
                                         range->text_len, "first ") == 1001ull + total
            && aotx_conversation_number((const unsigned char *)range->text,
                                         range->text_len, "last ") == 1000ull + total + count
            && aotx_conversation_number((const unsigned char *)range->text,
                                         range->text_len, "count ") == count;
        total += count;
        prior = aotx_transcript[0].summary_seq;
    } while (aotx_transcript[0].compact_left != 0u);
    unsigned long long first_summary = aotx_transcript[0].summary_seq;
    out->folded = aotx_transcript[0].folded;
    out->warm = aotx_transcript[0].warm;
    out->range_finding = ranges && total == 64u;

    aotx_agent_gear[0].kind = AOTX_AGENT_TURN_MESSAGE;
    const unsigned char next[] = "continue";
    unsigned int length = aotx_agent_prompt(0u, 0, next,
        (unsigned int)sizeof next - 1u, 0, 0, 0u, 0, 0u);
    out->summary_prompt = aotx_conversation_find(aotx_say.prompt[0], length,
                                                  "summary amber");
    aotx_transcript_compact(0u);
    out->manual_compact = aotx_transcript_needs_compact(0u);

    aotx_say.slot[0].wanted = 0u;
    aotx_agent_gear[0].kind = AOTX_AGENT_TURN_COMPACT;
    aotx_transcript_prepare(0u, instruction, (unsigned int)sizeof instruction - 1u,
                            131u, 0u);
    const unsigned char second[] = "summary two";
    aotx_transcript_summary(0u, second, (unsigned int)sizeof second - 1u, aotx_time_tick);
    const aotx_bus_body *body = aotx_bus_body_of(aotx_transcript[0].summary_seq);
    out->corrected_summary = (body != 0 && body->corrects_seq == first_summary) ? 1u : 0u;
}

__global__ void aotx_conversation_stress(unsigned int count, aotx_conversation_result *out)
{
    unsigned int agent = threadIdx.x;
    if (blockIdx.x != 0u || agent >= count) return;
    aotx_agents.agent[agent].state = AOTX_AGENT_STATE_IDLE;
    aotx_agents.agent[agent].role = AOTX_ROLE_NONE;
    aotx_agents.agent[agent].task = ~0u;
    aotx_transcript[agent].pages = 32u;
    unsigned char *text = aotx_say.prompt[agent];
    for (unsigned int i = 0u; i < 4000u; ++i) text[i] = (unsigned char)('a' + agent % 26u);
    const unsigned char reply[] = "ok";
    const unsigned char summary[] = "pool summary";
    const unsigned char instruction[] = "fold old turns";
    for (unsigned int turn = 1u; turn <= 300u; ++turn) {
        text[0] = (unsigned char)(turn & 0xffu);
        text[1] = (unsigned char)(turn >> 8u);
        aotx_agents.agent[agent].turn = turn;
        aotx_agent_gear[agent].source_seq = 100000ull + 400ull * agent + turn;
        aotx_agent_gear[agent].call.entry = AOTX_CATALOG_NO_ENTRY;
        aotx_transcript_maintain(agent);
        aotx_transcript_finish(agent, text, 4000u, reply, 2u, 112u,
                               200000ull + 400ull * agent + turn);
        unsigned int last = (aotx_transcript[agent].first
            + aotx_transcript[agent].count - 1u) % AOTX_MEMORY_TURNS;
        while (aotx_transcript[agent].turn[last].number != turn) {
            aotx_agent_gear[agent].kind = AOTX_AGENT_TURN_COMPACT;
            aotx_transcript_prepare(agent, instruction, sizeof instruction - 1u,
                                    turn, 0u);
            aotx_transcript_summary(agent, summary, sizeof summary - 1u, aotx_time_tick);
            aotx_agent_gear[agent].kind = AOTX_AGENT_TURN_MESSAGE;
            aotx_transcript_maintain(agent);
            aotx_transcript_finish(agent, text, 4000u, reply, 2u, 112u,
                                   200000ull + 400ull * agent + turn);
            last = (aotx_transcript[agent].first
                + aotx_transcript[agent].count - 1u) % AOTX_MEMORY_TURNS;
        }
    }
    aotx_transcript_maintain(agent);
    int good = 1;
    for (unsigned int i = 0u; i < aotx_transcript[agent].count; ++i) {
        unsigned int which = (aotx_transcript[agent].first + i) % AOTX_MEMORY_TURNS;
        const aotx_transcript_turn *turn = &aotx_transcript[agent].turn[which];
        if (turn->tier == AOTX_MEMORY_HOT) {
            good &= turn->text_live != 0u && turn->text_len == 4000u
                 && turn->number > 300u - aotx_transcript[agent].hot;
        }
    }
    atomicAdd(&out->stress_newest, good ? 1u : 0u);
    atomicMax(&out->stress_pool, aotx_transcript[agent].text_used);
}

__global__ void aotx_conversation_commands(aotx_conversation_result *out)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_agents.live = 1u;
    aotx_agents.agent[0].state = AOTX_AGENT_STATE_IDLE;
    aotx_agents.agent[0].role = AOTX_ROLE_NONE;
    const unsigned char automatic[] = "agent 0 pages auto";
    const unsigned char fixed[] = "agent 0 pages 32";
    const unsigned char compact[] = "agent 0 compact";
    const unsigned char show[] = "agent 0";
    const unsigned char memory[] = "memory";
    const unsigned char bad[] = "agent 0 pages 0";
    aotx_cli_line(automatic, (unsigned int)sizeof automatic - 1u, aotx_time_tick);
    out->command_auto = (aotx_transcript[0].pages == AOTX_TRANSCRIPT_AUTO) ? 1u : 0u;
    aotx_cli_line(fixed, (unsigned int)sizeof fixed - 1u, aotx_time_tick);
    out->command_pages = aotx_transcript[0].pages;
    aotx_cli_line(compact, (unsigned int)sizeof compact - 1u, aotx_time_tick);
    out->command_compact = aotx_transcript[0].compact;
    aotx_cli_line(show, (unsigned int)sizeof show - 1u, aotx_time_tick);
    aotx_cli_line(memory, (unsigned int)sizeof memory - 1u, aotx_time_tick);
    unsigned int before = aotx_cli_count.refused;
    aotx_cli_line(bad, (unsigned int)sizeof bad - 1u, aotx_time_tick);
    out->command_refused = (aotx_cli_count.refused == before + 1u) ? 1u : 0u;
}

static void aotx_conversation_memory_case(unsigned int count)
{
    aotx_conversation_fixture fixture;
    aotx_conversation_result *device = NULL;
    aotx_conversation_result result;
    memset(&result, 0, sizeof result);
    aotx_conversation_clear();
    aotx_conversation_fixture_open(&fixture);
    aotx_check_runtime(cudaMalloc(&device, sizeof result), "cudaMalloc");
    aotx_check_runtime(cudaMemset(device, 0, sizeof result), "cudaMemset");
    aotx_conversation_hot<<<1, 1>>>(count, device);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&result, device, sizeof result, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_conversation_check(result.wrong == 0u, "each worker keeps the same hot window");
    aotx_conversation_check(result.worker_hot == 3u && result.worker_warm == 1u,
                            "the fourth worker turn moves the first turn to warm memory");
    aotx_conversation_check(result.conductor_hot == 4u,
                            "the conductor keeps all four turns");
    aotx_conversation_check(result.auto_free == AOTX_KV_PAGES_EACH,
                            "automatic pages grow to the profile bound on a free pool");
    aotx_conversation_check(result.auto_full == 16u,
                            "automatic pages give way at the full agent count");
    printf("conversation: %u agents hold independent hot windows\n", count);
    cudaFree(device);
    aotx_conversation_fixture_close(&fixture);
}

static void aotx_conversation_prompt_case(void)
{
    aotx_conversation_fixture fixture;
    aotx_conversation_result *device = NULL;
    aotx_conversation_result result;
    memset(&result, 0, sizeof result);
    aotx_conversation_clear();
    aotx_conversation_fixture_open(&fixture);
    aotx_check_runtime(cudaMalloc(&device, sizeof result), "cudaMalloc");
    aotx_check_runtime(cudaMemset(device, 0, sizeof result), "cudaMemset");
    aotx_conversation_prompt<<<1, 1>>>(device);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&result, device, sizeof result, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_conversation_check(result.prompt_has_first != 0u,
                            "a later prompt contains the earlier transcript turn");
    aotx_conversation_check(result.index_records != 0u,
                            "the turn index keeps line, reply, token, call and result runs");
    aotx_conversation_check(result.prompt_has_tools != 0u,
                            "hot transcript text keeps calls, results, grants and refusals");
    aotx_conversation_check(result.prompt_tail != 0u && result.prompt_hash != 0u,
                            "an accepted line reaches the model prompt through its tail");
    aotx_conversation_check(result.prompt_refused != 0u && result.prompt_reason != 0u,
                            "an oversized prompt is refused with a reason and a counter");
    cudaFree(device);
    aotx_conversation_fixture_close(&fixture);
}

static void aotx_conversation_recall_case(void)
{
    aotx_conversation_fixture fixture;
    aotx_conversation_result *device = NULL;
    aotx_conversation_result result;
    memset(&result, 0, sizeof result);
    aotx_conversation_clear();
    aotx_conversation_fixture_open(&fixture);
    aotx_check_runtime(cudaMalloc(&device, sizeof result), "cudaMalloc");
    aotx_check_runtime(cudaMemset(device, 0, sizeof result), "cudaMemset");
    aotx_conversation_recall<<<1, 1>>>(device);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&result, device, sizeof result, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_conversation_check(result.recall_count == 1u && result.recall_turn == 2u,
                            "turn 40 selects the fact from turn 2");
    aotx_conversation_check(result.recall_prompt != 0u,
                            "the recalled turn is marked in the prompt");
    aotx_conversation_check(result.replay_searches == 0u && result.replay_choices == 1u,
                            "a replay applies the selection and runs no search");
    aotx_conversation_check(result.same_hash != 0u,
                            "the live and replayed prompt hashes agree");
    aotx_seam_state seam;
    aotx_check_runtime(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam),
                       "cudaMemcpyFromSymbol");
    unsigned int selections = 0u;
    for (unsigned long long seq = 1ull; seq <= seam.dev.tail; ++seq) {
        const aotx_record_header *record = (const aotx_record_header *)(fixture.ring
            + ((seq - 1ull) & (AOTX_CONVERSATION_RING - 1ull)) * AOTX_SLOT_BYTES);
        if (record->seq != seq || record->type != AOTX_REC_SELECTION) {
            continue;
        }
        const aotx_selection_body *body = (const aotx_selection_body *)(
            (const unsigned char *)record + AOTX_HEADER_BYTES);
        selections += 1u;
        aotx_conversation_check(record->cls == AOTX_CLASS_A
                                && record->writer == AOTX_WRITER_AGENT_BASE
                                && body->pages == 32u && body->count == 1u
                                && body->seq[0] == 1002ull,
                                "the agent writes its page limit and recall in selection");
        unsigned long long want = aotx_conversation_hash(
            AOTX_FNV_BASIS, (const unsigned char *)body, (unsigned int)sizeof *body);
        aotx_conversation_check(seam.apply.state_hash == want,
                                "the state hash folds the selection body");
    }
    aotx_conversation_check(selections == 1u, "one selection record closes the prompt");
    aotx_selection_body invalid[3] = {};
    const aotx_selection_body *saved = (const aotx_selection_body *)(fixture.ring
        + AOTX_HEADER_BYTES);
    invalid[0] = *saved;
    invalid[1] = *saved;
    invalid[2] = *saved;
    invalid[0].current_seq = 10000ull;
    invalid[1].current_seq = 10000ull;
    invalid[2].current_seq = 10000ull;
    invalid[0].count = AOTX_SELECTION_MAX + 1u;
    invalid[1].pages = AOTX_KV_PAGES_EACH + 1u;
    invalid[2].count = 1u;
    invalid[2].seq[0] = invalid[2].current_seq;
    unsigned long long prior_hash = seam.apply.state_hash;
    unsigned long long prior_applied = seam.apply.applied_count;
    for (unsigned int i = 0u; i < 3u; ++i) {
        aotx_conversation_put_type(&fixture, i, AOTX_REC_SELECTION, 0u,
            (const unsigned char *)&invalid[i], sizeof invalid[i]);
    }
    __atomic_store_n(&fixture.preamble->head, 3ull, __ATOMIC_RELEASE);
    aotx_conversation_apply(3ull, 3ull);
    aotx_check_runtime(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam),
                       "cudaMemcpyFromSymbol");
    aotx_conversation_check(seam.apply.rejected == 3ull
        && seam.apply.applied_count == prior_applied && seam.apply.state_hash == prior_hash,
        "invalid selections from the inbound ring are counted and fold nothing");
    cudaFree(device);
    aotx_conversation_fixture_close(&fixture);
}

static void aotx_conversation_compact_case(void)
{
    aotx_conversation_fixture fixture;
    aotx_conversation_result *device = NULL;
    aotx_conversation_result result;
    memset(&result, 0, sizeof result);
    aotx_conversation_clear();
    aotx_conversation_fixture_open(&fixture);
    aotx_check_runtime(cudaMalloc(&device, sizeof result), "cudaMalloc");
    aotx_check_runtime(cudaMemset(device, 0, sizeof result), "cudaMemset");
    aotx_conversation_compact<<<1, 1>>>(device);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&result, device, sizeof result, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_conversation_check(result.automatic_compact != 0u,
                            "the warm threshold asks for compaction");
    unsigned int kept = (AOTX_MEMORY_TURNS < 129u) ? AOTX_MEMORY_TURNS : 129u, dropped = (kept == AOTX_MEMORY_TURNS) ? 1u : 0u;
    aotx_conversation_check(result.folded + dropped == 64u && result.warm == kept - 64u,
                            "compaction folds the oldest half of warm memory"); printf("conversation: compaction folded %u and left %u warm of %u kept\n", result.folded, result.warm, kept);
    aotx_conversation_check(result.range_finding != 0u,
                            "the compaction finding names its first, last and count");
    aotx_conversation_check(result.summary_prompt != 0u,
                            "the prompt after compaction contains the summary");
    aotx_conversation_check(result.manual_compact != 0u,
                            "the direct compact command asks for a turn");
    aotx_conversation_check(result.corrected_summary != 0u,
                            "a new summary folds the prior summary");
    cudaFree(device);
    aotx_conversation_fixture_close(&fixture);
}

static void aotx_conversation_stress_case(unsigned int count)
{
    aotx_conversation_fixture fixture;
    aotx_conversation_result *device = NULL;
    aotx_conversation_result result = {};
    aotx_conversation_clear();
    aotx_conversation_fixture_open(&fixture);
    aotx_check_runtime(cudaMalloc(&device, sizeof result), "cudaMalloc");
    aotx_check_runtime(cudaMemset(device, 0, sizeof result), "cudaMemset");
    aotx_conversation_stress<<<1, count>>>(count, device);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&result, device, sizeof result, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_conversation_check(result.stress_newest == count,
                            "three hundred long turns keep every newest hot turn");
    aotx_conversation_check(result.stress_pool <= AOTX_TRANSCRIPT_TEXT_BYTES,
                            "the text ring stays inside its profile pool");
    printf("conversation: %u agents kept turn 300 in %u of %u text bytes\n", count,
           result.stress_pool, AOTX_TRANSCRIPT_TEXT_BYTES);
    cudaFree(device);
    aotx_conversation_fixture_close(&fixture);
}

static void aotx_conversation_command_case(void)
{
    aotx_conversation_fixture fixture;
    aotx_conversation_result *device = NULL;
    aotx_conversation_result result;
    memset(&result, 0, sizeof result);
    aotx_conversation_clear();
    aotx_conversation_fixture_open(&fixture);
    aotx_check_runtime(cudaMalloc(&device, sizeof result), "cudaMalloc");
    aotx_check_runtime(cudaMemset(device, 0, sizeof result), "cudaMemset");
    aotx_conversation_commands<<<1, 1>>>(device);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&result, device, sizeof result, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_conversation_check(result.command_auto != 0u,
                            "the pages command accepts the automatic form");
    aotx_conversation_check(result.command_pages == 32u,
                            "the pages command sets a fixed bound");
    aotx_conversation_check(result.command_compact != 0u,
                            "the compact command asks for one compaction turn");
    aotx_conversation_check(result.command_refused != 0u,
                            "the pages command refuses a zero bound");
    cudaFree(device);
    aotx_conversation_fixture_close(&fixture);
}

int main(void)
{
    CUdevice device;
    CUcontext context;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device),
                      "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    aotx_test_wrap_open();

    aotx_conversation_line_case(1u);
    aotx_conversation_line_case(AOTX_SLOTS);
    aotx_conversation_part_guards();
    aotx_conversation_memory_case(1u);
    aotx_conversation_memory_case(AOTX_SLOTS);
    aotx_conversation_prompt_case();
    aotx_conversation_recall_case();
    aotx_conversation_compact_case();
    aotx_conversation_stress_case(1u);
    aotx_conversation_stress_case(AOTX_SLOTS);
    aotx_conversation_command_case();

    printf("conversation: %u checks, %u failed\n", aotx_conversation_applied,
           aotx_conversation_failed);
    aotx_check_driver(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    return (aotx_conversation_failed == 0u) ? 0 : 1;
}
