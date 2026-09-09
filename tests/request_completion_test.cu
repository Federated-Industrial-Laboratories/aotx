/* Purpose: Check terminal prompt refusal, recovery and recorded-choice replay.
 * Owns: Temporary record rings and distinct agent fixtures without model weights.
 * Launch shape: One thread per agent; cases use one and all profile slots.
 * Lifetime: One test process. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agent/prompt.cuh"
#include "agent/transcript.cuh"
#include "boot/check.h"
#include "settings/settings.cuh"
#include "wrap_fixture.h"

#define REQUEST_RING 4096ull

static unsigned int applied;
static unsigned int failed;

static void check(int pass, const char *text, unsigned int agent)
{
    applied++;
    if (!pass) {
        failed++;
        printf("request completion: FAILED agent %u %s\n", agent, text);
    }
}

#define CLEAR(symbol) do { \
    void *address = NULL; \
    aotx_check_runtime(cudaGetSymbolAddress(&address, symbol), "cudaGetSymbolAddress"); \
    aotx_check_runtime(cudaMemset(address, 0, sizeof(symbol)), "cudaMemset"); \
} while (0)

static void reset_state(void)
{
    CLEAR(aotx_agents);
    CLEAR(aotx_agent_gear);
    CLEAR(aotx_agent_count);
    CLEAR(aotx_task_used);
    CLEAR(aotx_transcript);
    CLEAR(aotx_transcript_text);
    CLEAR(aotx_transcript_count);
    CLEAR(aotx_transcript_replay_tick);
    CLEAR(aotx_say);
    CLEAR(aotx_seqs);
    CLEAR(aotx_requests);
    CLEAR(aotx_tool_done);
    CLEAR(aotx_tool_embed);
    CLEAR(aotx_console);
    CLEAR(aotx_catalog);
}

static unsigned char *open_ring(void)
{
    unsigned char *ring = NULL;
    size_t bytes = (size_t)REQUEST_RING * AOTX_SLOT_BYTES;
    aotx_check_runtime(cudaMallocManaged(&ring, bytes), "cudaMallocManaged");
    memset(ring, 0, bytes);
    aotx_seam_state seam = {};
    seam.dev.base = ring;
    seam.dev.slot_count = REQUEST_RING;
    seam.dev.mask = REQUEST_RING - 1ull;
    seam.apply.state_hash = AOTX_FNV_BASIS;
    seam.boot_id = 35ull;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof seam), "cudaMemcpyToSymbol");
    return ring;
}

__global__ void aotx_request_prepare(unsigned int count, unsigned int mode, unsigned int stage)
{
    if (threadIdx.x == 0u) {
        aotx_settings_reset();
        aotx_setting_table.row[AOTX_SET_RECALL_K].value = 0ll;
        aotx_setting_table.row[AOTX_SET_COMPACT_AT].value = 0ll;
        aotx_catalog.conductor = AOTX_CATALOG_NO_ENTRY;
        aotx_agents.live = count;
        aotx_model[AOTX_MODEL_LANGUAGE].layers = 36u;
        aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, 36u, 8u, 128u);
    }
    __syncthreads();
    unsigned int id = threadIdx.x;
    if (id >= count) return;
    aotx_agent *agent = &aotx_agents.agent[id];
    agent->state = AOTX_AGENT_STATE_IDLE;
    agent->role = AOTX_ROLE_NONE;
    agent->task = ~0u;
    agent->turn = 1u;
    agent->budget_left = 4u;
    aotx_agent_work *gear = &aotx_agent_gear[id];
    gear->kind = AOTX_AGENT_TURN_MESSAGE;
    gear->call.entry = AOTX_CATALOG_NO_ENTRY;
    gear->source_seq = 2000ull + id;
    gear->message_len = stage == 0u ? AOTX_SAY_BYTES - 1u - id % 8u : 8u + id % 8u;
    gear->has_message = 1u;
    for (unsigned int i = 0u; i < gear->message_len; ++i)
        gear->message[i] = (unsigned char)('a' + (i + id) % 26u);
    aotx_transcript_agent *history = &aotx_transcript[id];
    history->pages = 1u;
    history->count = 1u;
    history->text_used = 32u;
    history->text_head = 32u;
    history->turn[0].seq = 1000ull + id;
    history->turn[0].number = 1u;
    history->turn[0].text_len = 32u;
    history->turn[0].stored_len = 32u;
    history->turn[0].text_live = 1u;
    history->turn[0].vector_ready = 1u;
    history->turn[0].tier = AOTX_MEMORY_HOT;
    history->turn[0].tokens = 1u;
    for (unsigned int i = 0u; i < 32u; ++i)
        aotx_transcript_text[id][i] = (unsigned char)('A' + (i + id) % 26u);
    if (mode != 0u) {
        agent->task = id;
        gear->kind = mode == 1u ? AOTX_AGENT_TURN_TASK : AOTX_AGENT_TURN_VERIFY;
        gear->has_message = 0u;
        aotx_task_used[id] = 1u;
        aotx_task *task = &aotx_agents.task[id];
        task->agent = id;
        task->verifier = id;
        task->state = mode == 1u ? AOTX_TASK_ASSIGNED : AOTX_TASK_VERIFYING;
        task->text_len = gear->message_len;
        task->source_seq = gear->source_seq;
        for (unsigned int i = 0u; i < task->text_len; ++i) task->text[i] = gear->message[i];
        task->result_len = mode == 2u ? 1u : 0u;
        task->result[0] = (char)('a' + id % 26u);
    }
}

__global__ void aotx_request_tick(unsigned int replay)
{
    if (threadIdx.x != 0u) return;
    aotx_time_tick += 1ull;
    aotx_seam.replaying = replay;
}

__global__ void aotx_request_commit(void)
{
    if (threadIdx.x == 0u) aotx_transcript_commit(aotx_time_tick);
}

__global__ void aotx_request_replay_choices(const aotx_selection_body *choices, unsigned int count)
{
    if (threadIdx.x < count) aotx_transcript_selection_apply(&choices[threadIdx.x]);
}

__global__ void aotx_request_next(unsigned int count, unsigned int complete)
{
    unsigned int id = threadIdx.x;
    if (id >= count) return;
    aotx_agent_work *gear = &aotx_agent_gear[id];
    if (complete == 0u) {
        unsigned char text[8] = {'n', 'e', 'x', 't', ' ', '0', '0', '.'};
        text[5] += (unsigned char)(id / 10u);
        text[6] += (unsigned char)(id % 10u);
        aotx_agent_queue_message(id, text, sizeof text, 3000ull + id);
    } else {
        /* Supply a fixed reply to test completion independently of model arithmetic. */
        aotx_say.slot[id].wanted = 0u;
        aotx_agents.agent[id].state = AOTX_AGENT_STATE_POST;
        gear->reply[0] = (unsigned char)('a' + id % 26u);
        gear->reply_len = 1u;
        gear->out_tokens = 1u;
        gear->last_token = 1u;
    }
}

/* Model a tokenizer or page-allocation refusal after prompt construction. */
__global__ void aotx_request_deny_open(void)
{
    unsigned int id = threadIdx.x;
    if (aotx_agents.agent[id].state == AOTX_AGENT_STATE_PROMPT) {
        aotx_say.slot[id].wanted = 0u;
        aotx_say.slot[id].ready = 0u;
    }
}

static void ticks(unsigned int count, unsigned int replay, unsigned int deny = 0u)
{
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_request_tick<<<1, 1>>>(replay);
        if (deny != 0u) aotx_request_deny_open<<<1, AOTX_SLOTS>>>();
        aotx_agent_step<<<1, AOTX_SLOTS>>>(0ull);
        aotx_request_commit<<<1, 1>>>();
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

static void refused_state(unsigned int count, unsigned int mode, unsigned int stage)
{
    aotx_agent_table *agents = new aotx_agent_table;
    aotx_agent_work *gear = new aotx_agent_work[AOTX_SLOTS];
    aotx_transcript_agent *history = new aotx_transcript_agent[AOTX_SLOTS];
    aotx_check_runtime(cudaMemcpyFromSymbol(agents, aotx_agents, sizeof *agents), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(gear, aotx_agent_gear,
        sizeof(*gear) * AOTX_SLOTS), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(history, aotx_transcript,
        sizeof(*history) * AOTX_SLOTS), "cudaMemcpyFromSymbol");
    for (unsigned int id = 0u; id < count; ++id) {
        check(agents->agent[id].state == AOTX_AGENT_STATE_IDLE
              && agents->agent[id].turn == 2u && agents->agent[id].task == ~0u,
              "refusal ends one attempted turn", id);
        check(gear[id].has_message == 0u && gear[id].continuable == 0u
              && gear[id].wrote == 0u && gear[id].stop_requested != 0u
              && agents->agent[id].request == 0u && agents->agent[id].deadline == 0ull,
              "refusal leaves no pending work ownership", id);
        check(history[id].count == 1u && history[id].text_used == 32u
              && history[id].turn[0].seq == 1000ull + id,
              "earlier conversation records remain", id);
        check(gear[id].message_len == (stage == 0u ? AOTX_SAY_BYTES - 1u - id % 8u : 8u + id % 8u)
              && gear[id].message[0] == (unsigned char)('a' + id % 26u)
              && gear[id].source_seq == 2000ull + id,
              "the refused input remains available", id);
        if (mode != 0u) check(agents->task[id].state == AOTX_TASK_FAILED,
                              "a refused task does not remain assigned", id);
        if (mode == 2u) check(agents->task[id].result_len == 1u
              && agents->task[id].result[0] == (char)('a' + id % 26u)
              && agents->agent[id].verdict == AOTX_VERDICT_UNCERTAIN,
              "a refused verifier preserves the result and states uncertainty", id);
    }
    delete[] history;
    delete[] gear;
    delete agents;
}

static unsigned int records(const unsigned char *ring, unsigned int count,
                            aotx_selection_body *choices)
{
    unsigned int ends[AOTX_SLOTS] = {};
    unsigned int selected[AOTX_SLOTS] = {};
    for (unsigned int i = 0u; i < REQUEST_RING; ++i) {
        const aotx_record_header *h = (const aotx_record_header *)(ring + i * AOTX_SLOT_BYTES);
        if (h->seq == 0ull) continue;
        if (h->type == AOTX_REC_MANIFEST) {
            const aotx_manifest_body *m = (const aotx_manifest_body *)
                ((const unsigned char *)h + AOTX_HEADER_BYTES);
            if (m->agent < count && m->finish == AOTX_TURN_REFUSED) {
                ends[m->agent]++;
                check(m->turn == 2u && m->output_tokens == 0u && m->input_hash == 0ull
                      && m->tool == AOTX_TOOL_NONE && m->request == 0u,
                      "refusal states no generation or executed call", m->agent);
            }
        } else if (h->type == AOTX_REC_SELECTION) {
            const aotx_selection_body *choice = (const aotx_selection_body *)
                ((const unsigned char *)h + AOTX_HEADER_BYTES);
            if (choice->agent < count) {
                selected[choice->agent]++;
                choices[choice->agent] = *choice;
                check(h->cls == AOTX_CLASS_A && choice->turn == 2u,
                      "the refusal memory choice is authoritative", choice->agent);
            }
        }
    }
    unsigned int total = 0u;
    for (unsigned int id = 0u; id < count; ++id) {
        check(ends[id] == 1u, "one refusal is recorded across repeated ticks", id);
        check(selected[id] == 1u, "one choice can reconstruct the refused turn", id);
        total += selected[id] == 1u;
    }
    return total;
}

static void run_case(unsigned int count, unsigned int mode, unsigned int stage)
{
    reset_state();
    unsigned char *ring = open_ring();
    aotx_request_prepare<<<1, AOTX_SLOTS>>>(count, mode, stage);
    ticks(1u, 1u);
    aotx_agent_counts counts = {};
    aotx_check_runtime(cudaMemcpyFromSymbol(&counts, aotx_agent_count, sizeof counts), "cudaMemcpyFromSymbol");
    check(counts.opens_refused == 0u, "replay waits for a missing recorded choice", 0u);
    ticks(12u, 0u, stage);
    refused_state(count, mode, stage);
    aotx_selection_body *choices = NULL;
    aotx_check_runtime(cudaMallocManaged(&choices, sizeof(*choices) * AOTX_SLOTS), "cudaMallocManaged");
    memset(choices, 0, sizeof(*choices) * AOTX_SLOTS);
    unsigned int selected = records(ring, count, choices);
    cudaFree(ring);
    reset_state();
    ring = open_ring();
    aotx_request_prepare<<<1, AOTX_SLOTS>>>(count, mode, stage);
    if (selected == count) {
        aotx_request_replay_choices<<<1, AOTX_SLOTS>>>(choices, count);
        ticks(8u, 1u, stage);
        refused_state(count, mode, stage);
        ticks(8u, 0u, stage);
        refused_state(count, mode, stage);
        aotx_request_next<<<1, AOTX_SLOTS>>>(count, 0u);
        ticks(1u, 0u);
        aotx_agent_table *agents = new aotx_agent_table;
        aotx_check_runtime(cudaMemcpyFromSymbol(agents, aotx_agents, sizeof *agents), "cudaMemcpyFromSymbol");
        for (unsigned int id = 0u; id < count; ++id)
            check(agents->agent[id].state == AOTX_AGENT_STATE_PROMPT
                  && agents->agent[id].turn == 3u, "new input is admitted after restore", id);
        aotx_request_next<<<1, AOTX_SLOTS>>>(count, 1u);
        ticks(1u, 0u);
        aotx_check_runtime(cudaMemcpyFromSymbol(agents, aotx_agents, sizeof *agents), "cudaMemcpyFromSymbol");
        for (unsigned int id = 0u; id < count; ++id)
            check(agents->agent[id].state == AOTX_AGENT_STATE_IDLE
                  && agents->agent[id].turn == 3u, "the admitted fixed reply completes", id);
        delete agents;
    }
    cudaFree(choices);
    cudaFree(ring);
}

__global__ void aotx_request_stop(unsigned int prior)
{
    if (threadIdx.x != 0u) return;
    if (prior != 0u) {
        /* A completed sequence can remain in the slot until the next admission. */
        aotx_say.slot[0].ready = 1u;
        aotx_seqs.slot[0].state = AOTX_SEQ_STATE_DONE;
    }
    aotx_cli_line((const unsigned char *)"stop", 4u, aotx_time_tick);
}

static void stop_case(unsigned int count, unsigned int mode, unsigned int prior)
{
    reset_state();
    unsigned char *ring = open_ring();
    aotx_request_prepare<<<1, AOTX_SLOTS>>>(count, mode, 1u);
    ticks(1u, 0u);
    aotx_request_stop<<<1, 1>>>(prior);
    ticks(8u, 0u);
    aotx_agent_table *agents = new aotx_agent_table;
    aotx_agent_work gear = {};
    aotx_agent_counts counts = {};
    aotx_check_runtime(cudaMemcpyFromSymbol(agents, aotx_agents, sizeof *agents), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&gear, aotx_agent_gear, sizeof gear), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&counts, aotx_agent_count, sizeof counts), "cudaMemcpyFromSymbol");
    check(agents->agent[0].state == AOTX_AGENT_STATE_IDLE && agents->agent[0].turn == 2u
          && agents->agent[0].task == ~0u, "stop ends one pending turn", 0u);
    check(gear.stopped == 1u && gear.prompt_refused == 0u && gear.has_message == 0u
          && gear.message_len == 8u && gear.source_seq == 2000ull,
          "stop retains the input and its stated reason", 0u);
    check(counts.opens_refused == 0u && counts.done == 0u,
          "stop states no capacity failure or task success", 0u);
    if (mode != 0u) check(agents->task[0].state == AOTX_TASK_FAILED, "stopped task ends", 0u);
    if (mode == 2u) check(agents->task[0].result_len == 1u && agents->task[0].result[0] == 'a'
          && agents->agent[0].verdict == AOTX_VERDICT_UNCERTAIN,
          "stopped verifier preserves the result without a verdict", 0u);
    for (unsigned int id = 1u; id < count; ++id)
        check(agents->agent[id].state == AOTX_AGENT_STATE_PROMPT && agents->agent[id].turn == 2u,
              "stop leaves other pending agents alone", id);
    unsigned int stopped = 0u, refused = 0u;
    for (unsigned int i = 0u; i < REQUEST_RING; ++i) {
        const aotx_record_header *h = (const aotx_record_header *)(ring + i * AOTX_SLOT_BYTES);
        if (h->seq == 0ull || h->type != AOTX_REC_MANIFEST) continue;
        const aotx_manifest_body *m = (const aotx_manifest_body *)
            ((const unsigned char *)h + AOTX_HEADER_BYTES);
        stopped += m->finish == AOTX_TURN_STOPPED;
        refused += m->finish == AOTX_TURN_REFUSED;
        check(m->agent == 0u && m->turn == 2u && m->output_tokens == 0u
              && m->input_hash == 0ull && m->tool == AOTX_TOOL_NONE && m->request == 0u,
              "the stopped manifest states no generation", m->agent);
    }
    check(stopped == 1u && refused == 0u, "repeated ticks retain one stopped outcome", 0u);
    aotx_request_next<<<1, AOTX_SLOTS>>>(1u, 0u);
    ticks(1u, 0u);
    aotx_check_runtime(cudaMemcpyFromSymbol(agents, aotx_agents, sizeof *agents), "cudaMemcpyFromSymbol");
    check(agents->agent[0].state == AOTX_AGENT_STATE_PROMPT && agents->agent[0].turn == 3u,
          "new input starts after a pending stop", 0u);
    delete agents;
    cudaFree(ring);
}

int main(void)
{
    aotx_check_runtime(cudaSetDevice(0), "cudaSetDevice");
    aotx_test_wrap_open();
    for (unsigned int stage = 0u; stage < 2u; ++stage) {
        for (unsigned int mode = 0u; mode < 3u; ++mode) {
            run_case(1u, mode, stage);
            run_case(AOTX_SLOTS, mode, stage);
        }
    }
    for (unsigned int prior = 0u; prior < 2u; ++prior)
        for (unsigned int mode = 0u; mode < 3u; ++mode) {
            stop_case(1u, mode, prior);
            stop_case(AOTX_SLOTS, mode, prior);
        }
    printf("request completion: %u checks, %u failed\n", applied, failed);
    return failed == 0u ? 0 : 1;
}
