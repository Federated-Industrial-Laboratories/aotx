/* Purpose: Check the per-agent sampler table, its command, replay and stop paths.
 * Owns: Nothing; the command fixture owns the seam and console state.
 * Launch shape: One thread for each agent row under test.
 * Lifetime: One part of the command check. */
#ifndef AOTX_TEST_CLI_SAMPLER_H
#define AOTX_TEST_CLI_SAMPLER_H

#include "model/sampler.cuh"

__global__ void aotx_test_sampler_rows(unsigned int count)
{
    unsigned int agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent >= count) {
        return;
    }
    aotx_agents.agent[agent].state = AOTX_AGENT_STATE_IDLE;
    aotx_sampler_reset(agent);
    aotx_sampler_set(agent, "decode.temperature", 18u, "0.8", 3u);
    aotx_sampler_set(agent, "decode.top_k", 12u, "23", 2u);
    aotx_sampler_set(agent, "decode.top_p", 12u, "0.9", 3u);
    aotx_sampler_set(agent, "decode.min_p", 12u, "0.05", 4u);
    aotx_sampler_set(agent, "decode.repeat_penalty", 21u, "1.1", 3u);
    aotx_sampler_set(agent, "decode.repeat_window", 20u, "64", 2u);
    aotx_sampler_set(agent, "decode.presence_penalty", 23u, "0.2", 3u);
    aotx_sampler_set(agent, "decode.frequency_penalty", 24u, "0.3", 3u);
    aotx_sampler_set(agent, "decode.seed", 11u, "42", 2u);
    aotx_sampler_set(agent, "decode.think_limit", 18u, "7", 1u);
}

__global__ void aotx_test_sampler_temperature(float value, unsigned int count)
{
    unsigned int agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent < count) {
        aotx_sampler.row[agent].temperature = value;
    }
}

__global__ void aotx_test_sampler_sequences(unsigned int count)
{
    unsigned int agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent < count) {
        aotx_seqs.slot[agent].state = AOTX_SEQ_STATE_DECODE;
        aotx_seqs.slot[agent].flags = 0u;
    }
}

static void aotx_test_sampler_commands(const char *key, const char *value,
                                       unsigned int count)
{
    char text[AOTX_TEST_BATCH][AOTX_BODY_BYTES] = { { 0 } };
    unsigned int length[AOTX_TEST_BATCH] = { 0u };
    for (unsigned int i = 0u; i < count; ++i) {
        int used = snprintf(text[i], sizeof(text[i]), "agent %u %s%s%s", i, key,
                            (value != NULL) ? " " : "", (value != NULL) ? value : "");
        length[i] = (used > 0) ? (unsigned int)used : 0u;
    }
    aotx_test_lines(text, length, count);
}

static void aotx_test_sampler_table(void)
{
    aotx_sampler_table table;
    static const unsigned int counts[2] = { 1u, AOTX_TEST_BATCH };
    for (unsigned int c = 0u; c < 2u; ++c) {
        unsigned int count = counts[c];
        aotx_test_sampler_rows<<<1, count>>>(count);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_sampler, sizeof table),
                           "cudaMemcpyFromSymbol");
        unsigned int same = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            const aotx_model_how *row = &table.row[i];
            same += (row->temperature == 0.8f && row->top_k == 23u && row->top_p == 0.9f
                     && row->min_p == 0.05f && row->repeat_penalty == 1.1f
                     && row->repeat_window == 64u && row->presence_penalty == 0.2f
                     && row->frequency_penalty == 0.3f && row->seed == 42ull
                     && row->think_limit == 7 && table.changed[i] == 10u) ? 1u : 0u;
        }
        aotx_test_check(same == count, "every sampler row takes all ten fields");
        printf("cli: %u sampler rows took ten fields\n", count);

        aotx_test_sampler_temperature<<<1, count>>>(0.0f, count);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_seam_set_replaying(1);
        aotx_test_sampler_commands("decode.temperature", "0.8", count);
        aotx_seam_set_replaying(0);
        aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_sampler, sizeof table),
                           "cudaMemcpyFromSymbol");
        same = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            same += (table.row[i].temperature == 0.8f && table.changed[i] == 11u) ? 1u : 0u;
        }
        aotx_test_check(same == count, "replayed agent set lines restore every sampler row");

        aotx_test_sampler_sequences<<<1, count>>>(count);
        aotx_test_sampler_commands("stop", NULL, count);
        aotx_seq_table *seqs = (aotx_seq_table *)malloc(sizeof *seqs);
        aotx_check_runtime(cudaMemcpyFromSymbol(seqs, aotx_seqs, sizeof *seqs),
                           "cudaMemcpyFromSymbol");
        same = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            same += ((seqs->slot[i].flags & AOTX_DECODE_MARK_STOP) != 0u) ? 1u : 0u;
        }
        aotx_test_check(same == count, "agent stop lines raise every decode flag");
        printf("cli: %u sampler rows replayed and %u replies stopped\n", count, same);
        free(seqs);
    }
}

#endif
