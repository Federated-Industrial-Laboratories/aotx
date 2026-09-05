/* Purpose: Compare every fixed-state byte after carried decode and prompt replay.
 * Owns: Host snapshots and one copy of the real decode child graph.
 * Threading: One host caller; device passes take distinct sequences.
 * Lifetime: One architecture check. */
#ifndef AOTX_TEST_ARCH_DELTA_H
#define AOTX_TEST_ARCH_DELTA_H
#include "model/hybrid.cuh"

static int aotx_arch_delta_pass(aotx_arch_gear *gear, aotx_kv_map *map,
                                 unsigned int role, unsigned int seqs,
                                 const unsigned int *agents, unsigned int mode,
                                 unsigned int part, cudaGraphExec_t decode)
{
    int ids[AOTX_ARCH_IDS];
    unsigned int offsets[AOTX_SLOTS + 1u] = {};
    for (unsigned int s = 0u; s < seqs; ++s) {
        unsigned int prefix = 2u + agents[s] % 3u;
        unsigned int first = 0u, count = prefix + 2u;
        if (mode == 0u) {
            first = part ? prefix + part - 1u : 0u;
            count = part ? 1u : prefix;
        } else if (mode == 2u) {
            first = part;
            count = part < 2u ? 1u : prefix;
        }
        offsets[s + 1u] = offsets[s] + count;
        if (offsets[s + 1u] > AOTX_ARCH_IDS) return 1;
        for (unsigned int p = 0u; p < count; ++p)
            ids[offsets[s] + p] = (int)(1000u + agents[s] * 19u + (first + p) * 7u);
    }
    aotx_check_runtime(cudaMemcpy(gear->ids, ids, offsets[seqs] * sizeof(int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->offset, offsets, (seqs + 1u) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->agent, agents, seqs * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    if (aotx_model_pages(role, gear->offset, seqs, gear->agent)) return 1;
    aotx_kv_serve(map, 0);
    aotx_model_run run = {};
    run.ids = gear->ids; run.offset = gear->offset; run.agent = gear->agent;
    run.seqs = seqs; run.tokens = offsets[seqs]; run.rows = seqs;
    run.select = AOTX_MODEL_ROWS_LAST;
    if (mode == 0u && part != 0u) {
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &run, sizeof run,
                                              (size_t)role * sizeof run), "cudaMemcpyToSymbol");
        aotx_check_runtime(cudaGraphLaunch(decode, 0), "cudaGraphLaunch");
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    } else if (aotx_model_launch(role, &run)) return 1;
    return aotx_model_faulted() != 0u;
}

static int aotx_arch_delta_snapshot(const aotx_delta_work *delta,
                                     const unsigned int *agents, unsigned int seqs,
                                     unsigned char *snapshot, unsigned char *scratch,
                                     unsigned int compare, unsigned long long *checked)
{
    size_t matrix = (size_t)(delta->state_elements / AOTX_SLOTS) * sizeof(float);
    size_t history = (size_t)(delta->history_elements / AOTX_SLOTS) * sizeof(float);
    size_t stride = matrix + history;
    int bad = 0;
    for (unsigned int s = 0u; s < seqs; ++s) {
        unsigned char *target = compare ? scratch : snapshot + s * stride;
        aotx_check_runtime(cudaMemcpy(target, (const char *)delta->state + agents[s] * matrix,
                                      matrix, cudaMemcpyDeviceToHost), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(target + matrix,
                                      (const char *)delta->history + agents[s] * history,
                                      history, cudaMemcpyDeviceToHost), "cudaMemcpy");
        if (!compare) {
            const float *values = (const float *)target;
            unsigned int matrix_live = 0u, history_live = 0u;
            for (size_t i = 0u; i < stride / sizeof(float); ++i) {
                bad |= !isfinite(values[i]);
                if (i < matrix / sizeof(float)) matrix_live |= values[i] != 0.0f;
                else history_live |= values[i] != 0.0f;
            }
            bad |= !matrix_live || !history_live;
        }
        if (compare) {
            const unsigned char *expected = snapshot + s * stride;
            if (memcmp(expected, target, stride) != 0) {
                size_t first = 0u;
                while (first < stride && expected[first] == target[first]) ++first;
                if (!bad) printf("arch: fixed state first difference slot %u byte %zu\n", agents[s], first);
                bad = 1;
            }
            *checked += stride;
        }
    }
    return bad;
}

static void aotx_arch_delta(aotx_arch_gear *gear, aotx_kv_map *map, unsigned int role,
                             const char *file)
{
    aotx_model_hold *hold = aotx_model_hold_of(role);
    if (hold->work.delta.layers == 0u) return;
    printf("arch: fixed allocation %llu bytes, descriptor %zu bytes, workspace %zu bytes\n",
           aotx_model_hybrid_bytes(&hold->desc, hold->max_tokens),
           sizeof(hold->desc), sizeof(hold->work));
    const aotx_delta_work delta = hold->work.delta;
    size_t stride = (size_t)((delta.state_elements + delta.history_elements) / AOTX_SLOTS)
                  * sizeof(float);
    unsigned char *scratch = (unsigned char *)malloc(stride);
    unsigned char *snapshot = (unsigned char *)malloc(stride * AOTX_SLOTS);
    cudaGraphExec_t decode = aotx_arch_decode_graph();
    if (!scratch || !snapshot || !decode) {
        aotx_arch_check(0, file, "fixed-state replay buffers and decode graph open");
    } else for (unsigned int arm = 0u; arm < 2u; ++arm) {
        unsigned int seqs = arm ? AOTX_SLOTS : 1u;
        unsigned int agents[AOTX_SLOTS];
        for (unsigned int s = 0u; s < seqs; ++s) agents[s] = (37u * s + 11u) % AOTX_SLOTS;
        unsigned int before = aotx_arch_mapped();
        unsigned long long checked = 0ull;
        int bad = 0;
        for (unsigned int mode = 0u; mode < 3u && !bad; ++mode) {
            aotx_model_forget();
            unsigned int parts = mode == 1u ? 1u : 3u;
            for (unsigned int part = 0u; part < parts && !bad; ++part) {
                unsigned int limit = AOTX_ARCH_IDS / ((mode == 0u && part != 0u) ? 1u : 6u);
                for (unsigned int first = 0u; first < seqs && !bad; first += limit) {
                    unsigned int count = seqs - first < limit ? seqs - first : limit;
                    bad = aotx_arch_delta_pass(gear, map, role, count, agents + first,
                                                mode, part, decode);
                }
            }
            if (!bad) bad = aotx_arch_delta_snapshot(&delta, agents, seqs, snapshot, scratch,
                                                      mode != 0u, &checked);
        }
        for (unsigned int s = 0u; s < seqs; ++s) aotx_arch_release<<<1, 1>>>(agents[s]);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_kv_serve(map, 0);
        printf("arch: fixed state N=%u bytes=%zu compared=%llu; carried decode, whole replay, split replay\n",
               seqs, stride * seqs, checked);
        aotx_arch_check(!bad && checked == 2ull * stride * seqs
                        && aotx_arch_mapped() == before, file,
                        arm ? "fixed state replay is byte identical for distinct batch slots"
                            : "fixed state replay is byte identical for one slot");
    }
    if (decode) cudaGraphExecDestroy(decode);
    aotx_decode_close();
    free(snapshot);
    free(scratch);
}
#endif
