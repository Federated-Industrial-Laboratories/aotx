/* Purpose: Check QKV biases before the rotary turn and the half cache writes.
 * Owns: Independent projection rows, head biases and page buffers for one and all sequence slots.
 * Launch shape: Token and head grids, with fewer blocks than tokens and with extra blocks.
 * Lifetime: One test run. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "model/forward.cuh"

static unsigned int aotx_bias_checks;
static unsigned int aotx_bias_failed;

static void aotx_bias_check(int ok, const char *name, unsigned int seqs)
{
    ++aotx_bias_checks;
    aotx_bias_failed += !ok;
    printf("bias: %s %s N=%u\n", ok ? "ok" : "FAIL", name, seqs);
}

static void *aotx_bias_copy(const void *host, size_t bytes)
{
    void *device = NULL;
    aotx_check_runtime(cudaMalloc(&device, bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
    return device;
}

/* Read at independently supplied positions, not the forward call's sequence search. */
__global__ void aotx_bias_read(const unsigned int *agents, const unsigned int *positions,
                               unsigned int tokens, half *keys, half *values)
{
    unsigned int t = blockIdx.x;
    const aotx_model_work *work = &aotx_model_space[0];
    if (t >= tokens) return;
    for (unsigned int d = threadIdx.x; d < 128u; d += blockDim.x) {
        const half *key = aotx_kvl_key(&work->shape, agents[t], 0u, d / 64u, positions[t]);
        const half *value = aotx_kvl_value(&work->shape, agents[t], 0u, d / 64u, positions[t]);
        keys[(size_t)t * 128u + d] = key ? key[d % 64u] : __float2half(NAN);
        values[(size_t)t * 128u + d] = value ? value[d % 64u] : __float2half(NAN);
    }
}

static int aotx_bias_turned(const half *got, const float *src, const float *bias,
                            const unsigned int *positions, unsigned int tokens,
                            unsigned int width)
{
    int ok = 1;
    for (unsigned int t = 0u; t < tokens; ++t) {
        for (unsigned int head = 0u; head < width / 64u; ++head) {
            for (unsigned int p = 0u; p < 32u; ++p) {
                unsigned int low = head * 64u + p, high = low + 32u;
                float a = src[(size_t)t * width + low] + bias[low];
                float b = src[(size_t)t * width + high] + bias[high];
                double angle = (double)positions[t] * pow(1000000.0, -(double)p / 32.0);
                double want[] = { a * cos(angle) - b * sin(angle),
                                  a * sin(angle) + b * cos(angle) };
                unsigned int at[] = { low, high };
                for (unsigned int side = 0u; side < 2u; ++side) {
                    double value = __half2float(got[(size_t)t * width + at[side]]);
                    ok &= isfinite(value) && fabs(value - want[side])
                        <= 0.0006 * fabs(want[side]) + 0.0001;
                }
            }
        }
    }
    return ok;
}

static void aotx_bias_case(unsigned int seqs)
{
    const unsigned int qwidth = 896u, kwidth = 128u;
    unsigned int offsets[AOTX_SLOTS + 1u] = {}, agents[AOTX_SLOTS] = {}, bases[AOTX_SLOTS] = {};
    unsigned int token_agent[3u * AOTX_SLOTS], positions[3u * AOTX_SLOTS];
    for (unsigned int s = 0u; s < seqs; ++s) {
        agents[s] = (s * 17u + 3u) % AOTX_SLOTS;
        bases[s] = 3u + (s * 7u + seqs) % 37u;
        offsets[s + 1u] = offsets[s] + 1u + s % 3u;
        for (unsigned int t = offsets[s]; t < offsets[s + 1u]; ++t) {
            token_agent[t] = agents[s];
            positions[t] = bases[s] + t - offsets[s];
        }
    }
    unsigned int tokens = offsets[seqs], capacity = tokens + 2u;
    size_t qcount = (size_t)capacity * qwidth, kcount = (size_t)capacity * kwidth;
    float *q = (float *)malloc(qcount * sizeof *q);
    float *k = (float *)malloc(kcount * sizeof *k);
    float *v = (float *)malloc(kcount * sizeof *v);
    float bias[8u + 896u + 128u + 128u];
    for (unsigned int d = 0u; d < sizeof bias / sizeof bias[0]; ++d)
        bias[d] = -32.0f + (float)((d * 11u + d / 64u * 19u) % 97u) / 101.0f;
    for (unsigned int t = 0u; t < capacity; ++t) {
        for (unsigned int d = 0u; d < qwidth; ++d)
            q[(size_t)t * qwidth + d] = 32.0f + (float)((t * 31u + d * 7u + seqs * 13u) % 113u) / 127.0f;
        for (unsigned int d = 0u; d < kwidth; ++d) {
            k[(size_t)t * kwidth + d] = 32.0f + (float)((t * 17u + d * 13u + seqs * 5u) % 109u) / 131.0f;
            v[(size_t)t * kwidth + d] = 32.0f + (float)((t * 23u + d * 19u + seqs * 11u) % 107u) / 137.0f;
        }
    }
    half *qh = (half *)malloc(qcount * sizeof *qh);
    half *kh = (half *)malloc(kcount * sizeof *kh);
    half *vh = (half *)malloc(kcount * sizeof *vh);
    for (size_t i = 0u; i < qcount; ++i) qh[i] = __float2half(-123.0f);
    for (size_t i = 0u; i < kcount; ++i) kh[i] = vh[i] = __float2half(-123.0f);
    aotx_model_desc desc = {};
    desc.layers = 1u; desc.hidden = qwidth; desc.heads = 14u;
    desc.kv_heads = 2u; desc.head_dim = 64u;
    desc.rope_theta = 1000000.0f; desc.rope_pairs = AOTX_ROPE_PAIRS_SPLIT;
    desc.rope_freqs = AOTX_MODEL_ABSENT;
    desc.kind[0] = AOTX_LAYER_KIND_ATTENTION_BIAS;
    desc.layer[0].offset[AOTX_BIAS_Q] = 8u * sizeof(float);
    desc.layer[0].offset[AOTX_BIAS_K] = (8u + qwidth) * sizeof(float);
    desc.layer[0].offset[AOTX_BIAS_V] = (8u + qwidth + kwidth) * sizeof(float);
    aotx_model_work work = {};
    work.q = (float *)aotx_bias_copy(q, qcount * sizeof *q);
    work.k = (float *)aotx_bias_copy(k, kcount * sizeof *k);
    work.v = (float *)aotx_bias_copy(v, kcount * sizeof *v);
    work.qh = (half *)aotx_bias_copy(qh, qcount * sizeof *qh);
    work.base = (unsigned int *)aotx_bias_copy(bases, seqs * sizeof *bases);
    work.weights = (unsigned long long)aotx_bias_copy(bias, sizeof bias);
    aotx_kvl_make(&work.shape, 1u, 2u, 64u);
    aotx_model_run run = {};
    run.offset = (unsigned int *)aotx_bias_copy(offsets, (seqs + 1u) * sizeof *offsets);
    run.agent = (unsigned int *)aotx_bias_copy(agents, seqs * sizeof *agents);
    run.seqs = seqs; run.tokens = tokens;
    void *pages = NULL;
    aotx_check_runtime(cudaMalloc(&pages, seqs * AOTX_KV_PAGE_BYTES), "cache allocation");
    aotx_check_runtime(cudaMemset(pages, 0, seqs * AOTX_KV_PAGE_BYTES), "cache clear");
    aotx_kv_table *table = (aotx_kv_table *)calloc(1u, sizeof *table);
    for (unsigned int s = 0u; s < seqs; ++s) {
        table->page[agents[s]][0] = (unsigned long long)pages + s * AOTX_KV_PAGE_BYTES;
        table->count[agents[s]] = 1u;
    }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_kv, table, sizeof *table), "cache table");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &desc, sizeof desc), "descriptor");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_space, &work, sizeof work), "buffers");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &run, sizeof run), "call");
    unsigned int faults = 0u;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_faults, &faults, sizeof faults), "fault count");
    unsigned int *device_agents = (unsigned int *)aotx_bias_copy(token_agent, tokens * sizeof *token_agent);
    unsigned int *device_positions = (unsigned int *)aotx_bias_copy(positions, tokens * sizeof *positions);
    half *device_k = (half *)aotx_bias_copy(kh, kcount * sizeof *kh);
    half *device_v = (half *)aotx_bias_copy(vh, kcount * sizeof *vh);
    aotx_model_qkv_bias<<<dim3(seqs == 1u ? capacity : 7u, 17u), AOTX_MODEL_ROW_THREADS>>>(0u, 0u);
    aotx_check_runtime(cudaGetLastError(), "bias launch");
    aotx_bias_read<<<tokens, 128>>>(device_agents, device_positions, tokens, device_k, device_v);
    aotx_check_runtime(cudaGetLastError(), "cache read");
    aotx_check_runtime(cudaMemcpy(qh, work.qh, qcount * sizeof *qh, cudaMemcpyDeviceToHost), "query result");
    aotx_check_runtime(cudaMemcpy(kh, device_k, kcount * sizeof *kh, cudaMemcpyDeviceToHost), "key result");
    aotx_check_runtime(cudaMemcpy(vh, device_v, kcount * sizeof *vh, cudaMemcpyDeviceToHost), "value result");
    aotx_bias_check(aotx_bias_turned(qh, q, bias + 8u, positions, tokens, qwidth),
                    "query bias before turn", seqs);
    aotx_bias_check(aotx_bias_turned(kh, k, bias + 8u + qwidth, positions, tokens, kwidth),
                    "key bias before turn", seqs);
    int value_ok = 1, guard_ok = 1;
    for (unsigned int t = 0u; t < tokens; ++t) {
        for (unsigned int d = 0u; d < kwidth; ++d) {
            float want = __half2float(__float2half(v[(size_t)t * kwidth + d]
                                                + bias[8u + qwidth + kwidth + d]));
            value_ok &= __half2float(vh[(size_t)t * kwidth + d]) == want;
        }
    }
    for (size_t i = (size_t)tokens * qwidth; i < qcount; ++i)
        guard_ok &= __half2float(qh[i]) == -123.0f;
    aotx_check_runtime(cudaMemcpyFromSymbol(&faults, aotx_model_faults, sizeof faults), "fault result");
    aotx_bias_check(value_ok, "value bias before half cache write", seqs);
    aotx_bias_check(guard_ok && faults == 0u, "bounded token and head grids", seqs);
    cudaFree(device_agents); cudaFree(device_positions); cudaFree(device_k); cudaFree(device_v);
    cudaFree(pages); cudaFree(work.q); cudaFree(work.k); cudaFree(work.v); cudaFree(work.qh);
    cudaFree(work.base); cudaFree((void *)work.weights); cudaFree((void *)run.offset);
    cudaFree((void *)run.agent);
    free(q); free(k); free(v); free(qh); free(kh); free(vh); free(table);
}

int main(void)
{
    aotx_bias_case(1u);
    aotx_bias_case(AOTX_SLOTS);
    printf("bias: %u checks, %u failures\n", aotx_bias_checks, aotx_bias_failed);
    return aotx_bias_failed == 0u ? 0 : 1;
}
