/* Purpose: Check expert routing, selected weight slices and ordered weighted sums.
 * Owns: Synthetic weights and forward buffers for batches of 1 and 64 tokens.
 * Launch shape: The expert kernels with both short and oversized token grids.
 * Lifetime: One test run. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "model/experts.cuh"
#include "matrix_kref.h"

#define AOTX_EXPERT_TEST_HIDDEN 32u
#define AOTX_EXPERT_TEST_GUARD (-12345.0f)

static unsigned int aotx_expert_checks;
static unsigned int aotx_expert_failed;

static void aotx_expert_check(int ok, const char *name, unsigned int tokens,
                               unsigned int type, unsigned int used)
{
    ++aotx_expert_checks;
    if (!ok) {
        ++aotx_expert_failed;
        printf("experts: FAIL %s tokens=%u type=%u used=%u\n", name, tokens, type, used);
    }
}

static void *aotx_expert_copy(const void *host, size_t bytes)
{
    void *device = NULL;
    aotx_check_runtime(cudaMalloc(&device, bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
    return device;
}

static void aotx_expert_install(const aotx_model_desc *desc, const aotx_model_work *work,
                                 unsigned int tokens)
{
    aotx_model_run run = {};
    run.tokens = tokens;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, desc, sizeof *desc), "descriptor");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_space, work, sizeof *work), "buffers");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &run, sizeof run), "call");
}

static void aotx_expert_route_case(unsigned int tokens, unsigned int experts,
                                   unsigned int used, int precision)
{
    const unsigned int hidden = AOTX_EXPERT_TEST_HIDDEN;
    const unsigned int capacity = tokens + 2u;
    size_t values = (size_t)capacity * hidden;
    size_t probabilities = (size_t)capacity * experts;
    size_t selections = (size_t)capacity * used;
    half *x = (half *)calloc(values, sizeof *x);
    float *router = (float *)calloc((size_t)experts * hidden + 8u, sizeof *router);
    float *prob = (float *)malloc(probabilities * sizeof *prob);
    unsigned int *ids = (unsigned int *)malloc(selections * sizeof *ids);
    float *proj = (float *)malloc(values * sizeof *proj);
    float *sum = (float *)malloc(values * sizeof *sum);
    float *want_sum = (float *)calloc(values, sizeof *want_sum);
    for (size_t i = 0; i < probabilities; ++i) prob[i] = AOTX_EXPERT_TEST_GUARD;
    for (size_t i = 0; i < selections; ++i) ids[i] = 0xFFFFFFFFu;
    for (size_t i = 0; i < values; ++i) proj[i] = sum[i] = AOTX_EXPERT_TEST_GUARD;
    for (unsigned int e = 0; e < experts; ++e) {
        float *row = router + 8u + (size_t)e * hidden;
        float position = (float)e / (float)(experts - 1u);
        for (unsigned int d = 0; d < hidden; ++d) {
            row[d] = precision ? (e == 2u ? 1.0002f : 1.0f)
                               : (float)((int)(d % 5u) - 2) * 0.0012345f;
        }
        if (!precision) {
            row[0] = 3.0f * position;
            row[1] = -3.0f * position * position;
        }
    }
    /* Equal rows test lower-index ordering without an all-equal router. */
    memcpy(router + 8u + hidden, router + 8u, hidden * sizeof *router);
    for (unsigned int t = 0; t < tokens; ++t) {
        for (unsigned int d = 0; d < hidden; ++d) {
            x[(size_t)t * hidden + d] = __float2half(precision ? 1.0f
                : (float)((int)((t * 7u + d * 3u) % 13u) - 6) / 16.0f);
        }
        if (!precision) {
            x[(size_t)t * hidden] = __float2half(2.0f * (float)t / (float)tokens + 0.007f);
            x[(size_t)t * hidden + 1u] = __float2half(1.0f);
        }
    }
    aotx_model_desc desc = {};
    desc.hidden = hidden;
    desc.expert_count = experts;
    desc.expert_used_count = used;
    desc.layer[0].ffn_router = 8u * sizeof(float);
    aotx_model_work work = {};
    work.max_tokens = capacity;
    work.x = (half *)aotx_expert_copy(x, values * sizeof *x);
    work.weights = (unsigned long long)aotx_expert_copy(router,
        ((size_t)experts * hidden + 8u) * sizeof *router);
    work.expert_prob = (float *)aotx_expert_copy(prob, probabilities * sizeof *prob);
    work.expert_id = (unsigned int *)aotx_expert_copy(ids, selections * sizeof *ids);
    work.proj = (float *)aotx_expert_copy(proj, values * sizeof *proj);
    work.expert_sum = (float *)aotx_expert_copy(sum, values * sizeof *sum);
    aotx_expert_install(&desc, &work, tokens);
    unsigned int grid = tokens == 1u ? capacity : 7u;
    aotx_model_expert_route<<<grid, AOTX_EXPERT_ROUTE_THREADS>>>(0u, 0u);
    aotx_check_runtime(cudaGetLastError(), "route launch");
    aotx_check_runtime(cudaMemcpy(prob, work.expert_prob, probabilities * sizeof *prob,
                                  cudaMemcpyDeviceToHost), "probabilities");
    aotx_check_runtime(cudaMemcpy(ids, work.expert_id, selections * sizeof *ids,
                                  cudaMemcpyDeviceToHost), "selections");
    int route_ok = 1, mass_ok = 1, guard_ok = 1, different = 0;
    for (unsigned int t = 0; t < tokens; ++t) {
        double logits[AOTX_LAYER_EXPERTS_MAX];
        double expected[AOTX_LAYER_EXPERTS_MAX];
        unsigned int order[AOTX_LAYER_EXPERTS_MAX];
        double maximum = -INFINITY, total = 0.0;
        for (unsigned int e = 0; e < experts; ++e) {
            double dot = 0.0;
            for (unsigned int d = 0; d < hidden; ++d) {
                dot += (double)router[8u + (size_t)e * hidden + d]
                     * (double)__half2float(x[(size_t)t * hidden + d]);
            }
            logits[e] = dot;
            if (dot > maximum) maximum = dot;
            order[e] = e;
        }
        for (unsigned int e = 0; e < experts; ++e) {
            expected[e] = exp(logits[e] - maximum);
            total += expected[e];
        }
        for (unsigned int e = 0; e < experts; ++e) {
            expected[e] /= total;
            route_ok &= isfinite(prob[(size_t)t * experts + e])
                && fabs(prob[(size_t)t * experts + e] - expected[e]) < 2e-6;
        }
        for (unsigned int r = 0; r < experts; ++r) {
            unsigned int best = r;
            for (unsigned int j = r + 1u; j < experts; ++j) {
                if (expected[order[j]] > expected[order[best]]
                    || (expected[order[j]] == expected[order[best]] && order[j] < order[best])) best = j;
            }
            unsigned int saved = order[r]; order[r] = order[best]; order[best] = saved;
        }
        double mass = 0.0;
        for (unsigned int r = 0; r < used; ++r) {
            route_ok &= ids[(size_t)t * used + r] == order[r];
            mass += prob[(size_t)t * experts + order[r]];
        }
        mass_ok &= used == experts ? fabs(mass - 1.0) < 2e-5 : mass < 0.999;
        different |= ids[(size_t)t * used] != ids[0];
    }
    for (size_t i = (size_t)tokens * experts; i < probabilities; ++i)
        guard_ok &= prob[i] == AOTX_EXPERT_TEST_GUARD;
    for (size_t i = (size_t)tokens * used; i < selections; ++i)
        guard_ok &= ids[i] == 0xFFFFFFFFu;
    aotx_expert_check(route_ok, precision ? "F32 router" : "route order", tokens, experts, used);
    aotx_expert_check(mass_ok, "full softmax mass", tokens, experts, used);
    aotx_expert_check(guard_ok, "route token boundary", tokens, experts, used);
    if (tokens > 1u && !precision)
        aotx_expert_check(different, "per-token winners", tokens, experts, used);

    int add_ok = 1;
    for (unsigned int r = 0; r < used; ++r) {
        for (unsigned int t = 0; t < tokens; ++t) {
            unsigned int selected = ids[(size_t)t * used + r];
            /* A failed selection remains a test failure and cannot index host memory. */
            if (selected >= experts) selected = 0u;
            float weight = prob[(size_t)t * experts + selected];
            for (unsigned int d = 0; d < hidden; ++d) {
                size_t at = (size_t)t * hidden + d;
                proj[at] = (float)((int)((t * 11u + d * 7u + r * 3u) % 41u) - 20)
                         * (float)(r + 1u) * 0.03125f;
                volatile float product = proj[at] * weight;
                volatile float accumulated = r == 0u ? product : want_sum[at] + product;
                want_sum[at] = accumulated;
            }
        }
        aotx_check_runtime(cudaMemcpy(work.proj, proj, values * sizeof *proj,
                                      cudaMemcpyHostToDevice), "projection");
        aotx_model_expert_add<<<grid, AOTX_EXPERT_ADD_THREADS>>>(0u, r);
        aotx_check_runtime(cudaGetLastError(), "add launch");
        aotx_check_runtime(cudaMemcpy(sum, work.expert_sum, values * sizeof *sum,
                                      cudaMemcpyDeviceToHost), "weighted sum");
        for (size_t at = 0; at < (size_t)tokens * hidden; ++at)
            add_ok &= sum[at] == want_sum[at];
    }
    aotx_check_runtime(cudaMemcpy(proj, work.proj, values * sizeof *proj,
                                  cudaMemcpyDeviceToHost), "final projection");
    for (size_t at = 0; at < values; ++at) {
        add_ok &= at < (size_t)tokens * hidden ? proj[at] == want_sum[at]
            : proj[at] == AOTX_EXPERT_TEST_GUARD && sum[at] == AOTX_EXPERT_TEST_GUARD;
    }
    aotx_expert_check(add_ok, "ordered weighted sum", tokens, experts, used);
    cudaFree(work.x); cudaFree((void *)work.weights); cudaFree(work.expert_prob);
    cudaFree(work.expert_id); cudaFree(work.proj); cudaFree(work.expert_sum);
    free(x); free(router); free(prob); free(ids); free(proj); free(sum); free(want_sum);
}

static void aotx_expert_half(unsigned char *at, float value)
{
    half word = __float2half(value);
    memcpy(at, &word, sizeof word);
}

/* Packed bytes vary by expert, row, block and column. Scales are finite binary fractions. */
static void aotx_expert_weights(unsigned char *weights, unsigned int type,
                                 unsigned int rows, unsigned int k, size_t stride)
{
    for (unsigned int r = 0; r < rows; ++r) {
        unsigned char *row = weights + (size_t)r * stride;
        if (type == AOTX_WEIGHT_F32 || type == AOTX_WEIGHT_F16) {
            for (unsigned int d = 0; d < k; ++d) {
                float value = (float)((int)((r * 17u + d * 13u + r * d) % 101u) - 50) * 0.013579f;
                if (type == AOTX_WEIGHT_F32) memcpy(row + (size_t)d * 4u, &value, 4u);
                else aotx_expert_half(row + (size_t)d * 2u, value);
            }
            continue;
        }
        unsigned int block_width = type == AOTX_WEIGHT_Q8_0 || type == AOTX_WEIGHT_Q4_0 ? 32u : 256u;
        size_t block_bytes = stride / (k / block_width);
        for (unsigned int b = 0; b < k / block_width; ++b) {
            unsigned char *one = row + b * block_bytes;
            for (size_t j = 0; j < block_bytes; ++j)
                one[j] = (unsigned char)(r * 37u + b * 73u + j * 19u + r * j * 3u);
            float scale = (float)(1u + (r + b * 3u) % 7u) / 128.0f;
            if (type == AOTX_WEIGHT_Q6_K) {
                for (unsigned int j = 0; j < 16u; ++j)
                    one[192u + j] = (unsigned char)(signed char)((int)((r + b + j * 3u) % 15u) - 7);
                aotx_expert_half(one + 208u, scale);
            } else {
                aotx_expert_half(one, scale);
                if (block_width == 256u) aotx_expert_half(one + 2u, scale * 0.5f);
            }
        }
    }
}

static double aotx_expert_weight(const unsigned char *row, unsigned int type, unsigned int d)
{
    if (type == AOTX_WEIGHT_Q4_K || type == AOTX_WEIGHT_Q5_K || type == AOTX_WEIGHT_Q6_K)
        return aotx_kref_weight(row, type, d);
    if (type == AOTX_WEIGHT_F16) return aotx_kref_half(row + (size_t)d * 2u);
    if (type == AOTX_WEIGHT_F32) {
        float value; memcpy(&value, row + (size_t)d * 4u, 4u); return value;
    }
    const unsigned char *one = row + (size_t)(d / 32u) * (type == AOTX_WEIGHT_Q8_0 ? 34u : 18u);
    int code;
    if (type == AOTX_WEIGHT_Q8_0) code = (signed char)one[2u + d % 32u];
    else {
        unsigned int pair = one[2u + d % 16u];
        code = (int)(d % 32u < 16u ? pair & 15u : pair >> 4) - 8;
    }
    return aotx_kref_half(one) * (double)code;
}

static void aotx_expert_matrix_case(unsigned int tokens, unsigned int type)
{
    const unsigned int experts = 5u, used = 3u, n = 19u, k = 512u, capacity = tokens + 2u;
    size_t stride = type == AOTX_WEIGHT_F32 ? k * 4u : type == AOTX_WEIGHT_F16 ? k * 2u
        : type == AOTX_WEIGHT_Q8_0 ? (k / 32u) * 34u : type == AOTX_WEIGHT_Q4_0 ? (k / 32u) * 18u
        : aotx_kref_row_bytes(type, k);
    size_t bytes = (size_t)experts * n * stride;
    unsigned char *weights = (unsigned char *)malloc(bytes);
    half *x = (half *)malloc((size_t)capacity * k * sizeof *x);
    float *y = (float *)malloc((size_t)capacity * n * sizeof *y);
    unsigned int *ids = (unsigned int *)malloc((size_t)capacity * used * sizeof *ids);
    aotx_expert_weights(weights, type, experts * n, k, stride);
    for (unsigned int t = 0; t < capacity; ++t) {
        for (unsigned int d = 0; d < k; ++d)
            x[(size_t)t * k + d] = __float2half((float)((int)((t * 11u + d * 7u + t * d) % 29u) - 14) / 32.0f);
        for (unsigned int r = 0; r < used; ++r) ids[(size_t)t * used + r] = (t * 3u + r * 2u + 1u) % experts;
    }
    aotx_model_desc desc = {};
    desc.expert_count = experts; desc.expert_used_count = used;
    aotx_model_work work = {};
    work.expert_id = (unsigned int *)aotx_expert_copy(ids, (size_t)capacity * used * sizeof *ids);
    aotx_expert_install(&desc, &work, tokens);
    void *device_weights = aotx_expert_copy(weights, bytes);
    half *device_x = (half *)aotx_expert_copy(x, (size_t)capacity * k * sizeof *x);
    for (size_t i = 0; i < (size_t)capacity * n; ++i) y[i] = AOTX_EXPERT_TEST_GUARD;
    float *device_y = (float *)aotx_expert_copy(y, (size_t)capacity * n * sizeof *y);
    int ok = 1;
    for (unsigned int rank = 0; rank < used; ++rank) {
        dim3 grid((n + AOTX_EXPERT_MATRIX_ROWS - 1u) / AOTX_EXPERT_MATRIX_ROWS,
                  tokens == 1u ? capacity : 7u);
        aotx_model_expert_matrix<<<grid, AOTX_EXPERT_MATRIX_THREADS>>>(0u, rank, device_weights,
            type, n, k, device_x, device_y);
        aotx_check_runtime(cudaGetLastError(), "matrix launch");
        aotx_check_runtime(cudaMemcpy(y, device_y, (size_t)capacity * n * sizeof *y,
                                      cudaMemcpyDeviceToHost), "matrix result");
        for (unsigned int t = 0; t < tokens; ++t) {
            for (unsigned int row = 0; row < n; ++row) {
                const unsigned char *w = weights + ((size_t)ids[(size_t)t * used + rank] * n + row) * stride;
                double want = 0.0, magnitude = 0.0;
                for (unsigned int d = 0; d < k; ++d) {
                    double product = aotx_expert_weight(w, type, d) * __half2float(x[(size_t)t * k + d]);
                    want += product; magnitude += fabs(product);
                }
                float got = y[(size_t)t * n + row];
                ok &= isfinite(got) && fabs((double)got - want) <= 2e-6 * magnitude + 2e-5;
            }
        }
        for (size_t i = (size_t)tokens * n; i < (size_t)capacity * n; ++i)
            ok &= y[i] == AOTX_EXPERT_TEST_GUARD;
    }
    aotx_expert_check(ok, "selected weight slices", tokens, type, used);
    cudaFree(device_weights); cudaFree(device_x); cudaFree(device_y); cudaFree(work.expert_id);
    free(weights); free(x); free(y); free(ids);
}

static void aotx_expert_norm_case(unsigned int tokens)
{
    const unsigned int qwidth = 384u, kwidth = 192u, capacity = tokens + 2u;
    size_t count = (size_t)capacity * (qwidth + kwidth);
    float *data = (float *)malloc(count * sizeof *data);
    float *got = (float *)malloc(count * sizeof *got);
    float *weight = (float *)malloc((qwidth + kwidth) * sizeof *weight);
    for (unsigned int d = 0; d < qwidth + kwidth; ++d)
        weight[d] = 0.4f + (float)((d * 13u) % 97u) / 64.0f;
    for (unsigned int kind = 0; kind < 2u; ++kind) {
        unsigned int width = kind == 0u ? qwidth : kwidth;
        size_t first = kind == 0u ? 0u : (size_t)capacity * qwidth;
        for (unsigned int t = 0; t < capacity; ++t) {
            for (unsigned int d = 0; d < width; ++d)
                data[first + (size_t)t * width + d] = t >= tokens ? AOTX_EXPERT_TEST_GUARD
                    : (float)((int)((t * 7u + d * 3u) % 19u) - 9) * (float)(1u + d / 32u) / 16.0f;
        }
    }
    aotx_model_desc desc = {};
    desc.heads = 12u; desc.kv_heads = 6u; desc.head_dim = 32u; desc.rms_eps = 1e-5f;
    desc.layer[0].attn_q_norm = 0u;
    desc.layer[0].attn_k_norm = qwidth * sizeof(float);
    aotx_model_work work = {};
    work.weights = (unsigned long long)aotx_expert_copy(weight, (qwidth + kwidth) * sizeof *weight);
    work.q = (float *)aotx_expert_copy(data, count * sizeof *data);
    work.k = work.q + (size_t)capacity * qwidth;
    aotx_expert_install(&desc, &work, tokens);
    aotx_model_qk_norm<<<dim3(capacity, 2u), AOTX_MODEL_ROW_THREADS>>>(0u, 0u);
    aotx_check_runtime(cudaGetLastError(), "QK norm launch");
    aotx_check_runtime(cudaMemcpy(got, work.q, count * sizeof *got, cudaMemcpyDeviceToHost), "QK norm");
    int ok = 1;
    for (unsigned int kind = 0; kind < 2u; ++kind) {
        unsigned int width = kind == 0u ? qwidth : kwidth;
        size_t first = kind == 0u ? 0u : (size_t)capacity * qwidth;
        const float *w = weight + (kind == 0u ? 0u : qwidth);
        for (unsigned int t = 0; t < capacity; ++t) {
            const float *row = data + first + (size_t)t * width;
            double total = 0.0;
            for (unsigned int d = 0; d < width; ++d) total += (double)row[d] * row[d];
            double scale = 1.0 / sqrt(total / width + desc.rms_eps);
            for (unsigned int d = 0; d < width; ++d) {
                double want = t >= tokens ? AOTX_EXPERT_TEST_GUARD : scale * row[d] * w[d];
                float value = got[first + (size_t)t * width + d];
                ok &= isfinite(value) && fabs(value - want) < 2e-6 * fabs(want) + 1e-6;
            }
        }
    }
    aotx_expert_check(ok, "full-width QK norm", tokens, 0u, 0u);
    cudaFree(work.q); cudaFree((void *)work.weights);
    free(data); free(got); free(weight);
}

int main(void)
{
    const unsigned int types[] = { AOTX_WEIGHT_F32, AOTX_WEIGHT_F16, AOTX_WEIGHT_Q8_0,
        AOTX_WEIGHT_Q4_0, AOTX_WEIGHT_Q4_K, AOTX_WEIGHT_Q5_K, AOTX_WEIGHT_Q6_K };
    for (unsigned int tokens = 1u; tokens <= 64u; tokens *= 64u) {
        aotx_expert_norm_case(tokens);
        aotx_expert_route_case(tokens, 7u, 1u, 0);
        aotx_expert_route_case(tokens, 7u, 3u, 0);
        aotx_expert_route_case(tokens, 7u, 7u, 0);
        aotx_expert_route_case(tokens, 64u, 8u, 0);
        aotx_expert_route_case(tokens, AOTX_LAYER_EXPERTS_MAX, AOTX_LAYER_EXPERTS_MAX, 0);
        aotx_expert_route_case(tokens, 3u, 1u, 1);
        for (unsigned int t = 0; t < sizeof types / sizeof types[0]; ++t)
            aotx_expert_matrix_case(tokens, types[t]);
    }
    printf("experts: %u checks, %u failures\n", aotx_expert_checks, aotx_expert_failed);
    return aotx_expert_failed == 0u ? 0 : 1;
}
