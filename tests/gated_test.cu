/* Purpose: Check gated query rotation, causal attention, and compact page writes against float64 arithmetic.
 * Owns: Synthetic projections, learned norms, and mapped page storage.
 * Launch shape: One sequence and AOTX_SLOTS sequences with distinct slots and unequal lengths.
 * Lifetime: One test process. */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "boot/check.h"
#include "model/conduct.cuh"
#include "model/hybrid.cuh"

static unsigned int aotx_test_bad;
static unsigned int aotx_test_cases;
static const unsigned int aotx_test_layer = 23u;
static const unsigned int aotx_test_role = AOTX_MODEL_LANGUAGE_Q4;

static void aotx_test_note(const char *name, int good, double error = 0.0)
{
    ++aotx_test_cases;
    aotx_test_bad += !good;
    printf("%-44s %s %.9g\n", name, good ? "ok" : "BAD", error);
}

static void *aotx_test_take(size_t bytes)
{
    void *ptr = 0;
    aotx_check_runtime(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
}

static void aotx_test_upload(void *dst, const void *src, size_t bytes)
{
    aotx_check_runtime(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
}

static void aotx_test_download(void *dst, const void *src, size_t bytes)
{
    aotx_check_runtime(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
}

static double aotx_test_half(double x)
{
    return (double)__half2float(__float2half((float)x));
}

/* The oracle uses a physical layer count of 24 and six interleaved paged layers. */
static size_t aotx_test_address(unsigned int position, unsigned int head,
                                unsigned int dim, unsigned int value)
{
    size_t bytes = 2u * 2u * AOTX_KVL_BLOCK * dim * sizeof(half);
    size_t capacity = (AOTX_KV_PAGE_BYTES - AOTX_KVL_HEADER) / bytes;
    size_t block = (position / AOTX_KVL_BLOCK) * 6u + 5u;
    size_t page = block / capacity;
    size_t at = (page * AOTX_KV_PAGE_BYTES + AOTX_KVL_HEADER
                 + (block % capacity) * bytes) / sizeof(half);
    return at + ((value * 2u + head) * AOTX_KVL_BLOCK
                 + position % AOTX_KVL_BLOCK) * dim;
}

static void aotx_test_norm(const float *src, const float *weight, unsigned int dim,
                            unsigned int rope, unsigned int position, double eps,
                            double theta, half *dst)
{
    double square = 0.0;
    for (unsigned int d = 0u; d < dim; ++d) square += (double)src[d] * src[d];
    double inv = 1.0 / sqrt(square / dim + eps);
    for (unsigned int d = 0u; d < dim; ++d)
        dst[d] = __float2half((float)(src[d] * inv * weight[d]));
    for (unsigned int i = 0u; i < rope / 2u; ++i) {
        unsigned int j = i + rope / 2u;
        double angle = position * pow(theta, -2.0 * i / rope);
        double a = src[i] * inv * weight[i];
        double b = src[j] * inv * weight[j];
        dst[i] = __float2half((float)(a * cos(angle) - b * sin(angle)));
        dst[j] = __float2half((float)(a * sin(angle) + b * cos(angle)));
    }
}

static double aotx_test_error(const half *got, const half *want, size_t count)
{
    double error = 0.0, scale = 1.0e-6;
    for (size_t i = 0; i < count; ++i) {
        double a = __half2float(got[i]), b = __half2float(want[i]);
        if (!isfinite(a)) return INFINITY;
        error = fmax(error, fabs(a - b));
        scale = fmax(scale, fabs(b));
    }
    return error / scale;
}

static void aotx_test_publish(const aotx_model_desc &desc, const aotx_model_work &work,
                               const aotx_model_run &run, const aotx_kv_table &table)
{
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &desc, sizeof desc,
                       aotx_test_role * sizeof desc), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_space, &work, sizeof work,
                       aotx_test_role * sizeof work), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &run, sizeof run,
                       aotx_test_role * sizeof run), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_kv, &table, sizeof table), "cudaMemcpyToSymbol");
    unsigned int zero = 0u;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_faults, &zero, sizeof zero),
                       "cudaMemcpyToSymbol");
    static float mass[AOTX_SLOTS][AOTX_KV_PAGES_EACH] = {};
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_page_mass, mass, sizeof mass),
                       "cudaMemcpyToSymbol");
}

static unsigned int aotx_test_faults(void)
{
    unsigned int faults;
    aotx_check_runtime(cudaMemcpyFromSymbol(&faults, aotx_model_faults, sizeof faults),
                       "cudaMemcpyFromSymbol");
    return faults;
}

static void aotx_test_batch(unsigned int seqs, unsigned int dim)
{
    printf("gated N=%u head=%u\n", seqs, dim);
    aotx_model_desc desc = {};
    desc.role = aotx_test_role;
    desc.layers = 24u;
    desc.heads = 8u;
    desc.kv_heads = 2u;
    desc.head_dim = dim;
    desc.rope_dim = 64u;
    desc.rope_theta = 10000.0f;
    desc.rms_eps = 1.0e-5f;
    desc.rope_freqs = AOTX_MODEL_ABSENT;
    desc.layer[aotx_test_layer].offset[AOTX_ATTENTION_Q_NORM] = 0u;
    desc.layer[aotx_test_layer].offset[AOTX_ATTENTION_K_NORM] = dim * sizeof(float);
    unsigned char states[24];
    for (unsigned int l = 0u; l < 24u; ++l)
        states[l] = l % 4u == 3u ? AOTX_STATE_KIND_KV_PAGES : AOTX_STATE_KIND_DELTA_STATE;
    aotx_model_work work = {};
    aotx_kvl_make_states(&work.shape, states, 24u, 2u, dim);
    unsigned int boundary = ((work.shape.blocks_page - 5u + 5u) / 6u) * AOTX_KVL_BLOCK;
    std::vector<unsigned int> offsets(seqs + 1u), agents(seqs), bases(seqs);
    for (unsigned int s = 0u; s < seqs; ++s) {
        agents[s] = (13u * s + 7u) % AOTX_SLOTS;
        bases[s] = boundary - 1u - s % 2u;
        offsets[s + 1u] = offsets[s] + 3u + s % 2u;
    }
    unsigned int tokens = offsets[seqs];
    unsigned int wide = desc.heads * dim, narrow = desc.kv_heads * dim;
    size_t page_cells = 2u * (size_t)AOTX_KV_PAGE_BYTES / sizeof(half);
    std::vector<float> qg((size_t)tokens * 2u * wide), k((size_t)tokens * narrow);
    std::vector<float> v(k.size()), weights(2u * dim);
    std::vector<half> queries((size_t)tokens * wide), attention(queries.size());
    std::vector<half> wanted_q(queries.size()), wanted_att(queries.size());
    std::vector<std::vector<half>> pages(seqs, std::vector<half>(page_cells, __float2half(-11.0f)));
    for (unsigned int d = 0u; d < dim; ++d) {
        weights[d] = 0.43f + 0.017f * (d % 37u);
        weights[dim + d] = 0.71f + 0.011f * ((d * 7u + 3u) % 41u);
    }
    for (unsigned int s = 0u; s < seqs; ++s) {
        for (unsigned int t = offsets[s]; t < offsets[s + 1u]; ++t) {
            for (unsigned int h = 0u; h < desc.heads; ++h) {
                for (unsigned int d = 0u; d < dim; ++d) {
                    size_t at = ((size_t)t * desc.heads + h) * 2u * dim + d;
                    qg[at] = (float)(0.17 + sin(0.071 * (d + 3u * h + 11u * t))
                                            + 0.003 * agents[s] + 0.02 * h);
                    qg[at + dim] = (float)(2.7 * cos(0.043 * (d + 19u * h + 5u * t))
                                                  - 0.4 + 0.007 * agents[s]);
                }
            }
            for (unsigned int h = 0u; h < desc.kv_heads; ++h) {
                for (unsigned int d = 0u; d < dim; ++d) {
                    size_t at = ((size_t)t * desc.kv_heads + h) * dim + d;
                    k[at] = (float)(cos(0.061 * (d + 23u * h + 7u * t))
                                           - 0.13 + 0.002 * agents[s]);
                    v[at] = (float)(0.2 * sin(0.037 * (2u * d + 13u * h + 17u * t))
                                           + 0.015 * h - 0.07 + 0.001 * agents[s]);
                }
            }
        }
        unsigned int end = bases[s] + offsets[s + 1u] - offsets[s] + 3u;
        for (unsigned int p = 0u; p < end; ++p) {
            for (unsigned int h = 0u; h < 2u; ++h) {
                size_t key = aotx_test_address(p, h, dim, 0u);
                size_t value = aotx_test_address(p, h, dim, 1u);
                for (unsigned int d = 0u; d < dim; ++d) {
                    pages[s][key + d] = __float2half((float)(0.3 * sin(0.023 * (d + 7u * p + 41u * h))
                                                                       + 0.001 * agents[s]));
                    pages[s][value + d] = __float2half((float)(0.15 * cos(0.041 * (d + 11u * p + 29u * h))
                                                                         + 0.0003 * agents[s]));
                }
            }
        }
    }
    static aotx_kv_table table;
    memset(&table, 0, sizeof table);
    void *storage = aotx_test_take((size_t)seqs * 2u * AOTX_KV_PAGE_BYTES);
    for (unsigned int s = 0u; s < seqs; ++s) {
        for (unsigned int p = 0u; p < 2u; ++p) {
            size_t physical = (2u * (seqs - 1u - s) + (1u - p)) * (size_t)AOTX_KV_PAGE_BYTES;
            table.page[agents[s]][p] = (unsigned long long)((char *)storage + physical);
            aotx_test_upload((void *)table.page[agents[s]][p],
                             pages[s].data() + p * AOTX_KV_PAGE_BYTES / sizeof(half),
                             AOTX_KV_PAGE_BYTES);
        }
        table.count[agents[s]] = 2u;
    }
    work.weights = (unsigned long long)aotx_test_take(weights.size() * sizeof(float));
    work.qgate = (float *)aotx_test_take(qg.size() * sizeof(float));
    work.k = (float *)aotx_test_take(k.size() * sizeof(float));
    work.v = (float *)aotx_test_take(v.size() * sizeof(float));
    work.qh = (half *)aotx_test_take(queries.size() * sizeof(half));
    work.att = (half *)aotx_test_take(attention.size() * sizeof(half));
    work.base = (unsigned int *)aotx_test_take(bases.size() * sizeof(unsigned int));
    unsigned int *offset = (unsigned int *)aotx_test_take(offsets.size() * sizeof(unsigned int));
    unsigned int *agent = (unsigned int *)aotx_test_take(agents.size() * sizeof(unsigned int));
    aotx_test_upload((void *)work.weights, weights.data(), weights.size() * sizeof(float));
    aotx_test_upload(work.qgate, qg.data(), qg.size() * sizeof(float));
    aotx_test_upload(work.k, k.data(), k.size() * sizeof(float));
    aotx_test_upload(work.v, v.data(), v.size() * sizeof(float));
    aotx_test_upload(work.base, bases.data(), bases.size() * sizeof(unsigned int));
    aotx_test_upload(offset, offsets.data(), offsets.size() * sizeof(unsigned int));
    aotx_test_upload(agent, agents.data(), agents.size() * sizeof(unsigned int));
    aotx_model_run run = {};
    run.offset = offset;
    run.agent = agent;
    run.tokens = tokens;
    run.seqs = seqs;
    run.telemetry = 1u;
    aotx_test_publish(desc, work, run, table);
    /* A short grid also checks the row stride of each kernel. */
    aotx_model_gated_qkv<<<dim3(2u, desc.heads + desc.kv_heads), 32u>>>(aotx_test_role, aotx_test_layer);
    aotx_model_gated_attend<<<dim3(2u, desc.heads), AOTX_MODEL_ATTN_THREADS>>>(aotx_test_role, aotx_test_layer);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_note("mapped pages have no faults", aotx_test_faults() == 0u);
    aotx_test_download(queries.data(), work.qh, queries.size() * sizeof(half));
    aotx_test_download(attention.data(), work.att, attention.size() * sizeof(half));

    for (unsigned int s = 0u; s < seqs; ++s) {
        for (unsigned int t = offsets[s]; t < offsets[s + 1u]; ++t) {
            unsigned int position = bases[s] + t - offsets[s];
            for (unsigned int h = 0u; h < desc.heads; ++h)
                aotx_test_norm(qg.data() + ((size_t)t * desc.heads + h) * 2u * dim,
                    weights.data(), dim, desc.rope_dim, position, desc.rms_eps,
                    desc.rope_theta, wanted_q.data() + ((size_t)t * desc.heads + h) * dim);
            for (unsigned int h = 0u; h < desc.kv_heads; ++h) {
                size_t at = ((size_t)t * desc.kv_heads + h) * dim;
                size_t key = aotx_test_address(position, h, dim, 0u);
                size_t value = aotx_test_address(position, h, dim, 1u);
                aotx_test_norm(k.data() + at, weights.data() + dim, dim, desc.rope_dim,
                    position, desc.rms_eps, desc.rope_theta, pages[s].data() + key);
                for (unsigned int d = 0u; d < dim; ++d)
                    pages[s][value + d] = __float2half(v[at + d]);
            }
        }
    }
    double query_error = aotx_test_error(queries.data(), wanted_q.data(), queries.size());
    aotx_test_note("full head norm and partial split rotation", query_error < 1.5e-3, query_error);
    double prefix_error = 0.0, suffix_error = 0.0;
    for (size_t row = 0u; row < (size_t)tokens * desc.heads; ++row) {
        prefix_error = fmax(prefix_error, aotx_test_error(queries.data() + row * dim,
                              wanted_q.data() + row * dim, desc.rope_dim));
        if (dim > desc.rope_dim)
            suffix_error = fmax(suffix_error, aotx_test_error(queries.data() + row * dim + desc.rope_dim,
                wanted_q.data() + row * dim + desc.rope_dim, dim - desc.rope_dim));
    }
    aotx_test_note("rotary prefix uses its own pair boundary", prefix_error < 1.5e-3, prefix_error);
    aotx_test_note("normalized suffix does not rotate", suffix_error < 1.5e-3, suffix_error);

    double cache_error = 0.0;
    unsigned int changed = 0u;
    std::vector<half> copied(page_cells);
    for (unsigned int s = 0u; s < seqs; ++s) {
        for (unsigned int p = 0u; p < 2u; ++p)
            aotx_test_download(copied.data() + p * AOTX_KV_PAGE_BYTES / sizeof(half),
                              (void *)table.page[agents[s]][p], AOTX_KV_PAGE_BYTES);
        for (size_t i = 0u; i < page_cells; ++i) {
            double got = __half2float(copied[i]), want = __half2float(pages[s][i]);
            if (!isfinite(got)) cache_error = INFINITY;
            cache_error = fmax(cache_error, fabs(got - want));
            if (want == -11.0 && got != want) ++changed;
        }
    }
    aotx_test_note("compact cache keys and raw values", cache_error < 0.004, cache_error);
    aotx_test_note("other layers and page headers stay intact", changed == 0u, changed);
    static double wanted_mass[AOTX_SLOTS][AOTX_KV_PAGES_EACH];
    memset(wanted_mass, 0, sizeof wanted_mass);
    for (unsigned int s = 0u; s < seqs; ++s) {
        for (unsigned int t = offsets[s]; t < offsets[s + 1u]; ++t) {
            unsigned int position = bases[s] + t - offsets[s];
            std::vector<double> scores(position + 1u);
            for (unsigned int h = 0u; h < desc.heads; ++h) {
                unsigned int kh = h / (desc.heads / desc.kv_heads);
                size_t row = ((size_t)t * desc.heads + h) * dim;
                double top = -INFINITY, mass = 0.0;
                for (unsigned int p = 0u; p <= position; ++p) {
                    size_t key = aotx_test_address(p, kh, dim, 0u);
                    double dot = 0.0;
                    for (unsigned int d = 0u; d < dim; ++d)
                        dot += (double)__half2float(wanted_q[row + d]) * __half2float(pages[s][key + d]);
                    scores[p] = dot / sqrt((double)dim);
                    top = fmax(top, scores[p]);
                }
                for (unsigned int p = 0u; p <= position; ++p) {
                    scores[p] = exp(scores[p] - top);
                    mass += scores[p];
                }
                for (unsigned int p = 0u; p <= position; ++p) {
                    scores[p] /= mass;
                    size_t page = aotx_test_address(p, kh, dim, 0u) * sizeof(half) / AOTX_KV_PAGE_BYTES;
                    wanted_mass[agents[s]][page] += scores[p];
                }
                for (unsigned int d = 0u; d < dim; ++d) {
                    double sum = 0.0;
                    for (unsigned int p = 0u; p <= position; ++p)
                        sum += scores[p] * __half2float(pages[s][aotx_test_address(p, kh, dim, 1u) + d]);
                    double gate = qg[row * 2u + dim + d];
                    wanted_att[row + d] = __float2half((float)(sum / (1.0 + exp(-gate))));
                }
            }
        }
    }
    double error = aotx_test_error(attention.data(), wanted_att.data(), attention.size());
    aotx_test_note("causal float64 attention and per-head gate", error < 2.5e-3, error);
    static float mass[AOTX_SLOTS][AOTX_KV_PAGES_EACH];
    aotx_check_runtime(cudaMemcpyFromSymbol(mass, aotx_page_mass, sizeof mass), "cudaMemcpyFromSymbol");
    double mass_error = 0.0;
    for (unsigned int s = 0u; s < AOTX_SLOTS; ++s)
        for (unsigned int p = 0u; p < AOTX_KV_PAGES_EACH; ++p)
            mass_error = fmax(mass_error, fabs(mass[s][p] - wanted_mass[s][p]));
    aotx_test_note("page mass follows compact pages and slots", mass_error < 0.002, mass_error);
    std::vector<float> preserved(qg.size());
    aotx_test_download(preserved.data(), work.qgate, preserved.size() * sizeof(float));
    aotx_test_note("query and gate projections stay unchanged",
                   memcmp(preserved.data(), qg.data(), qg.size() * sizeof(float)) == 0);

    /* A missing history page must not return a partial softmax result. */
    unsigned long long saved = table.page[agents[0]][0];
    table.page[agents[0]][0] = 0ull;
    aotx_test_publish(desc, work, run, table);
    std::vector<half> sentinel(attention.size(), __float2half(7.0f));
    aotx_test_upload(work.att, sentinel.data(), sentinel.size() * sizeof(half));
    aotx_model_gated_attend<<<dim3(2u, desc.heads), AOTX_MODEL_ATTN_THREADS>>>(aotx_test_role, aotx_test_layer);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int faults = aotx_test_faults();
    aotx_test_download(attention.data(), work.att, attention.size() * sizeof(half));
    aotx_test_note("missing history counts each query head", faults == offsets[1] * desc.heads, faults);
    aotx_test_note("missing history does not emit partial output",
        memcmp(attention.data(), sentinel.data(), offsets[1] * wide * sizeof(half)) == 0);
    table.page[agents[0]][0] = saved;

    saved = table.page[agents[0]][1];
    table.page[agents[0]][1] = 0ull;
    aotx_test_publish(desc, work, run, table);
    aotx_model_gated_qkv<<<dim3(2u, desc.heads + desc.kv_heads), 32u>>>(aotx_test_role, aotx_test_layer);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int missing_rows = 0u;
    for (unsigned int t = 0u; t < offsets[1]; ++t)
        missing_rows += bases[0] + t >= boundary;
    faults = aotx_test_faults();
    aotx_test_note("missing write page counts each key head", faults == missing_rows * desc.kv_heads, faults);
    table.page[agents[0]][1] = saved;

    if (seqs == 1u && dim == 256u) {
        /* An exact midpoint exposes a half conversion before the sigmoid gate. */
        run.tokens = 1u;
        run.telemetry = 0u;
        bases[0] = 1u;
        offsets[1] = 1u;
        aotx_test_upload(work.base, bases.data(), sizeof(unsigned int));
        aotx_test_upload(offset, offsets.data(), 2u * sizeof(unsigned int));
        for (size_t i = 0u; i < queries.size(); ++i) queries[i] = __float2half(0.0f);
        for (size_t i = 0u; i < qg.size(); ++i) qg[i] = -0.3f;
        for (unsigned int p = 0u; p < 2u; ++p) {
            for (unsigned int h = 0u; h < desc.kv_heads; ++h) {
                size_t key = aotx_test_address(p, h, dim, 0u);
                size_t value = aotx_test_address(p, h, dim, 1u);
                for (unsigned int d = 0u; d < dim; ++d) {
                    pages[0][key + d] = __float2half(0.0f);
                    pages[0][value + d] = __float2half(p == 0u ? 1.0f : 1.0009765625f);
                }
            }
        }
        aotx_test_upload((void *)table.page[agents[0]][0], pages[0].data(), AOTX_KV_PAGE_BYTES);
        aotx_test_upload(work.qh, queries.data(), queries.size() * sizeof(half));
        aotx_test_upload(work.qgate, qg.data(), qg.size() * sizeof(float));
        aotx_test_publish(desc, work, run, table);
        aotx_model_gated_attend<<<dim3(1u, desc.heads), AOTX_MODEL_ATTN_THREADS>>>(aotx_test_role, aotx_test_layer);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_test_download(attention.data(), work.att, wide * sizeof(half));
        double gate = 1.0 / (1.0 + exp(-(double)qg[dim]));
        double want = aotx_test_half(1.00048828125 * gate);
        double early = aotx_test_half(aotx_test_half(1.00048828125) * gate);
        unsigned int bad = 0u;
        for (unsigned int d = 0u; d < wide; ++d) bad += __half2float(attention[d]) != want;
        aotx_test_note("gate precedes the half output conversion", bad == 0u && want != early, bad);
        aotx_check_runtime(cudaMemcpyFromSymbol(mass, aotx_page_mass, sizeof mass), "cudaMemcpyFromSymbol");
        double sum = 0.0;
        for (unsigned int s = 0u; s < AOTX_SLOTS; ++s)
            for (unsigned int p = 0u; p < AOTX_KV_PAGES_EACH; ++p) sum += fabs(mass[s][p]);
        aotx_test_note("disabled telemetry leaves page mass unchanged", sum == 0.0, sum);
    }
    cudaFree(agent);
    cudaFree(offset);
    cudaFree(work.base);
    cudaFree(work.att);
    cudaFree(work.qh);
    cudaFree(work.v);
    cudaFree(work.k);
    cudaFree(work.qgate);
    cudaFree((void *)work.weights);
    cudaFree(storage);
}

int main(void)
{
    aotx_check_runtime(cudaSetDevice(0), "cudaSetDevice");
    aotx_test_batch(1u, 256u);
    aotx_test_batch(AOTX_SLOTS, 256u);
    aotx_test_batch(1u, 128u);
    aotx_test_batch(1u, 64u);
    printf("gated: %u cases, %u failed\n", aotx_test_cases, aotx_test_bad);
    return aotx_test_bad ? 1 : 0;
}
