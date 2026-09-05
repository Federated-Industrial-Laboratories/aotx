/* Purpose: Check delta state, convolution, gates, slot routing, and chunk equality.
 * Owns: Synthetic weights, distinct slot state, and an independent float64 reference.
 * Launch shape: Batches of 1 and 64 sequences with unequal token counts and permuted slots.
 * Lifetime: One test run. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#include "boot/check.h"
#include "model/forward.cuh"
#include "model/delta.cuh"
#include "model/kinds_data.h"

static const unsigned int aotx_delta_test_layer = 5u;
static const unsigned int aotx_delta_test_steps = 7u;
static const float aotx_delta_test_guard = -12345.0f;
static unsigned int aotx_delta_checks;
static unsigned int aotx_delta_failed;

static void aotx_delta_check(bool ok, const char *name, unsigned int n, unsigned int dim)
{
    ++aotx_delta_checks;
    if (!ok) {
        ++aotx_delta_failed;
        printf("delta: FAIL %s sequences=%u dim=%u\n", name, n, dim);
    }
}

static float aotx_delta_data(size_t index, unsigned int seed)
{
    unsigned int x = (unsigned int)index * 1664525u + seed * 1013904223u;
    x ^= x >> 13u;
    x *= 2246822519u;
    return (float)((int)(x % 2001u) - 1000) / 2048.0f;
}

static double aotx_delta_logistic(double value)
{
    if (value >= 0.0) return 1.0 / (1.0 + exp(-value));
    double e = exp(value);
    return e / (1.0 + e);
}

static bool aotx_delta_close(const std::vector<float> &got, const std::vector<double> &want,
                              double absolute, double relative)
{
    if (got.size() != want.size()) return false;
    for (size_t i = 0u; i < got.size(); ++i) {
        if (!isfinite(got[i]) || fabs((double)got[i] - want[i]) > absolute + relative * fabs(want[i])) {
            printf("delta: mismatch index=%zu got=%.9g want=%.17g\n", i, got[i], want[i]);
            return false;
        }
    }
    return true;
}

struct aotx_delta_fixture {
    aotx_model_desc desc = {};
    aotx_model_work work = {};
    unsigned int n, dim, heads, keys, width, channels, inner, capacity;
    bool reset;
    std::vector<float> weights, initial_state, initial_history, input, z, alpha, beta;
    std::vector<unsigned int> agent, length;
    std::vector<void *> allocations;

    void *copy(const void *source, size_t bytes)
    {
        void *device = NULL;
        aotx_check_runtime(cudaMalloc(&device, bytes), "delta allocate");
        if (source) aotx_check_runtime(cudaMemcpy(device, source, bytes, cudaMemcpyHostToDevice), "delta upload");
        allocations.push_back(device);
        return device;
    }

    aotx_delta_fixture(unsigned int batch, unsigned int dimension, unsigned int convolution,
                         bool clear) : n(batch), dim(dimension), heads(4u), keys(2u),
                         width(convolution), channels((2u * keys + heads) * dim),
                         inner(heads * dim), capacity(n * aotx_delta_test_steps + 2u), reset(clear)
    {
        desc.layers = 7u;
        desc.delta_dim = dim;
        desc.delta_heads = heads;
        desc.delta_key_heads = keys;
        desc.delta_inner = inner;
        desc.delta_conv = width;
        desc.rms_eps = 1e-5f;
        work.max_tokens = capacity;
        work.delta.layers = 3u;
        memset(work.delta.layer, 0xff, sizeof work.delta.layer);
        work.delta.layer[0] = 0u;
        work.delta.layer[aotx_delta_test_layer] = 1u;
        work.delta.layer[6] = 2u;
        work.delta.state_elements = (size_t)AOTX_SLOTS * 3u * heads * dim * dim;
        work.delta.history_elements = (size_t)AOTX_SLOTS * 3u * channels * (width - 1u);
        initial_state.resize(work.delta.state_elements);
        initial_history.resize(work.delta.history_elements);
        for (size_t i = 0u; i < initial_state.size(); ++i) initial_state[i] = 0.2f * aotx_delta_data(i, 11u);
        for (size_t i = 0u; i < initial_history.size(); ++i) initial_history[i] = aotx_delta_data(i, 13u);
        weights.resize(8u + channels * width + 2u * heads + dim);
        size_t offset = 8u;
        desc.layer[aotx_delta_test_layer].offset[AOTX_DELTA_CONV] = offset * sizeof(float);
        for (unsigned int c = 0u; c < channels; ++c)
            for (unsigned int tap = 0u; tap < width; ++tap)
                weights[offset + c * width + tap] = aotx_delta_data(c * width + tap, 17u) *
                                                     (0.3f + 0.1f * tap);
        offset += channels * width;
        desc.layer[aotx_delta_test_layer].offset[AOTX_DELTA_A] = offset * sizeof(float);
        for (unsigned int h = 0u; h < heads; ++h) weights[offset + h] = -0.03f * (h + 1u);
        offset += heads;
        desc.layer[aotx_delta_test_layer].offset[AOTX_DELTA_DT] = offset * sizeof(float);
        for (unsigned int h = 0u; h < heads; ++h) weights[offset + h] = 0.2f * ((int)h - 2);
        offset += heads;
        desc.layer[aotx_delta_test_layer].offset[AOTX_DELTA_NORM] = offset * sizeof(float);
        for (unsigned int d = 0u; d < dim; ++d) weights[offset + d] = 0.7f + aotx_delta_data(d, 23u);
        size_t rows = (size_t)n * aotx_delta_test_steps;
        input.resize(rows * channels);
        z.resize(rows * inner);
        alpha.resize(rows * heads);
        beta.resize(rows * heads);
        for (size_t i = 0u; i < input.size(); ++i) {
            unsigned int c = i % channels;
            float scale = c < dim ? 1e-7f : 1.0f;
            input[i] = aotx_delta_data(i, 29u) * scale;
            if (c >= keys * dim && c < (keys + 1u) * dim) input[i] = 0.0f;
        }
        for (size_t i = 0u; i < z.size(); ++i) z[i] = aotx_delta_data(i, 31u) * 5.0f;
        for (size_t i = 0u; i < alpha.size(); ++i) {
            alpha[i] = aotx_delta_data(i, 37u) * 4.0f;
            beta[i] = aotx_delta_data(i, 41u) * 6.0f;
            if (i % 19u == 0u) alpha[i] = 1000.0f;
            if (i % 23u == 0u) alpha[i] = -1000.0f;
            if (i % 17u == 0u) beta[i] = 90.0f;
            if (i % 29u == 0u) beta[i] = -90.0f;
        }
        agent.resize(n);
        length.resize(n);
        for (unsigned int s = 0u; s < n; ++s) {
            agent[s] = (s * 37u + 11u) % AOTX_SLOTS;
            length[s] = 4u + s % 3u;
        }
        work.weights = (unsigned long long)copy(weights.data(), weights.size() * sizeof(float));
        work.delta.state = (float *)copy(initial_state.data(), initial_state.size() * sizeof(float));
        work.delta.history = (float *)copy(initial_history.data(), initial_history.size() * sizeof(float));
        work.delta.qkv = (float *)copy(NULL, (size_t)capacity * channels * sizeof(float));
        work.delta.conv = (float *)copy(NULL, (size_t)capacity * channels * sizeof(float));
        work.delta.z = (float *)copy(NULL, (size_t)capacity * inner * sizeof(float));
        work.delta.out = (float *)copy(NULL, (size_t)capacity * inner * sizeof(float));
        work.delta.act = (half *)copy(NULL, (size_t)capacity * inner * sizeof(half));
        work.delta.alpha = (float *)copy(NULL, (size_t)capacity * heads * sizeof(float));
        work.delta.beta = (float *)copy(NULL, (size_t)capacity * heads * sizeof(float));
        work.base = (unsigned int *)copy(NULL, n * sizeof(unsigned int));
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &desc, sizeof desc,
            AOTX_MODEL_LANGUAGE * sizeof desc), "delta descriptor");
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_space, &work, sizeof work,
            AOTX_MODEL_LANGUAGE * sizeof work), "delta workspace");
    }

    ~aotx_delta_fixture()
    {
        for (void *pointer : allocations) aotx_check_runtime(cudaFree(pointer), "delta release");
    }
};

struct aotx_delta_reference {
    std::vector<double> state, history;
    explicit aotx_delta_reference(const aotx_delta_fixture &f) :
        state(f.initial_state.begin(), f.initial_state.end()),
        history(f.initial_history.begin(), f.initial_history.end()) {}
};

/* The reference uses scalar matrix math and no device reduction or state helpers. */
static void aotx_delta_cpu(const aotx_delta_fixture &f, aotx_delta_reference &ref,
                            const std::vector<unsigned int> &offset,
                            const std::vector<unsigned int> &base,
                            const std::vector<unsigned int> &source,
                            std::vector<double> &conv, std::vector<double> &out,
                            std::vector<double> &act)
{
    const unsigned int d = f.dim, h = f.heads, k = f.keys, c = f.channels;
    const float *kernel = f.weights.data() + f.desc.layer[aotx_delta_test_layer].offset[AOTX_DELTA_CONV] / 4u;
    const float *a = f.weights.data() + f.desc.layer[aotx_delta_test_layer].offset[AOTX_DELTA_A] / 4u;
    const float *dt = f.weights.data() + f.desc.layer[aotx_delta_test_layer].offset[AOTX_DELTA_DT] / 4u;
    const float *norm = f.weights.data() + f.desc.layer[aotx_delta_test_layer].offset[AOTX_DELTA_NORM] / 4u;
    conv.resize(source.size() * c);
    out.resize(source.size() * f.inner);
    act.resize(source.size() * f.inner);
    for (unsigned int s = 0u; s < f.n; ++s) {
        size_t si = ((size_t)f.agent[s] * 3u + 1u) * h * d * d;
        size_t hi = ((size_t)f.agent[s] * 3u + 1u) * c * (f.width - 1u);
        if (offset[s] == offset[s + 1u]) continue;
        if (base[s] == 0u) {
            std::fill(ref.state.begin() + si, ref.state.begin() + si + h * d * d, 0.0);
            std::fill(ref.history.begin() + hi, ref.history.begin() + hi + c * (f.width - 1u), 0.0);
        }
        for (unsigned int t = offset[s]; t < offset[s + 1u]; ++t) {
            unsigned int src = source[t];
            for (unsigned int channel = 0u; channel < c; ++channel) {
                double *past = ref.history.data() + hi + channel * (f.width - 1u);
                double sum = 0.0;
                for (unsigned int tap = 0u; tap < f.width; ++tap) {
                    double x = tap + 1u == f.width ? f.input[(size_t)src * c + channel] : past[tap];
                    sum += x * kernel[channel * f.width + tap];
                }
                conv[(size_t)t * c + channel] = sum * aotx_delta_logistic(sum);
                for (unsigned int tap = 1u; tap < f.width - 1u; ++tap) past[tap - 1u] = past[tap];
                past[f.width - 2u] = f.input[(size_t)src * c + channel];
            }
            for (unsigned int head = 0u; head < 2u * k; ++head) {
                double *row = conv.data() + (size_t)t * c + head * d;
                double square = 0.0;
                for (unsigned int j = 0u; j < d; ++j) square += row[j] * row[j];
                double divisor = std::max(sqrt(square), (double)f.desc.rms_eps);
                for (unsigned int j = 0u; j < d; ++j) row[j] /= divisor;
            }
            for (unsigned int head = 0u; head < h; ++head) {
                double alpha = (double)f.alpha[(size_t)src * h + head] + dt[head];
                double softplus = alpha > 0.0 ? alpha + log1p(exp(-alpha)) : log1p(exp(alpha));
                double decay = exp(a[head] * softplus);
                double beta = aotx_delta_logistic(f.beta[(size_t)src * h + head]);
                const double *q = conv.data() + (size_t)t * c + (head % k) * d;
                const double *key = q + k * d;
                const double *value = conv.data() + (size_t)t * c + 2u * k * d + head * d;
                double *result = out.data() + (size_t)t * f.inner + head * d;
                for (unsigned int v = 0u; v < d; ++v) {
                    double *matrix = ref.state.data() + si + head * d * d + v * d;
                    double prediction = 0.0;
                    for (unsigned int j = 0u; j < d; ++j) {
                        matrix[j] *= decay;
                        prediction += matrix[j] * key[j];
                    }
                    double change = beta * (value[v] - prediction);
                    double product = 0.0;
                    for (unsigned int j = 0u; j < d; ++j) {
                        matrix[j] += key[j] * change;
                        product += matrix[j] * q[j];
                    }
                    result[v] = product / sqrt((double)d);
                }
                double square = 0.0;
                for (unsigned int v = 0u; v < d; ++v) square += result[v] * result[v];
                double divisor = sqrt(square / d + f.desc.rms_eps);
                for (unsigned int v = 0u; v < d; ++v) {
                    size_t row = (size_t)t * f.inner + head * d + v;
                    double z = f.z[(size_t)src * f.inner + head * d + v];
                    act[row] = result[v] / divisor * norm[v] * z * aotx_delta_logistic(z);
                }
            }
        }
    }
}

struct aotx_delta_result {
    std::vector<float> state, history, conv, out;
    std::vector<half> act;
};

static void aotx_delta_upload(void *device, const void *host, size_t bytes)
{
    aotx_check_runtime(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice), "delta upload");
}

static void aotx_delta_download(void *host, const void *device, size_t bytes)
{
    aotx_check_runtime(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost), "delta download");
}

static aotx_delta_result aotx_delta_run(aotx_delta_fixture &f, bool split)
{
    aotx_delta_upload(f.work.delta.state, f.initial_state.data(), f.initial_state.size() * sizeof(float));
    aotx_delta_upload(f.work.delta.history, f.initial_history.data(), f.initial_history.size() * sizeof(float));
    aotx_delta_reference reference(f);
    aotx_delta_result result;
    result.state.resize(f.initial_state.size());
    result.history.resize(f.initial_history.size());
    result.conv.resize((size_t)f.n * aotx_delta_test_steps * f.channels, aotx_delta_test_guard);
    result.out.resize((size_t)f.n * aotx_delta_test_steps * f.inner, aotx_delta_test_guard);
    result.act.resize(result.out.size(), __float2half(aotx_delta_test_guard));
    std::vector<unsigned int> base(f.n), next(f.n, 0u), offset(f.n + 1u);
    for (unsigned int s = 0u; s < f.n; ++s) base[s] = f.reset || s % 3u == 1u ? 0u : 17u + s;
    unsigned int *device_offset = (unsigned int *)f.copy(NULL, offset.size() * sizeof(unsigned int));
    unsigned int *device_agent = (unsigned int *)f.copy(f.agent.data(), f.agent.size() * sizeof(unsigned int));
    unsigned int passes = split ? 4u : 2u;
    const unsigned int chunks[] = {1u, 2u, 3u};
    for (unsigned int pass = 0u; pass < passes; ++pass) {
        bool decode = pass + 1u == passes;
        unsigned int count = decode ? 1u : (split ? chunks[pass] : 6u);
        std::vector<unsigned int> source;
        for (unsigned int s = 0u; s < f.n; ++s) {
            if (decode && (f.reset || s % 5u == 1u)) base[s] = 0u;
            offset[s] = source.size();
            unsigned int end = decode ? f.length[s] + 1u : std::min(next[s] + count, f.length[s]);
            for (unsigned int t = next[s]; t < end; ++t) source.push_back(s * aotx_delta_test_steps + t);
            next[s] = end;
        }
        offset[f.n] = source.size();
        unsigned int tokens = source.size();
        std::vector<float> input((size_t)f.capacity * f.channels, aotx_delta_test_guard);
        std::vector<float> z((size_t)f.capacity * f.inner, aotx_delta_test_guard);
        std::vector<float> alpha((size_t)f.capacity * f.heads, aotx_delta_test_guard);
        std::vector<float> beta(alpha.size(), aotx_delta_test_guard);
        std::vector<float> conv(input.size(), aotx_delta_test_guard);
        std::vector<float> out(z.size(), aotx_delta_test_guard);
        std::vector<half> act(z.size(), __float2half(aotx_delta_test_guard));
        for (unsigned int t = 0u; t < tokens; ++t) {
            memcpy(input.data() + (size_t)t * f.channels, f.input.data() + (size_t)source[t] * f.channels,
                   f.channels * sizeof(float));
            memcpy(z.data() + (size_t)t * f.inner, f.z.data() + (size_t)source[t] * f.inner, f.inner * sizeof(float));
            memcpy(alpha.data() + (size_t)t * f.heads, f.alpha.data() + (size_t)source[t] * f.heads,
                   f.heads * sizeof(float));
            memcpy(beta.data() + (size_t)t * f.heads, f.beta.data() + (size_t)source[t] * f.heads,
                   f.heads * sizeof(float));
        }
        aotx_delta_upload(device_offset, offset.data(), offset.size() * sizeof(unsigned int));
        aotx_delta_upload(f.work.base, base.data(), base.size() * sizeof(unsigned int));
        aotx_delta_upload(f.work.delta.qkv, input.data(), input.size() * sizeof(float));
        aotx_delta_upload(f.work.delta.z, z.data(), z.size() * sizeof(float));
        aotx_delta_upload(f.work.delta.alpha, alpha.data(), alpha.size() * sizeof(float));
        aotx_delta_upload(f.work.delta.beta, beta.data(), beta.size() * sizeof(float));
        aotx_delta_upload(f.work.delta.conv, conv.data(), conv.size() * sizeof(float));
        aotx_delta_upload(f.work.delta.out, out.data(), out.size() * sizeof(float));
        aotx_delta_upload(f.work.delta.act, act.data(), act.size() * sizeof(half));
        aotx_model_run run = {};
        run.seqs = f.n;
        run.tokens = tokens;
        run.offset = device_offset;
        run.agent = device_agent;
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &run, sizeof run,
            AOTX_MODEL_LANGUAGE * sizeof run), "delta call");
        aotx_model_delta_conv<<<dim3(AOTX_SLOTS, (f.channels + AOTX_DELTA_THREADS - 1u) /
                                     AOTX_DELTA_THREADS), AOTX_DELTA_THREADS>>>(
                                         AOTX_MODEL_LANGUAGE, aotx_delta_test_layer);
        aotx_model_delta_qk<<<dim3(f.capacity, 2u * f.keys), 32u>>>(AOTX_MODEL_LANGUAGE, aotx_delta_test_layer);
        aotx_model_delta_scan<<<dim3(AOTX_SLOTS, f.heads, f.dim / AOTX_DELTA_VALUES), AOTX_DELTA_THREADS>>>(
            AOTX_MODEL_LANGUAGE, aotx_delta_test_layer);
        aotx_model_delta_gate<<<dim3(f.capacity, f.heads), 32u>>>(AOTX_MODEL_LANGUAGE, aotx_delta_test_layer);
        aotx_check_runtime(cudaGetLastError(), "delta launch");
        aotx_delta_download(conv.data(), f.work.delta.conv, conv.size() * sizeof(float));
        aotx_delta_download(out.data(), f.work.delta.out, out.size() * sizeof(float));
        aotx_delta_download(act.data(), f.work.delta.act, act.size() * sizeof(half));
        aotx_delta_download(result.state.data(), f.work.delta.state, result.state.size() * sizeof(float));
        aotx_delta_download(result.history.data(), f.work.delta.history, result.history.size() * sizeof(float));
        bool guard = true;
        for (size_t i = (size_t)tokens * f.channels; i < conv.size(); ++i) guard &= conv[i] == aotx_delta_test_guard;
        for (size_t i = (size_t)tokens * f.inner; i < out.size(); ++i)
            guard &= out[i] == aotx_delta_test_guard && __half2float(act[i]) == __half2float(__float2half(aotx_delta_test_guard));
        aotx_delta_check(guard, "inactive token rows", f.n, f.dim);
        conv.resize((size_t)tokens * f.channels);
        out.resize((size_t)tokens * f.inner);
        act.resize((size_t)tokens * f.inner);
        std::vector<double> want_conv, want_out, want_act;
        aotx_delta_cpu(f, reference, offset, base, source, want_conv, want_out, want_act);
        std::vector<float> actual_act(act.size());
        for (size_t i = 0u; i < act.size(); ++i) actual_act[i] = __half2float(act[i]);
        aotx_delta_check(aotx_delta_close(conv, want_conv, 3e-6, 3e-5), "convolution and L2", f.n, f.dim);
        aotx_delta_check(aotx_delta_close(out, want_out, 3e-6, 4e-5), "float64 recurrence output", f.n, f.dim);
        aotx_delta_check(aotx_delta_close(actual_act, want_act, 3e-5, 8e-4), "RMS norm and SiLU gate", f.n, f.dim);
        aotx_delta_check(aotx_delta_close(result.state, reference.state, 4e-6, 4e-5), "float64 carried state", f.n, f.dim);
        aotx_delta_check(aotx_delta_close(result.history, reference.history, 0.0, 0.0), "exact convolution history", f.n, f.dim);
        bool untouched = true;
        for (unsigned int slot = 0u; slot < AOTX_SLOTS; ++slot) {
            bool used = std::find(f.agent.begin(), f.agent.end(), slot) != f.agent.end();
            for (unsigned int layer = 0u; layer < 3u; ++layer) {
                if (used && layer == 1u) continue;
                size_t si = ((size_t)slot * 3u + layer) * f.heads * f.dim * f.dim;
                size_t hi = ((size_t)slot * 3u + layer) * f.channels * (f.width - 1u);
                untouched &= memcmp(result.state.data() + si, f.initial_state.data() + si,
                                    (size_t)f.heads * f.dim * f.dim * sizeof(float)) == 0;
                untouched &= memcmp(result.history.data() + hi, f.initial_history.data() + hi,
                                    (size_t)f.channels * (f.width - 1u) * sizeof(float)) == 0;
            }
        }
        aotx_delta_check(untouched, "other slots and compact layers", f.n, f.dim);
        for (unsigned int t = 0u; t < tokens; ++t) {
            memcpy(result.conv.data() + (size_t)source[t] * f.channels, conv.data() + (size_t)t * f.channels,
                   f.channels * sizeof(float));
            memcpy(result.out.data() + (size_t)source[t] * f.inner, out.data() + (size_t)t * f.inner,
                   f.inner * sizeof(float));
            memcpy(result.act.data() + (size_t)source[t] * f.inner, act.data() + (size_t)t * f.inner,
                   f.inner * sizeof(half));
        }
        for (unsigned int s = 0u; s < f.n; ++s) base[s] += offset[s + 1u] - offset[s];
    }
    return result;
}

template <typename T>
static bool aotx_delta_equal(const std::vector<T> &a, const std::vector<T> &b)
{
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

int main(void)
{
    const unsigned int batches[] = {1u, 64u};
    const unsigned int dims[] = {32u, 64u, 96u, 128u};
    const unsigned int widths[] = {2u, 4u, 16u, 4u};
    if (AOTX_SLOTS < 64u) {
        fprintf(stderr, "delta: requires at least 64 slots\n");
        return 1;
    }
    for (unsigned int n : batches) {
        for (unsigned int shape = 0u; shape < 4u; ++shape) {
            aotx_delta_fixture fixture(n, dims[shape], widths[shape], shape == 2u);
            aotx_delta_result whole = aotx_delta_run(fixture, false);
            aotx_delta_result split = aotx_delta_run(fixture, true);
            aotx_delta_check(aotx_delta_equal(whole.state, split.state), "partition state bytes", n, dims[shape]);
            aotx_delta_check(aotx_delta_equal(whole.history, split.history), "partition history bytes", n, dims[shape]);
            aotx_delta_check(aotx_delta_equal(whole.conv, split.conv), "partition convolution bytes", n, dims[shape]);
            aotx_delta_check(aotx_delta_equal(whole.out, split.out), "partition output bytes", n, dims[shape]);
            aotx_delta_check(aotx_delta_equal(whole.act, split.act), "partition gate bytes", n, dims[shape]);
        }
    }
    printf("delta: %u checks, %u failed\n", aotx_delta_checks, aotx_delta_failed);
    return aotx_delta_failed ? 1 : 0;
}
