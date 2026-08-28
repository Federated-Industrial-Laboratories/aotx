/* Purpose: Check the matrix kernels: dequantization, the tensor core product, the memory
 * bound product, the module of hand written assembly, and the rate gates.
 * Owns: The test fixtures and the counts of the cases.
 * Launch shape: The kernels of the model module, at one row of x and at 64.
 * Lifetime: The program. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "boot/check.h"
#include "model/matrix.cuh"

extern "C" {
#include "disk/modelfile/modelfile.h"
}

/* The seed of every fixture. The content of each row differs, so a wrong row index or a
 * turned operand cannot pass. */
#define AOTX_TEST_SEED     0x243F6A8885A308D3ull
#define AOTX_TEST_TOL      1e-2
#define AOTX_TEST_SPOTS    64u
#define AOTX_TEST_WARM     3u
#define AOTX_TEST_RUNS     20u
#define AOTX_TEST_FLOOR    30u

/* The rate gates. The peak is the rate the card gives for half products with single
 * precision sums. The band is the memory rate of the card. Both are figures of the card
 * and not measurements of this system.
 *
 * The gates hold a margin of about 25 percent under the lowest figure measured. The
 * display shares this card, and the clock of the card moves with the work of the display.
 *
 * Measured here: 17.6, 18.4 and 18.5 TFLOPS for the three product shapes. A tile of half
 * the width and half the height gives 12.9, 12.2 and 12.6, which the gate refuses. The
 * memory bound product gives 218 and 238 GB a second at one row of x. It gives 107 and 125
 * GB a second at 8 rows. A run under a gate is a defect of the kernel. */
#define AOTX_TEST_PEAK     51.0    /* half products with single precision sums, TFLOPS */
#define AOTX_TEST_BAND     360.0   /* memory rate of the card, GB a second */
#define AOTX_TEST_GEMM_MIN 13.5    /* 26.5 percent of the peak */
#define AOTX_TEST_GEMV_ONE 150.0   /* one row of x, 41.7 percent of the band */
#define AOTX_TEST_GEMV_MIN 75.0    /* 8 rows of x, 20.8 percent of the band */

static unsigned long long aotx_test_next(unsigned long long *state)
{
    unsigned long long z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static float aotx_test_unit(unsigned long long *state)
{
    unsigned int bits = (unsigned int)(aotx_test_next(state) >> 40);
    return (float)bits / 8388608.0f - 1.0f;
}

static double aotx_test_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* One weight of one row, read on the host from the block layout the model file gives.
 * This is a second reading of the layout. It is written from the definition and not from
 * the device code, so a fault of one is not a fault of both. */
static float aotx_test_weight(const unsigned char *row, unsigned int type, unsigned int j)
{
    if (type == AOTX_TENSOR_Q8_0) {
        const unsigned char *one = row + (size_t)(j / 32u) * 34u;
        __half scale;
        memcpy(&scale, one, sizeof scale);
        int q = (int)(signed char)one[2u + (j % 32u)];
        return __half2float(__float2half((float)q * __half2float(scale)));
    }
    if (type == AOTX_TENSOR_Q4_0) {
        const unsigned char *one = row + (size_t)(j / 32u) * 18u;
        __half scale;
        memcpy(&scale, one, sizeof scale);
        unsigned int at = j % 32u;
        unsigned int pair = one[2u + (at % 16u)];
        unsigned int nibble = (at < 16u) ? (pair & 0x0Fu) : (pair >> 4);
        return __half2float(__float2half(((float)(int)nibble - 8.0f)
                                         * __half2float(scale)));
    }
    if (type == AOTX_TENSOR_F16) {
        __half one;
        memcpy(&one, row + (size_t)j * 2u, sizeof one);
        return __half2float(one);
    }
    return ((const float *)row)[j];
}

typedef struct aotx_test_tensor {
    unsigned char *host;
    unsigned char *device;
    unsigned long long stride;
    unsigned long long bytes;
    unsigned int type;
    unsigned int n;
    unsigned int k;
} aotx_test_tensor;

static void aotx_test_free(aotx_test_tensor *w)
{
    free(w->host);
    if (w->device != NULL) {
        cudaFree(w->device);
    }
    memset(w, 0, sizeof *w);
}

/* Build a tensor of random blocks. Every block takes its own scale and its own values, so
 * no two rows and no two blocks hold the same content. */
static void aotx_test_build(aotx_test_tensor *w, unsigned int type, unsigned int n,
                            unsigned int k, unsigned long long seed)
{
    memset(w, 0, sizeof *w);
    w->type = type;
    w->n = n;
    w->k = k;
    w->stride = aotx_matrix_row_bytes(type, k);
    w->bytes = w->stride * (unsigned long long)n;
    w->host = (unsigned char *)calloc(1u, (size_t)w->bytes);
    unsigned long long state = seed;
    for (unsigned int r = 0u; r < n; ++r) {
        unsigned char *row = w->host + w->stride * (unsigned long long)r;
        for (unsigned int b = 0u; b < k / 32u; ++b) {
            float scale = 0.004f + 0.03f * (aotx_test_unit(&state) * 0.5f + 0.5f);
            __half packed = __float2half(scale);
            if (type == AOTX_TENSOR_Q8_0) {
                unsigned char *one = row + (size_t)b * 34u;
                memcpy(one, &packed, sizeof packed);
                for (unsigned int i = 0u; i < 32u; ++i) {
                    int q = (int)(aotx_test_next(&state) % 255u) - 127;
                    one[2u + i] = (unsigned char)(signed char)q;
                }
            } else {
                unsigned char *one = row + (size_t)b * 18u;
                memcpy(one, &packed, sizeof packed);
                for (unsigned int i = 0u; i < 16u; ++i) {
                    unsigned int lo = (unsigned int)(aotx_test_next(&state) % 16u);
                    unsigned int hi = (unsigned int)(aotx_test_next(&state) % 16u);
                    one[2u + i] = (unsigned char)(lo | (hi << 4));
                }
            }
        }
    }
    aotx_check_runtime(cudaMalloc((void **)&w->device, (size_t)w->bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(w->device, w->host, (size_t)w->bytes,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
}

/* Read a run of rows of a real tensor of a model file. The bytes go to the device without
 * a change, so the kernels read what the file holds. */
static int aotx_test_real(const char *path, const char *name, unsigned int first,
                          unsigned int rows, aotx_test_tensor *w)
{
    aotx_modelfile *file = NULL;
    aotx_tensor_info info;
    memset(w, 0, sizeof *w);
    if (aotx_modelfile_open(path, &file) != 0) {
        return 1;
    }
    if (aotx_modelfile_find(file, name, &info) != 0) {
        aotx_modelfile_close(file);
        return 1;
    }
    w->type = info.type;
    w->k = (unsigned int)info.dims[0];
    w->n = rows;
    w->stride = aotx_matrix_row_bytes(info.type, w->k);
    w->bytes = w->stride * (unsigned long long)rows;
    w->host = (unsigned char *)malloc((size_t)w->bytes);
    int bad = aotx_modelfile_read(file, info.offset + w->stride * (unsigned long long)first,
                                  w->bytes, w->host);
    aotx_modelfile_close(file);
    if (bad != 0) {
        free(w->host);
        memset(w, 0, sizeof *w);
        return 1;
    }
    aotx_check_runtime(cudaMalloc((void **)&w->device, (size_t)w->bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(w->device, w->host, (size_t)w->bytes,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    return 0;
}

/* The rows of x. Each row holds its own random content. */
static half *aotx_test_input(unsigned int m, unsigned int k, unsigned long long seed,
                             half **device)
{
    half *host = (half *)malloc((size_t)m * k * sizeof *host);
    unsigned long long state = seed;
    for (size_t i = 0u; i < (size_t)m * k; ++i) {
        host[i] = __float2half(aotx_test_unit(&state));
    }
    aotx_check_runtime(cudaMalloc((void **)device, (size_t)m * k * sizeof *host),
                       "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(*device, host, (size_t)m * k * sizeof *host,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    return host;
}

/* The reference product on the host. A swap of the operand turns the weight tensor, which
 * needs n equal to k, and gives the answer a right kernel does not give. */
static void aotx_test_ref(const aotx_test_tensor *w, const half *x, unsigned int m,
                          double *y, int swap)
{
    for (unsigned int i = 0u; i < m; ++i) {
        for (unsigned int j = 0u; j < w->n; ++j) {
            double sum = 0.0;
            for (unsigned int c = 0u; c < w->k; ++c) {
                const unsigned char *row = w->host
                    + w->stride * (unsigned long long)(swap ? c : j);
                float wv = aotx_test_weight(row, w->type, swap ? j : c);
                sum += (double)__half2float(x[(size_t)i * w->k + c]) * (double)wv;
            }
            y[(size_t)i * w->n + j] = sum;
        }
    }
}

/* Compare a result with the reference. The measure is the difference over the larger of
 * two figures: the reference value, and the root mean square of the reference. A value
 * near zero therefore does not give a false finding. The tolerance catches a wrong index,
 * a turned operand and a wrong block layout. The rounding to half gives about 5e-4. */
static unsigned int aotx_test_diff(const float *got, const double *want, size_t count,
                                   double tol, double *worst)
{
    double sum = 0.0;
    for (size_t i = 0u; i < count; ++i) {
        sum += want[i] * want[i];
    }
    double rms = sqrt(sum / (double)count);
    unsigned int bad = 0u;
    double high = 0.0;
    for (size_t i = 0u; i < count; ++i) {
        double base = fabs(want[i]);
        if (base < rms) {
            base = rms;
        }
        double rel = (base > 0.0) ? fabs((double)got[i] - want[i]) / base : 0.0;
        if (rel > high) {
            high = rel;
        }
        if (rel > tol) {
            bad += 1u;
        }
    }
    *worst = high;
    return bad;
}

/* Check a few results of a large product against a dot product of the same bytes. A whole
 * reference of a large shape takes minutes on the host. A sample of 64 results takes
 * milliseconds and finds the same class of defect. */
static unsigned int aotx_test_spot(const aotx_test_tensor *w, const half *x, unsigned int m,
                                   const float *y, unsigned int count, double *worst)
{
    unsigned long long state = AOTX_TEST_SEED ^ 0x5A5A5A5Aull;
    unsigned int bad = 0u;
    double high = 0.0;
    for (unsigned int s = 0u; s < count; ++s) {
        unsigned int i = (unsigned int)(aotx_test_next(&state) % m);
        unsigned int j = (unsigned int)(aotx_test_next(&state) % w->n);
        const unsigned char *row = w->host + w->stride * (unsigned long long)j;
        double sum = 0.0;
        double square = 0.0;
        for (unsigned int c = 0u; c < w->k; ++c) {
            double one = (double)__half2float(x[(size_t)i * w->k + c])
                       * (double)aotx_test_weight(row, w->type, c);
            sum += one;
            square += one * one;
        }
        double base = fabs(sum);
        double spread = sqrt(square * (double)w->k);
        if (base < spread * 1e-3) {
            base = spread * 1e-3;
        }
        double rel = fabs((double)y[(size_t)i * w->n + j] - sum) / base;
        if (rel > high) {
            high = rel;
        }
        if (rel > AOTX_TEST_TOL) {
            bad += 1u;
        }
    }
    *worst = high;
    return bad;
}

static void aotx_test_gemm(const aotx_test_tensor *w, const half *x, unsigned int m,
                           float *y)
{
    dim3 grid((w->n + AOTX_GEMM_TILE_N - 1u) / AOTX_GEMM_TILE_N,
              (m + AOTX_GEMM_TILE_M - 1u) / AOTX_GEMM_TILE_M);
    aotx_model_gemm<<<grid, AOTX_GEMM_THREADS>>>(w->device, w->type, w->n, w->k, x, m, y);
}

static void aotx_test_gemv(const aotx_test_tensor *w, const half *x, unsigned int m,
                           float *y)
{
    unsigned int blocks = (w->n + AOTX_GEMV_ROWS_CTA - 1u) / AOTX_GEMV_ROWS_CTA;
    aotx_model_gemv<<<blocks, AOTX_GEMV_THREADS>>>(w->device, w->type, w->n, w->k, x, m, y);
}

/* Compare a run of dequantized rows with the host reading of the same bytes. The values
 * must be equal, because both sides round one product to half. */
static unsigned int aotx_test_same(const aotx_test_tensor *w, unsigned int first,
                                   unsigned int rows, const half *got)
{
    unsigned int bad = 0u;
    for (unsigned int r = 0u; r < rows; ++r) {
        const unsigned char *row = w->host + w->stride * (unsigned long long)(first + r);
        for (unsigned int j = 0u; j < w->k; ++j) {
            float want = aotx_test_weight(row, w->type, j);
            if (__half2float(got[(size_t)r * w->k + j]) != want) {
                bad += 1u;
            }
        }
    }
    return bad;
}

static unsigned int aotx_test_case_dequant(unsigned int *applied)
{
    static const unsigned int types[2] = { AOTX_TENSOR_Q8_0, AOTX_TENSOR_Q4_0 };
    static const char *names[2] = { "q8_0", "q4_0" };
    static const unsigned int batch[2] = { 1u, 64u };
    unsigned int failed = 0u;
    for (unsigned int t = 0u; t < 2u; ++t) {
        aotx_test_tensor w;
        aotx_test_build(&w, types[t], 96u, 256u, AOTX_TEST_SEED + t);
        for (unsigned int s = 0u; s < 2u; ++s) {
            unsigned int rows = batch[s];
            unsigned int first = 16u;
            half *out = NULL;
            aotx_check_runtime(cudaMalloc((void **)&out,
                                          (size_t)rows * w.k * sizeof *out), "cudaMalloc");
            aotx_model_dequant<<<rows, 256u>>>(w.device, w.type, w.k, first, rows, out);
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            half *host = (half *)malloc((size_t)rows * w.k * sizeof *host);
            aotx_check_runtime(cudaMemcpy(host, out, (size_t)rows * w.k * sizeof *host,
                                          cudaMemcpyDeviceToHost), "cudaMemcpy");
            unsigned int bad = aotx_test_same(&w, first, rows, host);
            printf("matrix: dequant %s at %u rows of 256 columns gives %u wrong values\n",
                   names[t], rows, bad);
            failed += (bad != 0u) ? 1u : 0u;
            *applied += 1u;
            free(host);
            cudaFree(out);
        }
        aotx_test_free(&w);
    }
    return failed;
}

/* The product cases. The shape 80 by 128 is not a multiple of the tile. The guard of the
 * load and the guard of the store therefore both take part. */
static unsigned int aotx_test_case_product(unsigned int *applied, int tensor_core)
{
    static const unsigned int types[2] = { AOTX_TENSOR_Q8_0, AOTX_TENSOR_Q4_0 };
    static const char *names[2] = { "q8_0", "q4_0" };
    static const unsigned int batch[2] = { 1u, 64u };
    const char *kind = tensor_core ? "gemm" : "gemv";
    unsigned int failed = 0u;
    for (unsigned int t = 0u; t < 2u; ++t) {
        aotx_test_tensor w;
        aotx_test_build(&w, types[t], 80u, 128u, AOTX_TEST_SEED + 8u + t);
        for (unsigned int s = 0u; s < 2u; ++s) {
            unsigned int m = batch[s];
            half *dx = NULL;
            half *x = aotx_test_input(m, w.k, AOTX_TEST_SEED + 16u + s, &dx);
            float *dy = NULL;
            aotx_check_runtime(cudaMalloc((void **)&dy, (size_t)m * w.n * sizeof *dy),
                               "cudaMalloc");
            aotx_check_runtime(cudaMemset(dy, 0, (size_t)m * w.n * sizeof *dy),
                               "cudaMemset");
            if (tensor_core) {
                aotx_test_gemm(&w, dx, m, dy);
            } else {
                aotx_test_gemv(&w, dx, m, dy);
            }
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            float *y = (float *)malloc((size_t)m * w.n * sizeof *y);
            aotx_check_runtime(cudaMemcpy(y, dy, (size_t)m * w.n * sizeof *y,
                                          cudaMemcpyDeviceToHost), "cudaMemcpy");
            double *want = (double *)malloc((size_t)m * w.n * sizeof *want);
            aotx_test_ref(&w, x, m, want, 0);
            double worst = 0.0;
            unsigned int bad = aotx_test_diff(y, want, (size_t)m * w.n, AOTX_TEST_TOL,
                                              &worst);
            printf("matrix: %s %s at %u rows of x, 80 by 128, gives %u values over %.0e, "
                   "worst %.2e\n", kind, names[t], m, bad, AOTX_TEST_TOL, worst);
            failed += (bad != 0u) ? 1u : 0u;
            *applied += 1u;
            free(want);
            free(y);
            free(x);
            cudaFree(dx);
            cudaFree(dy);
        }
        aotx_test_free(&w);
    }
    return failed;
}

/* The negative arm. A turned weight tensor must give a different answer, or the compare
 * cannot see a turned operand. The shape is square, so the turn is defined. */
static unsigned int aotx_test_case_turn(unsigned int *applied, int tensor_core)
{
    const char *kind = tensor_core ? "gemm" : "gemv";
    aotx_test_tensor w;
    unsigned int m = 64u;
    aotx_test_build(&w, AOTX_TENSOR_Q8_0, 64u, 64u, AOTX_TEST_SEED + 32u);
    half *dx = NULL;
    half *x = aotx_test_input(m, w.k, AOTX_TEST_SEED + 33u, &dx);
    float *dy = NULL;
    aotx_check_runtime(cudaMalloc((void **)&dy, (size_t)m * w.n * sizeof *dy), "cudaMalloc");
    if (tensor_core) {
        aotx_test_gemm(&w, dx, m, dy);
    } else {
        aotx_test_gemv(&w, dx, m, dy);
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    float *y = (float *)malloc((size_t)m * w.n * sizeof *y);
    aotx_check_runtime(cudaMemcpy(y, dy, (size_t)m * w.n * sizeof *y,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    double *want = (double *)malloc((size_t)m * w.n * sizeof *want);
    double worst = 0.0;
    aotx_test_ref(&w, x, m, want, 0);
    unsigned int right = aotx_test_diff(y, want, (size_t)m * w.n, AOTX_TEST_TOL, &worst);
    aotx_test_ref(&w, x, m, want, 1);
    unsigned int turned = aotx_test_diff(y, want, (size_t)m * w.n, AOTX_TEST_TOL, &worst);
    printf("matrix: %s against a turned weight tensor gives %u values over %.0e of %u, "
           "and %u against the right one\n", kind, turned, AOTX_TEST_TOL, m * w.n, right);
    *applied += 1u;
    unsigned int failed = (turned == 0u || right != 0u) ? 1u : 0u;
    free(want);
    free(y);
    free(x);
    cudaFree(dx);
    cudaFree(dy);
    aotx_test_free(&w);
    return failed;
}

/* One product against the reference, for a tensor which is already on the device. */
static unsigned int aotx_test_one(const aotx_test_tensor *w, unsigned int m, int tensor_core,
                                  const char *note, unsigned int *applied)
{
    half *dx = NULL;
    half *x = aotx_test_input(m, w->k, AOTX_TEST_SEED + 64u + m, &dx);
    float *dy = NULL;
    aotx_check_runtime(cudaMalloc((void **)&dy, (size_t)m * w->n * sizeof *dy), "cudaMalloc");
    if (tensor_core) {
        aotx_test_gemm(w, dx, m, dy);
    } else {
        aotx_test_gemv(w, dx, m, dy);
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    float *y = (float *)malloc((size_t)m * w->n * sizeof *y);
    aotx_check_runtime(cudaMemcpy(y, dy, (size_t)m * w->n * sizeof *y,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    double *want = (double *)malloc((size_t)m * w->n * sizeof *want);
    aotx_test_ref(w, x, m, want, 0);
    double worst = 0.0;
    unsigned int bad = aotx_test_diff(y, want, (size_t)m * w->n, AOTX_TEST_TOL, &worst);
    printf("matrix: %s at %u rows of x, %u by %u, gives %u values over %.0e, worst %.2e\n",
           note, m, w->n, w->k, bad, AOTX_TEST_TOL, worst);
    *applied += 1u;
    free(want);
    free(y);
    free(x);
    cudaFree(dx);
    cudaFree(dy);
    return (bad != 0u) ? 1u : 0u;
}

/* The real rows of one tensor of one model file. The bytes come from the file, and the
 * reference reads the same bytes on the host. */
static unsigned int aotx_test_case_rows(const char *path, const char *name,
                                        unsigned int first, unsigned int rows,
                                        const char *label, unsigned int *applied,
                                        unsigned int *skipped)
{
    aotx_test_tensor w;
    if (aotx_test_real(path, name, first, rows, &w) != 0) {
        printf("matrix: skipped 5 real row cases of %s; the file %s did not read\n",
               label, path);
        *skipped += 5u;
        return 0u;
    }
    unsigned int failed = 0u;
    half *out = NULL;
    aotx_check_runtime(cudaMalloc((void **)&out, (size_t)w.n * w.k * sizeof *out),
                       "cudaMalloc");
    aotx_model_dequant<<<w.n, 256u>>>(w.device, w.type, w.k, 0u, w.n, out);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    half *host = (half *)malloc((size_t)w.n * w.k * sizeof *host);
    aotx_check_runtime(cudaMemcpy(host, out, (size_t)w.n * w.k * sizeof *host,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int bad = aotx_test_same(&w, 0u, w.n, host);
    printf("matrix: dequant of %u real %s rows of %u columns gives %u wrong values\n",
           w.n, label, w.k, bad);
    failed += (bad != 0u) ? 1u : 0u;
    *applied += 1u;
    free(host);
    cudaFree(out);

    char note[64];
    snprintf(note, sizeof note, "gemm real %s", label);
    failed += aotx_test_one(&w, 1u, 1, note, applied);
    failed += aotx_test_one(&w, 64u, 1, note, applied);
    snprintf(note, sizeof note, "gemv real %s", label);
    failed += aotx_test_one(&w, 1u, 0, note, applied);
    failed += aotx_test_one(&w, 64u, 0, note, applied);
    aotx_test_free(&w);
    return failed;
}

/* The real rows of the model files, and one larger run checked at 64 places. */
static unsigned int aotx_test_case_real(const char *models, unsigned int *applied,
                                        unsigned int *skipped)
{
    char embed[512];
    char language[512];
    snprintf(embed, sizeof embed, "%s/Qwen3-Embedding-0.6B-Q8_0.gguf", models);
    snprintf(language, sizeof language, "%s/Qwen3-4B-Q4_0.gguf", models);
    unsigned int failed = 0u;
    failed += aotx_test_case_rows(embed, "token_embd.weight", 1000u, 64u, "q8_0",
                                  applied, skipped);
    failed += aotx_test_case_rows(language, "blk.0.ffn_gate.weight", 512u, 64u, "q4_0",
                                  applied, skipped);

    aotx_test_tensor w;
    if (aotx_test_real(embed, "token_embd.weight", 4096u, 1024u, &w) != 0) {
        printf("matrix: skipped 1 real row case; the file did not read\n");
        *skipped += 1u;
        return failed;
    }
    unsigned int m = 256u;
    half *dx = NULL;
    half *x = aotx_test_input(m, w.k, AOTX_TEST_SEED + 96u, &dx);
    float *dy = NULL;
    aotx_check_runtime(cudaMalloc((void **)&dy, (size_t)m * w.n * sizeof *dy), "cudaMalloc");
    aotx_test_gemm(&w, dx, m, dy);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    float *y = (float *)malloc((size_t)m * w.n * sizeof *y);
    aotx_check_runtime(cudaMemcpy(y, dy, (size_t)m * w.n * sizeof *y,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    double worst = 0.0;
    unsigned int wrong = aotx_test_spot(&w, x, m, y, AOTX_TEST_SPOTS, &worst);
    printf("matrix: gemm real at 256 rows of x, 1024 by %u, gives %u of %u places over "
           "%.0e, worst %.2e\n", w.k, wrong, AOTX_TEST_SPOTS, AOTX_TEST_TOL, worst);
    failed += (wrong != 0u) ? 1u : 0u;
    *applied += 1u;
    free(y);
    free(x);
    cudaFree(dx);
    cudaFree(dy);
    aotx_test_free(&w);
    return failed;
}

/* The rate of the tensor core product, with a check of 64 places of the result. */
static unsigned int aotx_test_rate_gemm(unsigned int m, unsigned int n, unsigned int k,
                                        unsigned int *applied)
{
    aotx_test_tensor w;
    aotx_test_build(&w, AOTX_TENSOR_Q8_0, n, k, AOTX_TEST_SEED + n + k);
    half *dx = NULL;
    half *x = aotx_test_input(m, k, AOTX_TEST_SEED + 128u, &dx);
    float *dy = NULL;
    aotx_check_runtime(cudaMalloc((void **)&dy, (size_t)m * n * sizeof *dy), "cudaMalloc");
    for (unsigned int i = 0u; i < AOTX_TEST_WARM; ++i) {
        aotx_test_gemm(&w, dx, m, dy);
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double from = aotx_test_now();
    for (unsigned int i = 0u; i < AOTX_TEST_RUNS; ++i) {
        aotx_test_gemm(&w, dx, m, dy);
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double spent = (aotx_test_now() - from) / (double)AOTX_TEST_RUNS;
    double rate = 2.0 * (double)m * (double)n * (double)k / spent / 1e12;
    float *y = (float *)malloc((size_t)m * n * sizeof *y);
    aotx_check_runtime(cudaMemcpy(y, dy, (size_t)m * n * sizeof *y, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    double worst = 0.0;
    unsigned int wrong = aotx_test_spot(&w, x, m, y, AOTX_TEST_SPOTS, &worst);
    printf("matrix: gemm q8_0 m=%u n=%u k=%u gives %.2f TFLOPS, %.1f percent of %.1f, "
           "%.0f us a launch, gate %.2f; %u of %u places over %.0e, worst %.2e\n",
           m, n, k, rate, rate / AOTX_TEST_PEAK * 100.0, AOTX_TEST_PEAK, spent * 1e6,
           AOTX_TEST_GEMM_MIN, wrong, AOTX_TEST_SPOTS, AOTX_TEST_TOL, worst);
    *applied += 2u;
    unsigned int failed = (wrong != 0u) ? 1u : 0u;
    if (rate < AOTX_TEST_GEMM_MIN) {
        printf("matrix: the gemm rate gate refuses %.2f TFLOPS under %.2f\n", rate,
               AOTX_TEST_GEMM_MIN);
        failed += 1u;
    }
    free(y);
    free(x);
    cudaFree(dx);
    cudaFree(dy);
    aotx_test_free(&w);
    return failed;
}

/* The rate of the memory bound product, as the weight bytes the kernel reads. A batch of
 * more than 8 rows of x reads the weights again for each group of 8. */
static unsigned int aotx_test_rate_gemv(unsigned int m, unsigned int n, unsigned int k,
                                        unsigned int *applied)
{
    aotx_test_tensor w;
    aotx_test_build(&w, AOTX_TENSOR_Q8_0, n, k, AOTX_TEST_SEED + n + k + 1u);
    half *dx = NULL;
    half *x = aotx_test_input(m, k, AOTX_TEST_SEED + 160u, &dx);
    float *dy = NULL;
    aotx_check_runtime(cudaMalloc((void **)&dy, (size_t)m * n * sizeof *dy), "cudaMalloc");
    for (unsigned int i = 0u; i < AOTX_TEST_WARM; ++i) {
        aotx_test_gemv(&w, dx, m, dy);
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double from = aotx_test_now();
    for (unsigned int i = 0u; i < AOTX_TEST_RUNS; ++i) {
        aotx_test_gemv(&w, dx, m, dy);
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double spent = (aotx_test_now() - from) / (double)AOTX_TEST_RUNS;
    double groups = (double)((m + AOTX_GEMV_BATCH - 1u) / AOTX_GEMV_BATCH);
    double rate = (double)w.bytes * groups / spent / 1e9;
    float *y = (float *)malloc((size_t)m * n * sizeof *y);
    aotx_check_runtime(cudaMemcpy(y, dy, (size_t)m * n * sizeof *y, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    double worst = 0.0;
    unsigned int wrong = aotx_test_spot(&w, x, m, y, AOTX_TEST_SPOTS, &worst);
    double gate = (m == 1u) ? AOTX_TEST_GEMV_ONE : AOTX_TEST_GEMV_MIN;
    printf("matrix: gemv q8_0 m=%u n=%u k=%u gives %.1f GB a second, %.1f percent of %.1f, "
           "%.0f us a launch, gate %.1f; %u of %u places over %.0e, worst %.2e\n",
           m, n, k, rate, rate / AOTX_TEST_BAND * 100.0, AOTX_TEST_BAND, spent * 1e6,
           gate, wrong, AOTX_TEST_SPOTS, AOTX_TEST_TOL, worst);
    *applied += 2u;
    unsigned int failed = (wrong != 0u) ? 1u : 0u;
    if (rate < gate) {
        printf("matrix: the gemv rate gate refuses %.1f GB a second under %.1f\n", rate,
               gate);
        failed += 1u;
    }
    free(y);
    free(x);
    cudaFree(dx);
    cudaFree(dy);
    aotx_test_free(&w);
    return failed;
}

/* Read the text of a module. The driver compiles the text at load. */
static char *aotx_test_module(const char *path)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *text = (char *)malloc((size_t)size + 1u);
    if (text == NULL || fread(text, 1u, (size_t)size, file) != (size_t)size) {
        free(text);
        fclose(file);
        return NULL;
    }
    text[size] = '\0';
    fclose(file);
    return text;
}

/* The module of hand written assembly, as a node of a graph through the driver. The result
 * and the rate both go against the kernel the compiler makes. */
static unsigned int aotx_test_case_ptx(unsigned int *applied, unsigned int *skipped)
{
    char *text = aotx_test_module(AOTX_PTX_DIR "/gemv_q8.ptx");
    if (text == NULL) {
        printf("matrix: skipped 2 module cases; the module text did not read\n");
        *skipped += 2u;
        return 0u;
    }
    CUmodule module;
    CUfunction function;
    aotx_check_driver(cuModuleLoadData(&module, text), "cuModuleLoadData");
    free(text);
    aotx_check_driver(cuModuleGetFunction(&function, module, "aotx_gemv_q8"),
                      "cuModuleGetFunction");

    unsigned int n = 2560u;
    unsigned int k = 2560u;
    unsigned int m = 1u;
    aotx_test_tensor w;
    aotx_test_build(&w, AOTX_TENSOR_Q8_0, n, k, AOTX_TEST_SEED + 200u);
    half *dx = NULL;
    half *x = aotx_test_input(m, k, AOTX_TEST_SEED + 201u, &dx);
    float *dy = NULL;
    float *dz = NULL;
    aotx_check_runtime(cudaMalloc((void **)&dy, (size_t)n * sizeof *dy), "cudaMalloc");
    aotx_check_runtime(cudaMalloc((void **)&dz, (size_t)n * sizeof *dz), "cudaMalloc");
    aotx_check_runtime(cudaMemset(dz, 0, (size_t)n * sizeof *dz), "cudaMemset");

    CUdeviceptr pw = (CUdeviceptr)w.device;
    CUdeviceptr px = (CUdeviceptr)dx;
    CUdeviceptr pz = (CUdeviceptr)dz;
    void *params[] = { &pw, &n, &k, &px, &pz };
    CUDA_KERNEL_NODE_PARAMS node_params = {};
    node_params.func = function;
    node_params.gridDimX = (n + AOTX_GEMV_ROWS_CTA - 1u) / AOTX_GEMV_ROWS_CTA;
    node_params.gridDimY = 1u;
    node_params.gridDimZ = 1u;
    node_params.blockDimX = AOTX_GEMV_THREADS;
    node_params.blockDimY = 1u;
    node_params.blockDimZ = 1u;
    node_params.kernelParams = params;

    CUgraph graph;
    CUgraphExec exec;
    CUgraphNode node;
    aotx_check_driver(cuGraphCreate(&graph, 0), "cuGraphCreate");
    aotx_check_driver(cuGraphAddKernelNode(&node, graph, NULL, 0, &node_params),
                      "cuGraphAddKernelNode");
    aotx_check_driver(cuGraphInstantiate(&exec, graph, 0), "cuGraphInstantiate");

    for (unsigned int i = 0u; i < AOTX_TEST_WARM; ++i) {
        aotx_check_driver(cuGraphLaunch(exec, 0), "cuGraphLaunch");
        aotx_test_gemv(&w, dx, m, dy);
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    double from = aotx_test_now();
    for (unsigned int i = 0u; i < AOTX_TEST_RUNS; ++i) {
        aotx_check_driver(cuGraphLaunch(exec, 0), "cuGraphLaunch");
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double raw = (aotx_test_now() - from) / (double)AOTX_TEST_RUNS;

    from = aotx_test_now();
    for (unsigned int i = 0u; i < AOTX_TEST_RUNS; ++i) {
        aotx_test_gemv(&w, dx, m, dy);
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double made = (aotx_test_now() - from) / (double)AOTX_TEST_RUNS;

    float *a = (float *)malloc((size_t)n * sizeof *a);
    float *b = (float *)malloc((size_t)n * sizeof *b);
    aotx_check_runtime(cudaMemcpy(a, dy, (size_t)n * sizeof *a, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(b, dz, (size_t)n * sizeof *b, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    double *want = (double *)malloc((size_t)n * sizeof *want);
    aotx_test_ref(&w, x, m, want, 0);
    double one = 0.0;
    double two = 0.0;
    unsigned int wrong = aotx_test_diff(b, want, n, AOTX_TEST_TOL, &one);
    unsigned int bad = aotx_test_diff(a, want, n, AOTX_TEST_TOL, &two);
    double rate_raw = (double)w.bytes / raw / 1e9;
    double rate_made = (double)w.bytes / made / 1e9;
    printf("matrix: the module gives %u of %u sums over %.0e of the reference, worst %.2e; "
           "the kernel gives %u, worst %.2e\n", wrong, n, AOTX_TEST_TOL, one, bad, two);
    printf("matrix: the module runs at %.1f GB a second and the kernel at %.1f, "
           "%.0f us against %.0f us\n", rate_raw, rate_made, raw * 1e6, made * 1e6);
    *applied += 2u;
    unsigned int failed = (wrong != 0u || bad != 0u) ? 1u : 0u;
    free(want);
    if (rate_raw < AOTX_TEST_GEMV_ONE) {
        printf("matrix: the gemv rate gate refuses the module at %.1f GB a second\n",
               rate_raw);
        failed += 1u;
    }

    free(a);
    free(b);
    free(x);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    aotx_test_free(&w);
    aotx_check_driver(cuGraphExecDestroy(exec), "cuGraphExecDestroy");
    aotx_check_driver(cuGraphDestroy(graph), "cuGraphDestroy");
    aotx_check_driver(cuModuleUnload(module), "cuModuleUnload");
    return failed;
}

int main(int argc, char **argv)
{
    const char *models = (argc > 1) ? argv[1] : "models";
    CUdevice device;
    CUcontext context;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device),
                      "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    printf("matrix: the seed of every fixture is 0x%llx\n",
           (unsigned long long)AOTX_TEST_SEED);

    unsigned int applied = 0u;
    unsigned int failed = 0u;
    unsigned int skipped = 0u;

    failed += aotx_test_case_dequant(&applied);
    failed += aotx_test_case_product(&applied, 1);
    failed += aotx_test_case_product(&applied, 0);
    failed += aotx_test_case_turn(&applied, 1);
    failed += aotx_test_case_turn(&applied, 0);
    failed += aotx_test_case_real(models, &applied, &skipped);
    failed += aotx_test_rate_gemm(256u, 2560u, 2560u, &applied);
    failed += aotx_test_rate_gemm(256u, 9728u, 2560u, &applied);
    failed += aotx_test_rate_gemm(256u, 2560u, 9728u, &applied);
    failed += aotx_test_rate_gemv(1u, 2560u, 2560u, &applied);
    failed += aotx_test_rate_gemv(8u, 2560u, 2560u, &applied);
    failed += aotx_test_rate_gemv(1u, 9728u, 2560u, &applied);
    failed += aotx_test_rate_gemv(8u, 9728u, 2560u, &applied);
    failed += aotx_test_case_ptx(&applied, &skipped);

    if (skipped != 0u) {
        printf("matrix: skipped %u cases\n", skipped);
    }
    printf("matrix: %u cases applied, %u failed, %u skipped\n", applied, failed, skipped);
    if (applied < AOTX_TEST_FLOOR) {
        printf("matrix: %u cases is under the floor of %u\n", applied, AOTX_TEST_FLOOR);
        return 1;
    }
    return (failed == 0u) ? 0 : 1;
}
