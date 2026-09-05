/* Purpose: Check packed weight readers against a host reading in double.
 * Owns: Packed fixtures and the counts of the block cases.
 * Launch shape: The dequantization kernel and the two product kernels of the model module.
 * Lifetime: One matrix test run.
 *
 * Three checks for each type. The whole tensor check reads one full tensor of a K model
 * file and dequantizes it on the device. It compares every weight with the host reading of
 * matrix_kref.h in double.
 *
 * The agreement check reads every position of every block of a run of rows through the
 * single value reader and the group reader. Both must give one weight. The product check
 * runs both products on real rows against the double reference. A store that holds no K
 * file gives a skip with a count. */

/* The two readers of one block, side by side. Each thread takes one group of 4 weights of
 * one block of one row. It reads the group, reads the 4 values, and counts the
 * differences. A difference at any position is a wrong sub block or a wrong shift. */
template <unsigned int TYPE>
__device__ __forceinline__ void aotx_kcase_agree_rows(const unsigned char *w, unsigned int k,
    unsigned int rows, unsigned int *bad, float *values, float *group_values)
{
    unsigned long long stride = aotx_matrix_row_bytes(TYPE, k);
    unsigned int groups = k / AOTX_MATRIX_GROUP;
    for (unsigned int r = blockIdx.x; r < rows; r += gridDim.x) {
        const unsigned char *row = w + stride * (unsigned long long)r;
        for (unsigned int g = threadIdx.x; g < groups; g += blockDim.x) {
            unsigned int block = (g * AOTX_MATRIX_GROUP) / AOTX_MATRIX_BLOCK;
            unsigned int at = (g * AOTX_MATRIX_GROUP) % AOTX_MATRIX_BLOCK;
            float four[AOTX_MATRIX_GROUP];
            aotx_matrix_group<TYPE>(row, block, at, four);
            unsigned int wrong = 0u;
            for (unsigned int v = 0u; v < AOTX_MATRIX_GROUP; ++v) {
                half one_half = aotx_matrix_value<TYPE>(row, block, at + v);
                half two_half = __float2half(four[v]);
                float one = __half2float(one_half);
                wrong += __half_as_ushort(one_half) != __half_as_ushort(two_half);
                size_t index = (size_t)r * k + g * AOTX_MATRIX_GROUP + v;
                values[index] = one;
                group_values[index] = four[v];
            }
            if (wrong != 0u) {
                atomicAdd(bad, wrong);
            }
        }
    }
}

__global__ void aotx_kcase_agree(const void *w, unsigned int type, unsigned int k,
    unsigned int rows, unsigned int *bad, float *values, float *groups)
{
    const unsigned char *base = (const unsigned char *)w;
    switch (type) {
    case AOTX_WEIGHT_Q4_1:
        aotx_kcase_agree_rows<AOTX_WEIGHT_Q4_1>(base, k, rows, bad, values, groups);
        break;
    case AOTX_WEIGHT_Q5_0:
        aotx_kcase_agree_rows<AOTX_WEIGHT_Q5_0>(base, k, rows, bad, values, groups);
        break;
    case AOTX_WEIGHT_Q5_1:
        aotx_kcase_agree_rows<AOTX_WEIGHT_Q5_1>(base, k, rows, bad, values, groups);
        break;
    case AOTX_WEIGHT_Q2_K:
        aotx_kcase_agree_rows<AOTX_WEIGHT_Q2_K>(base, k, rows, bad, values, groups);
        break;
    case AOTX_WEIGHT_Q3_K:
        aotx_kcase_agree_rows<AOTX_WEIGHT_Q3_K>(base, k, rows, bad, values, groups);
        break;
    case AOTX_WEIGHT_Q4_K:
        aotx_kcase_agree_rows<AOTX_WEIGHT_Q4_K>(base, k, rows, bad, values, groups);
        break;
    case AOTX_WEIGHT_Q5_K:
        aotx_kcase_agree_rows<AOTX_WEIGHT_Q5_K>(base, k, rows, bad, values, groups);
        break;
    case AOTX_WEIGHT_Q6_K:
        aotx_kcase_agree_rows<AOTX_WEIGHT_Q6_K>(base, k, rows, bad, values, groups);
        break;
    default:
        atomicAdd(bad, 1u);
        break;
    }
}

/* The flat reader of the blocks module, one weight for each thread, over a run of rows. */
__global__ void aotx_kcase_flat(const void *w, unsigned int type, unsigned int k,
                                unsigned int rows, half *out)
{
    unsigned long long count = (unsigned long long)rows * k;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        out[i] = __float2half(aotx_block_at(w, type, i));
    }
}

static const char *aotx_kcase_name(unsigned int type)
{
    return aotx_tensor_type_name(type);
}

/* Read a run of rows of a real K tensor. The stride of a row comes from the specification
 * and not from the device side. A wrong stride of the kernels therefore reads other bytes
 * than the host reading does. The check of the stride is a case of its own. The byte count
 * of the file side must agree as well, or the read takes bytes of the next tensor. */
static int aotx_kcase_real(const char *path, const char *name, unsigned int first,
                           unsigned int rows, aotx_test_tensor *w, unsigned int *applied,
                           unsigned int *failed)
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
    if (info.dim_count != 2u || info.dims[0] == 0u || info.dims[0] > UINT32_MAX
        || first > info.dims[1] || rows == 0u || rows > info.dims[1] - first) {
        aotx_modelfile_close(file);
        return 1;
    }
    w->type = info.type;
    w->k = (unsigned int)info.dims[0];
    w->n = rows;
    w->stride = aotx_kref_row_bytes(info.type, w->k);
    if (w->stride == 0u) {
        aotx_modelfile_close(file);
        return 1;
    }
    unsigned long long device_stride = aotx_matrix_row_bytes(info.type, w->k);
    unsigned long long file_bytes = w->stride * info.dims[1];
    if (device_stride != w->stride || file_bytes != info.bytes) {
        printf("matrix: %s row stride: the specification gives %llu bytes, the kernels give "
               "%llu, the file side gives %llu for %llu rows\n", aotx_kcase_name(info.type),
               w->stride, device_stride, (unsigned long long)info.bytes,
               (unsigned long long)info.dims[1]);
        *failed += 1u;
        aotx_modelfile_close(file);
        *applied += 1u;
        return 1;
    }
    *applied += 1u;
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

/* Relative error uses the absolute reference. At zero, an unequal result gives infinity.
 * The acceptance bound permits half rounding and one half subnormal step. */
template <typename T>
static void aotx_kcase_errors(const aotx_test_tensor *w, const T *got, double *abs_max,
                              double *rel_max, unsigned int *wrong)
{
    double base_floor = 5.9604644775390625e-08;
    *abs_max = 0.0;
    *rel_max = 0.0;
    *wrong = 0u;
    for (unsigned int r = 0u; r < w->n; ++r) {
        const unsigned char *row = w->host + w->stride * (unsigned long long)r;
        for (unsigned int j = 0u; j < w->k; ++j) {
            double want = aotx_kref_weight(row, w->type, j);
            double have = (double)(float)got[(size_t)r * w->k + j];
            double abs_err = isfinite(have) && isfinite(want) ? fabs(have - want) : INFINITY;
            double base = fabs(want);
            double rel = base > 0.0 ? abs_err / base : (abs_err == 0.0 ? 0.0 : INFINITY);
            if (abs_err > *abs_max) {
                *abs_max = abs_err;
            }
            if (rel > *rel_max) {
                *rel_max = rel;
            }
            if (!isfinite(have) || !isfinite(want)
                || (rel > 9.765625e-04 && abs_err > base_floor)) {
                *wrong += 1u;
            }
        }
    }
}

static unsigned int aotx_kcase_agreement(const aotx_test_tensor *w, unsigned int *applied,
                                         double *metrics = NULL);

/* Read one whole tensor of a K file in pieces of rows, dequantize each piece on the device
 * and compare it with the host reading. A whole embedding is 128256 rows, so the pieces
 * hold the device buffer at a bounded size. */
static unsigned int aotx_kcase_whole(const char *path, const char *name, unsigned int *applied,
                                     unsigned int *skipped)
{
    aotx_modelfile *file = NULL;
    aotx_tensor_info info;
    if (aotx_modelfile_open(path, &file) != 0) {
        printf("matrix: skipped 1 whole tensor case; the file %s did not read\n", path);
        *skipped += 1u;
        return 0u;
    }
    if (aotx_modelfile_find(file, name, &info) != 0) {
        aotx_modelfile_close(file);
        printf("matrix: skipped 1 whole tensor case; %s has no tensor %s\n", path, name);
        *skipped += 1u;
        return 0u;
    }
    aotx_modelfile_close(file);
    unsigned int rows = (unsigned int)info.dims[1];
    unsigned int k = (unsigned int)info.dims[0];
    unsigned int piece = 4096u;
    double abs_max = 0.0;
    double rel_max = 0.0;
    unsigned int wrong = 0u;
    unsigned long long elements = 0ull;
    unsigned int stride_bad = 0u;
    double metrics[4] = {};
    unsigned int reader_bad = 0u;
    for (unsigned int first = 0u; first < rows;) {
        unsigned int take = (rows - first < piece) ? (rows - first) : piece;
        aotx_test_tensor w;
        unsigned int stride_case = 0u;
        if (aotx_kcase_real(path, name, first, take, &w, &stride_case, &stride_bad) != 0) {
            printf("matrix: the rows %u of %s did not read\n", first, name);
            *applied += 1u;
            return 1u;
        }
        reader_bad += aotx_kcase_agreement(&w, applied, metrics);
        half *out = NULL;
        aotx_check_runtime(cudaMalloc((void **)&out, (size_t)w.n * w.k * sizeof *out),
                           "cudaMalloc");
        aotx_model_dequant<<<w.n, 256u>>>(w.device, w.type, w.k, 0u, w.n, out);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        half *host = (half *)malloc((size_t)w.n * w.k * sizeof *host);
        aotx_check_runtime(cudaMemcpy(host, out, (size_t)w.n * w.k * sizeof *host,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        double abs_one;
        double rel_one;
        unsigned int wrong_one;
        aotx_kcase_errors(&w, host, &abs_one, &rel_one, &wrong_one);
        abs_max = (abs_one > abs_max) ? abs_one : abs_max;
        rel_max = (rel_one > rel_max) ? rel_one : rel_max;
        wrong += wrong_one;
        elements += (unsigned long long)w.n * w.k;
        free(host);
        cudaFree(out);
        aotx_test_free(&w);
        first += take;
    }
    printf("matrix: %s %s of %u by %u, %llu weights against the host reading in double: "
           "max abs error %.3e, max rel error %.3e, %u wrong\n", aotx_kcase_name(info.type),
           name, rows, k, elements, abs_max, rel_max, wrong);
    printf("matrix: %s %s, %llu elements: value max abs %.3e rel %.3e; group max abs %.3e "
           "rel %.3e; relative base is abs(reference), zero gives 0 or infinity\n",
           aotx_kcase_name(info.type), name, elements, metrics[0], metrics[1], metrics[2], metrics[3]);
    *applied += 2u;
    return ((wrong != 0u) ? 1u : 0u) + ((stride_bad != 0u) ? 1u : 0u) + reader_bad;
}

/* The agreement of the two readers and of the flat reader, over a run of real rows. */
static unsigned int aotx_kcase_agreement(const aotx_test_tensor *w, unsigned int *applied,
                                         double *metrics)
{
    unsigned int *bad = NULL;
    unsigned int got = 0u;
    size_t bytes = (size_t)w->n * w->k * sizeof(float);
    float *values = NULL;
    float *groups = NULL;
    aotx_check_runtime(cudaMalloc((void **)&values, bytes), "cudaMalloc");
    aotx_check_runtime(cudaMalloc((void **)&groups, bytes), "cudaMalloc");
    aotx_check_runtime(cudaMalloc((void **)&bad, sizeof *bad), "cudaMalloc");
    aotx_check_runtime(cudaMemset(bad, 0, sizeof *bad), "cudaMemset");
    aotx_kcase_agree<<<w->n, 256u>>>(w->device, w->type, w->k, w->n, bad, values, groups);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&got, bad, sizeof got, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    cudaFree(bad);
    printf("matrix: %s value and group readers differ at %u of %llu positions over %u rows "
           "of %u\n", aotx_kcase_name(w->type), got, (unsigned long long)w->n * w->k, w->n,
           w->k);
    *applied += 1u;
    unsigned int failed = (got != 0u) ? 1u : 0u;
    float *readback = (float *)malloc(bytes);
    for (unsigned int reader = 0u; reader < 2u; ++reader) {
        aotx_check_runtime(cudaMemcpy(readback, reader ? groups : values, bytes,
            cudaMemcpyDeviceToHost), "cudaMemcpy");
        double abs_one;
        double rel_one;
        unsigned int wrong_one;
        aotx_kcase_errors(w, readback, &abs_one, &rel_one, &wrong_one);
        printf("matrix: %s %s reader, %llu elements: max abs %.3e rel %.3e, %u wrong\n",
            aotx_kcase_name(w->type), reader ? "group" : "value",
            (unsigned long long)w->n * w->k, abs_one, rel_one, wrong_one);
        if (metrics != NULL) {
            metrics[reader * 2u] = fmax(metrics[reader * 2u], abs_one);
            metrics[reader * 2u + 1u] = fmax(metrics[reader * 2u + 1u], rel_one);
        }
        failed += wrong_one != 0u;
        *applied += 1u;
    }
    free(readback);
    cudaFree(values);
    cudaFree(groups);

    half *out = NULL;
    aotx_check_runtime(cudaMalloc((void **)&out, (size_t)w->n * w->k * sizeof *out),
                       "cudaMalloc");
    aotx_kcase_flat<<<256u, 256u>>>(w->device, w->type, w->k, w->n, out);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    half *host = (half *)malloc((size_t)w->n * w->k * sizeof *host);
    aotx_check_runtime(cudaMemcpy(host, out, (size_t)w->n * w->k * sizeof *host,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    double abs_max;
    double rel_max;
    unsigned int wrong;
    aotx_kcase_errors(w, host, &abs_max, &rel_max, &wrong);
    printf("matrix: %s flat reader over %u rows of %u: max abs error %.3e, max rel error "
           "%.3e, %u wrong\n", aotx_kcase_name(w->type), w->n, w->k, abs_max, rel_max, wrong);
    *applied += 1u;
    failed += (wrong != 0u) ? 1u : 0u;
    free(host);
    cudaFree(out);
    return failed;
}

/* The product of real K rows against the double reference of the host reading. */
static unsigned int aotx_kcase_product(const aotx_test_tensor *w, unsigned int m,
                                       int tensor_core, unsigned int *applied)
{
    half *dx = NULL;
    half *x = aotx_test_input(m, w->k, AOTX_TEST_SEED + 200u + m, &dx);
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
    for (unsigned int i = 0u; i < m; ++i) {
        for (unsigned int j = 0u; j < w->n; ++j) {
            const unsigned char *row = w->host + w->stride * (unsigned long long)j;
            double sum = 0.0;
            for (unsigned int c = 0u; c < w->k; ++c) {
                sum += (double)__half2float(x[(size_t)i * w->k + c])
                     * aotx_kref_weight(row, w->type, c);
            }
            want[(size_t)i * w->n + j] = sum;
        }
    }
    double worst = 0.0;
    unsigned int bad = aotx_test_diff(y, want, (size_t)m * w->n, AOTX_TEST_TOL, &worst);
    printf("matrix: %s real %s at %u rows of x, %u by %u, gives %u values over %.0e, "
           "worst %.2e\n", tensor_core ? "gemm" : "gemv", aotx_kcase_name(w->type), m, w->n,
           w->k, bad, AOTX_TEST_TOL, worst);
    *applied += 1u;
    free(want);
    free(y);
    free(x);
    cudaFree(dx);
    cudaFree(dy);
    return (bad != 0u) ? 1u : 0u;
}

/* The row cases of one K tensor: the stride, agreement, the flat reader, and the four
 * products. */
static unsigned int aotx_kcase_rows(const char *path, const char *name, unsigned int first,
                                    unsigned int rows, unsigned int *applied,
                                    unsigned int *skipped)
{
    aotx_test_tensor w;
    unsigned int stride_bad = 0u;
    if (aotx_kcase_real(path, name, first, rows, &w, applied, &stride_bad) != 0) {
        printf("matrix: skipped 7 K row cases; the file %s has no tensor %s\n", path, name);
        *skipped += 7u;
        return 0u;
    }
    unsigned int failed = (stride_bad != 0u) ? 1u : 0u;
    failed += aotx_kcase_agreement(&w, applied);
    failed += aotx_kcase_product(&w, 1u, 1, applied);
    failed += aotx_kcase_product(&w, 64u, 1, applied);
    failed += aotx_kcase_product(&w, 1u, 0, applied);
    failed += aotx_kcase_product(&w, 64u, 0, applied);
    aotx_test_free(&w);
    return failed;
}

/* The rate of the memory bound product on one whole real K tensor, as the weight bytes the
 * kernel reads. The figure is a report and not a gate. The K reader unpacks a scale for
 * each group where Q8_0 reads one half value, and this line states what that costs. */
static void aotx_kcase_rate(const char *path, const char *name, unsigned int m,
                            unsigned int *skipped)
{
    aotx_modelfile *file = NULL;
    aotx_tensor_info info;
    if (aotx_modelfile_open(path, &file) != 0 || aotx_modelfile_find(file, name, &info) != 0) {
        if (file != NULL) {
            aotx_modelfile_close(file);
        }
        *skipped += 1u;
        return;
    }
    aotx_modelfile_close(file);
    aotx_test_tensor w;
    unsigned int applied = 0u;
    unsigned int failed = 0u;
    if (aotx_kcase_real(path, name, 0u, (unsigned int)info.dims[1], &w, &applied,
                        &failed) != 0) {
        *skipped += 1u;
        return;
    }
    half *dx = NULL;
    half *x = aotx_test_input(m, w.k, AOTX_TEST_SEED + 300u, &dx);
    float *dy = NULL;
    aotx_check_runtime(cudaMalloc((void **)&dy, (size_t)m * w.n * sizeof *dy), "cudaMalloc");
    for (unsigned int i = 0u; i < aotx_test_warm; ++i) {
        aotx_test_gemv(&w, dx, m, dy);
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double from = aotx_test_now();
    for (unsigned int i = 0u; i < aotx_test_runs; ++i) {
        aotx_test_gemv(&w, dx, m, dy);
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double spent = (aotx_test_now() - from) / (double)aotx_test_runs;
    double groups = (double)((m + AOTX_GEMV_BATCH - 1u) / AOTX_GEMV_BATCH);
    double rate = (double)w.bytes * groups / spent / 1e9;
    printf("matrix: gemv %s m=%u n=%u k=%u gives %.1f GB a second, %.0f us a launch, "
           "%llu weight bytes\n", aotx_kcase_name(w.type), m, w.n, w.k, rate, spent * 1e6,
           w.bytes);
    free(x);
    cudaFree(dx);
    cudaFree(dy);
    aotx_test_free(&w);
}

/* The K cases against the two K model files of the store. The Q4_K_M file gives Q4_K and
 * Q6_K tensors, and the Q5_K_M file gives Q5_K. The whole tensor checks take the down
 * projection of the first layer for Q6_K and the gate projection for Q4_K and Q5_K. */
static unsigned int aotx_test_case_k(const char *models, unsigned int *applied,
                                     unsigned int *skipped)
{
    char q4[512];
    char q5[512];
    snprintf(q4, sizeof q4, "%s/Llama-3.2-1B-Instruct-Q4_K_M.gguf", models);
    snprintf(q5, sizeof q5, "%s/Llama-3.2-1B-Instruct-Q5_K_M.gguf", models);
    unsigned int failed = 0u;
    failed += aotx_kcase_whole(q4, "blk.0.ffn_gate.weight", applied, skipped);
    failed += aotx_kcase_whole(q5, "blk.0.ffn_gate.weight", applied, skipped);
    failed += aotx_kcase_whole(q4, "blk.0.ffn_down.weight", applied, skipped);
    failed += aotx_kcase_rows(q4, "blk.0.ffn_gate.weight", 512u, 64u, applied, skipped);
    failed += aotx_kcase_rows(q5, "blk.0.ffn_gate.weight", 512u, 64u, applied, skipped);
    failed += aotx_kcase_rows(q4, "blk.0.ffn_down.weight", 128u, 64u, applied, skipped);
    aotx_kcase_rate(q4, "blk.0.ffn_gate.weight", 1u, skipped);
    aotx_kcase_rate(q5, "blk.0.ffn_gate.weight", 1u, skipped);
    aotx_kcase_rate(q4, "blk.0.ffn_down.weight", 1u, skipped);
    aotx_kcase_rate(q4, "blk.0.ffn_gate.weight", 8u, skipped);
    return failed;
}

/* Packed fixtures vary each row, scale, bit plane, and super block. */
static void aotx_kcase_build(aotx_test_tensor *w, unsigned int type, unsigned int rows)
{
    memset(w, 0, sizeof *w);
    w->type = type;
    w->n = rows;
    w->k = 512u;
    w->stride = aotx_kref_row_bytes(type, w->k);
    w->bytes = w->stride * rows;
    w->host = (unsigned char *)calloc(1u, (size_t)w->bytes);
    unsigned int width = type < AOTX_KREF_Q2K ? 32u : 256u;
    unsigned int bytes = (unsigned int)aotx_kref_row_bytes(type, width);
    unsigned long long state = AOTX_TEST_SEED + type;
    for (unsigned int r = 0u; r < rows; ++r) {
        for (unsigned int b = 0u; b < w->k / width; ++b) {
            unsigned char *one = w->host + r * w->stride + b * bytes;
            for (unsigned int i = 0u; i < bytes; ++i) {
                one[i] = (unsigned char)aotx_test_next(&state);
            }
            unsigned int d_at = type == AOTX_KREF_Q2K ? 80u
                : (type == AOTX_KREF_Q3K ? 108u : (type == AOTX_KREF_Q6K ? 208u : 0u));
            float scale = (float)(1u + ((r * 3u + b) % 15u)) / 1024.0f;
            if ((r + b) % 7u == 0u) scale = -scale;
            if ((r + b) % 13u == 0u) scale = 0.0f;
            half packed = __float2half(scale);
            memcpy(one + d_at, &packed, sizeof packed);
            if (type == AOTX_KREF_Q41 || type == AOTX_KREF_Q51
                || type == AOTX_KREF_Q2K || type == AOTX_KREF_Q4K || type == AOTX_KREF_Q5K) {
                unsigned int m_at = type == AOTX_KREF_Q2K ? 82u : 2u;
                packed = __float2half((float)((int)((r + b) % 9u) - 4) / 512.0f);
                memcpy(one + m_at, &packed, sizeof packed);
            }
            if (type == AOTX_KREF_Q2K) {
                for (unsigned int s = 0u; s < 16u; ++s) {
                    one[s] = (unsigned char)(((s + r + b) % 16u)
                        | (((s * 7u + r * 3u + b) % 16u) << 4));
                }
            }
            if (type == AOTX_KREF_Q3K) {
                memset(one + 96u, 0, 12u);
                for (unsigned int s = 0u; s < 16u; ++s) {
                    unsigned int code = (s * 13u + r * 3u + b * 7u) % 64u;
                    one[96u + s % 8u] |= (unsigned char)((code % 16u) << (4u * (s / 8u)));
                    one[104u + s % 4u] |= (unsigned char)((code / 16u) << (2u * (s / 4u)));
                }
            }
        }
    }
    aotx_check_runtime(cudaMalloc((void **)&w->device, (size_t)w->bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(w->device, w->host, (size_t)w->bytes,
        cudaMemcpyHostToDevice), "cudaMemcpy");
}

static unsigned int aotx_kcase_synthetic(unsigned int *applied)
{
    const unsigned int types[] = {AOTX_KREF_Q41, AOTX_KREF_Q50, AOTX_KREF_Q51,
        AOTX_KREF_Q2K, AOTX_KREF_Q3K, AOTX_KREF_Q4K, AOTX_KREF_Q5K, AOTX_KREF_Q6K};
    unsigned int failed = 0u;
    for (unsigned int type : types) {
        for (unsigned int rows = 1u; rows <= 64u; rows *= 64u) {
            aotx_test_tensor w;
            aotx_kcase_build(&w, type, rows);
            failed += aotx_matrix_row_bytes(type, w.k) != w.stride;
            *applied += 1u;
            failed += aotx_kcase_agreement(&w, applied);
            failed += aotx_kcase_product(&w, 1u, 0, applied);
            failed += aotx_kcase_product(&w, 64u, 0, applied);
            failed += aotx_kcase_product(&w, 1u, 1, applied);
            failed += aotx_kcase_product(&w, 64u, 1, applied);
            half *out = NULL;
            size_t bytes = (size_t)w.n * w.k * sizeof(half);
            aotx_check_runtime(cudaMalloc((void **)&out, bytes), "cudaMalloc");
            aotx_model_dequant<<<w.n, 256u>>>(w.device, type, w.k, 0u, w.n, out);
            half *host = (half *)malloc(bytes);
            aotx_check_runtime(cudaMemcpy(host, out, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
            double abs_max, rel_max;
            unsigned int wrong;
            aotx_kcase_errors(&w, host, &abs_max, &rel_max, &wrong);
            printf("matrix: synthetic %s dequant N=%u abs %.3e rel %.3e, %u wrong\n",
                aotx_kcase_name(type), rows, abs_max, rel_max, wrong);
            failed += wrong != 0u;
            *applied += 1u;
            free(host);
            cudaFree(out);
            aotx_test_free(&w);
        }
    }
    return failed;
}

/* An explicit file and type must supply a complete two-dimensional tensor. */
static unsigned int aotx_kcase_required(const char *path, const char *type_name,
    const char *tensor_name, unsigned int *applied)
{
    const unsigned int types[] = {AOTX_KREF_Q41, AOTX_KREF_Q50, AOTX_KREF_Q51,
        AOTX_KREF_Q2K, AOTX_KREF_Q3K, AOTX_KREF_Q4K, AOTX_KREF_Q5K, AOTX_KREF_Q6K};
    unsigned int type = ~0u;
    for (unsigned int candidate : types) {
        if (strcasecmp(type_name, aotx_kcase_name(candidate)) == 0) type = candidate;
    }
    aotx_modelfile *file = NULL;
    if (type == ~0u || aotx_modelfile_open(path, &file) != 0) {
        printf("matrix: required file or type cannot be read: %s %s\n", path, type_name);
        return 1u;
    }
    aotx_tensor_info info = {};
    int found = 0;
    for (uint64_t i = 0u; i < aotx_modelfile_tensor_count(file); ++i) {
        if (aotx_modelfile_tensor(file, i, &info) != 0) break;
        if (info.type == type && info.dim_count == 2u && info.dims[0] > 0u
            && info.dims[0] <= UINT32_MAX && info.dims[1] > 0u && info.dims[1] <= UINT32_MAX
            && (tensor_name == NULL || strcmp(tensor_name, info.name) == 0)) {
            found = 1;
            break;
        }
    }
    aotx_modelfile_close(file);
    if (!found) {
        printf("matrix: required %s tensor not found in %s\n", type_name, path);
        return 1u;
    }
    unsigned int skipped = 0u;
    unsigned int failed = aotx_kcase_whole(path, info.name, applied, &skipped);
    unsigned int rows = info.dims[1] < 64u ? (unsigned int)info.dims[1] : 64u;
    failed += aotx_kcase_rows(path, info.name, 0u, rows, applied, &skipped);
    printf("matrix: required %s tensor %s, %llu elements, %u skipped\n", type_name,
        info.name, (unsigned long long)(info.dims[0] * info.dims[1]), skipped);
    return failed + (skipped != 0u);
}
