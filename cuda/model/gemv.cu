/* Purpose: Multiply a small batch of activation rows by a weight tensor.
 * Owns: Nothing; the caller owns the input and the output.
 * Launch shape: One warp for each run of 2 rows of the weight tensor, 256 threads a block.
 * Lifetime: One launch.
 *
 * The product is y[m][n] equal to the sum over k of x[m][k] times w[n][k]. The work is
 * bound by the read of the weights. The kernel reads each weight one time and applies it
 * to every row of x. One warp takes a run of 2 rows of w. Eight lanes take one block of 32
 * weights, and each lane takes 4 weights that follow each other. One warp therefore takes
 * 4 blocks of each row in a turn.
 *
 * The 32 lanes hold parts of the same sum and add them at the end. A batch of more than 8
 * rows of x runs as groups of 8, and each group reads the weights again. The count of rows
 * in the group is a value of the code. A batch of one row does the work of one row.
 *
 * The kernel keeps the product of a weight and its scale in single precision. The
 * dequantization function rounds the same product to half. The difference is about 5e-4 of
 * the value. The accuracy gate of the product holds 1e-2.
 *
 * The depth k must be a multiple of 32, which is the block size of every quantized type.
 * The caller refuses any other k. */
#include "model/matrix.cuh"

#define AOTX_GEMV_FULL     0xFFFFFFFFu
#define AOTX_GEMV_STEP     (32u / AOTX_MATRIX_LANES)

/* One pass over the weights, for one block type and a known count of rows of x. */
template <unsigned int TYPE, unsigned int TAKE>
__device__ __forceinline__ void aotx_gemv_pass(const unsigned char *w, unsigned int n,
                                               unsigned int k, const half *x,
                                               unsigned int m, unsigned int m0, float *y)
{
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int warps = blockDim.x >> 5;
    unsigned int row0 = (blockIdx.x * warps + warp) * AOTX_GEMV_ROWS;
    if (row0 >= n) {
        return;
    }
    unsigned int blocks = k / AOTX_MATRIX_BLOCK;
    unsigned long long stride = aotx_matrix_row_bytes(TYPE, k);

    /* A run at the end of the tensor can pass the last row. Such a row reads the last row
     * again, which keeps the read in the tensor, and the store leaves it out. */
    const unsigned char *row[AOTX_GEMV_ROWS];
#pragma unroll
    for (unsigned int r = 0u; r < AOTX_GEMV_ROWS; ++r) {
        unsigned int at = (row0 + r < n) ? (row0 + r) : (n - 1u);
        row[r] = w + stride * (unsigned long long)at;
    }
    float acc[AOTX_GEMV_ROWS][TAKE];
#pragma unroll
    for (unsigned int r = 0u; r < AOTX_GEMV_ROWS; ++r) {
#pragma unroll
        for (unsigned int t = 0u; t < TAKE; ++t) {
            acc[r][t] = 0.0f;
        }
    }

    unsigned int sub = lane / AOTX_MATRIX_LANES;
    unsigned int pos = (lane % AOTX_MATRIX_LANES) * AOTX_MATRIX_GROUP;
    for (unsigned int b0 = 0u; b0 < blocks; b0 += AOTX_GEMV_STEP) {
        unsigned int block = b0 + sub;
        if (block >= blocks) {
            continue;
        }
        float xv[TAKE][AOTX_MATRIX_GROUP];
#pragma unroll
        for (unsigned int t = 0u; t < TAKE; ++t) {
            if (m0 + t < m) {
                aotx_matrix_four(x + (size_t)(m0 + t) * k + block * AOTX_MATRIX_BLOCK + pos,
                                 xv[t]);
            } else {
#pragma unroll
                for (unsigned int v = 0u; v < AOTX_MATRIX_GROUP; ++v) {
                    xv[t][v] = 0.0f;
                }
            }
        }
#pragma unroll
        for (unsigned int r = 0u; r < AOTX_GEMV_ROWS; ++r) {
            float wv[AOTX_MATRIX_GROUP];
            aotx_matrix_group<TYPE>(row[r], block, pos, wv);
#pragma unroll
            for (unsigned int v = 0u; v < AOTX_MATRIX_GROUP; ++v) {
#pragma unroll
                for (unsigned int t = 0u; t < TAKE; ++t) {
                    acc[r][t] += wv[v] * xv[t][v];
                }
            }
        }
    }

    /* The 32 lanes hold 32 parts of each sum. The exchange adds them in the lanes. */
#pragma unroll
    for (unsigned int step = 16u; step > 0u; step >>= 1) {
#pragma unroll
        for (unsigned int r = 0u; r < AOTX_GEMV_ROWS; ++r) {
#pragma unroll
            for (unsigned int t = 0u; t < TAKE; ++t) {
                acc[r][t] += __shfl_down_sync(AOTX_GEMV_FULL, acc[r][t], step);
            }
        }
    }
    if (lane == 0u) {
#pragma unroll
        for (unsigned int r = 0u; r < AOTX_GEMV_ROWS; ++r) {
            if (row0 + r >= n) {
                continue;
            }
#pragma unroll
            for (unsigned int t = 0u; t < TAKE; ++t) {
                if (m0 + t < m) {
                    y[(size_t)(m0 + t) * n + row0 + r] = acc[r][t];
                }
            }
        }
    }
}

/* The groups of rows of x. A batch of one row takes the code that makes one product for
 * each weight. A larger batch takes the code that makes 8. A group count of 2 and of 4 as
 * well costs more in registers than it saves in work. This card gave 116 GB a second at 8
 * rows with four group counts, against 125 with two. */
template <unsigned int TYPE>
__device__ __forceinline__ void aotx_gemv_batch(const unsigned char *w, unsigned int n,
                                                unsigned int k, const half *x,
                                                unsigned int m, float *y)
{
    for (unsigned int m0 = 0u; m0 < m; m0 += AOTX_GEMV_BATCH) {
        unsigned int left = m - m0;
        if (left == 1u) {
            aotx_gemv_pass<TYPE, 1u>(w, n, k, x, m, m0, y);
        } else {
            aotx_gemv_pass<TYPE, AOTX_GEMV_BATCH>(w, n, k, x, m, m0, y);
        }
    }
}

__global__ void aotx_model_gemv(const void *w, unsigned int type, unsigned int n,
                                unsigned int k, const half *x, unsigned int m, float *y)
{
    const unsigned char *base = (const unsigned char *)w;
    switch (type) {
    case AOTX_WEIGHT_Q8_0:
        aotx_gemv_batch<AOTX_WEIGHT_Q8_0>(base, n, k, x, m, y);
        break;
    case AOTX_WEIGHT_Q4_0:
        aotx_gemv_batch<AOTX_WEIGHT_Q4_0>(base, n, k, x, m, y);
        break;
    case AOTX_WEIGHT_F16:
        aotx_gemv_batch<AOTX_WEIGHT_F16>(base, n, k, x, m, y);
        break;
    case AOTX_WEIGHT_F32:
        aotx_gemv_batch<AOTX_WEIGHT_F32>(base, n, k, x, m, y);
        break;
    default:
        break;
    }
}
