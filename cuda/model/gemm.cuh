/* Purpose: Give the tensor core product of a batch of activation rows.
 * Owns: Nothing; the caller gives the two shared tiles.
 * Launch shape: Device function; the kernel that calls it sets the grid.
 * Lifetime: The whole run.
 *
 * The product is y[m][n] equal to the sum over k of x[m][k] times w[n][k]. The tensor core
 * operation mma.sync.aligned.m16n8k16 takes A as a 16 by 16 row major fragment. It takes B
 * as a 16 by 8 column major fragment. The row of x is A and the row of w is a column of B.
 * Both operands therefore keep the layout of the model file and need no transpose. The
 * operation ldmatrix puts the values in the registers the operation expects.
 *
 * The tile is 128 rows of x by 64 rows of the weight tensor, in 8 warps of 32 by 32. This
 * card gave 17.7 TFLOPS at m=256, n=2560, k=2560 and 18.5 TFLOPS at n=9728. A tile of 64
 * by 64 gave 15.7 and 14.2. The bound of 4 blocks holds the registers at 64, which puts 4
 * blocks on one multiprocessor. A tile of 64 rows of x is faster below 128 rows. Both
 * shapes are bound by the read of the weights there.
 *
 * The depth k must be a multiple of 32, which is the block size of every quantized type.
 * A K type takes a multiple of 256. The caller refuses any other k. The kernel reads one
 * block of each row for each step of k. It writes the half values into shared memory, so
 * the tensor cores see half values whatever the block type is. */
#ifndef AOTX_MODEL_GEMM_CUH
#define AOTX_MODEL_GEMM_CUH

#include "model/matrix.cuh"

#define AOTX_GEMM_TILE_K   AOTX_MATRIX_BLOCK
#define AOTX_GEMM_WARPS_M  4u
#define AOTX_GEMM_WARPS_N  2u
#define AOTX_GEMM_WARPS    (AOTX_GEMM_THREADS / 32u)
#define AOTX_GEMM_WARP_M   (AOTX_GEMM_TILE_M / AOTX_GEMM_WARPS_M)
#define AOTX_GEMM_WARP_N   (AOTX_GEMM_TILE_N / AOTX_GEMM_WARPS_N)

/* The shape of one tensor core operation, and the count of them one warp makes. */
#define AOTX_GEMM_STEP_M   16u
#define AOTX_GEMM_STEP_N   8u
#define AOTX_GEMM_STEP_K   16u
#define AOTX_GEMM_MMA_M    (AOTX_GEMM_WARP_M / AOTX_GEMM_STEP_M)
#define AOTX_GEMM_MMA_N    (AOTX_GEMM_WARP_N / AOTX_GEMM_STEP_N)

/* A tile row holds 8 half values more than the step of k. The pad puts the eight row
 * addresses of one ldmatrix in eight different banks of shared memory. */
#define AOTX_GEMM_PAD      8u
#define AOTX_GEMM_LINE     (AOTX_GEMM_TILE_K + AOTX_GEMM_PAD)

/* One thread of the fill takes 8 activations or 4 weights. */
#define AOTX_GEMM_WIDE     8u
#define AOTX_GEMM_SPAN_M   (AOTX_GEMM_TILE_K / AOTX_GEMM_WIDE)
#define AOTX_GEMM_SPAN_N   (AOTX_GEMM_TILE_K / AOTX_MATRIX_GROUP)
#define AOTX_GEMM_FILL_M   (AOTX_GEMM_TILE_M * AOTX_GEMM_SPAN_M / AOTX_GEMM_THREADS)
#define AOTX_GEMM_FILL_N   (AOTX_GEMM_TILE_N * AOTX_GEMM_SPAN_N / AOTX_GEMM_THREADS)

/* One tile of y, for one block type. The block type is a value of the launch. The switch
 * of the kernel picks the reader one time, and the inner loop holds no switch. */
template <unsigned int TYPE>
__device__ __forceinline__ void aotx_gemm_tile(const unsigned char *w, unsigned int n,
                                               unsigned int k, const half *x,
                                               unsigned int m, float *y,
                                               half *sx, half *sw)
{
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int warp_m = warp / AOTX_GEMM_WARPS_N;
    unsigned int warp_n = warp % AOTX_GEMM_WARPS_N;
    unsigned int m0 = blockIdx.y * AOTX_GEMM_TILE_M;
    unsigned int n0 = blockIdx.x * AOTX_GEMM_TILE_N;
    unsigned long long stride = aotx_matrix_row_bytes(TYPE, k);

    float acc[AOTX_GEMM_MMA_M][AOTX_GEMM_MMA_N][4];
#pragma unroll
    for (unsigned int mi = 0u; mi < AOTX_GEMM_MMA_M; ++mi) {
#pragma unroll
        for (unsigned int ni = 0u; ni < AOTX_GEMM_MMA_N; ++ni) {
#pragma unroll
            for (unsigned int e = 0u; e < 4u; ++e) {
                acc[mi][ni][e] = 0.0f;
            }
        }
    }

    unsigned int blocks = k / AOTX_MATRIX_BLOCK;
    for (unsigned int b = 0u; b < blocks; ++b) {
        /* Fill the activation tile, 8 values in one read. A row past the batch gives zero,
         * so the guard costs one compare and the tensor cores see a full tile. */
#pragma unroll
        for (unsigned int f = 0u; f < AOTX_GEMM_FILL_M; ++f) {
            unsigned int g = threadIdx.x + f * AOTX_GEMM_THREADS;
            unsigned int r = g / AOTX_GEMM_SPAN_M;
            unsigned int c = (g % AOTX_GEMM_SPAN_M) * AOTX_GEMM_WIDE;
            unsigned int row = m0 + r;
            float4 four = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (row < m) {
                four = *(const float4 *)(x + (size_t)row * k + b * AOTX_MATRIX_BLOCK + c);
            }
            *(float4 *)(&sx[r * AOTX_GEMM_LINE + c]) = four;
        }
        /* Fill the weight tile, 4 weights in one group. */
#pragma unroll
        for (unsigned int f = 0u; f < AOTX_GEMM_FILL_N; ++f) {
            unsigned int g = threadIdx.x + f * AOTX_GEMM_THREADS;
            unsigned int r = g / AOTX_GEMM_SPAN_N;
            unsigned int c = (g % AOTX_GEMM_SPAN_N) * AOTX_MATRIX_GROUP;
            unsigned int row = n0 + r;
            float raw[AOTX_MATRIX_GROUP] = { 0.0f, 0.0f, 0.0f, 0.0f };
            if (row < n) {
                aotx_matrix_group<TYPE>(w + stride * (unsigned long long)row, b, c, raw);
            }
            half pair[AOTX_MATRIX_GROUP];
#pragma unroll
            for (unsigned int v = 0u; v < AOTX_MATRIX_GROUP; ++v) {
                pair[v] = __float2half(raw[v]);
            }
            *(float2 *)(&sw[r * AOTX_GEMM_LINE + c]) = *(const float2 *)pair;
        }
        __syncthreads();

#pragma unroll
        for (unsigned int kk = 0u; kk < AOTX_GEMM_TILE_K; kk += AOTX_GEMM_STEP_K) {
            unsigned int a[AOTX_GEMM_MMA_M][4];
            unsigned int bb[AOTX_GEMM_MMA_N][2];
#pragma unroll
            for (unsigned int mi = 0u; mi < AOTX_GEMM_MMA_M; ++mi) {
                unsigned int row = warp_m * AOTX_GEMM_WARP_M + mi * AOTX_GEMM_STEP_M
                                 + (lane & 15u);
                unsigned int col = kk + (lane >> 4) * 8u;
                unsigned int at = (unsigned int)__cvta_generic_to_shared(
                    &sx[row * AOTX_GEMM_LINE + col]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                             "{%0,%1,%2,%3}, [%4];"
                             : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3])
                             : "r"(at));
            }
#pragma unroll
            for (unsigned int ni = 0u; ni < AOTX_GEMM_MMA_N; ++ni) {
                unsigned int row = warp_n * AOTX_GEMM_WARP_N + ni * AOTX_GEMM_STEP_N
                                 + (lane & 7u);
                unsigned int col = kk + ((lane >> 3) & 1u) * 8u;
                unsigned int at = (unsigned int)__cvta_generic_to_shared(
                    &sw[row * AOTX_GEMM_LINE + col]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(bb[ni][0]), "=r"(bb[ni][1])
                             : "r"(at));
            }
#pragma unroll
            for (unsigned int mi = 0u; mi < AOTX_GEMM_MMA_M; ++mi) {
#pragma unroll
                for (unsigned int ni = 0u; ni < AOTX_GEMM_MMA_N; ++ni) {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                                   "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                                 : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]),
                                   "r"(a[mi][3]), "r"(bb[ni][0]), "r"(bb[ni][1]));
                }
            }
        }
        __syncthreads();
    }

    /* The result of one operation holds four values in each lane. The values are two
     * columns of one row, and the same two columns of the row 8 lower. */
    unsigned int group = lane >> 2;
    unsigned int pair = (lane & 3u) * 2u;
#pragma unroll
    for (unsigned int mi = 0u; mi < AOTX_GEMM_MMA_M; ++mi) {
#pragma unroll
        for (unsigned int ni = 0u; ni < AOTX_GEMM_MMA_N; ++ni) {
            unsigned int row = m0 + warp_m * AOTX_GEMM_WARP_M + mi * AOTX_GEMM_STEP_M
                             + group;
            unsigned int col = n0 + warp_n * AOTX_GEMM_WARP_N + ni * AOTX_GEMM_STEP_N
                             + pair;
#pragma unroll
            for (unsigned int e = 0u; e < 2u; ++e) {
                if (col + e < n) {
                    if (row < m) {
                        y[(size_t)row * n + col + e] = acc[mi][ni][e];
                    }
                    if (row + 8u < m) {
                        y[(size_t)(row + 8u) * n + col + e] = acc[mi][ni][2u + e];
                    }
                }
            }
        }
    }
}

/* Take the block type of the launch and run the tile for that type. The two shared tiles
 * come from the kernel, so one kernel may hold two products. */
__device__ __forceinline__ void aotx_gemm_type(const void *w, unsigned int type,
                                               unsigned int n, unsigned int k,
                                               const half *x, unsigned int m, float *y,
                                               half *sx, half *sw)
{
    const unsigned char *base = (const unsigned char *)w;
    switch (type) {
    case AOTX_WEIGHT_Q8_0:
        aotx_gemm_tile<AOTX_WEIGHT_Q8_0>(base, n, k, x, m, y, sx, sw);
        break;
    case AOTX_WEIGHT_Q4_0:
        aotx_gemm_tile<AOTX_WEIGHT_Q4_0>(base, n, k, x, m, y, sx, sw);
        break;
    case AOTX_WEIGHT_Q4_K:
        aotx_gemm_tile<AOTX_WEIGHT_Q4_K>(base, n, k, x, m, y, sx, sw);
        break;
    case AOTX_WEIGHT_Q5_K:
        aotx_gemm_tile<AOTX_WEIGHT_Q5_K>(base, n, k, x, m, y, sx, sw);
        break;
    case AOTX_WEIGHT_Q6_K:
        aotx_gemm_tile<AOTX_WEIGHT_Q6_K>(base, n, k, x, m, y, sx, sw);
        break;
    case AOTX_WEIGHT_F16:
        aotx_gemm_tile<AOTX_WEIGHT_F16>(base, n, k, x, m, y, sx, sw);
        break;
    case AOTX_WEIGHT_F32:
        aotx_gemm_tile<AOTX_WEIGHT_F32>(base, n, k, x, m, y, sx, sw);
        break;
    default:
        break;
    }
}

#endif
