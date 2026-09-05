/* Purpose: Give the block layouts and the weight readers that the matrix kernels share.
 * Owns: Nothing; the layout constants and the read functions only.
 * Launch shape: Device functions; the kernel that calls them sets the grid.
 * Lifetime: The whole run.
 *
 * The model file holds a quantized tensor as blocks of 32 weights. A Q8_0 block holds a
 * half scale and 32 signed bytes in 34 bytes. A Q4_0 block holds a half scale and 16 bytes
 * of nibble pairs in 18 bytes. The low nibble of byte b is weight b. The high nibble of
 * byte b is weight b plus 16. Each nibble takes away 8 before the scale multiplies it.
 *
 * A K type holds a super block of 256 weights, which is 8 blocks of 32. The reader takes
 * the block index of the kernels. The super block is the index over 8. The sub block is
 * the index modulo 8. The kernels therefore step by blocks of 32 for every type, and the
 * depth k of a K tensor is a multiple of 256.
 *
 * The reader is one function for each block type. A new type takes a new function and
 * leaves the matrix kernels as they are. */
#ifndef AOTX_MATRIX_CUH
#define AOTX_MATRIX_CUH

#include <cuda_fp16.h>

#include "model/model.cuh"

#define AOTX_MATRIX_BLOCK     32u   /* weights of one block of a quantized type */
#define AOTX_MATRIX_Q8_BYTES  34u   /* bytes of one Q8_0 block */
#define AOTX_MATRIX_Q4_BYTES  18u   /* bytes of one Q4_0 block */
#define AOTX_MATRIX_Q41_BYTES 20u
#define AOTX_MATRIX_Q50_BYTES 22u
#define AOTX_MATRIX_Q51_BYTES 24u
#define AOTX_MATRIX_HALF      2u    /* bytes of one half value */
#define AOTX_MATRIX_WORD      4u    /* bytes of one single precision value */
#define AOTX_MATRIX_NIBBLES   16u   /* nibble bytes of one Q4_0 block */
#define AOTX_MATRIX_ZERO      8     /* the value a Q4_0 nibble takes away */

/* The K types. A super block of 256 weights holds 8 sub blocks of 32. Q4_K holds two half
 * values d and dmin, 12 scale bytes and 128 nibble bytes. Q5_K holds the same and 32 high
 * bit bytes before the nibbles. Q6_K holds 128 nibble bytes, 64 high bit bytes, 16 signed
 * scale bytes and one half value d. The offsets below are bytes from the super block. */
#define AOTX_MATRIX_SUPER     256u  /* weights of one super block */
#define AOTX_MATRIX_SUBS      (AOTX_MATRIX_SUPER / AOTX_MATRIX_BLOCK)
#define AOTX_MATRIX_Q2K_BYTES 84u
#define AOTX_MATRIX_Q3K_BYTES 110u
#define AOTX_MATRIX_Q4K_BYTES 144u
#define AOTX_MATRIX_Q5K_BYTES 176u
#define AOTX_MATRIX_Q6K_BYTES 210u
#define AOTX_MATRIX_K_SCALES  4u    /* the 12 scale bytes of Q4_K and Q5_K */
#define AOTX_MATRIX_Q4K_QS    16u   /* the nibbles of Q4_K */
#define AOTX_MATRIX_Q5K_QH    16u   /* the high bits of Q5_K */
#define AOTX_MATRIX_Q5K_QS    48u   /* the nibbles of Q5_K */
#define AOTX_MATRIX_Q6K_QH    128u  /* the high bit pairs of Q6_K */
#define AOTX_MATRIX_Q6K_SCALES 192u /* the 16 signed scales of Q6_K */
#define AOTX_MATRIX_Q6K_D     208u  /* the half scale of Q6_K */
#define AOTX_MATRIX_Q6K_ZERO  32    /* the value a Q6_K code takes away */

/* The launch shape of the two matrix kernels. The caller sets the grid from these, so the
 * kernel and the caller keep one definition of the tile. The tensor core kernel takes one
 * block for each tile of y. The memory bound kernel takes one warp for each run of rows.
 * One pass over the weights applies to 8 rows of x. */
#define AOTX_GEMM_TILE_M   128u
#define AOTX_GEMM_TILE_N   64u
#define AOTX_GEMM_THREADS  256u
#define AOTX_GEMV_ROWS     2u
#define AOTX_GEMV_BATCH    8u
#define AOTX_GEMV_WARPS    8u
#define AOTX_GEMV_THREADS  (AOTX_GEMV_WARPS * 32u)
#define AOTX_GEMV_ROWS_CTA (AOTX_GEMV_WARPS * AOTX_GEMV_ROWS)

/* One thread reads 4 weights that follow each other. A read of a quantized block therefore
 * takes two 2-byte reads, and a read of activations takes one 8-byte read. The position of
 * the group in the block is a multiple of 4. */
#define AOTX_MATRIX_GROUP  4u
#define AOTX_MATRIX_LANES  (AOTX_MATRIX_BLOCK / AOTX_MATRIX_GROUP)

/* Bytes of one row of k weights. The count of blocks is exact, because k is a multiple of
 * the block size for every quantized type. */
__host__ __device__ inline unsigned long long aotx_matrix_row_bytes(unsigned int type,
                                                                    unsigned int k)
{
    switch (type) {
    case AOTX_WEIGHT_Q8_0:
        return (unsigned long long)(k / AOTX_MATRIX_BLOCK) * AOTX_MATRIX_Q8_BYTES;
    case AOTX_WEIGHT_Q4_0:
        return (unsigned long long)(k / AOTX_MATRIX_BLOCK) * AOTX_MATRIX_Q4_BYTES;
    case AOTX_WEIGHT_Q4_1:
        return (unsigned long long)(k / AOTX_MATRIX_BLOCK) * AOTX_MATRIX_Q41_BYTES;
    case AOTX_WEIGHT_Q5_0:
        return (unsigned long long)(k / AOTX_MATRIX_BLOCK) * AOTX_MATRIX_Q50_BYTES;
    case AOTX_WEIGHT_Q5_1:
        return (unsigned long long)(k / AOTX_MATRIX_BLOCK) * AOTX_MATRIX_Q51_BYTES;
    case AOTX_WEIGHT_Q2_K:
        return (unsigned long long)(k / AOTX_MATRIX_SUPER) * AOTX_MATRIX_Q2K_BYTES;
    case AOTX_WEIGHT_Q3_K:
        return (unsigned long long)(k / AOTX_MATRIX_SUPER) * AOTX_MATRIX_Q3K_BYTES;
    case AOTX_WEIGHT_Q4_K:
        return (unsigned long long)(k / AOTX_MATRIX_SUPER) * AOTX_MATRIX_Q4K_BYTES;
    case AOTX_WEIGHT_Q5_K:
        return (unsigned long long)(k / AOTX_MATRIX_SUPER) * AOTX_MATRIX_Q5K_BYTES;
    case AOTX_WEIGHT_Q6_K:
        return (unsigned long long)(k / AOTX_MATRIX_SUPER) * AOTX_MATRIX_Q6K_BYTES;
    case AOTX_WEIGHT_F16:
        return (unsigned long long)k * AOTX_MATRIX_HALF;
    case AOTX_WEIGHT_F32:
        return (unsigned long long)k * AOTX_MATRIX_WORD;
    default:
        return 0ull;
    }
}

/* Report whether the kernels read this block type. */
__host__ __device__ inline int aotx_matrix_known(unsigned int type)
{
    return type == AOTX_WEIGHT_Q8_0 || type == AOTX_WEIGHT_Q4_0
        || type == AOTX_WEIGHT_Q4_1 || type == AOTX_WEIGHT_Q5_0
        || type == AOTX_WEIGHT_Q5_1 || type == AOTX_WEIGHT_Q2_K
        || type == AOTX_WEIGHT_Q3_K
        || type == AOTX_WEIGHT_Q4_K || type == AOTX_WEIGHT_Q5_K
        || type == AOTX_WEIGHT_Q6_K
        || type == AOTX_WEIGHT_F16 || type == AOTX_WEIGHT_F32;
}

/* Read a half value at a byte address which holds a multiple of 2. */
__device__ __forceinline__ float aotx_matrix_half(const unsigned char *at)
{
    return __half2float(*(const __half *)at);
}

/* The scale and the offset of sub block j of a Q4_K or Q5_K super block, as whole
 * numbers of 6 bits. Bytes 0 to 3 hold the low 6 bits of the first 4 scales. Bytes 4 to
 * 7 hold the low 6 bits of the first 4 offsets.
 *
 * Bytes 8 to 11 hold the low 4 bits of the last 4 scales in their low nibble. The same
 * bytes hold the low 4 bits of the last 4 offsets in their high nibble. The high 2 bits of
 * bytes 0 to 3 complete the last 4 scales. The high 2 bits of bytes 4 to 7 complete the
 * last 4 offsets. */
__device__ __forceinline__ void aotx_matrix_k_scale(const unsigned char *scales,
                                                    unsigned int j, unsigned int *sc,
                                                    unsigned int *m)
{
    if (j < 4u) {
        *sc = scales[j] & 63u;
        *m = scales[j + 4u] & 63u;
    } else {
        *sc = (scales[j + 4u] & 0x0Fu) | ((scales[j - 4u] >> 6) << 4);
        *m = (scales[j + 4u] >> 4) | ((scales[j] >> 6) << 4);
    }
}

/* One sub block of a K super block, as the reader sees it. It holds the scale and the
 * offset of the weights, the first code byte of the group, and the first high bit byte.
 * It holds the shifts that take the code and the high bits out of their bytes, and the
 * count of high bits. The two readers of a type take this one description, so the layout
 * is written one time for each type. */
typedef struct aotx_matrix_k_sub {
    float scale;
    float offset;
    const unsigned char *qs;
    const unsigned char *qh;
    unsigned int shift;
    unsigned int high;
    unsigned int bits;
} aotx_matrix_k_sub;

/* One weight of Q4_K or Q5_K from its code: the scale times the code, less the offset.
 * The one rounding of the fused operation gives the single value reader and the group
 * reader the same weight. */
__device__ __forceinline__ float aotx_matrix_k_weight(const aotx_matrix_k_sub *sub,
                                                      unsigned int q)
{
    return __fmaf_rn(sub->scale, (float)q, -sub->offset);
}

/* Q4_K: the two half values d and dmin, the 12 scale bytes, and 128 nibble bytes in 4
 * runs of 32. Run c holds sub block 2c in its low nibbles and sub block 2c plus 1 in its
 * high nibbles. The weight is d times the scale times the nibble, less dmin times the
 * offset. */
__device__ __forceinline__ void aotx_matrix_q4k_sub(const unsigned char *one, unsigned int j,
                                                    unsigned int at, aotx_matrix_k_sub *sub)
{
    unsigned int sc;
    unsigned int m;
    aotx_matrix_k_scale(one + AOTX_MATRIX_K_SCALES, j, &sc, &m);
    sub->scale = aotx_matrix_half(one) * (float)sc;
    sub->offset = aotx_matrix_half(one + AOTX_MATRIX_HALF) * (float)m;
    sub->qs = one + AOTX_MATRIX_Q4K_QS + (j >> 1) * AOTX_MATRIX_BLOCK + at;
    sub->qh = 0;
    sub->shift = (j & 1u) * 4u;
    sub->high = 0u;
    sub->bits = 0u;
}

/* Q5_K: as Q4_K, with 32 high bit bytes before the nibbles. High bit byte l holds bit j
 * for weight l of sub block j, and that bit adds 16 to the code. */
__device__ __forceinline__ void aotx_matrix_q5k_sub(const unsigned char *one, unsigned int j,
                                                    unsigned int at, aotx_matrix_k_sub *sub)
{
    unsigned int sc;
    unsigned int m;
    aotx_matrix_k_scale(one + AOTX_MATRIX_K_SCALES, j, &sc, &m);
    sub->scale = aotx_matrix_half(one) * (float)sc;
    sub->offset = aotx_matrix_half(one + AOTX_MATRIX_HALF) * (float)m;
    sub->qs = one + AOTX_MATRIX_Q5K_QS + (j >> 1) * AOTX_MATRIX_BLOCK + at;
    sub->qh = one + AOTX_MATRIX_Q5K_QH + at;
    sub->shift = (j & 1u) * 4u;
    sub->high = j;
    sub->bits = 1u;
}

/* Q6_K: 128 nibble bytes, 64 high bit bytes, 16 signed scale bytes and one half value d.
 * The super block holds two halves of 128 weights, and a half holds 4 sub blocks. In a
 * half, the 64 nibble bytes hold sub blocks 0 and 1 in their low nibbles. They hold sub
 * blocks 2 and 3 in their high nibbles.
 *
 * The 32 high bit bytes of the half give 2 bits to weight l of each sub block. Sub block 0
 * takes the low pair and sub block 3 the high pair. Each run of 16 weights takes one
 * signed scale byte, so the scale of a group depends on its position. The weight is d
 * times the scale times the code less 32. */
__device__ __forceinline__ void aotx_matrix_q6k_sub(const unsigned char *one, unsigned int j,
                                                    unsigned int at, aotx_matrix_k_sub *sub)
{
    unsigned int chunk = j >> 2;
    unsigned int quarter = j & 3u;
    int sc = (int)(signed char)one[AOTX_MATRIX_Q6K_SCALES + j * 2u + (at >> 4)];
    sub->scale = aotx_matrix_half(one + AOTX_MATRIX_Q6K_D) * (float)sc;
    sub->offset = sub->scale * (float)AOTX_MATRIX_Q6K_ZERO;
    sub->qs = one + chunk * 2u * AOTX_MATRIX_BLOCK + (quarter & 1u) * AOTX_MATRIX_BLOCK + at;
    sub->qh = one + AOTX_MATRIX_Q6K_QH + chunk * AOTX_MATRIX_BLOCK + at;
    sub->shift = (quarter >> 1) * 4u;
    sub->high = quarter * 2u;
    sub->bits = 2u;
}

/* The code of weight v of a group from the packed bytes. The low bytes hold the nibbles
 * and the high bytes hold the high bits, byte v for weight v. Q4_K gives no high bits.
 * Q5_K gives 1 bit, and Q6_K gives 2. */
__device__ __forceinline__ unsigned int aotx_matrix_k_code(const aotx_matrix_k_sub *sub,
                                                           unsigned int low,
                                                           unsigned int high,
                                                           unsigned int v)
{
    unsigned int q = (low >> (8u * v + sub->shift)) & 0x0Fu;
    unsigned int mask = (1u << sub->bits) - 1u;
    return q | (((high >> (8u * v + sub->high)) & mask) << 4);
}

/* Read 4 bytes at an address which holds a multiple of 2. A Q6_K super block takes 210
 * bytes, so a super block after the first lies on an odd multiple of 2. */
__device__ __forceinline__ unsigned int aotx_matrix_bytes4(const unsigned char *at)
{
    unsigned int low = *(const unsigned short *)at;
    unsigned int high = *(const unsigned short *)(at + 2u);
    return low | (high << 16);
}
/* Legacy affine blocks hold a minimum after d. Q5 blocks hold one high bit per weight. */
template <unsigned int TYPE>
__device__ __forceinline__ void aotx_matrix_legacy_sub(const unsigned char *row,
    unsigned int block, unsigned int at, aotx_matrix_k_sub *sub)
{
    const unsigned int bytes = TYPE == AOTX_WEIGHT_Q4_1 ? AOTX_MATRIX_Q41_BYTES
        : (TYPE == AOTX_WEIGHT_Q5_0 ? AOTX_MATRIX_Q50_BYTES : AOTX_MATRIX_Q51_BYTES);
    const unsigned int head = TYPE == AOTX_WEIGHT_Q5_0 ? 2u : 4u;
    const unsigned char *one = row + (size_t)block * bytes;
    const unsigned int qs = head + (TYPE == AOTX_WEIGHT_Q4_1 ? 0u : 4u);
    sub->qs = one + qs + (at & 15u);
    sub->qh = TYPE == AOTX_WEIGHT_Q4_1 ? 0 : one + head;
    sub->shift = (at >> 4) * 4u;
    sub->high = at;
    sub->scale = aotx_matrix_half(one);
    sub->offset = TYPE == AOTX_WEIGHT_Q5_0 ? 0.0f : aotx_matrix_half(one + 2u);
}

template <unsigned int TYPE>
__device__ __forceinline__ float aotx_matrix_legacy_weight(const aotx_matrix_k_sub *sub,
    unsigned int low, unsigned int high, unsigned int v)
{
    unsigned int q = (low >> (8u * v + sub->shift)) & 15u;
    if (TYPE != AOTX_WEIGHT_Q4_1) {
        q |= ((high >> (sub->high + v)) & 1u) << 4;
    }
    if (TYPE == AOTX_WEIGHT_Q5_0) {
        return sub->scale * (float)((int)q - 16);
    }
    return __fmaf_rn(sub->scale, (float)q, sub->offset);
}

template <unsigned int TYPE>
__device__ __forceinline__ float aotx_matrix_legacy(const unsigned char *row,
    unsigned int block, unsigned int at)
{
    aotx_matrix_k_sub sub;
    aotx_matrix_legacy_sub<TYPE>(row, block, at, &sub);
    unsigned int high = sub.qh != 0 ? aotx_matrix_bytes4(sub.qh) : 0u;
    return aotx_matrix_legacy_weight<TYPE>(&sub, *sub.qs, high, 0u);
}

template <unsigned int TYPE>
__device__ __forceinline__ void aotx_matrix_legacy_group(const unsigned char *row,
    unsigned int block, unsigned int at, float *out)
{
    aotx_matrix_k_sub sub;
    aotx_matrix_legacy_sub<TYPE>(row, block, at, &sub);
    unsigned int low = aotx_matrix_bytes4(sub.qs);
    unsigned int high = sub.qh != 0 ? aotx_matrix_bytes4(sub.qh) : 0u;
#pragma unroll
    for (unsigned int v = 0u; v < AOTX_MATRIX_GROUP; ++v) {
        out[v] = aotx_matrix_legacy_weight<TYPE>(&sub, low, high, v);
    }
}

/* Two-bit codes use one byte for each column of a 32-weight sub block.
 * Each aligned group of four weights stays within one 16-weight scale. */
template <unsigned int TYPE>
__device__ __forceinline__ void aotx_matrix_low_sub(const unsigned char *row,
    unsigned int block, unsigned int at, aotx_matrix_k_sub *sub)
{
    const unsigned int bytes = TYPE == AOTX_WEIGHT_Q2_K
        ? AOTX_MATRIX_Q2K_BYTES : AOTX_MATRIX_Q3K_BYTES;
    const unsigned char *one = row + (size_t)(block / AOTX_MATRIX_SUBS) * bytes;
    unsigned int j = block % AOTX_MATRIX_SUBS;
    unsigned int s = j * 2u + (at >> 4);
    sub->shift = (j & 3u) * 2u;
    sub->high = j;
    sub->qs = one + (TYPE == AOTX_WEIGHT_Q2_K ? 16u : 32u) + (j >> 2) * 32u + at;
    sub->qh = TYPE == AOTX_WEIGHT_Q2_K ? 0 : one + at;
    if (TYPE == AOTX_WEIGHT_Q2_K) {
        sub->scale = aotx_matrix_half(one + 80u) * (float)(one[s] & 15u);
        sub->offset = aotx_matrix_half(one + 82u) * (float)(one[s] >> 4);
    } else {
        unsigned int low = (one[96u + (s & 7u)] >> ((s >> 3) * 4u)) & 15u;
        unsigned int high = (one[104u + (s & 3u)] >> ((s >> 2) * 2u)) & 3u;
        sub->scale = aotx_matrix_half(one + 108u) * (float)((int)(low | (high << 4)) - 32);
        sub->offset = 0.0f;
    }
}

template <unsigned int TYPE>
__device__ __forceinline__ float aotx_matrix_low_weight(const aotx_matrix_k_sub *sub,
    unsigned int low, unsigned int high, unsigned int v)
{
    int q = (int)((low >> (8u * v + sub->shift)) & 3u);
    if (TYPE == AOTX_WEIGHT_Q3_K) {
        q -= ((high >> (8u * v + sub->high)) & 1u) ? 0 : 4;
    }
    return __fmaf_rn(sub->scale, (float)q, -sub->offset);
}

/* The sub block of a K type at block index of the kernels. The super block is the index
 * over 8 and the sub block is the index modulo 8. */
template <unsigned int TYPE>
__device__ __forceinline__ void aotx_matrix_k_sub_of(const unsigned char *row,
                                                     unsigned int block, unsigned int at,
                                                     aotx_matrix_k_sub *sub);

template <>
__device__ __forceinline__ void aotx_matrix_k_sub_of<AOTX_WEIGHT_Q4_K>(
    const unsigned char *row, unsigned int block, unsigned int at, aotx_matrix_k_sub *sub)
{
    const unsigned char *one = row + (size_t)(block / AOTX_MATRIX_SUBS) * AOTX_MATRIX_Q4K_BYTES;
    aotx_matrix_q4k_sub(one, block % AOTX_MATRIX_SUBS, at, sub);
}

template <>
__device__ __forceinline__ void aotx_matrix_k_sub_of<AOTX_WEIGHT_Q5_K>(
    const unsigned char *row, unsigned int block, unsigned int at, aotx_matrix_k_sub *sub)
{
    const unsigned char *one = row + (size_t)(block / AOTX_MATRIX_SUBS) * AOTX_MATRIX_Q5K_BYTES;
    aotx_matrix_q5k_sub(one, block % AOTX_MATRIX_SUBS, at, sub);
}

template <>
__device__ __forceinline__ void aotx_matrix_k_sub_of<AOTX_WEIGHT_Q6_K>(
    const unsigned char *row, unsigned int block, unsigned int at, aotx_matrix_k_sub *sub)
{
    const unsigned char *one = row + (size_t)(block / AOTX_MATRIX_SUBS) * AOTX_MATRIX_Q6K_BYTES;
    aotx_matrix_q6k_sub(one, block % AOTX_MATRIX_SUBS, at, sub);
}

/* One weight of a K type. */
template <unsigned int TYPE>
__device__ __forceinline__ half aotx_matrix_k_value(const unsigned char *row,
                                                    unsigned int block, unsigned int at)
{
    aotx_matrix_k_sub sub;
    aotx_matrix_k_sub_of<TYPE>(row, block, at, &sub);
    unsigned int low = *sub.qs;
    unsigned int high = (sub.qh != 0) ? *sub.qh : 0u;
    return __float2half(aotx_matrix_k_weight(&sub, aotx_matrix_k_code(&sub, low, high, 0u)));
}

/* A group of 4 weights of a K type. The scale unpack runs one time for the 4. */
template <unsigned int TYPE>
__device__ __forceinline__ void aotx_matrix_k_group(const unsigned char *row,
                                                    unsigned int block, unsigned int at,
                                                    float *out)
{
    aotx_matrix_k_sub sub;
    aotx_matrix_k_sub_of<TYPE>(row, block, at, &sub);
    unsigned int low = aotx_matrix_bytes4(sub.qs);
    unsigned int high = (sub.qh != 0) ? aotx_matrix_bytes4(sub.qh) : 0u;
#pragma unroll
    for (unsigned int v = 0u; v < AOTX_MATRIX_GROUP; ++v) {
        out[v] = aotx_matrix_k_weight(&sub, aotx_matrix_k_code(&sub, low, high, v));
    }
}

/* Read one weight of one row. The row pointer is the first byte of the row. The block is
 * the index of the block of 32 weights, and at is the position in that block. A type
 * which holds no blocks takes the two together as one column index. */
template <unsigned int TYPE>
__device__ __forceinline__ half aotx_matrix_value(const unsigned char *row,
                                                  unsigned int block, unsigned int at);

template <>
__device__ __forceinline__ half aotx_matrix_value<AOTX_WEIGHT_Q8_0>(const unsigned char *row,
                                                                    unsigned int block,
                                                                    unsigned int at)
{
    const unsigned char *one = row + (size_t)block * AOTX_MATRIX_Q8_BYTES;
    float scale = __half2float(*(const __half *)one);
    int q = (int)(signed char)one[AOTX_MATRIX_HALF + at];
    return __float2half((float)q * scale);
}

template <>
__device__ __forceinline__ half aotx_matrix_value<AOTX_WEIGHT_Q4_0>(const unsigned char *row,
                                                                    unsigned int block,
                                                                    unsigned int at)
{
    const unsigned char *one = row + (size_t)block * AOTX_MATRIX_Q4_BYTES;
    float scale = __half2float(*(const __half *)one);
    unsigned int pair = one[AOTX_MATRIX_HALF + (at & (AOTX_MATRIX_NIBBLES - 1u))];
    unsigned int nibble = (at < AOTX_MATRIX_NIBBLES) ? (pair & 0x0Fu) : (pair >> 4);
    return __float2half(((float)(int)nibble - (float)AOTX_MATRIX_ZERO) * scale);
}

template <>
__device__ __forceinline__ half aotx_matrix_value<AOTX_WEIGHT_F16>(const unsigned char *row,
                                                                   unsigned int block,
                                                                   unsigned int at)
{
    const __half *one = (const __half *)row;
    return one[(size_t)block * AOTX_MATRIX_BLOCK + at];
}

template <>
__device__ __forceinline__ half aotx_matrix_value<AOTX_WEIGHT_F32>(const unsigned char *row,
                                                                   unsigned int block,
                                                                   unsigned int at)
{
    const float *one = (const float *)row;
    return __float2half(one[(size_t)block * AOTX_MATRIX_BLOCK + at]);
}

template <>
__device__ __forceinline__ half aotx_matrix_value<AOTX_WEIGHT_Q4_K>(const unsigned char *row,
                                                                    unsigned int block,
                                                                    unsigned int at)
{
    return aotx_matrix_k_value<AOTX_WEIGHT_Q4_K>(row, block, at);
}

template <>
__device__ __forceinline__ half aotx_matrix_value<AOTX_WEIGHT_Q5_K>(const unsigned char *row,
                                                                    unsigned int block,
                                                                    unsigned int at)
{
    return aotx_matrix_k_value<AOTX_WEIGHT_Q5_K>(row, block, at);
}

template <>
__device__ __forceinline__ half aotx_matrix_value<AOTX_WEIGHT_Q6_K>(const unsigned char *row,
                                                                    unsigned int block,
                                                                    unsigned int at)
{
    return aotx_matrix_k_value<AOTX_WEIGHT_Q6_K>(row, block, at);
}

/* Read 4 half values that follow each other. The address holds a multiple of 8 bytes. */
__device__ __forceinline__ void aotx_matrix_four(const half *from, float *out)
{
    float2 raw = *(const float2 *)from;
    unsigned int low = __float_as_uint(raw.x);
    unsigned int high = __float_as_uint(raw.y);
    out[0] = __half2float(__ushort_as_half((unsigned short)(low & 0xFFFFu)));
    out[1] = __half2float(__ushort_as_half((unsigned short)(low >> 16)));
    out[2] = __half2float(__ushort_as_half((unsigned short)(high & 0xFFFFu)));
    out[3] = __half2float(__ushort_as_half((unsigned short)(high >> 16)));
}

/* Read a group of 4 weights of one row. The position at is a multiple of 4 and stays in
 * one block. A group takes fewer read operations than 4 single values. The scale of the
 * block is read one time. */
template <unsigned int TYPE>
__device__ __forceinline__ void aotx_matrix_group(const unsigned char *row,
                                                  unsigned int block, unsigned int at,
                                                  float *out);

template <>
__device__ __forceinline__ void aotx_matrix_group<AOTX_WEIGHT_Q8_0>(const unsigned char *row,
                                                                    unsigned int block,
                                                                    unsigned int at,
                                                                    float *out)
{
    const unsigned char *one = row + (size_t)block * AOTX_MATRIX_Q8_BYTES;
    float scale = __half2float(*(const __half *)one);
    unsigned int low = *(const unsigned short *)(one + AOTX_MATRIX_HALF + at);
    unsigned int high = *(const unsigned short *)(one + AOTX_MATRIX_HALF + at + 2u);
    out[0] = (float)(int)(signed char)(low & 0xFFu) * scale;
    out[1] = (float)(int)(signed char)(low >> 8) * scale;
    out[2] = (float)(int)(signed char)(high & 0xFFu) * scale;
    out[3] = (float)(int)(signed char)(high >> 8) * scale;
}

template <>
__device__ __forceinline__ void aotx_matrix_group<AOTX_WEIGHT_Q4_0>(const unsigned char *row,
                                                                    unsigned int block,
                                                                    unsigned int at,
                                                                    float *out)
{
    const unsigned char *one = row + (size_t)block * AOTX_MATRIX_Q4_BYTES;
    float scale = __half2float(*(const __half *)one);
    unsigned int byte = at & (AOTX_MATRIX_NIBBLES - 1u);
    unsigned int low = *(const unsigned short *)(one + AOTX_MATRIX_HALF + byte);
    unsigned int high = *(const unsigned short *)(one + AOTX_MATRIX_HALF + byte + 2u);
    unsigned int shift = (at < AOTX_MATRIX_NIBBLES) ? 0u : 4u;
    out[0] = ((float)(int)((low >> shift) & 0x0Fu) - (float)AOTX_MATRIX_ZERO) * scale;
    out[1] = ((float)(int)((low >> (8u + shift)) & 0x0Fu) - (float)AOTX_MATRIX_ZERO) * scale;
    out[2] = ((float)(int)((high >> shift) & 0x0Fu) - (float)AOTX_MATRIX_ZERO) * scale;
    out[3] = ((float)(int)((high >> (8u + shift)) & 0x0Fu) - (float)AOTX_MATRIX_ZERO) * scale;
}

template <>
__device__ __forceinline__ void aotx_matrix_group<AOTX_WEIGHT_F16>(const unsigned char *row,
                                                                   unsigned int block,
                                                                   unsigned int at,
                                                                   float *out)
{
    aotx_matrix_four((const half *)row + (size_t)block * AOTX_MATRIX_BLOCK + at, out);
}

template <>
__device__ __forceinline__ void aotx_matrix_group<AOTX_WEIGHT_F32>(const unsigned char *row,
                                                                   unsigned int block,
                                                                   unsigned int at,
                                                                   float *out)
{
    const float4 raw = *(const float4 *)((const float *)row
                                         + (size_t)block * AOTX_MATRIX_BLOCK + at);
    out[0] = raw.x;
    out[1] = raw.y;
    out[2] = raw.z;
    out[3] = raw.w;
}

template <>
__device__ __forceinline__ void aotx_matrix_group<AOTX_WEIGHT_Q4_K>(const unsigned char *row,
                                                                    unsigned int block,
                                                                    unsigned int at,
                                                                    float *out)
{
    aotx_matrix_k_group<AOTX_WEIGHT_Q4_K>(row, block, at, out);
}

template <>
__device__ __forceinline__ void aotx_matrix_group<AOTX_WEIGHT_Q5_K>(const unsigned char *row,
                                                                    unsigned int block,
                                                                    unsigned int at,
                                                                    float *out)
{
    aotx_matrix_k_group<AOTX_WEIGHT_Q5_K>(row, block, at, out);
}

template <>
__device__ __forceinline__ void aotx_matrix_group<AOTX_WEIGHT_Q6_K>(const unsigned char *row,
                                                                    unsigned int block,
                                                                    unsigned int at,
                                                                    float *out)
{
    aotx_matrix_k_group<AOTX_WEIGHT_Q6_K>(row, block, at, out);
}

template <>
__device__ __forceinline__ half aotx_matrix_value<AOTX_WEIGHT_Q4_1>(
    const unsigned char *row, unsigned int block, unsigned int at)
{
    return __float2half(aotx_matrix_legacy<AOTX_WEIGHT_Q4_1>(row, block, at));
}

template <>
__device__ __forceinline__ void aotx_matrix_group<AOTX_WEIGHT_Q4_1>(
    const unsigned char *row, unsigned int block, unsigned int at, float *out)
{
    aotx_matrix_legacy_group<AOTX_WEIGHT_Q4_1>(row, block, at, out);
}

template <>
__device__ __forceinline__ half aotx_matrix_value<AOTX_WEIGHT_Q5_0>(
    const unsigned char *row, unsigned int block, unsigned int at)
{
    return __float2half(aotx_matrix_legacy<AOTX_WEIGHT_Q5_0>(row, block, at));
}

template <>
__device__ __forceinline__ void aotx_matrix_group<AOTX_WEIGHT_Q5_0>(
    const unsigned char *row, unsigned int block, unsigned int at, float *out)
{
    aotx_matrix_legacy_group<AOTX_WEIGHT_Q5_0>(row, block, at, out);
}

template <>
__device__ __forceinline__ half aotx_matrix_value<AOTX_WEIGHT_Q5_1>(
    const unsigned char *row, unsigned int block, unsigned int at)
{
    return __float2half(aotx_matrix_legacy<AOTX_WEIGHT_Q5_1>(row, block, at));
}

template <>
__device__ __forceinline__ void aotx_matrix_group<AOTX_WEIGHT_Q5_1>(
    const unsigned char *row, unsigned int block, unsigned int at, float *out)
{
    aotx_matrix_legacy_group<AOTX_WEIGHT_Q5_1>(row, block, at, out);
}

template <>
__device__ __forceinline__ half aotx_matrix_value<AOTX_WEIGHT_Q2_K>(
    const unsigned char *row, unsigned int block, unsigned int at)
{
    aotx_matrix_k_sub sub;
    aotx_matrix_low_sub<AOTX_WEIGHT_Q2_K>(row, block, at, &sub);
    unsigned int high = sub.qh != 0 ? *sub.qh : 0u;
    return __float2half(aotx_matrix_low_weight<AOTX_WEIGHT_Q2_K>(&sub, *sub.qs, high, 0u));
}

template <>
__device__ __forceinline__ void aotx_matrix_group<AOTX_WEIGHT_Q2_K>(
    const unsigned char *row, unsigned int block, unsigned int at, float *out)
{
    aotx_matrix_k_sub sub;
    aotx_matrix_low_sub<AOTX_WEIGHT_Q2_K>(row, block, at, &sub);
    unsigned int low = aotx_matrix_bytes4(sub.qs);
    unsigned int high = sub.qh != 0 ? aotx_matrix_bytes4(sub.qh) : 0u;
#pragma unroll
    for (unsigned int v = 0u; v < AOTX_MATRIX_GROUP; ++v) {
        out[v] = aotx_matrix_low_weight<AOTX_WEIGHT_Q2_K>(&sub, low, high, v);
    }
}

template <>
__device__ __forceinline__ half aotx_matrix_value<AOTX_WEIGHT_Q3_K>(
    const unsigned char *row, unsigned int block, unsigned int at)
{
    aotx_matrix_k_sub sub;
    aotx_matrix_low_sub<AOTX_WEIGHT_Q3_K>(row, block, at, &sub);
    unsigned int high = sub.qh != 0 ? *sub.qh : 0u;
    return __float2half(aotx_matrix_low_weight<AOTX_WEIGHT_Q3_K>(&sub, *sub.qs, high, 0u));
}

template <>
__device__ __forceinline__ void aotx_matrix_group<AOTX_WEIGHT_Q3_K>(
    const unsigned char *row, unsigned int block, unsigned int at, float *out)
{
    aotx_matrix_k_sub sub;
    aotx_matrix_low_sub<AOTX_WEIGHT_Q3_K>(row, block, at, &sub);
    unsigned int low = aotx_matrix_bytes4(sub.qs);
    unsigned int high = sub.qh != 0 ? aotx_matrix_bytes4(sub.qh) : 0u;
#pragma unroll
    for (unsigned int v = 0u; v < AOTX_MATRIX_GROUP; ++v) {
        out[v] = aotx_matrix_low_weight<AOTX_WEIGHT_Q3_K>(&sub, low, high, v);
    }
}

/* Read the weight at column j of one row. */
template <unsigned int TYPE>
__device__ __forceinline__ half aotx_matrix_weight(const unsigned char *row, unsigned int j)
{
    return aotx_matrix_value<TYPE>(row, j / AOTX_MATRIX_BLOCK, j % AOTX_MATRIX_BLOCK);
}

#endif
