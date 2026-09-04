/* Purpose: Read one weight of a tensor in the block type that the model file gives.
 * Owns: Nothing; the weights region holds the bytes.
 * Launch shape: Device functions only; the caller gives the block and the thread.
 * Lifetime: From model load to the end of the run. */
#ifndef AOTX_MODEL_BLOCKS_CUH
#define AOTX_MODEL_BLOCKS_CUH

#include <cuda_fp16.h>

#include "model/matrix.cuh"

/* Weights of one quantized block, and the bytes that block takes. A Q8_0 block holds a half
 * scale and 32 signed bytes. A Q4_0 block holds a half scale and 32 nibbles in 16 bytes.
 * The low nibble of byte j gives weight j, and the high nibble gives weight j plus 16. A K
 * type holds a super block of 256 weights; the readers of the matrix module give its
 * layout. */
#define AOTX_BLOCK_WIDTH   32u
#define AOTX_BLOCK_Q8_0    34u
#define AOTX_BLOCK_Q4_0    18u

/* The bytes that a tensor of the given type and element count takes. */
__device__ __host__ __forceinline__ unsigned long long aotx_block_bytes(unsigned int type,
                                                                        unsigned long long count)
{
    if (type == AOTX_WEIGHT_Q8_0) {
        return count / AOTX_BLOCK_WIDTH * AOTX_BLOCK_Q8_0;
    }
    if (type == AOTX_WEIGHT_Q4_0) {
        return count / AOTX_BLOCK_WIDTH * AOTX_BLOCK_Q4_0;
    }
    if (type == AOTX_WEIGHT_Q4_K) {
        return count / AOTX_MATRIX_SUPER * AOTX_MATRIX_Q4K_BYTES;
    }
    if (type == AOTX_WEIGHT_Q5_K) {
        return count / AOTX_MATRIX_SUPER * AOTX_MATRIX_Q5K_BYTES;
    }
    if (type == AOTX_WEIGHT_Q6_K) {
        return count / AOTX_MATRIX_SUPER * AOTX_MATRIX_Q6K_BYTES;
    }
    if (type == AOTX_WEIGHT_F16) {
        return count * 2ull;
    }
    return count * 4ull;
}

/* One weight of a K type by its position in the whole tensor. The rows are whole super
 * blocks, so the super block is the position over 256 and the sub block is the rest over
 * 32. The super block address is computed in 64 bits, because a large embedding holds more
 * blocks than 32 bits count. */
__device__ __forceinline__ float aotx_block_k(const void *weights, unsigned int type,
                                              unsigned long long index)
{
    unsigned int in = (unsigned int)(index % AOTX_MATRIX_SUPER);
    unsigned int j = in / AOTX_BLOCK_WIDTH;
    unsigned int at = in % AOTX_BLOCK_WIDTH;
    unsigned long long super = index / AOTX_MATRIX_SUPER;
    const unsigned char *base = (const unsigned char *)weights;
    aotx_matrix_k_sub sub;
    if (type == AOTX_WEIGHT_Q4_K) {
        aotx_matrix_q4k_sub(base + super * AOTX_MATRIX_Q4K_BYTES, j, at, &sub);
    } else if (type == AOTX_WEIGHT_Q5_K) {
        aotx_matrix_q5k_sub(base + super * AOTX_MATRIX_Q5K_BYTES, j, at, &sub);
    } else {
        aotx_matrix_q6k_sub(base + super * AOTX_MATRIX_Q6K_BYTES, j, at, &sub);
    }
    unsigned int low = *sub.qs;
    unsigned int high = (sub.qh != 0) ? *sub.qh : 0u;
    return aotx_matrix_k_weight(&sub, aotx_matrix_k_code(&sub, low, high, 0u));
}

/* One weight of a tensor, by the position of the weight in the whole tensor. The tensor is
 * row major, so the position of element i of row r is r times the row length plus i. */
__device__ __forceinline__ float aotx_block_at(const void *weights, unsigned int type,
                                               unsigned long long index)
{
    if (type == AOTX_WEIGHT_Q8_0) {
        const unsigned char *block = (const unsigned char *)weights
                                   + (index / AOTX_BLOCK_WIDTH) * AOTX_BLOCK_Q8_0;
        half scale = __ushort_as_half((unsigned short)(block[0] | (block[1] << 8)));
        int q = (int)(signed char)block[2u + (unsigned int)(index % AOTX_BLOCK_WIDTH)];
        return __half2float(scale) * (float)q;
    }
    if (type == AOTX_WEIGHT_Q4_0) {
        const unsigned char *block = (const unsigned char *)weights
                                   + (index / AOTX_BLOCK_WIDTH) * AOTX_BLOCK_Q4_0;
        half scale = __ushort_as_half((unsigned short)(block[0] | (block[1] << 8)));
        unsigned int at = (unsigned int)(index % AOTX_BLOCK_WIDTH);
        unsigned char byte = block[2u + (at & 15u)];
        int q = (at < 16u) ? (int)(byte & 15u) : (int)(byte >> 4);
        return __half2float(scale) * (float)(q - 8);
    }
    if (type == AOTX_WEIGHT_Q4_K || type == AOTX_WEIGHT_Q5_K || type == AOTX_WEIGHT_Q6_K) {
        return aotx_block_k(weights, type, index);
    }
    if (type == AOTX_WEIGHT_F16) {
        return __half2float(((const half *)weights)[index]);
    }
    return ((const float *)weights)[index];
}

/* The first byte of a tensor in the weights region, or a null pointer when the model does
 * not have it. The offset in the descriptor counts from the start of the region. */
__device__ __forceinline__ const void *aotx_block_tensor(unsigned long long base,
                                                         unsigned long long offset)
{
    if (offset == AOTX_MODEL_ABSENT) {
        return 0;
    }
    return (const void *)(base + offset);
}

#endif
