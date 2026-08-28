/* Purpose: Read one weight of a tensor in the block type that the model file gives.
 * Owns: Nothing; the weights region holds the bytes.
 * Launch shape: Device functions only; the caller gives the block and the thread.
 * Lifetime: From model load to the end of the run. */
#ifndef AOTX_MODEL_BLOCKS_CUH
#define AOTX_MODEL_BLOCKS_CUH

#include <cuda_fp16.h>

#include "model/model.cuh"

/* Weights of one quantized block, and the bytes that block takes. A Q8_0 block holds a half
 * scale and 32 signed bytes. A Q4_0 block holds a half scale and 32 nibbles in 16 bytes.
 * The low nibble of byte j gives weight j, and the high nibble gives weight j plus 16. */
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
    if (type == AOTX_WEIGHT_F16) {
        return count * 2ull;
    }
    return count * 4ull;
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
