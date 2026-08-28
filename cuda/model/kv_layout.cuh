/* Purpose: Give the place of the key and value state of one token in the page cache.
 * Owns: Nothing; the pages belong to the page cache.
 * Launch shape: Device functions only; the caller gives the block and the thread.
 * Lifetime: The whole run. */
#ifndef AOTX_KV_LAYOUT_CUH
#define AOTX_KV_LAYOUT_CUH

#include <cuda_fp16.h>

#include "kvcache/kvcache.cuh"

/* Positions of one layout block. A block holds the key rows of one layer and then the value
 * rows of the same layer. In each half the head comes first and the position comes second.
 * The rows of one head over a run of positions are therefore one run of bytes. Attention
 * reads exactly that run, which is why the position is not the first index. */
#define AOTX_KVL_BLOCK    16u

/* Bytes at the start of a page that the layout does not use. The page header of the cache
 * takes 16 bytes. The round up to 256 keeps every block on a 256 byte boundary. */
#define AOTX_KVL_HEADER   256u

/* The shape that the layout needs, from the model descriptor. The host glue and the device
 * both build one, so the two agree on the place of every row. */
typedef struct aotx_kvl_shape {
    unsigned int layers;
    unsigned int kv_heads;
    unsigned int head_dim;
    unsigned int head_bytes;   /* bytes of one head over one block of positions */
    unsigned int block_bytes;  /* bytes of one block: the keys and then the values */
    unsigned int blocks_page;  /* blocks that one page holds */
} aotx_kvl_shape;

/* Fill the shape. A model with a large head count can give a block that no page holds.
 * The blocks_page value is then zero, and every address call gives zero. */
__device__ __host__ __forceinline__ void aotx_kvl_make(aotx_kvl_shape *shape,
                                                       unsigned int layers,
                                                       unsigned int kv_heads,
                                                       unsigned int head_dim)
{
    shape->layers = layers;
    shape->kv_heads = kv_heads;
    shape->head_dim = head_dim;
    shape->head_bytes = AOTX_KVL_BLOCK * head_dim * (unsigned int)sizeof(half);
    shape->block_bytes = 2u * kv_heads * shape->head_bytes;
    shape->blocks_page = (shape->block_bytes == 0u)
        ? 0u
        : (unsigned int)((AOTX_KV_PAGE_BYTES - AOTX_KVL_HEADER) / shape->block_bytes);
}

/* Bytes that one token of one layer takes. */
__device__ __host__ __forceinline__ unsigned int aotx_kvl_token_bytes(const aotx_kvl_shape *shape)
{
    return 2u * shape->kv_heads * shape->head_dim * (unsigned int)sizeof(half);
}

/* Pages that a context of the given tokens needs. */
__device__ __host__ __forceinline__ unsigned int aotx_kvl_pages(const aotx_kvl_shape *shape,
                                                                unsigned int context)
{
    if (shape->blocks_page == 0u) {
        return 0u;
    }
    unsigned int blocks = ((context + AOTX_KVL_BLOCK - 1u) / AOTX_KVL_BLOCK) * shape->layers;
    return (blocks + shape->blocks_page - 1u) / shape->blocks_page;
}

/* The first row of one layout block, or a null pointer when the slot has no such page.
 * The block number counts the position blocks of a layer, and the layers of one position
 * block come one after the other. A sequence which grows takes the pages in order. */
__device__ __forceinline__ half *aotx_kvl_block(const aotx_kvl_shape *shape, unsigned int agent,
                                                unsigned int layer, unsigned int position)
{
    if (shape->blocks_page == 0u) {
        return 0;
    }
    unsigned int block = (position / AOTX_KVL_BLOCK) * shape->layers + layer;
    unsigned long long page = aotx_kv_page(agent, block / shape->blocks_page);
    if (page == 0ull) {
        return 0;
    }
    unsigned int at = (block % shape->blocks_page) * shape->block_bytes;
    return (half *)(page + (unsigned long long)AOTX_KVL_HEADER + (unsigned long long)at);
}

/* The key row of one position of one head, or a null pointer. */
__device__ __forceinline__ half *aotx_kvl_key(const aotx_kvl_shape *shape, unsigned int agent,
                                              unsigned int layer, unsigned int kv_head,
                                              unsigned int position)
{
    half *block = aotx_kvl_block(shape, agent, layer, position);
    if (block == 0) {
        return 0;
    }
    unsigned int row = kv_head * AOTX_KVL_BLOCK + (position % AOTX_KVL_BLOCK);
    return block + (unsigned long long)row * shape->head_dim;
}

/* The value row of one position of one head, or a null pointer. */
__device__ __forceinline__ half *aotx_kvl_value(const aotx_kvl_shape *shape, unsigned int agent,
                                                unsigned int layer, unsigned int kv_head,
                                                unsigned int position)
{
    half *block = aotx_kvl_block(shape, agent, layer, position);
    if (block == 0) {
        return 0;
    }
    unsigned int row = (shape->kv_heads + kv_head) * AOTX_KVL_BLOCK
                     + (position % AOTX_KVL_BLOCK);
    return block + (unsigned long long)row * shape->head_dim;
}

#endif
