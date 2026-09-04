/* Purpose: Give the place of the key and value state of one token in the page cache.
 * Owns: Nothing; the pages belong to the page cache.
 * Launch shape: Device functions only; the caller gives the block and the thread.
 * Lifetime: The whole run. */
#ifndef AOTX_KV_LAYOUT_CUH
#define AOTX_KV_LAYOUT_CUH

#include <cuda_fp16.h>

#include "model/kinds.h"
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
    unsigned int state_layers;
    unsigned int kv_heads;
    unsigned int head_dim;
    unsigned int head_bytes;   /* bytes of one head over one block of positions */
    unsigned int block_bytes;  /* bytes of one block: the keys and then the values */
    unsigned int blocks_page;  /* blocks that one page holds */
    unsigned char state_layer[AOTX_MODEL_MAX_LAYERS]; /* compact layer, or 0xff */
} aotx_kvl_shape;

/* Fill the shape. A model with a large head count can give a block that no page holds.
 * The blocks_page value is then zero, and every address call gives zero. */
__device__ __host__ __forceinline__ void aotx_kvl_make(aotx_kvl_shape *shape,
                                                       unsigned int layers,
                                                       unsigned int kv_heads,
                                                       unsigned int head_dim)
{
    shape->state_layers = layers;
    shape->kv_heads = kv_heads;
    shape->head_dim = head_dim;
    shape->head_bytes = AOTX_KVL_BLOCK * head_dim * (unsigned int)sizeof(half);
    shape->block_bytes = 2u * kv_heads * shape->head_bytes;
    shape->blocks_page = (shape->block_bytes == 0u)
        ? 0u
        : (unsigned int)((AOTX_KV_PAGE_BYTES - AOTX_KVL_HEADER) / shape->block_bytes);
    for (unsigned int layer = 0u; layer < AOTX_MODEL_MAX_LAYERS; ++layer) {
        shape->state_layer[layer] = (layer < layers) ? (unsigned char)layer : 0xffu;
    }
}

/* Fill a compact key and value layout from one state kind for each model layer. */
static inline void aotx_kvl_make_states(aotx_kvl_shape *shape, const unsigned char *state,
                                        unsigned int layers, unsigned int kv_heads,
                                        unsigned int head_dim)
{
    unsigned int state_layers = 0u;
    for (unsigned int layer = 0u; layer < layers; ++layer) {
        state_layers += state[layer] == AOTX_STATE_KIND_KV_PAGES;
    }
    aotx_kvl_make(shape, state_layers, kv_heads, head_dim);
    for (unsigned int layer = 0u; layer < AOTX_MODEL_MAX_LAYERS; ++layer) {
        shape->state_layer[layer] = 0xffu;
    }
    unsigned int state_layer = 0u;
    for (unsigned int layer = 0u; layer < layers; ++layer) {
        if (state[layer] == AOTX_STATE_KIND_KV_PAGES) {
            shape->state_layer[layer] = (unsigned char)state_layer++;
        }
    }
}

/* Read the selected layer kinds into the state sequence of a key and value layout. */
static inline void aotx_kvl_make_desc(aotx_kvl_shape *shape, const aotx_model_desc *desc)
{
    unsigned char state[AOTX_MODEL_MAX_LAYERS];
    for (unsigned int layer = 0u; layer < desc->layers; ++layer) {
        const aotx_layer_kind *kind = aotx_layer_kind_of(desc->kind[layer]);
        state[layer] = (kind != NULL) ? (unsigned char)kind->state : AOTX_STATE_KIND_NONE;
    }
    aotx_kvl_make_states(shape, state, desc->layers, desc->kv_heads, desc->head_dim);
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
    unsigned int blocks_layer = (context + AOTX_KVL_BLOCK - 1u) / AOTX_KVL_BLOCK;
    unsigned int blocks = blocks_layer * shape->state_layers;
    return (blocks + shape->blocks_page - 1u) / shape->blocks_page;
}

/* Give the page of one model layer and position, or an invalid index for another state. */
__device__ __host__ __forceinline__ unsigned int aotx_kvl_page_of(
    const aotx_kvl_shape *shape, unsigned int layer, unsigned int position)
{
    if (shape->blocks_page == 0u || layer >= AOTX_MODEL_MAX_LAYERS
        || shape->state_layer[layer] == 0xffu) {
        return ~0u;
    }
    unsigned int block = (position / AOTX_KVL_BLOCK) * shape->state_layers
                       + (unsigned int)shape->state_layer[layer];
    return block / shape->blocks_page;
}

/* The first row of one layout block, or a null pointer when the slot has no such page.
 * The block number counts the position blocks of a layer, and the layers of one position
 * block come one after the other. A sequence which grows takes the pages in order. */
__device__ __forceinline__ half *aotx_kvl_block(const aotx_kvl_shape *shape, unsigned int agent,
                                                unsigned int layer, unsigned int position)
{
    unsigned int page_index = aotx_kvl_page_of(shape, layer, position);
    if (page_index == ~0u) {
        return 0;
    }
    unsigned int block = (position / AOTX_KVL_BLOCK) * shape->state_layers
                       + (unsigned int)shape->state_layer[layer];
    unsigned long long page = aotx_kv_page(agent, page_index);
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
