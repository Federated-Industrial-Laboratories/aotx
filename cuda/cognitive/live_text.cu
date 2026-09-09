/* Purpose: Prepare live text queries through the existing device embedding batch.
 * Owns: Bounded slot leases and preparation status; the typed store stays unchanged.
 * Launch shape: One 64-thread block; completion uses one thread for each tool slot.
 * Lifetime: One admitted text batch through its recorded choice. */
#include "cognitive/live_text.cuh"
#include "model/load.cuh"
#include "tool/tool_state.cuh"

/* This identity names the exact text and pooling contract in docs/21-text-memory.md. */
__device__ const unsigned char aotx_live_processor[32] = {0x7d, 0x12, 0xaf, 0x1d, 0x2c, 0xd1, 0xe5, 0x19, 0x4d, 0xef, 0x98, 0x3d, 0x1f, 0xd8, 0x07, 0x3d, 0x1c, 0x36, 0xc4, 0x44, 0xee, 0xa3, 0x9c, 0x2d, 0xcf, 0x9b, 0xbe, 0x39, 0x4e, 0x75, 0x89, 0x2d};
static __device__ uint32_t aotx_live_text_ticks;

__device__ void aotx_live_text_fail(uint32_t slot, uint32_t status) {
    if (!aotx_live_text_pending(slot)) return;
    aotx_live.text_status[slot] = (status + 1) | 0x80000000u;
    aotx_tool_embed.state[slot] = AOTX_TOOL_EMBED_NONE;
}
__device__ void aotx_live_text_done(uint32_t slot) {
    if (aotx_sched.held || aotx_seam.replaying || !aotx_live_text_pending(slot) ||
        aotx_tool_embed.state[slot] != AOTX_TOOL_EMBED_RUN) return;
    uint32_t place = aotx_tool_embed.place[slot], width = aotx_tool_embed.width;
    if (place >= aotx_tool_embed.seqs || width > AOTX_RECALL_WIDTH || !width) {
        aotx_live_text_fail(slot, AOTX_COG_LAYOUT); return;
    }
    unsigned char *q = aotx_live.requests + 64 + (aotx_live.text_row[slot] - 1) * AOTX_RECALL_QUERY;
    for (uint32_t j = 0; j < width; ++j)
        aotx_cog_put(q + 160 + j * 4, __float_as_uint(aotx_tool_embed.vector[(uint64_t)place * width + j]), 4);
    aotx_live.text_status[slot] = (aotx_recall_query_check(q) + 1) | 0x80000000u;
    aotx_tool_embed.state[slot] = AOTX_TOOL_EMBED_NONE;
    atomicAdd(&aotx_live.encoded, 1ull);
}
static __device__ uint32_t aotx_live_encoder(void) {
    const aotx_model_resident_row *r = aotx_model_load.resident + AOTX_MODEL_EMBEDDING;
    if (!r->active || r->slot != AOTX_MODEL_EMBEDDING || aotx_cog_zero(r->body.digest, 32) ||
        !aotx_tool_embed.ready || aotx_tool_embed.role != AOTX_MODEL_EMBEDDING ||
        !aotx_tool_embed.width || aotx_tool_embed.width > AOTX_RECALL_WIDTH ||
        aotx_tool_embed.width != aotx_model[AOTX_MODEL_EMBEDDING].hidden ||
        !aotx_model[AOTX_MODEL_EMBEDDING].layers || aotx_model_load.pending_count) return AOTX_COG_LAYOUT;
    for (uint32_t i = 0; i < aotx_live.count; ++i) {
        uint32_t slot = aotx_cog_u32(aotx_live.prefixes[i]);
        if (aotx_requests.slot[slot].request || aotx_tool_embed.state[slot] != AOTX_TOOL_EMBED_NONE ||
            aotx_transcript[slot].embed_kind != AOTX_MEMORY_EMBED_NONE) return AOTX_COG_DENIED;
    }
    return AOTX_COG_OK;
}
__device__ void aotx_live_text_begin(void) {
    aotx_live.status = aotx_live_encoder();
    if (aotx_live.status) return;
    for (uint32_t i = 0; i < AOTX_SLOTS; ++i) {
        aotx_live.text_row[i] = 0; aotx_live.text_status[i] = 0;
    }
    const unsigned char *model = aotx_model_load.resident[AOTX_MODEL_EMBEDDING].body.digest;
    for (uint32_t i = 0; i < aotx_live.count; ++i) {
        uint32_t slot = aotx_cog_u32(aotx_live.prefixes[i]);
        unsigned char *q = aotx_live.requests + 64 + i * AOTX_RECALL_QUERY;
        uint32_t bytes = aotx_cog_u32(q + 148);
        for (uint32_t j = 0; j < 32; ++j) { q[64 + j] = model[j]; q[96 + j] = aotx_live_processor[j]; }
        aotx_cog_put(q + 128, aotx_tool_embed.width, 4);
        for (uint32_t j = 0; j < bytes; ++j) aotx_tool_gear.text[slot * AOTX_TOOL_TEXT_BYTES + j] = q[4640 + j];
        aotx_tool_gear.bytes[slot] = bytes;
        aotx_tool_embed.asked[slot] = aotx_tool_embed.starved[slot] = 0;
        aotx_tool_embed.state[slot] = AOTX_TOOL_EMBED_WAIT;
        aotx_live.text_row[slot] = i + 1;
    }
    aotx_live_text_ticks = 0;
    aotx_live.phase = AOTX_LIVE_ENCODING;
}
__global__ void aotx_live_prepare(void) {
    if (aotx_sched.held || aotx_seam.replaying || aotx_live.phase != AOTX_LIVE_ENCODING || threadIdx.x) return;
    uint32_t done = 0, error = aotx_live.status;
    for (uint32_t i = 0; i < aotx_live.count; ++i) {
        uint32_t slot = aotx_cog_u32(aotx_live.prefixes[i]);
        uint32_t status = aotx_live.text_status[slot];
        if (status & 0x80000000u) {
            if (aotx_kv_release(slot)) aotx_live.text_status[slot] = status &= 0x7fffffffu;
            else continue;
        }
        done += status != 0;
        if (status > 1 && !error) error = status - 1;
    }
    if (++aotx_live_text_ticks > AOTX_LIVE_TEXT_TICKS && done != aotx_live.count) error = AOTX_COG_CAPACITY;
    if (!error && done != aotx_live.count) return;
    for (uint32_t i = 0; i < aotx_live.count; ++i)
        aotx_live_text_fail(aotx_cog_u32(aotx_live.prefixes[i]), error ? error : AOTX_COG_CAPACITY);
    aotx_live.status = error;
    for (uint32_t i = 0; i < aotx_live.count; ++i)
        if (aotx_live.text_status[aotx_cog_u32(aotx_live.prefixes[i])] & 0x80000000u) return;
    aotx_live.phase = AOTX_LIVE_SEARCH;
}
