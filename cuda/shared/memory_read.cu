/* Purpose: Read memory metadata and exact payload bytes within current space rights.
 * Owns: No memory state; results are bounded mailbox copies.
 * Launch shape: One ordered read batch with transitive device scope checks.
 * Lifetime: One current store view; cursors count visible rows only. */
#include "shared/internal.cuh"
#include "shared/memory_lookup.cuh"
__device__ unsigned aotx_shared_memory_marks[2][AOTX_COG_WORDS];
static __device__ bool aotx_shared_memory_visible(unsigned index, const aotx_shared_space &space)
{
    const unsigned char *r = aotx_live_store.objects[index];
    unsigned scope = aotx_cog_u32(r + AOTX_CO_SCOPE);
    if (scope != AOTX_COG_INSTANCE && (scope != space.scope ||
        !aotx_shared_id(r + AOTX_CO_OWNER, space.id))) return false;
    aotx_cognitive_query q = {};
    aotx_service_bytes(q.id, r + AOTX_CO_ID, 16);
    aotx_service_bytes(q.principal, space.id, 16);
    if (space.scope == AOTX_COG_ROOM) aotx_service_bytes(q.room, space.id, 16);
    q.version = aotx_cog_u64(r + AOTX_CO_VERSION);
    return aotx_shared_resolve(&aotx_live_store, &q, false, aotx_live_store.sequence).status == AOTX_COG_OK;
}
static __device__ void aotx_shared_memory_row(unsigned char *out, const unsigned char *r)
{
    aotx_shared_zero(out, 128);
    aotx_service_bytes(out, r + AOTX_CO_ID, 16);
    aotx_service_put(out + 16, aotx_cog_u64(r + AOTX_CO_VERSION), 8);
    aotx_service_put(out + 24, aotx_cog_u16(r + AOTX_CO_KIND), 4);
    aotx_service_put(out + 28, aotx_cog_u32(r + AOTX_CO_SCOPE), 4);
    aotx_service_bytes(out + 32, r + AOTX_CO_OWNER, 16);
    aotx_service_bytes(out + 48, r + AOTX_CO_ROOM, 16);
    aotx_service_put(out + 64, aotx_cog_u64(r + AOTX_CO_BYTES), 8);
    aotx_service_bytes(out + 72, r + AOTX_CO_SOURCE, 16);
    aotx_service_bytes(out + 88, r + AOTX_CO_SUBJECT, 16);
}
__device__ void aotx_shared_memory_read(unsigned channel, const aotx_service_grant *grant,
                                       const unsigned char *read, unsigned space_index)
{
    unsigned participant = aotx_shared_participant_find(grant->principal);
    if (!(grant->actions & AOTX_SHARED_READ_ACTION) ||
        !aotx_shared_visible(participant, space_index, 1)) { aotx_service_answer(channel, 404, 0); return; }
    if (!aotx_live.ready || aotx_live.fatal) { aotx_service_answer(channel, 503, 0); return; }
    const aotx_shared_space &space = aotx_shared.spaces[space_index];
    unsigned char *out = aotx_shared_reply(channel, AOTX_SHARED_MEMORY_READ);
    aotx_service_bytes(out + 48, space.id, 16);
    aotx_service_put(out + 228, space.scope, 4);
    aotx_service_put(out + 232, aotx_shared_rights(participant, space_index), 4);
    bool detail = aotx_service_nonzero(read + 48, 16);
    unsigned long long cursor = aotx_shared_u64(read + 64), byte = aotx_shared_u64(read + 72);
    unsigned limit = aotx_shared_u32(read + 80), rows = 0, data = 0;
    if (detail) {
        if (cursor) { aotx_service_answer(channel, 400, 0); return; }
        int index = aotx_cog_latest(&aotx_live_store, read + 48);
        if (index < 0 || !aotx_shared_memory_visible((unsigned)index, space)) {
            aotx_service_answer(channel, 404, 0); return;
        }
        const unsigned char *r = aotx_live_store.objects[index];
        unsigned long long bytes = aotx_cog_u64(r + AOTX_CO_BYTES);
        if (byte > bytes) { aotx_service_answer(channel, 409, 0); return; }
        data = (unsigned)min(bytes - byte, (unsigned long long)(AOTX_SERVICE_DATA - AOTX_SHARED_REPLY_HEAD - 128));
        aotx_shared_memory_row(out + AOTX_SHARED_REPLY_HEAD, r);
        aotx_service_bytes(out + AOTX_SHARED_REPLY_HEAD + 128,
            aotx_live_store.payload + aotx_cog_u64(r + AOTX_CO_OFFSET) + byte, data);
        aotx_service_put(out + 208, byte + data, 8); rows = 1;
    } else {
        if (byte || !limit || limit > 64) { aotx_service_answer(channel, 400, 0); return; }
        unsigned long long seen = 0;
        bool more = false;
        for (unsigned i = 0; i < aotx_live_store.count; ++i) {
            if (!aotx_shared_memory_visible(i, space)) continue;
            if (seen++ < cursor) continue;
            if (rows == limit) { more = true; break; }
            aotx_shared_memory_row(out + AOTX_SHARED_REPLY_HEAD + rows * 128, aotx_live_store.objects[i]);
            ++rows;
        }
        if (seen < cursor) { aotx_service_answer(channel, 409, 0); return; }
        aotx_service_put(out + 200, more ? cursor + rows : 0, 8);
    }
    aotx_service_put(out + 192, rows, 4); aotx_service_put(out + 196, 128, 4);
    aotx_service_answer(channel, 200, 0, AOTX_SHARED_REPLY_HEAD + rows * 128 + data);
}
