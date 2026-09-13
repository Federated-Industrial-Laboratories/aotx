/* Purpose: Publish an exact native memory source into an explicitly permitted shared space.
 * Owns: A staged batch of source, vector and working objects with independent destination IDs.
 * Launch shape: The serial shared admission thread validates and copies all three objects.
 * Lifetime: The recorded admission defines one atomic store change and terminal receipt. */
#include "shared/internal.cuh"
#include "shared/memory_lookup.cuh"
#include "cognitive/checkpoint.cuh"
#include "cognitive/maintenance.cuh"
#include "cognitive/recall_format.cuh"
#include "cognitive/validate.cuh"

static __device__ bool aotx_shared_publish_empty(const unsigned char *r, unsigned offset)
{ return aotx_cog_zero(r + offset, 24); }

static __device__ unsigned aotx_shared_publish_graph(const aotx_shared_receipt *r,
    const aotx_service_grant *grant, unsigned indices[3])
{
    const aotx_cognitive_store *s = &aotx_live_store;
    if (!aotx_live.ready || aotx_live.fatal || aotx_live.phase != AOTX_LIVE_IDLE ||
        aotx_live.received || aotx_maintenance.running || !aotx_checkpoint_foreign_quiet()) return 429;
    for (unsigned i = 0; i < AOTX_SLOTS; ++i) if (aotx_shared.slot[i]) return 429;
    if (s->count > AOTX_COG_OBJECTS || s->bytes > AOTX_COG_PAYLOAD ||
        s->sequence > UINT64_MAX - 3 || s->tick == UINT64_MAX || aotx_live.accepted == UINT64_MAX) return 429;
    if (r->space >= aotx_shared.space_capacity || !aotx_shared.spaces[r->space].active ||
        !aotx_shared_id(r->command + 72, aotx_shared.spaces[r->space].id)) return 404;
    int working = aotx_cog_latest(s, r->command + 56);
    if (working < 0) return 404;
    const unsigned char *w = s->objects[working];
    unsigned source_space = aotx_shared_space_find(w + AOTX_CO_OWNER);
    if (source_space == AOTX_SHARED_NONE ||
        aotx_shared.spaces[source_space].scope != aotx_cog_u32(w + AOTX_CO_SCOPE)) return 404;
    if (grant && (!aotx_shared_id(grant->principal, r->actor) ||
        (grant->actions & (AOTX_SHARED_READ_ACTION | AOTX_SHARED_WRITE_ACTION | AOTX_SHARED_MANAGE_ACTION)) !=
        (AOTX_SHARED_READ_ACTION | AOTX_SHARED_WRITE_ACTION | AOTX_SHARED_MANAGE_ACTION) ||
        !aotx_shared_visible(r->participant, source_space, 5) ||
        !aotx_shared_visible(r->participant, r->space, 6))) return 404;
    if (aotx_cog_u16(w + AOTX_CO_KIND) != AOTX_COG_WORKING) return 409;
    aotx_cognitive_query q = {};
    aotx_service_bytes(q.id, r->command + 56, 16);
    aotx_service_bytes(q.principal, aotx_shared.spaces[source_space].id, 16);
    if (aotx_shared.spaces[source_space].scope == AOTX_COG_ROOM)
        aotx_service_bytes(q.room, q.principal, 16);
    q.version = aotx_shared_u64(r->command + 128);
    aotx_cognitive_match found = aotx_shared_resolve(s, &q, true, s->sequence + 3);
    if (found.status) return found.status == AOTX_COG_DENIED ? 404 : 409;
    int source = aotx_cog_find(s, w + AOTX_CO_SOURCE, aotx_cog_u64(w + AOTX_CO_SOURCE_VERSION));
    int vector = aotx_cog_find(s, w + AOTX_CO_EMBEDDING, aotx_cog_u64(w + AOTX_CO_EMBED_VERSION));
    if (source < 0 || vector < 0 || source == vector || source == working || vector == working) return 409;
    indices[0] = (unsigned)source; indices[1] = (unsigned)vector; indices[2] = (unsigned)working;
    const unsigned char *event = s->objects[source], *embedding = s->objects[vector];
    if (aotx_cog_u16(event + AOTX_CO_KIND) != AOTX_COG_EVENT ||
        aotx_cog_u16(embedding + AOTX_CO_KIND) != AOTX_COG_COMPONENT ||
        !aotx_shared_publish_empty(event, AOTX_CO_SOURCE) ||
        !aotx_shared_publish_empty(event, AOTX_CO_SUPERSEDES) ||
        !aotx_shared_publish_empty(event, AOTX_CO_EMBEDDING) ||
        !aotx_shared_publish_empty(embedding, AOTX_CO_SUPERSEDES) ||
        !aotx_shared_publish_empty(embedding, AOTX_CO_EMBEDDING) ||
        !aotx_cog_equal(embedding + AOTX_CO_SOURCE, w + AOTX_CO_SOURCE, 24)) return 409;
    for (unsigned k = 0; k < 3; ++k) {
        const unsigned char *row = s->objects[indices[k]];
        uint64_t offset = aotx_cog_u64(row + AOTX_CO_OFFSET), bytes = aotx_cog_u64(row + AOTX_CO_BYTES);
        if (offset > s->bytes || bytes > s->bytes - offset ||
            !aotx_cog_visible(row, &q, true, s->sequence + 3) ||
            aotx_cog_latest(s, row + AOTX_CO_ID) != (int)indices[k] || aotx_cog_superseded(s, row)) return 409;
        const unsigned char *p = s->payload + offset;
        if (k != 1) {
            if (bytes < 32 || !aotx_recall_magic(p, "AOTXMEM1") || aotx_recall_text(s, row) <= 0) return 409;
        } else {
            if (bytes < 128 || !aotx_recall_magic(p, "AOTXVEC2") || aotx_cog_u32(p + 8) != 2 ||
                aotx_cog_u32(p + 16) != 4 || aotx_cog_u32(p + 20) != 1 ||
                aotx_cog_zero(p + 24, 32) || aotx_cog_zero(p + 56, 32) ||
                !aotx_cog_equal(p + 88, row + AOTX_CO_SOURCE, 24) || !aotx_cog_zero(p + 112, 16)) return 409;
            unsigned width = aotx_cog_u32(p + 12);
            if (!width || width > AOTX_RECALL_WIDTH || bytes != 128 + 4 * width) return 409;
            bool nonzero = false;
            for (unsigned j = 0; j < width; ++j) {
                if (!aotx_recall_finite(p + 128 + j * 4)) return 409;
                nonzero |= (aotx_cog_u32(p + 128 + j * 4) & 0x7fffffffu) != 0;
            }
            if (!nonzero) return 409;
        }
    }
    return 200;
}

static __device__ unsigned aotx_shared_publish_stage(const aotx_shared_receipt *receipt,
    const aotx_service_grant *grant, unsigned long long serial)
{
    unsigned indices[3];
    unsigned status = aotx_shared_publish_graph(receipt, grant, indices);
    if (status != 200) return status;
    const aotx_cognitive_store *live = &aotx_live_store;
    aotx_cognitive_store *stage = &aotx_live_scratch;
    if (!serial || serial > (UINT64_MAX - 3) / 4 || 3u > AOTX_COG_OBJECTS - live->count) return 429;
    uint64_t payload = 0;
    for (unsigned k = 0; k < 3; ++k) payload += aotx_cog_u64(live->objects[indices[k]] + AOTX_CO_BYTES);
    if (payload > AOTX_COG_PAYLOAD - live->bytes) return 429;
    stage->count = 3; stage->bytes = (unsigned)payload;
    stage->sequence = live->sequence + 3; stage->tick = live->tick + 1;
    stage->pressure_percent = live->pressure_percent; stage->root_sequence = live->root_sequence;
    aotx_service_bytes(stage->lineage, live->lineage, 16);
    const aotx_shared_space *space = aotx_shared.spaces + receipt->space;
    uint64_t offset = 0;
    for (unsigned k = 0; k < 3; ++k) {
        const unsigned char *source = live->objects[indices[k]];
        unsigned char *row = stage->objects[k];
        uint64_t bytes = aotx_cog_u64(source + AOTX_CO_BYTES), sequence = live->sequence + k + 1;
        aotx_service_bytes(row, source, AOTX_COG_OBJECT);
        aotx_service_bytes(row + AOTX_CO_ID, (const unsigned char *)"AOTXSHP1", 8);
        aotx_cog_put(row + AOTX_CO_ID + 8, serial * 4 + k + 1, 8);
        if (aotx_cog_latest(live, row + AOTX_CO_ID) >= 0) return 409;
        aotx_cog_put(row + AOTX_CO_VERSION, live->pressure_percent ? sequence : 1, 8);
        aotx_cog_put(row + AOTX_CO_CREATED, sequence, 8); aotx_cog_put(row + AOTX_CO_UPDATED, sequence, 8);
        aotx_service_bytes(row + AOTX_CO_OWNER, space->id, 16);
        aotx_shared_zero(row + AOTX_CO_ROOM, 16);
        if (space->scope == AOTX_COG_ROOM) aotx_service_bytes(row + AOTX_CO_ROOM, space->id, 16);
        aotx_cog_put(row + AOTX_CO_SCOPE, space->scope, 4);
        aotx_shared_zero(row + AOTX_CO_SUPERSEDES, 24);
        aotx_cog_put(row + AOTX_CO_OFFSET, offset, 8);
        unsigned char *p = stage->payload + offset;
        aotx_service_bytes(p, live->payload + aotx_cog_u64(source + AOTX_CO_OFFSET), (unsigned)bytes);
        if (k) {
            aotx_service_bytes(row + AOTX_CO_SOURCE, stage->objects[0] + AOTX_CO_ID, 16);
            aotx_cog_put(row + AOTX_CO_SOURCE_VERSION, live->pressure_percent ? live->sequence + 1 : 1, 8);
        }
        if (k == 1) {
            aotx_service_bytes(p + 88, row + AOTX_CO_SOURCE, 24);
        } else if (k == 2) {
            aotx_service_bytes(row + AOTX_CO_EMBEDDING, stage->objects[1] + AOTX_CO_ID, 16);
            aotx_cog_put(row + AOTX_CO_EMBED_VERSION, live->pressure_percent ? live->sequence + 2 : 1, 8);
        }
        offset += bytes;
    }
    for (unsigned k = 0; k < 3; ++k) if (aotx_cog_validate(stage, k)) return 409;
    return 200;
}

__device__ unsigned aotx_shared_publish_check(const aotx_shared_receipt *receipt,
    const aotx_service_grant *grant)
{
    if (!grant || !aotx_checkpoint_quiet() || aotx_checkpoint_pressure() ||
        aotx_shared.kind || aotx_shared.received || aotx_shared.serial == UINT64_MAX) return 429;
    return aotx_shared_publish_stage(receipt, grant, aotx_shared.serial + 1);
}
__device__ bool aotx_shared_publish_apply(unsigned index, bool replay)
{
    if (index >= aotx_shared.receipt_capacity) return false;
    aotx_shared_receipt *receipt = aotx_shared.receipts + index;
    if (receipt->operation != AOTX_SHARED_PUBLISH || receipt->phase != AOTX_SHARED_QUEUED ||
        !receipt->admission_source || (!replay && aotx_shared.fatal) ||
        aotx_shared_publish_stage(receipt, NULL, aotx_shared.transfer_serial) != 200) return false;
    aotx_cognitive_store *live = &aotx_live_store, *stage = &aotx_live_scratch;
    unsigned base = live->bytes, count = live->count;
    aotx_service_bytes(live->payload + base, stage->payload, stage->bytes);
    for (unsigned k = 0; k < 3; ++k) {
        unsigned char *row = live->objects[count + k];
        aotx_service_bytes(row, stage->objects[k], AOTX_COG_OBJECT);
        aotx_cog_put(row + AOTX_CO_OFFSET, base + aotx_cog_u64(row + AOTX_CO_OFFSET), 8);
    }
    __threadfence();
    live->sequence = stage->sequence; live->tick = stage->tick;
    live->bytes = base + stage->bytes; live->count = count + 3;
    ++aotx_live.accepted;
    receipt->phase = AOTX_SHARED_DONE; receipt->status = 200;
    receipt->terminal_source = receipt->admission_source;
    return true;
}
