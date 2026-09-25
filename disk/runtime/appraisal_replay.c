/* Purpose: Check saved model identities across fragmented appraisal result records.
 * Owns: One result header and row buffer; memory bytes are not assembled.
 * Threading: One disk reader processes the complete journal block batch in order.
 * Lifetime: One complete runtime admission or checkpoint publication. */
#include "disk/runtime/appraisal.h"
#include "cognitive/format.h"
#include "disk/runtime/replay.h"
#include <stdlib.h>

typedef struct aotx_appraisal_replay {
    aotx_runtime_appraisal_models models;
    unsigned char header[64], row[AOTX_APPRAISAL_RESULT_ROW], correlation[16];
    uint32_t total, offset, count, status, required, declared, version, stride;
    int replace, error;
} aotx_appraisal_replay;
static int result_header(aotx_appraisal_replay *s) {
    const unsigned char *h = s->header;
    s->version = aotx_ccir_u32(h + 8);
    if (memcmp(h, "AOTXAPS1", 8) || (s->version != 1 && s->version != 2)) return AOTX_CCIR_UNSUPPORTED;
    s->stride = s->version == 1 ? AOTX_APPRAISAL_LEGACY_ROW : AOTX_APPRAISAL_RESULT_ROW;
    s->count = aotx_ccir_u32(h + 12); s->status = aotx_ccir_u32(h + 32);
    uint64_t start = 64 + (uint64_t)s->count * s->stride;
    if (s->count > 64 || s->status > AOTX_COG_UNAVAILABLE || !aotx_ccir_zero(h + 36, 28) || start > s->total ||
        aotx_ccir_u64(h + 24) != s->total - start ||
        (!s->count && (!s->status || s->total != 64)) ||
        (s->replace && s->status != AOTX_COG_DENIED)) return AOTX_CCIR_INVALID;
    return 0;
}
static int model_digest(const aotx_appraisal_replay *s, const unsigned char *digest) {
    if (aotx_ccir_zero(digest, 32)) return s->status ? 0 : AOTX_CCIR_INVALID;
    for (unsigned i = 0; i < s->models.count; ++i)
        if (!memcmp(digest, s->models.digest[i], 32)) return 0;
    return AOTX_CCIR_UNSUPPORTED;
}
static int result_row(const aotx_appraisal_replay *s) {
    const unsigned char *p = s->row;
    int rc = model_digest(s, p + 24);
    if (rc || s->version == 1) return rc;
    uint32_t bytes = aotx_ccir_u32(p + 56), first = aotx_ccir_u32(p + 4160);
    uint32_t second = aotx_ccir_u32(p + 4164), phase = aotx_ccir_u32(p + 4168);
    int model = !aotx_ccir_zero(p + 24, 32), first_model = !aotx_ccir_zero(p + 4192, 32);
    if (aotx_ccir_zero(p, 16) || !aotx_ccir_u64(p + 16) || aotx_ccir_u32(p + 60) != s->status ||
        bytes > 4096 || first > 4096 || second > 1 || phase > 2 ||
        !aotx_ccir_zero(p + 4172, 20) || !aotx_ccir_zero(p + 64 + bytes, 4096 - bytes) ||
        !aotx_ccir_zero(p + 4224 + first, 4096 - first) || (bytes && !model) || (first && !first_model) ||
        (first_model && memcmp(p + 4192, p + 24, 32)) ||
        (!phase && (bytes || first || second || first_model)) || (phase == 1 && (bytes || second)) ||
        (phase == 2 && (!first || !first_model || (bytes && !second))) ||
        (!s->status && (!bytes || phase != 2 || !second))) return AOTX_CCIR_INVALID;
    return first_model ? model_digest(s, p + 4192) : 0;
}
static int result_part(aotx_appraisal_replay *s, const aotx_record_header *r) {
    const unsigned char *p = (const unsigned char *)r + AOTX_HEADER_BYTES;
    if (r->body_len <= 32 || r->body_len > 192) return AOTX_CCIR_INVALID;
    if (aotx_ccir_u32(p) != 1) return AOTX_CCIR_UNSUPPORTED;
    unsigned flags = r->flags & ~(AOTX_FLAG_REPLAYED | AOTX_FLAG_REPLAY);
    if (flags & ~AOTX_FLAG_ADMISSION) return AOTX_CCIR_INVALID;
    uint32_t total = aotx_ccir_u32(p + 24), offset = aotx_ccir_u32(p + 28), bytes = r->body_len - 32;
    if (total < 64 || offset > total || bytes > total - offset) return AOTX_CCIR_INVALID;
    if (!offset) {
        int partial = s->offset != s->total;
        if (partial && !(flags & AOTX_FLAG_ADMISSION)) return AOTX_CCIR_INVALID;
        s->replace = !!(flags & AOTX_FLAG_ADMISSION);
        s->total = total; s->offset = s->count = s->status = 0;
        memset(s->header, 0, sizeof(s->header)); memcpy(s->correlation, p + 8, 16);
    }
    if (total != s->total || offset != s->offset || memcmp(s->correlation, p + 8, 16) ||
        (offset && flags)) return AOTX_CCIR_INVALID;
    p += 32;
    uint64_t end = (uint64_t)offset + bytes;
    if (offset < 64) {
        uint32_t take = bytes < 64 - offset ? bytes : 64 - offset;
        memcpy(s->header + offset, p, take);
        if (end >= 64) { int rc = result_header(s); if (rc) return rc; }
    }
    if (end < 64) { s->offset += bytes; return 0; }
    uint64_t row = offset < 64 ? 0 : ((uint64_t)offset - 64) / s->stride;
    for (; row < s->count; ++row) {
        uint64_t first = 64 + row * s->stride, last = first + s->stride;
        if (first >= end) break;
        if (last <= offset) continue;
        uint64_t from = first > offset ? first : offset, to = last < end ? last : end;
        memcpy(s->row + from - first, p + from - offset, (size_t)(to - from));
        if (from < first + 56 && to >= first + 56) {
            int rc = model_digest(s, s->row + 24); if (rc) return rc;
        }
        if (s->version == 2 && from < first + 4224 && to >= first + 4224) {
            int rc = model_digest(s, s->row + 4192); if (rc) return rc;
        }
        if (to == last) { int rc = result_row(s); if (rc) return rc; }
    }
    s->offset += bytes;
    return 0;
}
static int block(void *context, const unsigned char *bytes, uint64_t index) {
    aotx_appraisal_replay *s = context;
    const aotx_block_header *b = (const aotx_block_header *)bytes;
    for (uint32_t i = 0; !s->error && i < b->record_count; ++i) {
        const aotx_record_header *r = aotx_block_record(bytes, i);
        s->required |= aotx_runtime_appraisal_record(r);
        if (s->required & ~s->declared) s->error = AOTX_CCIR_UNSUPPORTED;
        else if (r->cls == AOTX_CLASS_A && r->type == 33 && r->body_len >= 8 &&
            aotx_ccir_u32((const unsigned char *)r + AOTX_HEADER_BYTES + 4) == AOTX_APPRAISAL_RESULT)
            s->error = result_part(s, r);
    }
    (void)index;
    return 0;
}
int aotx_runtime_appraisal_replay_check(const aotx_ccir_view *replay, const aotx_ccir_view *assets,
    const aotx_runtime_index *index) {
    aotx_appraisal_replay state = {0};
    state.declared = aotx_ccir_u32(index->header + 20);
    int rc = state.declared & AOTX_RUNTIME_APPRAISAL ?
        aotx_runtime_appraisal_model_view(assets, index, &state.models) : 0;
    unsigned char *buffer = !rc ? malloc(16u * 1024u * 1024u) : NULL;
    if (!rc && !buffer) rc = AOTX_CCIR_IO;
    aotx_journal_scan scan;
    if (!rc) rc = aotx_runtime_replay_walk(replay, buffer, 16u * 1024u * 1024u, block, &state, &scan);
    free(buffer);
    if (!rc) rc = state.error;
    return rc ? rc : state.offset == state.total ? 0 : AOTX_CCIR_INVALID;
}
