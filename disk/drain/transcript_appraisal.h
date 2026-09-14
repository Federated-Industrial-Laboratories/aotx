/* Purpose: Check recorded appraisal additions before transcript projection.
 * Owns: Bounded comparisons of query defaults and pending queue byte extents.
 * Threading: One disk reader; cognitive admission remains on the device.
 * Lifetime: One complete recorded query and choice batch. */
#ifndef AOTX_TRANSCRIPT_APPRAISAL_H
#define AOTX_TRANSCRIPT_APPRAISAL_H
#include "appraisal/format.h"

static int audit_context(const unsigned char *raw, const unsigned char *prepared) {
    if (!memcmp(raw + 4640, prepared + 4640, AOTX_RECALL_QUERY - 4640)) return 1;
    if (memcmp(raw + 4640, prepared + 4640, AOTX_RECALL_EXTENSION - 4640)) return 0;
    const unsigned char *a = raw + AOTX_RECALL_EXTENSION, *b = prepared + AOTX_RECALL_EXTENSION;
    uint32_t flags = (uint32_t)audit_get(a + 12, 4);
    if (flags & ~AOTX_RECALL_TASKS || memcmp(b, "AOTXCTX1", 8) || audit_get(b + 8, 4) != 1 ||
        audit_get(b + 12, 4) != (flags | AOTX_RECALL_APPRAISE) ||
        audit_get(b + 36, 4) > 1000000 || audit_get(b + 40, 4) > 1000000) return 0;
    if (!flags) return audit_zero(a, AOTX_RECALL_QUERY - AOTX_RECALL_EXTENSION) &&
        audit_zero(b + 16, 20) && audit_get(b + 44, 4) == 1 &&
        audit_zero(b + 48, AOTX_RECALL_QUERY - AOTX_RECALL_EXTENSION - 48);
    return !memcmp(a, b, 12) && !memcmp(a + 16, b + 16, 20) &&
        !memcmp(a + 44, b + 44, AOTX_RECALL_QUERY - AOTX_RECALL_EXTENSION - 44);
}
static int audit_queues(const aotx_transcript_live *s, const unsigned char *tail,
                       unsigned first, unsigned count, unsigned retained, uint64_t bytes) {
    if (count < first || count - first > retained) return 0;
    if (count == first) return 1;
    uint64_t base = 128 + (uint64_t)count * 256, payload = audit_get(tail + 24, 8);
    unsigned char seen[AOTX_RECALL_BATCH] = {0};
    if (base > bytes || payload != bytes - base || payload < (count - first) * AOTX_APPRAISAL_QUEUE_BYTES) return 0;
    for (unsigned i = first; i < count; ++i) {
        const unsigned char *r = tail + 128 + i * 256;
        uint64_t at = audit_get(r + AOTX_CO_OFFSET, 8);
        if (at != payload - (count - i) * AOTX_APPRAISAL_QUEUE_BYTES ||
            audit_get(r, 2) != 1 || audit_get(r + AOTX_CO_KIND, 2) != AOTX_COG_POLICY ||
            !audit_zero(r + AOTX_CO_FLAGS, 4) || audit_zero(r + AOTX_CO_ID, 16) ||
            memcmp(r + AOTX_CO_LINEAGE, tail + 48, 16) || !audit_get(r + AOTX_CO_VERSION, 8) ||
            audit_get(r + AOTX_CO_BYTES, 8) != AOTX_APPRAISAL_QUEUE_BYTES ||
            audit_get(r + AOTX_CO_SOURCE_KIND, 4) != AOTX_COG_INFERRED ||
            audit_get(r + AOTX_CO_RETENTION, 4) != 2) return 0;
        const unsigned char *p = tail + base + at;
        if (memcmp(p, "AOTXAPQ1", 8) || audit_get(p + 8, 4) != 1 || audit_get(p + 12, 4) ||
            audit_zero(p + 16, 16) || !audit_get(p + 32, 8) || !audit_zero(p + 56, 8) ||
            audit_zero(p + 64, 32) || !audit_zero(p + 96, 32) || !audit_zero(p + 152, 8)) return 0;
        unsigned match = AOTX_RECALL_BATCH, source_row = 0;
        for (unsigned j = 0; j < audit_get(s->query + 8, 4); ++j) {
            const unsigned char *q = s->query + 128 + j * AOTX_LIVE_QUERY_ROW;
            const unsigned char *held = s->choice + 64 + j * audit_choice_row(s) + AOTX_LIVE_TEXT_CHOICE_ROW;
            if (audit_zero(held, AOTX_LIVE_RETAINED_ROW)) continue;
            uint64_t version = audit_get(tail + 8, 4) == 2 ? audit_get(tail + 32, 8) + 3 * source_row : 1;
            ++source_row;
            if (memcmp(r + AOTX_CO_SOURCE, held + 32, 16)) continue;
            if (seen[j] || audit_zero(held + 112, 16) || memcmp(r + AOTX_CO_SUBJECT, held + 112, 16) ||
                memcmp(r + AOTX_CO_OWNER, q + 16, 32) ||
                audit_get(r + AOTX_CO_SCOPE, 4) != audit_get(q + 152, 4) ||
                audit_get(r + AOTX_CO_SOURCE_VERSION, 8) != version) return 0;
            match = j; break;
        }
        if (match == AOTX_RECALL_BATCH) return 0;
        seen[match] = 1;
    }
    return 1;
}
#endif
