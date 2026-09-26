/* Purpose: Check policy record framing and saved revision intervals without state execution.
 * Owns: One small metadata header across an ordered fragment batch.
 * Threading: One complete-file disk reader; state payload bytes remain opaque.
 * Lifetime: One admission or explicit compatible update. */
#include "disk/runtime/policy.h"
#include "disk/runtime/replay.h"
#include "disk/ccir/internal.h"
#include <stdlib.h>
#include <string.h>

typedef struct aotx_policy_scan {
    const aotx_policy_file *selected;
    const aotx_policy_history *history;
    unsigned char header[64];
    uint32_t total, offset;
    uint64_t decision, last;
    int error;
} aotx_policy_scan;
static int part(aotx_policy_scan *s, const aotx_record_header *r) {
    const unsigned char *p = aotx_record_body(r);
    if (r->body_len <= 32 || r->body_len > AOTX_BODY_BYTES) return AOTX_CCIR_INVALID;
    uint32_t total = aotx_ccir_u32(p + 4), offset = aotx_ccir_u32(p + 8), count = aotx_ccir_u32(p + 12);
    uint64_t decision = aotx_ccir_u64(p + 16);
    if (aotx_ccir_u32(p) != 1 || aotx_ccir_u64(p + 24) ||
        total != 256u + s->selected->config.state_bytes || count != r->body_len - 32 ||
        offset > total || count > total - offset || offset != s->offset ||
        !decision || s->last == UINT64_MAX || decision != s->last + 1) return AOTX_CCIR_INVALID;
    if (!offset) { s->total = total; s->decision = decision; memset(s->header, 0, 64); }
    if (total != s->total || decision != s->decision) return AOTX_CCIR_INVALID;
    if (offset < 64) {
        uint32_t n = count < 64 - offset ? count : 64 - offset;
        memcpy(s->header + offset, p + 32, n);
    }
    s->offset += count;
    if (s->offset != total) return 0;
    const aotx_policy_config *config = &s->selected->config;
    const unsigned char *digest = s->selected->digest;
    for (uint32_t i = 0; i < s->history->count; ++i) {
        if (decision > s->history->rows[i].last_decision) continue;
        config = &s->history->rows[i].config; digest = s->history->rows[i].digest; break;
    }
    const unsigned char *h = s->header;
    uint32_t abi = config->abi >= AOTX_POLICY_APPRAISAL_ABI ? config->abi : 0;
    if (memcmp(h, "AOTXPD01", 8) || aotx_ccir_u32(h + 8) != 1 ||
        aotx_ccir_u32(h + 12) != config->state_schema || aotx_ccir_u32(h + 16) != config->state_bytes ||
        aotx_ccir_u32(h + 20) != abi || aotx_ccir_u64(h + 24) != decision ||
        memcmp(h + 32, digest, 32)) return AOTX_CCIR_INVALID;
    s->last = decision; s->offset = s->total = 0;
    return 0;
}
static int block(void *context, const unsigned char *bytes, uint64_t index) {
    aotx_policy_scan *s = context;
    const aotx_block_header *b = (const aotx_block_header *)bytes;
    for (uint32_t i = 0; !s->error && i < b->record_count; ++i) {
        const aotx_record_header *r = aotx_block_record(bytes, i);
        if (r->cls == AOTX_CLASS_A && r->type == AOTX_REC_POLICY) s->error = part(s, r);
    }
    (void)index;
    return 0;
}
int aotx_runtime_policy_replay(const aotx_ccir_view *view, const aotx_policy_file *selected,
    const aotx_policy_history *history, uint64_t *last) {
    *last = 0;
    const aotx_ccir_section *section = NULL;
    for (uint32_t i = 0; i < view->count; ++i)
        if (view->sections[i].type == AOTX_CCIR_REPLAY) section = view->sections + i;
    unsigned char header[128];
    int rc = aotx_runtime_replay_header(view->fd, section, header);
    aotx_policy_scan state = {0}; state.selected = selected; state.history = history;
    if (!rc && aotx_ccir_u32(header + 12) == 2) {
        unsigned char *buffer = malloc(16u * 1024u * 1024u);
        if (!buffer) return AOTX_CCIR_IO;
        aotx_journal_scan scan;
        rc = aotx_runtime_replay_walk(view, buffer, 16u * 1024u * 1024u, block, &state, &scan);
        free(buffer);
        if (!rc) rc = state.error;
        if (!rc && state.offset) rc = AOTX_CCIR_INVALID;
    }
    if (!rc && history->count && state.last < history->rows[history->count - 1].last_decision)
        rc = AOTX_CCIR_INVALID;
    if (!rc) *last = state.last;
    return rc;
}
