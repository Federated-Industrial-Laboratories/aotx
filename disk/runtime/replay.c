/* Purpose: Read validated replay headers and complete packaged journal blocks.
 * Owns: Framing checks and the recovery summary for one extent.
 * Threading: One reader holds the source lease and passes blocks in order.
 * Lifetime: One inspection or replay; this reader executes no embedded code. */
#include "disk/runtime/replay.h"
#include "disk/ccir/internal.h"
#include <string.h>

int aotx_runtime_replay_header(int fd, const aotx_ccir_section *s, unsigned char h[128]) {
    if (!s || s->type != AOTX_CCIR_REPLAY || s->bytes < 128) return AOTX_CCIR_INVALID;
    int rc = aotx_ccir_pread(fd, h, 128, s->offset);
    if (rc) return rc;
    uint32_t mode = aotx_ccir_u32(h + 12);
    if (memcmp(h, "AOTXRPL1", 8) || aotx_ccir_u32(h + 8) != 1 ||
        (mode != 1 && mode != 2) || !aotx_ccir_zero(h + 80, 48) ||
        aotx_ccir_u64(h + 56) != s->bytes - 128) return AOTX_CCIR_INVALID;
    if (mode == 1) return s->bytes == 128 && aotx_ccir_zero(h + 16, 112) ? 0 : AOTX_CCIR_INVALID;
    if (!aotx_ccir_u64(h + 16) || !aotx_ccir_u64(h + 24) || !aotx_ccir_u64(h + 48) ||
        !aotx_ccir_u64(h + 64) || !aotx_ccir_u64(h + 72))
        return AOTX_CCIR_INVALID;
    return 0;
}
int aotx_runtime_replay_walk(const aotx_ccir_view *view, unsigned char *buffer, uint32_t bytes,
    aotx_block_fn fn, void *context, aotx_journal_scan *scan) {
    const aotx_ccir_section *s = NULL;
    for (uint32_t i = 0; i < view->count; ++i)
        if (view->sections[i].type == AOTX_CCIR_REPLAY &&
            (view->sections[i].flags & AOTX_CCIR_REQUIRED)) s = view->sections + i;
    unsigned char h[128];
    int rc = aotx_runtime_replay_header(view->fd, s, h);
    if (rc || aotx_ccir_u32(h + 12) != 2) return rc ? rc : AOTX_CCIR_UNSUPPORTED;
    memset(scan, 0, sizeof(*scan));
    scan->boot_id = aotx_ccir_u64(h + 16); scan->last_tick = aotx_ccir_u64(h + 24);
    scan->state_hash = aotx_ccir_u64(h + 32); scan->blocks = aotx_ccir_u64(h + 48);
    scan->last_block = scan->blocks - 1;
    uint64_t cursor = 128, records = 0, last_seq = 0, last_block = 0, last_tick = 0;
    for (uint64_t i = 0; i < scan->blocks; ++i) {
        unsigned char size[8];
        if (cursor > s->bytes || s->bytes - cursor < 8) return AOTX_CCIR_INVALID;
        rc = aotx_ccir_pread(view->fd, size, 8, s->offset + cursor);
        if (rc) return rc;
        uint64_t take = aotx_ccir_u64(size); cursor += 8;
        if (take > bytes || take > s->bytes - cursor) return AOTX_CCIR_LIMIT;
        rc = aotx_ccir_pread(view->fd, buffer, (size_t)take, s->offset + cursor);
        const char *reason = NULL;
        if (rc || aotx_block_valid(buffer, (uint32_t)take, &reason)) return rc ? rc : AOTX_CCIR_INVALID;
        const aotx_block_header *b = (const aotx_block_header *)buffer;
        if (b->kind || b->boot_id != scan->boot_id || b->tick > scan->last_tick ||
            b->tick < last_tick || b->block_seq <= last_block) return AOTX_CCIR_INVALID;
        last_block = b->block_seq; last_tick = b->tick;
        for (uint32_t j = 0; j < b->record_count; ++j) {
            const aotx_record_header *r = aotx_block_record(buffer, j);
            if (!r->seq || r->boot_id != scan->boot_id ||
                (last_seq && r->seq != last_seq + 1)) return AOTX_CCIR_INVALID;
            last_seq = r->seq;
            records += r->cls == AOTX_CLASS_A && r->type != AOTX_REC_BOOT && r->type != AOTX_REC_TICK_COMMIT;
        }
        if (i + 1 == scan->blocks) {
            if (!b->record_count) return AOTX_CCIR_INVALID;
            const aotx_record_header *last = aotx_block_record(buffer, b->record_count - 1);
            if (last->cls != AOTX_CLASS_A || last->type != AOTX_REC_TICK_COMMIT ||
                last->body_len != sizeof(aotx_commit_body) ||
                last->tick != scan->last_tick) return AOTX_CCIR_INVALID;
            aotx_commit_body commit; memcpy(&commit, aotx_record_body(last), sizeof(commit));
            if (commit.state_hash != scan->state_hash) return AOTX_CCIR_INVALID;
        }
        if (fn) { int step = fn(context, buffer, i); if (step < 0) return AOTX_CCIR_IO; }
        cursor += take;
    }
    return cursor == s->bytes && records == aotx_ccir_u64(h + 40) &&
        aotx_ccir_u64(h + 72) <= last_seq ? 0 : AOTX_CCIR_INVALID;
}
