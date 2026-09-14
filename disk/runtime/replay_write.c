/* Purpose: Copy one complete journal prefix into a runtime replay extent.
 * Owns: A temporary file and a bounded block buffer.
 * Threading: The drain calls this after its journal writes, with no concurrent writer.
 * Lifetime: The returned file stays open until the CCIR transaction ends. */
#include "disk/runtime/replay.h"
#include "disk/runtime/appraisal.h"
#include "disk/ccir/internal.h"
#include <stdlib.h>
#include <string.h>

typedef struct aotx_runtime_collect {
    FILE *file;
    uint64_t boot, tick, blocks, records, bytes, hash, last_seq, limit;
    int complete, status;
    uint32_t features;
} aotx_runtime_collect;
static int collect(void *context, const unsigned char *block, uint64_t index) {
    aotx_runtime_collect *c = context;
    const aotx_block_header *b = (const aotx_block_header *)block;
    if (c->complete || b->boot_id != c->boot || b->tick > c->tick) return -1;
    if (8 + (uint64_t)b->byte_len > c->limit - c->bytes) { c->status = AOTX_CCIR_LIMIT; return -1; }
    unsigned char size[8]; aotx_ccir_put(size, b->byte_len, 8);
    if (fwrite(size, 1, 8, c->file) != 8 || fwrite(block, 1, b->byte_len, c->file) != b->byte_len) return -1;
    c->bytes += 8 + b->byte_len; ++c->blocks;
    for (uint32_t i = 0; i < b->record_count; ++i) {
        const aotx_record_header *r = aotx_block_record(block, i);
        if (c->last_seq && r->seq != c->last_seq + 1) return -1;
        c->last_seq = r->seq; c->features |= aotx_runtime_appraisal_record(r);
        c->records += r->cls == AOTX_CLASS_A && r->type != AOTX_REC_BOOT && r->type != AOTX_REC_TICK_COMMIT;
    }
    if (b->tick == c->tick && b->record_count) {
        const aotx_record_header *last = aotx_block_record(block, b->record_count - 1);
        if (last->type == AOTX_REC_TICK_COMMIT && last->body_len == sizeof(aotx_commit_body)) {
            aotx_commit_body commit; memcpy(&commit, aotx_record_body(last), sizeof(commit));
            c->hash = commit.state_hash; c->complete = 1;
            return 1;
        }
    }
    (void)index;
    return 0;
}
int aotx_runtime_replay_collect_features(const char *journal, uint64_t boot, uint64_t tick,
    uint64_t memory_revision, uint64_t runtime_sequence, uint64_t limit, FILE **file, uint64_t *bytes, uint32_t *features) {
    *file = NULL; *bytes = 0;
    if (features) *features = 0;
    if (!journal || !boot || !memory_revision) return AOTX_CCIR_INVALID;
    if (limit < 128) return AOTX_CCIR_LIMIT;
    unsigned char *buffer = malloc(16u * 1024u * 1024u), header[128] = {0};
    FILE *out = buffer ? tmpfile() : NULL;
    if (!out) { free(buffer); return AOTX_CCIR_IO; }
    aotx_runtime_collect c = {0}; c.file = out; c.boot = boot; c.tick = tick;
    c.limit = limit - 128;
    int rc = fwrite(header, 1, sizeof(header), out) == sizeof(header) ? 0 : AOTX_CCIR_IO;
    uint64_t blocks = 0; int torn = 0;
    if (!rc && aotx_journal_walk(journal, buffer, 16u * 1024u * 1024u, collect, &c, &blocks, &torn))
        rc = c.status ? c.status : AOTX_CCIR_INVALID;
    if (!rc && torn) rc = AOTX_CCIR_INVALID;
    if (!rc && !c.complete) rc = AOTX_CCIR_BUSY;
    if (!rc) {
        memcpy(header, "AOTXRPL1", 8); aotx_ccir_put(header + 8, 1, 4); aotx_ccir_put(header + 12, 2, 4);
        aotx_ccir_put(header + 16, boot, 8); aotx_ccir_put(header + 24, tick, 8);
        aotx_ccir_put(header + 32, c.hash, 8); aotx_ccir_put(header + 40, c.records, 8);
        aotx_ccir_put(header + 48, c.blocks, 8); aotx_ccir_put(header + 56, c.bytes, 8);
        aotx_ccir_put(header + 64, memory_revision, 8); aotx_ccir_put(header + 72, runtime_sequence, 8);
        if (fflush(out)) rc = AOTX_CCIR_IO;
        if (!rc) rc = aotx_ccir_pwrite(fileno(out), header, sizeof(header), 0);
    }
    free(buffer);
    if (rc) { fclose(out); return rc; }
    *file = out; *bytes = 128 + c.bytes;
    if (features) *features = c.features;
    return 0;
}

int aotx_runtime_replay_collect(const char *journal, uint64_t boot, uint64_t tick,
    uint64_t memory_revision, uint64_t runtime_sequence, uint64_t limit, FILE **file, uint64_t *bytes) {
    return aotx_runtime_replay_collect_features(journal, boot, tick, memory_revision, runtime_sequence,
        limit, file, bytes, NULL);
}
