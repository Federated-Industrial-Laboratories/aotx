/* Purpose: Supply typed memory and language assets for runtime disk admission checks.
 * Owns: Distinct encoded rows and journal block batches.
 * Threading: One disk process checks batches of one and 64 source records.
 * Lifetime: The caller releases each byte buffer after its file checks. */
#ifndef AOTX_APPRAISAL_RUNTIME_DISK_FIXTURE_H
#define AOTX_APPRAISAL_RUNTIME_DISK_FIXTURE_H
#include "runtime_dependency_fixture.h"
#include "disk/runtime/appraisal.h"
#include "disk/runtime/pack.h"
#include "disk/runtime/activate.h"
#include "disk/runtime/replay.h"
#include "cognitive/checkpoint_io.h"
#include <sys/stat.h>

static const unsigned char processor[32] = AOTX_APPRAISAL_PROCESSOR_BYTES;
static void profile(aotx_dependency_fixture *f, unsigned enabled) {
    unsigned features = aotx_ccir_u32(f->index.header + 20) & ~AOTX_RUNTIME_APPRAISAL;
    aotx_ccir_put(f->index.header + 20, features, 4); memset(f->index.header + 188, 0, 68);
    if (enabled) {
        aotx_runtime_appraisal_models models = {0};
        memcpy(models.selected, f->index.rows[0] + 32, 32);
        aotx_runtime_appraisal_require(f->index.header, &models);
    }
    f->input[3].section.schema = aotx_runtime_schema(aotx_ccir_u32(f->index.header + 20));
    aotx_ccir_put(f->index.header + 16, f->index.count, 4);
    f->input[3].section.bytes = AOTX_RUNTIME_HEADER + f->index.count * AOTX_RUNTIME_ROW;
}
static void prepared(aotx_dependency_fixture *f, uint64_t bytes) {
    static const char phrases[] = "I cannot\n";
    asset(f, "quality/refusal-phrases.txt", 1, phrases, sizeof(phrases) - 1);
#ifdef AOTX_AFFECT
    aotx_ccir_put(f->index.header + 20, AOTX_RUNTIME_AFFECT, 4);
#endif
    memcpy(f->live, "AOTXLCP1", 8); aotx_ccir_put(f->live + 8, 1, 4);
    aotx_ccir_put(f->live + 12, AOTX_CP_ROW, 4); aotx_ccir_put(f->live + 24, bytes, 8);
    f->live[32] = 73;
    for (unsigned i = 48; i <= 72; i += 8) aotx_ccir_put(f->live + i, 1, 8);
}
static unsigned char *row_payload(unsigned char *memory, unsigned row) {
    uint64_t start = aotx_ccir_u64(memory + 72);
    return memory + start + aotx_ccir_u64(memory + AOTX_COG_HEADER + row * AOTX_COG_OBJECT + AOTX_CO_OFFSET);
}
static unsigned char *memory_batch(aotx_dependency_fixture *f, unsigned n, uint64_t *bytes) {
    unsigned count = 1 + 3 * n;
    uint64_t start = AOTX_COG_HEADER + (uint64_t)count * AOTX_COG_OBJECT;
    *bytes = start + AOTX_APPRAISAL_CONFIG_BYTES + (uint64_t)n *
        (AOTX_APPRAISAL_QUEUE_BYTES + AOTX_APPRAISAL_ASSESS_BYTES + AOTX_APPRAISAL_RELATION_BYTES);
    unsigned char *memory = calloc(1, (size_t)*bytes); CHECK(memory != NULL); if (!memory) return NULL;
    memcpy(memory, "AOTXOBJ1", 8); aotx_ccir_put(memory + 8, 1, 4);
    aotx_ccir_put(memory + 12, AOTX_COG_HEADER, 4); aotx_ccir_put(memory + 16, AOTX_COG_OBJECT, 4);
    aotx_ccir_put(memory + 20, count, 4); aotx_ccir_put(memory + 24, *bytes - start, 8);
    aotx_ccir_put(memory + 32, 1, 8); aotx_ccir_put(memory + 40, 1, 8); memory[48] = 73;
    aotx_ccir_put(memory + 64, AOTX_COG_HEADER, 8); aotx_ccir_put(memory + 72, start, 8);
    aotx_ccir_put(memory + 80, *bytes, 8); aotx_ccir_put(memory + 88, 1, 4);
    uint64_t at = 0;
    for (unsigned i = 0; i < count; ++i) {
        unsigned type = i ? (i - 1) % 3 + 1 : 0;
        unsigned sizes[] = {AOTX_APPRAISAL_CONFIG_BYTES, AOTX_APPRAISAL_QUEUE_BYTES,
            AOTX_APPRAISAL_ASSESS_BYTES, AOTX_APPRAISAL_RELATION_BYTES};
        unsigned char *row = memory + AOTX_COG_HEADER + i * AOTX_COG_OBJECT, *p = memory + start + at;
        aotx_ccir_put(row + AOTX_CO_KIND, type < 2 ? AOTX_COG_POLICY :
            type == 2 ? AOTX_COG_APPRAISAL : AOTX_COG_RELATIONSHIP, 2);
        aotx_ccir_put(row + 8, i + 101, 8); aotx_ccir_put(row + AOTX_CO_OFFSET, at, 8);
        aotx_ccir_put(row + AOTX_CO_BYTES, sizes[type], 8); at += sizes[type];
        if (!type) {
            memcpy(p, "AOTXAPC1", 8); aotx_ccir_put(p + 8, 1, 4); memcpy(p + 40, processor, 32);
        } else if (type == 1) {
            memcpy(p, "AOTXAPQ1", 8); aotx_ccir_put(p + 8, 1, 4);
            aotx_ccir_put(p + 12, AOTX_APPRAISAL_COMPLETE, 4);
            memcpy(p + 64, processor, 32); memcpy(p + 96, f->index.rows[0] + 32, 32);
        } else if (type == 2) {
            aotx_ccir_put(p, 2, 4); memcpy(p + 32, processor, 32); memcpy(p + 64, f->index.rows[0] + 32, 32);
        } else {
            memcpy(p, "AOTXREL1", 8); aotx_ccir_put(p + 8, 1, 4);
            memcpy(p + 72, processor, 32); memcpy(p + 104, f->index.rows[0] + 32, 32);
        }
    }
    f->input[1].data = memory; f->input[1].section.bytes = *bytes;
    prepared(f, *bytes); profile(f, 1);
    return memory;
}
static void historical(aotx_dependency_fixture *f, unsigned char model[96], unsigned char digest[32]) {
    memcpy(model, f->model, 96); model[95] ^= 3;
    asset(f, "weights/prior.gguf", 1, model, 96);
    memcpy(digest, f->index.rows[f->index.count - 1] + 32, 32);
    aotx_manifest_entry e = {0}; strcpy(e.name, "prior"); strcpy(e.role, "language-q4");
    strcpy(e.path, "weights/prior.gguf"); strcpy(e.source, "local:prior");
    strcpy(e.revision, "1"); strcpy(e.license, "Apache-2.0"); e.bytes = 96;
    aotx_sha256_text(digest, e.sha256); size_t used = strlen(f->models);
    CHECK(!aotx_manifest_write_line(f->models + used, sizeof(f->models) - used, &e));
    for (unsigned i = 0; i < f->index.count; ++i) if (!strcmp((char *)f->index.rows[i] + 64, "manifest.jsonl")) {
        aotx_ccir_put(f->index.rows[i] + 24, strlen(f->models), 8);
        aotx_ccir_hash(f->models, strlen(f->models), f->index.rows[i] + 32);
        f->input[i + 5].section.bytes = strlen(f->models);
    }
    profile(f, 1);
}
#define APPRAISAL_BLOCK (AOTX_BLOCK_HEADER_BYTES + 2 * AOTX_SLOT_BYTES)
static void block_batch(unsigned char *blocks, unsigned n, unsigned operation) {
    memset(blocks, 0, (size_t)n * APPRAISAL_BLOCK);
    for (unsigned i = 0; i < n; ++i) {
        aotx_block_header *b = (aotx_block_header *)(blocks + i * APPRAISAL_BLOCK);
        b->magic = AOTX_BLOCK_MAGIC; b->layout = AOTX_WIRE_LAYOUT; b->byte_len = APPRAISAL_BLOCK;
        b->block_seq = i + 1; b->boot_id = 75; b->tick = i + 1; b->record_count = 2;
        for (unsigned j = 0; j < 2; ++j) {
            aotx_record_header *r = (aotx_record_header *)((unsigned char *)b + AOTX_BLOCK_HEADER_BYTES + j * AOTX_SLOT_BYTES);
            r->magic = AOTX_WIRE_MAGIC; r->layout = AOTX_WIRE_LAYOUT; r->header_bytes = AOTX_HEADER_BYTES;
            r->boot_id = 75; r->tick = i + 1; r->seq = 2 * i + j + 1; r->cls = AOTX_CLASS_A;
            r->type = j ? AOTX_REC_TICK_COMMIT : 33; r->body_len = j ? sizeof(aotx_commit_body) : 32;
            unsigned char *body = (unsigned char *)r + AOTX_HEADER_BYTES;
            if (j) ((aotx_commit_body *)body)->state_hash = 1235 + i;
            else { aotx_ccir_put(body, 1, 4); aotx_ccir_put(body + 4, operation, 4); aotx_ccir_put(body + 8, i + 101, 8); }
        }
    }
}
#endif
