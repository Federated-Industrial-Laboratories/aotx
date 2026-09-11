/* Purpose: Check complete runtime replay batches and damaged recovery boundaries.
 * Owns: Encoded blocks and one temporary file for each case.
 * Threading: One process validates batches of one and 64 complete ticks.
 * Lifetime: All source descriptors and buffers end with the case. */
#include "disk/runtime/replay.h"
#include "disk/ccir/internal.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static unsigned checks, failures;
#define CHECK(value) do { ++checks; if (!(value)) { ++failures; fprintf(stderr, "line %d failed\n", __LINE__); } } while (0)
#define BLOCK (AOTX_BLOCK_HEADER_BYTES + 2 * AOTX_SLOT_BYTES)
static int count_block(void *context, const unsigned char *bytes, uint64_t index) {
    unsigned *count = context;
    CHECK(index == *count);
    CHECK(((const aotx_block_header *)bytes)->record_count == 2);
    ++*count;
    return 0;
}
static void test(unsigned n) {
    size_t bytes = 128 + n * (8 + BLOCK);
    unsigned char *data = calloc(1, bytes), buffer[BLOCK];
    CHECK(data != NULL); if (!data) return;
    memcpy(data, "AOTXRPL1", 8); aotx_ccir_put(data + 8, 1, 4); aotx_ccir_put(data + 12, 2, 4);
    aotx_ccir_put(data + 16, 75, 8); aotx_ccir_put(data + 24, n, 8);
    aotx_ccir_put(data + 32, 1234 + n, 8); aotx_ccir_put(data + 40, n, 8);
    aotx_ccir_put(data + 48, n, 8); aotx_ccir_put(data + 56, bytes - 128, 8);
    aotx_ccir_put(data + 64, 7, 8); aotx_ccir_put(data + 72, 2 * n - 1, 8);
    for (unsigned i = 0; i < n; ++i) {
        unsigned char *frame = data + 128 + i * (8 + BLOCK);
        aotx_ccir_put(frame, BLOCK, 8);
        aotx_block_header *b = (aotx_block_header *)(frame + 8);
        b->magic = AOTX_BLOCK_MAGIC; b->layout = AOTX_WIRE_LAYOUT; b->byte_len = BLOCK;
        b->block_seq = i + 1; b->boot_id = 75; b->tick = i + 1; b->record_count = 2;
        for (unsigned j = 0; j < 2; ++j) {
            aotx_record_header *r = (aotx_record_header *)((unsigned char *)b + AOTX_BLOCK_HEADER_BYTES + j * AOTX_SLOT_BYTES);
            r->magic = AOTX_WIRE_MAGIC; r->layout = AOTX_WIRE_LAYOUT; r->header_bytes = AOTX_HEADER_BYTES;
            r->boot_id = 75; r->tick = i + 1; r->seq = 2 * i + j + 1; r->cls = AOTX_CLASS_A;
            r->type = j ? AOTX_REC_TICK_COMMIT : AOTX_REC_INPUT_LINE;
            r->body_len = j ? sizeof(aotx_commit_body) : 1;
            unsigned char *body = (unsigned char *)r + AOTX_HEADER_BYTES;
            if (j) ((aotx_commit_body *)body)->state_hash = 1235 + i;
            else *body = (unsigned char)('a' + i % 26);
        }
    }
    FILE *file = tmpfile(); CHECK(file != NULL); if (!file) { free(data); return; }
    aotx_ccir_view view = {0}; view.fd = fileno(file); view.count = 1;
    view.sections[0].type = AOTX_CCIR_REPLAY; view.sections[0].flags = AOTX_CCIR_REQUIRED;
    view.sections[0].bytes = bytes;
    CHECK(aotx_ccir_pwrite(view.fd, data, bytes, 0) == 0);
    aotx_journal_scan scan; unsigned seen = 0;
    CHECK(aotx_runtime_replay_walk(&view, buffer, sizeof(buffer), count_block, &seen, &scan) == 0);
    CHECK(seen == n && scan.state_hash == 1234 + n && scan.last_tick == n);
    for (unsigned defect = 0; defect < 9; ++defect) {
        unsigned char *bad = malloc(bytes); CHECK(bad != NULL); if (!bad) continue;
        memcpy(bad, data, bytes);
        aotx_block_header *first = (aotx_block_header *)(bad + 136);
        aotx_record_header *r = (aotx_record_header *)((unsigned char *)first + AOTX_BLOCK_HEADER_BYTES);
        if (defect == 0) bad[32] ^= 1;
        if (defect == 1) aotx_ccir_put(bad + 40, n + 1, 8);
        if (defect == 2) aotx_ccir_put(bad + 128, BLOCK + 1, 8);
        if (defect == 3) first->boot_id ^= 1;
        if (defect == 4) r->seq = 0;
        if (defect == 5) r->boot_id ^= 1;
        if (defect == 6) aotx_ccir_put(bad + 72, 2 * n + 1, 8);
        if (defect == 7) first->kind = AOTX_BLOCK_PAD;
        if (defect == 8) {
            aotx_block_header *last = (aotx_block_header *)(bad + 136 + (n - 1) * (8 + BLOCK));
            aotx_record_header *commit = (aotx_record_header *)((unsigned char *)last + AOTX_BLOCK_HEADER_BYTES + AOTX_SLOT_BYTES);
            commit->cls = AOTX_CLASS_B;
        }
        CHECK(aotx_ccir_pwrite(view.fd, bad, bytes, 0) == 0);
        CHECK(aotx_runtime_replay_walk(&view, buffer, sizeof(buffer), NULL, NULL, &scan) != 0);
        free(bad);
    }
    CHECK(aotx_ccir_pwrite(view.fd, data, bytes, 0) == 0);
    CHECK(ftruncate(view.fd, bytes - 1) == 0);
    CHECK(aotx_runtime_replay_walk(&view, buffer, sizeof(buffer), NULL, NULL, &scan) != 0);
    fclose(file); free(data);
}
int main(void) {
    test(1); test(64);
    printf("runtime replay: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
