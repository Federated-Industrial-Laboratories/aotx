/* Purpose: Check historical model dependencies in split appraisal result records.
 * Owns: Distinct result rows, interruption markers and complete journal frames.
 * Threading: One disk process checks batches of one and 64 model identities.
 * Lifetime: The caller removes every source and destination file. */
#ifndef AOTX_APPRAISAL_RUNTIME_DISK_REPLAY_H
#define AOTX_APPRAISAL_RUNTIME_DISK_REPLAY_H
static void result_frame(unsigned char *frame, unsigned number, unsigned total, unsigned offset,
    const unsigned char *data, unsigned bytes, unsigned flags) {
    aotx_ccir_put(frame, APPRAISAL_BLOCK, 8);
    block_batch(frame + 8, 1, AOTX_APPRAISAL_RESULT);
    aotx_block_header *b = (aotx_block_header *)(frame + 8);
    b->block_seq = b->tick = number + 1;
    for (unsigned j = 0; j < 2; ++j) {
        aotx_record_header *r = (aotx_record_header *)((unsigned char *)b + AOTX_BLOCK_HEADER_BYTES + j * AOTX_SLOT_BYTES);
        r->seq = 2 * number + j + 1; r->tick = number + 1;
        unsigned char *p = (unsigned char *)r + AOTX_HEADER_BYTES;
        if (j) ((aotx_commit_body *)p)->state_hash = 1235 + number;
        else {
            r->body_len = bytes + 32; r->flags = (uint16_t)flags;
            aotx_ccir_put(p + 8, 71, 8); aotx_ccir_put(p + 24, total, 4); aotx_ccir_put(p + 28, offset, 4);
            memcpy(p + 32, data, bytes);
        }
    }
}
static void replay_header(unsigned char *p, unsigned blocks) {
    memcpy(p, "AOTXRPL1", 8); aotx_ccir_put(p + 8, 1, 4); aotx_ccir_put(p + 12, 2, 4);
    aotx_ccir_put(p + 16, 75, 8); aotx_ccir_put(p + 24, blocks, 8);
    aotx_ccir_put(p + 32, 1234 + blocks, 8); aotx_ccir_put(p + 40, blocks, 8);
    aotx_ccir_put(p + 48, blocks, 8); aotx_ccir_put(p + 56, (uint64_t)blocks * (8 + APPRAISAL_BLOCK), 8);
    aotx_ccir_put(p + 64, 1, 8); aotx_ccir_put(p + 72, 2 * blocks - 1, 8);
}
static void replay_mirror(const char *root, aotx_dependency_fixture *f, unsigned char *replay,
    unsigned frames, int valid) {
    char path[256], journal[256], segment[288]; snprintf(path, sizeof(path), "%s/history-mirror.aotxccir", root);
    snprintf(journal, sizeof(journal), "%s/history-journal", root); snprintf(segment, sizeof(segment), "%s/seg-000000.seg", journal);
    CHECK(!mkdir(journal, 0700)); unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
    f->input[4].data = f->replay; f->input[4].section.bytes = 128;
    aotx_ccir_put(f->live + 72, 1, 8);
    CHECK(!aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL));
    aotx_checkpoint_ring ring = {0}; ring.boot = 75;
    aotx_checkpoint_disk disk = {0}; disk.view.fd = -1; disk.runtime = 1; disk.runtime_sequence = 2 * frames - 1;
    disk.path = path; disk.journal = journal; disk.ring = &ring;
    CHECK(!aotx_ccir_writer_open(path, NULL, &disk.view));
    aotx_runtime_revision(&disk.view, disk.runtime_revision);
    aotx_segment_writer writer; CHECK(!aotx_segment_open(&writer, journal, 0));
    for (unsigned i = 0; i < frames; ++i)
        CHECK(!aotx_segment_put(&writer, replay + 136 + i * (8 + APPRAISAL_BLOCK), APPRAISAL_BLOCK));
    CHECK(!aotx_segment_close(&writer));
    unsigned char image[256]; memcpy(image, f->live, 128); memcpy(image + 128, f->memory, 128);
    aotx_ccir_put(image + 72, frames, 8);
    uint64_t generation = disk.view.generation;
    int rc = aotx_checkpoint_file_write(&disk, image, sizeof(image));
    CHECK((rc == 0) == valid); CHECK(valid ? disk.view.generation > generation : disk.view.generation == generation);
    CHECK(!aotx_runtime_dependencies(&disk.view));
    unsigned char *saved = NULL; uint32_t saved_bytes = 0;
    CHECK(!aotx_checkpoint_file_read(path, &saved, &saved_bytes));
    CHECK(saved && saved_bytes == sizeof(image));
    if (saved && saved_bytes == sizeof(image)) {
        CHECK(!memcmp(saved + 128, image + 128, 128));
        CHECK(!memcmp(saved, image, 72) && !memcmp(saved + 80, image + 80, 48));
        CHECK(aotx_ccir_u64(saved + 72) == (valid ? frames : 1));
    }
    free(saved);
    aotx_ccir_close(&disk.view); CHECK(!unlink(segment)); CHECK(!rmdir(journal)); CHECK(!unlink(path));
}
static void fragmented(const char *root, unsigned n, unsigned defect, unsigned transport) {
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    make(f, n, 0); prepared(f, 128); profile(f, 1);
    memcpy(f->memory, "AOTXOBJ1", 8); aotx_ccir_put(f->memory + 8, 1, 4);
    aotx_ccir_put(f->memory + 12, 128, 4); aotx_ccir_put(f->memory + 16, 256, 4);
    aotx_ccir_put(f->memory + 32, 1, 8); aotx_ccir_put(f->memory + 40, 1, 8); f->memory[48] = 73;
    aotx_ccir_put(f->memory + 64, 128, 8); aotx_ccir_put(f->memory + 72, 128, 8); aotx_ccir_put(f->memory + 80, 128, 8);
    unsigned char prior[96], digest[32]; memcpy(digest, f->index.rows[0] + 32, 32);
    if (defect == 1) historical(f, prior, digest);
    unsigned version = defect >= 21 ? 2 : 1, kind = defect >= 21 ? defect - 21 : 0;
    unsigned stride = version == 2 ? AOTX_APPRAISAL_RESULT_ROW : AOTX_APPRAISAL_LEGACY_ROW;
    unsigned chunk = version == 2 && kind == 14 ? 37 : 101;
    unsigned total = 64 + n * stride, parts = (total + chunk - 1) / chunk;
    unsigned char *result = calloc(1, total); CHECK(result != NULL); if (!result) { free(f); return; }
    memcpy(result, "AOTXAPS1", 8); aotx_ccir_put(result + 8, version, 4); aotx_ccir_put(result + 12, n, 4);
    aotx_ccir_put(result + 16, 1, 8); aotx_ccir_put(result + 32, AOTX_COG_DENIED, 4);
    for (unsigned i = 0; i < n; ++i) {
        unsigned char *r = result + 64 + i * stride;
        aotx_ccir_put(r, 101 + i, 8); aotx_ccir_put(r + 16, 1, 8); memcpy(r + 24, digest, 32);
        aotx_ccir_put(r + 60, AOTX_COG_DENIED, 4);
    }
    if (defect == 2) result[64 + (n - 1) * stride + 24] ^= 1;
    if (defect == 3) result[88] ^= 1;
    if (defect == 10) aotx_ccir_put(result + 8, 2, 4);
    if (defect >= 13 && defect <= 15) {
        for (unsigned i = 0; i < n; ++i) {
            unsigned char *r = result + 64 + i * stride;
            if (defect != 15) memset(r + 24, 0, 32);
            if (defect >= 14) aotx_ccir_put(r + 60, 0, 4);
        }
        if (defect >= 14) aotx_ccir_put(result + 32, 0, 4);
    }
    if (defect == 20) aotx_ccir_put(result + 32, 0, 4);
    if (version == 2) {
        for (unsigned i = 0; i < n; ++i) {
            unsigned char *r = result + 64 + i * stride;
            unsigned phase = kind < 3 ? kind : 2, second = kind >= 3;
            if (phase) {
                char text[80]; int bytes = snprintf(text, sizeof(text), "[\"source %u\"]", i);
                aotx_ccir_put(r + 4160, (unsigned)bytes, 4); memcpy(r + 4224, text, (size_t)bytes);
                memcpy(r + 4192, digest, 32);
            }
            aotx_ccir_put(r + 4164, second, 4); aotx_ccir_put(r + 4168, phase, 4);
            if (second) { aotx_ccir_put(r + 56, 1, 4); r[64] = '{'; }
            if (kind == 4) aotx_ccir_put(r + 60, 0, 4);
            if (kind == 13) aotx_ccir_put(r + 60, AOTX_COG_UNAVAILABLE, 4);
        }
        if (kind == 4) aotx_ccir_put(result + 32, 0, 4);
        if (kind == 13) aotx_ccir_put(result + 32, AOTX_COG_UNAVAILABLE, 4);
        unsigned char *last = result + 64 + (n - 1) * stride;
        if (kind == 5) last[4192] ^= 1;
        if (kind == 6) memset(last + 4192, 0, 32);
        if (kind == 7) aotx_ccir_put(last + 4168, 3, 4);
        if (kind == 8) aotx_ccir_put(last + 4164, 2, 4);
        if (kind == 9) aotx_ccir_put(last + 4160, 4097, 4);
        if (kind == 10) last[8319] = 1;
        if (kind == 11) aotx_ccir_put(last + 4168, 1, 4);
        if (kind == 12) aotx_ccir_put(last + 4168, 0, 4);
        if (kind == 15) last[4172] = 1;
        if (kind == 16) last[4159] = 1;
        if (kind == 17) aotx_ccir_put(last + 56, 4097, 4);
        if (kind == 18) { memset(last + 24, 0, 32); memset(last + 4192, 0, 32); }
    }
    int marker = defect >= 3 && defect <= 6;
    if (marker) parts = 3;
    if (defect == 11 || defect == 12) parts = 2;
    size_t bytes = 128 + (size_t)parts * (8 + APPRAISAL_BLOCK);
    unsigned char *replay = calloc(1, bytes); CHECK(replay != NULL);
    if (!replay) { free(result); free(f); return; }
    for (unsigned i = 0; i < parts; ++i) {
        unsigned offset = i * chunk, take = total - offset < chunk ? total - offset : chunk;
        unsigned char denied[64] = {0}; const unsigned char *data = result + offset;
        unsigned length = total, flags = transport;
        if (marker && i == 2) {
            memcpy(denied, "AOTXAPS1", 8); aotx_ccir_put(denied + 8, 1, 4);
            aotx_ccir_put(denied + 32, defect == 6 ? AOTX_COG_FORMAT : AOTX_COG_DENIED, 4);
            offset = 0; take = length = 64; data = denied; flags |= defect == 5 ? 0 : AOTX_FLAG_ADMISSION;
        }
        if ((defect == 16 && !i) || (defect == 17 && i == 1)) flags |= 16;
        if (defect == 18 && i == 1) flags |= AOTX_FLAG_ADMISSION;
        if (defect == 19 && !i) flags |= AOTX_FLAG_FRAGMENT;
        if (defect == 20 && !i) flags |= AOTX_FLAG_ADMISSION;
        unsigned char *frame = replay + 128 + i * (8 + APPRAISAL_BLOCK);
        result_frame(frame, i, length, offset, data, take, flags);
        unsigned char *part = frame + 8 + AOTX_BLOCK_HEADER_BYTES + AOTX_HEADER_BYTES;
        if (i == 1 && defect == 7) part[8] ^= 1;
        if (i == 1 && defect == 8) aotx_ccir_put(part + 24, total + 1, 4);
        if (i == 1 && defect == 9) aotx_ccir_put(part + 28, offset + 1, 4);
        if (i == 1 && defect == 12) aotx_ccir_put(part + 4, AOTX_APPRAISAL_REQUEST, 4);
    }
    replay_header(replay, parts); f->input[4].data = replay; f->input[4].section.bytes = bytes;
    aotx_ccir_put(f->live + 72, parts, 8);
    int valid = defect == 0 || defect == 1 || defect == 4 || defect == 13 || defect == 15 ||
        (version == 2 && (kind <= 4 || kind == 13 || kind == 14));
    char path[256]; snprintf(path, sizeof(path), "%s/history.aotxccir", root);
    unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
    CHECK(!aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL));
    aotx_ccir_view view; int rc = aotx_ccir_open(path, NULL, &view); CHECK(!rc);
    if (!rc) { rc = aotx_runtime_dependencies(&view);
        if ((rc == 0) != valid) fprintf(stderr, "history n=%u case=%u flags=%u status=%d\n", n, defect, transport, rc);
        CHECK((rc == 0) == valid); aotx_ccir_close(&view); }
    CHECK(!unlink(path));
    if (defect == 0 || defect == 2 || defect == 4 || (defect >= 16 && defect <= 20)) replay_mirror(root, f, replay, parts, valid);
    free(replay); free(result); free(f);
}
#endif
