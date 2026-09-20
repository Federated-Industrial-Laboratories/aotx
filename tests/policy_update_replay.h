/* Purpose: Supply saved policy fragments and detecting disk admission faults.
 * Owns: Distinct decision batches across selected and historical revisions.
 * Threading: One disk test process; private state bytes are opaque.
 * Lifetime: Replay buffers and temporary files belong to each test case. */
#ifndef AOTX_POLICY_UPDATE_REPLAY_H
#define AOTX_POLICY_UPDATE_REPLAY_H
static unsigned char *saved_replay(aotx_dependency_fixture *f, unsigned n,
    const aotx_policy_file *policies, unsigned revisions, unsigned fault) {
    unsigned decisions = n * revisions, records = 2 * decisions + 1;
    size_t block_bytes = 64 + records * AOTX_SLOT_BYTES, bytes = 136 + block_bytes;
    unsigned char *p = calloc(1, bytes); CHECK(p != NULL); if (!p) exit(1);
    memcpy(p, "AOTXRPL1", 8); aotx_ccir_put(p + 8, 1, 4); aotx_ccir_put(p + 12, 2, 4);
    aotx_ccir_put(p + 16, 75, 8); aotx_ccir_put(p + 24, 17, 8); aotx_ccir_put(p + 32, 1234, 8);
    aotx_ccir_put(p + 40, records - 1, 8); aotx_ccir_put(p + 48, 1, 8);
    aotx_ccir_put(p + 56, bytes - 128, 8); aotx_ccir_put(p + 64, 7, 8);
    aotx_ccir_put(p + 72, 1, 8); aotx_ccir_put(p + 128, block_bytes, 8);
    aotx_block_header *block = (aotx_block_header *)(p + 136);
    block->magic = AOTX_BLOCK_MAGIC; block->layout = AOTX_WIRE_LAYOUT; block->byte_len = block_bytes;
    block->boot_id = 75; block->tick = 17; block->block_seq = 1; block->record_count = records;
    for (unsigned i = 0; i < records; ++i) {
        aotx_record_header *r = (aotx_record_header *)(p + 200 + i * AOTX_SLOT_BYTES);
        r->magic = AOTX_WIRE_MAGIC; r->layout = AOTX_WIRE_LAYOUT; r->header_bytes = 64;
        r->boot_id = 75; r->tick = 17; r->seq = i + 1; r->cls = AOTX_CLASS_A;
        unsigned char *body = (unsigned char *)r + 64;
        if (i + 1 == records) {
            r->type = AOTX_REC_TICK_COMMIT; r->body_len = sizeof(aotx_commit_body);
            ((aotx_commit_body *)body)->state_hash = 1234; continue;
        }
        unsigned decision = i / 2 + 1, revision = (decision - 1) / n;
        unsigned offset = i % 2 ? 160 : 0, count = i % 2 ? 112 : 160;
        unsigned char event[272] = {0}; memcpy(event, "AOTXPD01", 8);
        aotx_ccir_put(event + 8, 1, 4); aotx_ccir_put(event + 12, 1, 4);
        aotx_ccir_put(event + 16, 16, 4); aotx_ccir_put(event + 24, decision, 8);
        memcpy(event + 32, policies[revision].digest, 32);
        for (unsigned j = 64; j < sizeof(event); ++j) event[j] = (unsigned char)(decision * 13 + j * 7);
        if (fault >= 1 && fault <= revisions && revision + 1 == fault) event[32] ^= 1;
        r->type = AOTX_REC_POLICY; r->body_len = 32 + count;
        aotx_ccir_put(body, 1, 4); aotx_ccir_put(body + 4, sizeof(event), 4);
        aotx_ccir_put(body + 8, offset, 4); aotx_ccir_put(body + 12, count, 4);
        aotx_ccir_put(body + 16, decision, 8); memcpy(body + 32, event + offset, count);
        if (decision == n + 1) {
            if (fault == 4 && i % 2) aotx_ccir_put(body + 8, 159, 4);
            if (fault == 5) aotx_ccir_put(body + 16, decision + 1, 8);
            if (fault == 6 && i % 2) { r->body_len--; aotx_ccir_put(body + 12, count - 1, 4); }
            if (fault == 7 && i % 2) aotx_ccir_put(body + 4, sizeof(event) + 1, 4);
        }
    }
    f->input[4].data = p; f->input[4].section.bytes = bytes;
    memcpy(f->live, "AOTXLCP1", 8); aotx_ccir_put(f->live + 64, 7, 8); aotx_ccir_put(f->live + 72, 17, 8);
    return p;
}
static void replay_cases(unsigned n, unsigned mode) {
    char root[] = "/tmp/aotx-policy-replay-XXXXXX", paths[3][256], path[256];
    CHECK(mkdtemp(root) != NULL); snprintf(path, sizeof(path), "%s/replay", root);
    aotx_policy_file policies[3] = {0}; aotx_policy_history history = {0}; history.count = 2;
    for (unsigned i = 0; i < 3; ++i) {
        snprintf(paths[i], sizeof(paths[i]), "%s/policy-%u", root, i);
        bundle(paths[i], n + 29 + i, mode, 1, policies + i);
        if (i < 2) {
            history.rows[i].config = policies[i].config;
            memcpy(history.rows[i].digest, policies[i].digest, 32); history.rows[i].last_decision = (i + 1) * n;
        }
    }
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) exit(1);
    for (unsigned fault = 0; fault < 10; ++fault) {
        make(f, n, 0); unsigned char *p = saved_replay(f, n, policies, 3, fault);
        int fd = open(path, O_CREAT | O_EXCL | O_RDWR, 0600); CHECK(fd >= 0); if (fd < 0) exit(1);
        CHECK(!aotx_ccir_pwrite(fd, p, f->input[4].section.bytes, 0));
        aotx_ccir_view view = {0}; view.fd = fd; view.count = 1; view.sections[0] = f->input[4].section;
        view.sections[0].offset = 0;
        aotx_policy_history changed = history;
        if (fault == 8) changed.rows[0].last_decision = n + 1;
        if (fault == 9) changed.rows[1].last_decision = 3 * n + 1;
        uint64_t last = 0;
        int rc = aotx_runtime_policy_replay(&view, policies + 2, &changed, &last);
        CHECK(rc == (fault ? AOTX_CCIR_INVALID : 0)); CHECK(last == (fault ? 0 : 3 * n));
        CHECK(!close(fd)); CHECK(!unlink(path)); free(p);
    }
    free(f);
    for (unsigned i = 0; i < 3; ++i) { aotx_policy_file_close(policies + i); CHECK(!unlink(paths[i])); }
    CHECK(!rmdir(root));
}
#endif
