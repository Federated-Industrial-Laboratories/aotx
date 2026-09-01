/* Purpose: Give the drain check the arm of the successful model load note.
 * Owns: Nothing; the check that includes this file holds the drain and temporary files.
 * Threading: One process while the drain reads the fake device ring.
 * Lifetime: One case of the check.
 *
 * The file is included by tests/derive_test.c after the parts that every arm shares. */
#ifndef AOTX_TESTS_DERIVE_MODEL_H
#define AOTX_TESTS_DERIVE_MODEL_H

/* Model records are successful load results. Each complete derived line names the role,
 * file, record tick, model tick, source, and boot. A changed line must not compare equal. */
static void model_results(int n)
{
    run_ctx c;
    aotx_model_body model;
    aotx_commit_body commit;
    char path[1024];
    char stamp[64];
    char want[1024];
    char changed[1024];
    char *at;
    int i;

    start(&c, 0x00a05e0000000000ull + (uint64_t)n, "bus");
    for (i = 0; i < n; i++) {
        memset(&model, 0, sizeof(model));
        model.tick = 37u + (uint64_t)i;
        snprintf(model.role, sizeof(model.role), "language-%d", i);
        snprintf(model.file, sizeof(model.file), "model-%d-q8.gguf", i);
        c.device.writer = AOTX_WRITER_CONSOLE;
        aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_MODEL, &model, sizeof(model));
    }
    memset(&commit, 0, sizeof(commit));
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);

    bus_path(&c, path, sizeof(path));
    CHECK(slurp(path, text, sizeof(text)) > 0, "the model result lines do not read");
    CHECK(count_of(text, "\n") == n, "the file holds %d model results, not %d",
          count_of(text, "\n"), n);
    at = text;
    for (i = 0; i < n; i++) {
        char *end = strchr(at, '\n');
        size_t actual = (end != NULL) ? (size_t)(end - at) + 1u : 0u;
        CHECK(end != NULL, "model result %d has no end byte", i);
        CHECK(aotx_json_text(at, "\"ts\":\"", stamp, sizeof(stamp)),
              "model result %d has no time", i);
        snprintf(want, sizeof(want),
                 "{\"v\":1,\"run\":\"aotx\",\"agent\":\"console\",\"seq\":%d,"
                 "\"ts\":\"%s\",\"type\":\"note\",\"body\":{\"text\":"
                 "\"model language-%d loaded model-%d-q8.gguf at tick %d\",\"tick\":1,"
                 "\"boot\":\"%016llx\",\"lag_ms\":null}}\n",
                 i + 1, stamp, i, i, 37 + i,
                 (unsigned long long)(0x00a05e0000000000ull + (uint64_t)n));
        CHECK(actual == strlen(want) && strncmp(at, want, actual) == 0,
              "model result %d differs as a whole", i);
        if (i == 0 && actual < sizeof(changed)) {
            char *loaded;
            memcpy(changed, at, actual);
            changed[actual] = '\0';
            loaded = strstr(changed, " loaded ");
            CHECK(loaded != NULL, "the model result word is not present");
            if (loaded != NULL) loaded[1] = 'x';
            CHECK(strcmp(changed, want) != 0,
                  "the model result check accepted a changed line");
        }
        at = (end != NULL) ? end + 1 : at;
    }
    printf("model results %d: exact lines %d, changed lines refused 1\n", n, n);
    aotx_remove_tree(c.dir);
}

#endif
