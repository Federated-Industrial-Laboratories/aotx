/* Purpose: Check the affect and quality streams at one and many agents.
 * Owns: Nothing; the derive fixture owns the fake ring and temporary files.
 * Threading: Two processes while the drain reads the fake device ring.
 * Lifetime: One part of the derive check. */
#ifndef AOTX_TEST_DERIVE_AFFECT_H
#define AOTX_TEST_DERIVE_AFFECT_H

#ifdef AOTX_AFFECT
static void affect_streams(int n)
{
    run_ctx c;
    aotx_affect_trace_body affect;
    aotx_quality_body quality;
    aotx_commit_body commit;
    char path[1024];
    char want[1024];
    int i;

    start(&c, 0x00a0af0000000000ull + (uint64_t)n, "affect,quality");
    for (i = 0; i < n; i++) {
        memset(&affect, 0, sizeof(affect));
        affect.agent = (uint32_t)i;
        affect.turn = (uint32_t)(i + 1);
        affect.prompt[0] = 0.25f;
        affect.prompt[1] = -0.5f;
        affect.reply[0] = 0.75f;
        affect.reply[1] = 0.125f;
        affect.guard[0] = 0.5f;
        affect.guard[1] = -0.25f;
        affect.logprob = -0.75f;
        affect.entropy = 1.5f;
        affect.rows = (uint32_t)(20 + i);
        affect.think = (uint32_t)(i & 1);
        affect.reason = (1u << 0) | (1u << 4);
        affect.effective[0] = 8192;
        affect.effective[1] = -4096;
        affect.flags = 1u;
        c.device.writer = AOTX_WRITER_AGENT_BASE + (uint32_t)i;
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_AFFECT_TRACE,
                         &affect, sizeof(affect));

        memset(&quality, 0, sizeof(quality));
        quality.agent = (uint32_t)i;
        quality.turn = (uint32_t)(i + 1);
        quality.repetition = 0.25f;
        quality.tokens = 12u;
        quality.limit = 12u;
        quality.refusal = (uint32_t)(i & 1);
        quality.guard[0] = 0.5f;
        quality.guard[1] = -0.25f;
        if ((i & 1) == 0) {
            quality.coherence_prompt = 0.75f;
            quality.flags = 5u;
        } else {
            quality.coherence_turn = -0.5f;
            quality.flags = 6u;
        }
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_QUALITY,
                         &quality, sizeof(quality));
    }

    /* Invalid bodies make no line in either stream. */
    memset(&affect, 0, sizeof(affect));
    affect.entropy = 1.0f;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_AFFECT_TRACE,
                     &affect, sizeof(affect));
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_AFFECT_TRACE,
                     &affect, sizeof(affect) - 1u);
    affect.agent = 64u;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_AFFECT_TRACE,
                     &affect, sizeof(affect));
    affect.agent = 0u;
    affect.prompt[0] = NAN;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_AFFECT_TRACE,
                     &affect, sizeof(affect));

    memset(&quality, 0, sizeof(quality));
    quality.limit = 1u;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_QUALITY,
                     &quality, sizeof(quality));
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_QUALITY,
                     &quality, sizeof(quality) - 1u);
    quality.agent = 64u;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_QUALITY,
                     &quality, sizeof(quality));
    quality.agent = 0u;
    quality.repetition = NAN;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_QUALITY,
                     &quality, sizeof(quality));

    memset(&commit, 0, sizeof(commit));
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT,
                     &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);

    snprintf(path, sizeof(path), "%s/affect.jsonl", c.boot_dir);
    CHECK(slurp(path, text, sizeof(text)) > 0, "the affect stream does not read");
    CHECK(count_of(text, "\n") == n, "the affect stream holds %d lines and %d were asked for",
          count_of(text, "\n"), n);
    for (i = 0; i < n; i++) {
        snprintf(want, sizeof(want),
                 "{\"tick\":1,\"agent\":%d,\"turn\":%d,\"kind\":\"trace\","
                 "\"prompt\":[0.25,-0.5,0,0],\"reply\":[0.75,0.125,0,0],"
                 "\"guard\":[0.5,-0.25],\"logprob\":-0.75,\"entropy\":1.5,"
                 "\"rows\":%d,\"think\":%d,\"reason\":[\"stop\",\"tool_ok\"],"
                 "\"effective\":[0.25,-0.125,0,0],\"flags\":1}\n",
                 i, i + 1, 20 + i, i & 1);
        CHECK(strstr(text, want) != NULL, "affect line %d is not exact", i);
    }

    snprintf(path, sizeof(path), "%s/quality.jsonl", c.boot_dir);
    CHECK(slurp(path, text, sizeof(text)) > 0, "the quality stream does not read");
    CHECK(count_of(text, "\n") == n, "the quality stream holds %d lines and %d were asked for",
          count_of(text, "\n"), n);
    for (i = 0; i < n; i++) {
        snprintf(want, sizeof(want),
                 "{\"tick\":1,\"agent\":%d,\"turn\":%d,\"coherence_prompt\":%s,"
                 "\"coherence_turn\":%s,\"repetition\":0.25,\"tokens\":12,\"limit\":12,"
                 "\"limit_hit\":1,\"refusal\":%d,\"guard\":[0.5,-0.25],\"flags\":%d}\n",
                 i, i + 1, (i & 1) ? "null" : "0.75", (i & 1) ? "-0.5" : "null",
                 i & 1, (i & 1) ? 6 : 5);
        CHECK(strstr(text, want) != NULL, "quality line %d is not exact", i);
    }
    printf("affect streams %d: exact affect and quality lines %d\n", n, n);
    aotx_remove_tree(c.dir);
}
#else
/* A build without the option journals the layouts and makes no optional stream. */
static void affect_streams_pass_over(void)
{
    run_ctx c;
    aotx_affect_trace_body affect;
    aotx_quality_body quality;
    aotx_commit_body commit;
    char path[1024];
    int records;
    start(&c, 0x00a0af0000000001ull, NULL);
    memset(&affect, 0, sizeof(affect));
    memset(&quality, 0, sizeof(quality));
    c.device.writer = AOTX_WRITER_AGENT_BASE;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_AFFECT_TRACE,
                     &affect, sizeof(affect));
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_QUALITY,
                     &quality, sizeof(quality));
    memset(&commit, 0, sizeof(commit));
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT,
                     &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);
    records = journal_records(c.boot_dir);
    snprintf(path, sizeof(path), "%s/affect.jsonl", c.boot_dir);
    CHECK(access(path, F_OK) != 0, "the build without the option made an affect stream");
    snprintf(path, sizeof(path), "%s/quality.jsonl", c.boot_dir);
    CHECK(access(path, F_OK) != 0, "the build without the option made a quality stream");
    CHECK(records == 3, "the journal holds %d records and 3 were written", records);
    printf("affect streams off: two optional records passed to the journal\n");
    aotx_remove_tree(c.dir);
}

/* The optional stream names are unknown when the option is not in the build. */
static void affect_names_refused(void)
{
    static const char *names[2] = { "affect", "quality" };
    int i;
    for (i = 0; i < 2; i++) {
        char *args[8];
        int child;
        args[0] = arguments[1];
        args[1] = (char *)"--ring-fd";
        args[2] = (char *)"0";
        args[3] = (char *)"--journal";
        args[4] = (char *)"unused";
        args[5] = (char *)"--derive";
        args[6] = (char *)names[i];
        args[7] = NULL;
        child = aotx_spawn(args, -1, -1);
        CHECK(child > 0, "the drain refusal does not start");
        CHECK(aotx_wait(child) != 0, "the build without the option took the name %s", names[i]);
    }
    printf("affect streams off: affect and quality names refused\n");
}
#endif

#endif
