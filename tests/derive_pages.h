/* Purpose: Check the derived page map at one and many agents.
 * Owns: Nothing; the derive fixture owns the fake ring and temporary files.
 * Threading: Two processes while the drain reads the fake device ring.
 * Lifetime: One part of the derive check. */
#ifndef AOTX_TEST_DERIVE_PAGES_H
#define AOTX_TEST_DERIVE_PAGES_H

static void page_stats(int n)
{
    run_ctx c;
    char path[1024];
    char want[256];
    aotx_commit_body commit;
    aotx_page_stats_body body;
    int i;

    start(&c, 0x00a0260000000000ull + (uint64_t)n, "pages");
    for (i = 0; i < n; ++i) {
        memset(&body, 0, sizeof(body));
        body.agent = (uint32_t)i;
        body.page = (uint32_t)(i + 2);
        body.residency = (uint32_t)(i & 1);
        body.cadence = 64u;
        body.mass = 0.5f + (float)i;
        body.slots = 160u;
        c.device.writer = AOTX_WRITER_AGENT_BASE + (uint32_t)i;
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_PAGE_STATS,
                         &body, sizeof(body));
    }

    /* Each mutation is invalid and must make no derived line. */
    memset(&body, 0, sizeof(body));
    body.cadence = 64u;
    body.mass = 0.5f;
    body.slots = 160u;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_PAGE_STATS, &body, sizeof(body));
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_PAGE_STATS,
                     &body, sizeof(body) - 1u);
    body.agent = 64u;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_PAGE_STATS, &body, sizeof(body));
    body.agent = 0u; body.residency = 2u;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_PAGE_STATS, &body, sizeof(body));
    body.residency = 0u; body.cadence = 63u;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_PAGE_STATS, &body, sizeof(body));
    body.cadence = 64u; body.mass = -1.0f;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_PAGE_STATS, &body, sizeof(body));
    body.mass = 0.5f; body.slots = 0u;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_PAGE_STATS, &body, sizeof(body));
    body.slots = 160u; body.page = 160u;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_PAGE_STATS, &body, sizeof(body));
    body.page = 0u; body.slots = 4097u;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_PAGE_STATS, &body, sizeof(body));

    memset(&commit, 0, sizeof(commit));
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT,
                     &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);

    snprintf(path, sizeof(path), "%s/pages.jsonl", c.boot_dir);
    CHECK(slurp(path, text, sizeof(text)) > 0, "the page map does not read");
    CHECK(count_of(text, "\n") == n, "the page map holds %d lines and %d were asked for",
          count_of(text, "\n"), n);
    for (i = 0; i < n; ++i) {
        snprintf(want, sizeof(want),
                 "{\"tick\":1,\"agent\":%d,\"page\":%d,\"residency\":%d,\"slots\":160,"
                 "\"mass\":%.9g}\n", i, i + 2, i & 1, 0.5 + (double)i);
        CHECK(strstr(text, want) != NULL, "page map line %d is not exact", i);
    }
    printf("page map %d: exact lines %d, invalid mutations refused 9\n", n, n);
    aotx_remove_tree(c.dir);
}

#endif
