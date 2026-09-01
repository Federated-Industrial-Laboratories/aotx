/* Purpose: Check the derived token statistics stream at one and many agents.
 * Owns: Nothing; the derive fixture owns the fake ring and temporary files.
 * Threading: Two processes while the drain reads the fake device ring.
 * Lifetime: One part of the derive check. */
#ifndef AOTX_TEST_DERIVE_TOKENS_H
#define AOTX_TEST_DERIVE_TOKENS_H

#include <math.h>
#include "tests/derive_pages.h"

static void token_stats(int n)
{
    run_ctx c;
    char path[1024];
    char want[256];
    aotx_commit_body commit;
    aotx_token_stats_body body;
    int i;

    start(&c, 0x00a0250000000000ull + (uint64_t)n, "tokens");
    c.device.writer = AOTX_WRITER_AGENT_BASE;
    for (i = 0; i < n; ++i) {
        memset(&body, 0, sizeof(body));
        body.agent = (uint32_t)i;
        body.turn = (uint32_t)(i + 1);
        body.index = (uint32_t)(i * 2);
        body.token = (uint32_t)(100 + i);
        body.flags = (i & 1) ? AOTX_TOKEN_STATS_THINK : 0u;
        body.logprob = -0.25f - (float)i;
        body.entropy = 0.5f + (float)i;
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TOKEN_STATS,
                         &body, sizeof(body));
    }

    /* Each mutation is invalid and must make no derived line. */
    memset(&body, 0, sizeof(body));
    body.logprob = -0.25f;
    body.entropy = 0.5f;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TOKEN_STATS, &body, sizeof(body));
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TOKEN_STATS,
                     &body, sizeof(body) - 1u);
    body.agent = 64u;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TOKEN_STATS, &body, sizeof(body));
    body.agent = 0u;
    body.flags = 2u;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TOKEN_STATS, &body, sizeof(body));
    body.flags = 0u;
    body.logprob = NAN;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TOKEN_STATS, &body, sizeof(body));
    body.logprob = -0.25f;
    body.entropy = -1.0f;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TOKEN_STATS, &body, sizeof(body));

    memset(&commit, 0, sizeof(commit));
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT,
                     &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);

    snprintf(path, sizeof(path), "%s/tokens.jsonl", c.boot_dir);
    CHECK(slurp(path, text, sizeof(text)) > 0, "the token statistics file does not read");
    CHECK(count_of(text, "\n") == n,
          "the token statistics file holds %d lines and %d were asked for",
          count_of(text, "\n"), n);
    for (i = 0; i < n; ++i) {
        snprintf(want, sizeof(want),
                 "{\"tick\":1,\"agent\":%d,\"turn\":%d,\"index\":%d,\"token\":%d,"
                 "\"logprob\":%.9g,\"entropy\":%.9g,\"think\":%s}\n",
                 i, i + 1, i * 2, 100 + i, -0.25 - (double)i, 0.5 + (double)i,
                 (i & 1) ? "true" : "false");
        CHECK(strstr(text, want) != NULL, "token statistics line %d is not exact", i);
    }
    printf("token statistics %d: exact lines %d, invalid mutations refused 6\n", n, n);
    aotx_remove_tree(c.dir);
    page_stats(n);
}

#endif
