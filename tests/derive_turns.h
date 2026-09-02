/* Purpose: Check manifest chains and task and agent event lines.
 * Owns: Nothing; the derive fixture owns the fake ring and temporary files.
 * Threading: Two processes while the drain reads the fake device ring.
 * Lifetime: One part of the derive check. */
#ifndef AOTX_TEST_DERIVE_TURNS_H
#define AOTX_TEST_DERIVE_TURNS_H

/* Every line carries the digest of the line before it. */
static void turns(int n)
{
    run_ctx c;
    char path[1024];
    char want[512];
    char digest_text[65];
    unsigned char digest[AOTX_SHA256_DIGEST];
    aotx_sha256 state;
    aotx_commit_body commit;
    aotx_manifest_body m;
    uint64_t boot_id = 0x00f05e0000000001ull + (uint64_t)n;
    char *at;
    int line = 0;
    int i;

    start(&c, boot_id, NULL);
    for (i = 0; i < n; i++) {
        c.device.writer = AOTX_WRITER_AGENT_BASE + (uint32_t)(i % 3);
        aotx_fake_manifest(i, &m);
        if (i == 0) m.finish = AOTX_TURN_STOPPED;
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_MANIFEST, &m, sizeof(m));
    }
    aotx_fake_manifest(0, &m);
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_MANIFEST, &m, 8u);
    memset(&commit, 0, sizeof(commit));
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);

    snprintf(path, sizeof(path), "%s/manifest/%016llx.jsonl", c.dir,
             (unsigned long long)boot_id);
    CHECK(slurp(path, text, sizeof(text)) > 0, "the chain file does not read");
    CHECK(count_of(text, "\n") == n, "the chain holds %d lines and %d turns were written",
          count_of(text, "\n"), n);
    CHECK(count_of(text, "\"finish\":\"stopped\"") == 1,
          "the chain holds %d stopped turns and one was asked for",
          count_of(text, "\"finish\":\"stopped\""));
    memset(digest_text, '0', 64);
    digest_text[64] = '\0';
    at = text;
    while (*at != '\0' && line < n) {
        char *end = strchr(at, '\n');
        size_t bytes;
        if (end == NULL) break;
        bytes = (size_t)(end - at) + 1u;
        aotx_fake_manifest(line, &m);
        snprintf(want, sizeof(want),
                 "{\"agent\":%u,\"turn\":%u,\"input_hash\":\"%016llx\","
                 "\"output_hash\":\"%016llx\",\"tokens\":%u,",
                 m.agent, m.turn, (unsigned long long)m.input_hash,
                 (unsigned long long)m.output_hash, m.output_tokens);
        CHECK(strncmp(at, want, strlen(want)) == 0, "line %d does not hold turn %d", line,
              line);
        snprintf(want, sizeof(want), "\"prev\":\"%s\"}", digest_text);
        CHECK(strstr(at, want) != NULL && strstr(at, want) < end,
              "line %d does not name the digest of the line before it", line);
        aotx_sha256_init(&state);
        aotx_sha256_update(&state, at, bytes);
        aotx_sha256_final(&state, digest);
        aotx_sha256_text(digest, digest_text);
        at = end + 1;
        line++;
    }
    CHECK(line == n, "the walk read %d lines of %d", line, n);
    printf("turns %d: lines %d\n", n, count_of(text, "\n"));
    aotx_remove_tree(c.dir);
}

/* Done tasks give handoffs. Other task states and agent events give notes. */
static void events(int n)
{
    run_ctx c;
    char path[1024];
    char want[512];
    aotx_commit_body commit;
    aotx_clock_body clock;
    aotx_task_body task;
    aotx_agent_body event;
    int i;

    start(&c, 0x00905e0000000001ull + (uint64_t)n, NULL);
    clock.wall_ns = aotx_wall_ns();
    c.device.writer = AOTX_WRITER_FEEDER;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_START, &clock, sizeof(clock));
    for (i = 0; i < n; i++) {
        c.device.writer = AOTX_WRITER_AGENT_BASE + (uint32_t)(i % 3);
        aotx_fake_task(i, AOTX_TASK_RUNNING, &task);
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TASK, &task, sizeof(task));
        aotx_fake_task(i, AOTX_TASK_DONE, &task);
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TASK, &task, sizeof(task));
        aotx_fake_agent(i, AOTX_AGENT_SPAWNED, &event);
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_AGENT, &event, sizeof(event));
    }
    aotx_fake_agent(0, AOTX_AGENT_TURN, &event);
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_AGENT, &event, 8u);
    memset(&commit, 0, sizeof(commit));
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);

    bus_path(&c, path, sizeof(path));
    CHECK(slurp(path, text, sizeof(text)) > 0, "the message file does not read");
    CHECK(count_of(text, "\n") == 3 * n, "the file holds %d lines and %d were asked for",
          count_of(text, "\n"), 3 * n);
    CHECK(count_of(text, "\"type\":\"handoff\"") == n, "the file holds %d handoffs",
          count_of(text, "\"type\":\"handoff\""));
    CHECK(count_of(text, "\"type\":\"note\"") == 2 * n, "the file holds %d notes",
          count_of(text, "\"type\":\"note\""));
    for (i = 0; i < n; i++) {
        aotx_fake_task(i, AOTX_TASK_DONE, &task);
        snprintf(want, sizeof(want),
                 "\"path\":\"task %u\",\"status\":\"ready\",\"note\":\"result %d of the run\"",
                 task.task, i);
        CHECK(strstr(text, want) != NULL, "the handoff of task %d is not in the file", i);
        snprintf(want, sizeof(want),
                 "\"text\":\"task %u running agent %u attempts %u ticks %llu result %d of"
                 " the run\"", task.task, task.agent, task.attempts,
                 (unsigned long long)task.ticks, i);
        CHECK(strstr(text, want) != NULL, "the note of task %d is not in the file", i);
        aotx_fake_agent(i, AOTX_AGENT_SPAWNED, &event);
        snprintf(want, sizeof(want),
                 "\"text\":\"agent %u spawned role %u parent %u state %u turn %u ticks %llu\"",
                 event.agent, event.role, event.parent, event.state, event.turn,
                 (unsigned long long)event.ticks);
        CHECK(strstr(text, want) != NULL, "the note of agent event %d is not in the file", i);
        snprintf(want, sizeof(want), "\"agent\":\"agent-%d\"", i % 3);
        CHECK(strstr(text, want) != NULL, "no line names the writer of event %d", i);
    }
    validate(path);
    printf("events %d: lines %d, handoffs %d\n", n, count_of(text, "\n"),
           count_of(text, "\"type\":\"handoff\""));
    aotx_remove_tree(c.dir);
}

#endif
