/* Purpose: Give the drain check the arms of the setting note and of the card note.
 * Owns: Nothing; the check that includes this file holds the drain and the temporary files.
 * Threading: One process at a time; the check runs the drain and reads what it wrote.
 * Lifetime: One case of the check.
 *
 * The file is included by tests/derive_test.c after the parts that every arm shares. */
#ifndef AOTX_TESTS_DERIVE_SETTINGS_H
#define AOTX_TESTS_DERIVE_SETTINGS_H

/* Writes the text that the setting note of one record must hold. */
static void setting_text(int number, char *out, size_t bytes)
{
    aotx_setting_body body;
    char value[32];
    aotx_fake_setting(number, &body);
    aotx_settings_format(body.value, (int)body.scale, value, sizeof(value));
    snprintf(out, bytes, "\"text\":\"setting %s %s\"", body.key, value);
}

/* Writes the text that the card note of one record must hold. */
static void card_text(int number, char *out, size_t bytes)
{
    aotx_card_body body;
    aotx_fake_card(number, &body);
    snprintf(out, bytes,
             "\"text\":\"card %s, %llu MB, %llu MB free, sm_%u%u, profile %s, arch %u,"
             " slots %u\"", body.name, (unsigned long long)(body.memory_total >> 20),
             (unsigned long long)(body.memory_free >> 20), body.compute_major,
             body.compute_minor, body.profile, body.arch, body.slots);
}

/* The setting records of the feeder and of the console, and the card record of the system,
 * each become one note line. A body that is shorter than the layout and a setting with no
 * key make no line. */
static void settings(int n)
{
    run_ctx c;
    aotx_setting_body setting;
    aotx_card_body card;
    aotx_commit_body commit;
    aotx_clock_body clock;
    char path[1024];
    char want[512];
    int i;

    start(&c, 0x00c05e0000000001ull + (uint64_t)n, NULL);
    clock.wall_ns = aotx_wall_ns();
    c.device.writer = AOTX_WRITER_FEEDER;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_START, &clock, sizeof(clock));
    for (i = 0; i < n; i++) {
        c.device.writer = AOTX_WRITER_FEEDER;
        aotx_fake_setting(i, &setting);
        aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_SETTING, &setting, sizeof(setting));
        c.device.writer = AOTX_WRITER_CONSOLE;
        aotx_fake_setting(1000 + i, &setting);
        aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_SETTING, &setting, sizeof(setting));
        c.device.writer = AOTX_WRITER_AGENT_BASE + (uint32_t)(i % 3);
        aotx_fake_card(i, &card);
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_CARD, &card, sizeof(card));
    }
    /* A body that is shorter than the layout holds no setting and no card. */
    c.device.writer = AOTX_WRITER_FEEDER;
    aotx_fake_setting(0, &setting);
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_SETTING, &setting, 8u);
    aotx_fake_card(0, &card);
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_CARD, &card, 8u);
    /* A setting record with no key names no setting, so it makes no line. */
    memset(&setting, 0, sizeof(setting));
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_SETTING, &setting, sizeof(setting));
    memset(&commit, 0, sizeof(commit));
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);

    bus_path(&c, path, sizeof(path));
    CHECK(slurp(path, text, sizeof(text)) > 0, "the message file does not read");
    CHECK(count_of(text, "\n") == 3 * n, "the file holds %d lines and %d were asked for",
          count_of(text, "\n"), 3 * n);
    CHECK(count_of(text, "\"type\":\"note\"") == 3 * n, "the file holds %d notes",
          count_of(text, "\"type\":\"note\""));
    for (i = 0; i < n; i++) {
        setting_text(i, want, sizeof(want));
        CHECK(strstr(text, want) != NULL, "the note of setting %d is not in the file: %s", i,
              want);
        setting_text(1000 + i, want, sizeof(want));
        CHECK(strstr(text, want) != NULL, "the note of console setting %d is not in the file",
              i);
        card_text(i, want, sizeof(want));
        CHECK(strstr(text, want) != NULL, "the note of card %d is not in the file: %s", i,
              want);
    }
    /* The writer of the record names the source of the line. The card line is a line of
     * the system, whatever agent wrote the record. */
    CHECK(count_of(text, "\"agent\":\"feeder\"") == n, "the file holds %d feeder lines",
          count_of(text, "\"agent\":\"feeder\""));
    CHECK(count_of(text, "\"agent\":\"console\"") == n, "the file holds %d console lines",
          count_of(text, "\"agent\":\"console\""));
    CHECK(count_of(text, "\"agent\":\"system\"") == n, "the file holds %d system lines",
          count_of(text, "\"agent\":\"system\""));
    validate(path);
    printf("settings %d: lines %d\n", n, count_of(text, "\n"));
    aotx_remove_tree(c.dir);
}

/* The setting note and the card note are under the bus bit of the switch. A record that
 * makes no line still reaches the journal. */
static void settings_filter(const char *derive, int want_lines, int number)
{
    run_ctx c;
    aotx_setting_body setting;
    aotx_card_body card;
    aotx_commit_body commit;
    char path[1024];
    int records;
    int i;

    start(&c, 0x00d05e0000000100ull + (uint64_t)number, derive);
    for (i = 0; i < 2; i++) {
        c.device.writer = AOTX_WRITER_FEEDER;
        aotx_fake_setting(i, &setting);
        aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_SETTING, &setting, sizeof(setting));
    }
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_card(0, &card);
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_CARD, &card, sizeof(card));
    memset(&commit, 0, sizeof(commit));
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);

    bus_path(&c, path, sizeof(path));
    text[0] = '\0';
    slurp(path, text, sizeof(text));
    CHECK(count_of(text, "\n") == want_lines, "the switch %s gives %d setting and card lines"
          " and %d were asked for", (derive == NULL) ? "of every type" : derive,
          count_of(text, "\n"), want_lines);
    records = journal_records(c.boot_dir);
    CHECK(records == 4, "the journal holds %d records and 4 were written", records);
    if (want_lines > 0) {
        validate(path);
    }
    printf("switch %s: setting and card lines %d, records %d\n",
           (derive == NULL) ? "console,note,bus,bulk,sequence" : derive, want_lines, records);
    aotx_remove_tree(c.dir);
}

#endif
