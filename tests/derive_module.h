/* Purpose: Give the drain check the arms of the import note and of the remove note.
 * Owns: Nothing; the check that includes this file holds the drain and the temporary files.
 * Threading: One process at a time; the check runs the drain and reads what it wrote.
 * Lifetime: One case of the check.
 *
 * The file is included by tests/derive_test.c after the parts that every arm shares. */
#ifndef AOTX_TESTS_DERIVE_MODULE_H
#define AOTX_TESTS_DERIVE_MODULE_H

/* Gives the name of a module kind, as the note line writes it. */
static const char *module_kind_name(uint32_t kind)
{
    static const char *names[3] = { "skill", "role", "tool" };
    return (kind >= 1u && kind <= 3u) ? names[kind - 1u] : "other";
}

/* Writes the text that the note of one import head must hold. */
static void import_text(int number, char *out, size_t bytes)
{
    aotx_import_head head;
    aotx_fake_import_head(number, &head);
    snprintf(out, bytes, "\"text\":\"module %s %s import %u from %s\"", head.name,
             module_kind_name(head.kind), head.import, head.path);
}

/* Writes the text that the note of one remove record must hold. */
static void remove_text(int number, char *out, size_t bytes)
{
    aotx_remove_body gone;
    aotx_fake_remove(number, &gone);
    snprintf(out, bytes, "\"text\":\"module %s removed\"", gone.name);
}

/* The head of an import and a remove record each become one note line. A part of an import
 * makes no line, because the head names the module and the parts carry only its bytes. A
 * body that is shorter than the layout, and a record with no name, make no line. */
static void modules(int n)
{
    run_ctx c;
    aotx_import_head head;
    aotx_import_part part;
    aotx_remove_body gone;
    aotx_commit_body commit;
    char path[1024];
    char want[1024];
    int i;

    start(&c, 0x00e05e0000000001ull + (uint64_t)n, NULL);
    for (i = 0; i < n; i++) {
        c.device.writer = AOTX_WRITER_FEEDER;
        aotx_fake_import_head(i, &head);
        aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_IMPORT, &head, sizeof(head));
        aotx_fake_import_part(i, 1u, &part);
        aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_IMPORT, &part, sizeof(part));
        if (i == 0) {
            /* A second part of one import proves that the count of lines follows the heads
             * and not the records. */
            aotx_fake_import_part(i, 2u, &part);
            aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_IMPORT, &part, sizeof(part));
        }
        c.device.writer = AOTX_WRITER_CONSOLE;
        aotx_fake_remove(1000 + i, &gone);
        aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_REMOVE, &gone, sizeof(gone));
    }
    /* A body that is shorter than the layout holds no head and no name. */
    c.device.writer = AOTX_WRITER_FEEDER;
    aotx_fake_import_head(0, &head);
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_IMPORT, &head, 8u);
    aotx_fake_remove(0, &gone);
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_REMOVE, &gone, 8u);
    /* A head with no name names no module, so it makes no line. */
    memset(&head, 0, sizeof(head));
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_IMPORT, &head, sizeof(head));
    memset(&gone, 0, sizeof(gone));
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_REMOVE, &gone, sizeof(gone));
    memset(&commit, 0, sizeof(commit));
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);

    bus_path(&c, path, sizeof(path));
    CHECK(slurp(path, text, sizeof(text)) > 0, "the message file does not read");
    CHECK(count_of(text, "\n") == 2 * n, "the file holds %d lines and %d were asked for",
          count_of(text, "\n"), 2 * n);
    CHECK(count_of(text, "\"type\":\"note\"") == 2 * n, "the file holds %d notes",
          count_of(text, "\"type\":\"note\""));
    for (i = 0; i < n; i++) {
        import_text(i, want, sizeof(want));
        CHECK(strstr(text, want) != NULL, "the note of import %d is not in the file: %s", i,
              want);
        remove_text(1000 + i, want, sizeof(want));
        CHECK(strstr(text, want) != NULL, "the note of the removed module %d is not in the"
              " file: %s", i, want);
    }
    /* The writer of the record names the source of the line. */
    CHECK(count_of(text, "\"agent\":\"feeder\"") == n, "the file holds %d feeder lines",
          count_of(text, "\"agent\":\"feeder\""));
    CHECK(count_of(text, "\"agent\":\"console\"") == n, "the file holds %d console lines",
          count_of(text, "\"agent\":\"console\""));
    validate(path);
    printf("modules %d: lines %d\n", n, count_of(text, "\n"));
    aotx_remove_tree(c.dir);
}

/* The import note and the remove note are under the bus bit of the switch. A record that
 * makes no line still reaches the journal. */
static void modules_filter(const char *derive, int want_lines, int number)
{
    run_ctx c;
    aotx_import_head head;
    aotx_import_part part;
    aotx_remove_body gone;
    aotx_commit_body commit;
    char path[1024];
    int records;
    int i;

    start(&c, 0x00f05e0000000100ull + (uint64_t)number, derive);
    c.device.writer = AOTX_WRITER_FEEDER;
    for (i = 0; i < 2; i++) {
        aotx_fake_import_head(i, &head);
        aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_IMPORT, &head, sizeof(head));
        aotx_fake_import_part(i, 1u, &part);
        aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_IMPORT, &part, sizeof(part));
        aotx_fake_remove(i, &gone);
        aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_REMOVE, &gone, sizeof(gone));
    }
    memset(&commit, 0, sizeof(commit));
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);

    bus_path(&c, path, sizeof(path));
    text[0] = '\0';
    slurp(path, text, sizeof(text));
    CHECK(count_of(text, "\n") == want_lines, "the switch %s gives %d module lines and %d"
          " were asked for", (derive == NULL) ? "of every type" : derive,
          count_of(text, "\n"), want_lines);
    records = journal_records(c.boot_dir);
    CHECK(records == 7, "the journal holds %d records and 7 were written", records);
    if (want_lines > 0) {
        validate(path);
    }
    printf("switch %s: module lines %d, records %d\n",
           (derive == NULL) ? "console,note,bus,bulk,sequence" : derive, want_lines, records);
    aotx_remove_tree(c.dir);
}

#endif
