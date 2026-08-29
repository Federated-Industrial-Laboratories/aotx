/* Purpose: Check the settings reader against the key list, the refusals and the writer.
 * Owns: One temporary directory and one table for each case.
 * Threading: One thread; the program reads and writes files of its own.
 * Lifetime: The run of the program. */
#include "disk/settings/settings.h"
#include "tests/disk_fake.h"

#include <dirent.h>
#include <fcntl.h>

#define AOTX_FILE_MAX 16384

/* The table that the key list must give. The test states every row. A change of a
 * default, a range or a scale in the key list gives a failure here. */
typedef struct want_number {
    const char  *name;
    int64_t      value;
    int64_t      least;
    int64_t      most;
    int          scale;
    unsigned int side;
    unsigned int effect;
} want_number;

static const want_number want_numbers[] = {
    { "window.on",             0,   0,  1,       1,     AOTX_SETTING_SIDE_BOOT,
      AOTX_SETTING_AT_BOOT },
    { "tui.on",                0,   0,  1,       1,     AOTX_SETTING_SIDE_BOOT,
      AOTX_SETTING_AT_BOOT },
    { "tui.escape_ms",         25,  5,  500,     1,     AOTX_SETTING_SIDE_TERMINAL,
      AOTX_SETTING_AT_READ },
    { "tick.period_ms",        10,  1,  1000,    1,     AOTX_SETTING_SIDE_DEVICE,
      AOTX_SETTING_AT_TICK },
    { "decode.budget_ms",      120, 10, 10000,   1,     AOTX_SETTING_SIDE_DEVICE,
      AOTX_SETTING_AT_TICK },
    { "decode.prefill_tokens", 512, 32, 512,    1,     AOTX_SETTING_SIDE_DEVICE,
      AOTX_SETTING_AT_TICK },
    { "decode.reply_limit",    256, 1,  8191,    1,     AOTX_SETTING_SIDE_DEVICE,
      AOTX_SETTING_AT_SEQUENCE },
    { "sample.temperature",    7000, 0, 20000,   10000, AOTX_SETTING_SIDE_DEVICE,
      AOTX_SETTING_AT_SEQUENCE },
    { "sample.top_p",          8000, 1, 10000,   10000, AOTX_SETTING_SIDE_DEVICE,
      AOTX_SETTING_AT_SEQUENCE },
    { "sample.top_k",          20,  1,  1000,    1,     AOTX_SETTING_SIDE_DEVICE,
      AOTX_SETTING_AT_SEQUENCE },
    { "agent.budget",          8,   1,  64,      1,     AOTX_SETTING_SIDE_DEVICE,
      AOTX_SETTING_AT_TASK },
    { "tool.deadline_ticks",   500, 1,  1000000, 1,     AOTX_SETTING_SIDE_DEVICE,
      AOTX_SETTING_AT_REQUEST },
    { "mirror.hz",             30,  1,  120,     1,     AOTX_SETTING_SIDE_DEVICE,
      AOTX_SETTING_AT_FRAME },
    { "agent.pages",           0,   0,  4096,    1,     AOTX_SETTING_SIDE_DEVICE,
      AOTX_SETTING_AT_TASK },
    { "agent.recall_k",        4,   0,  16,      1,     AOTX_SETTING_SIDE_DEVICE,
      AOTX_SETTING_AT_TASK },
    { "agent.compact_at",      128, 8,  1024,    1,     AOTX_SETTING_SIDE_DEVICE,
      AOTX_SETTING_AT_TASK }
};

typedef struct want_text {
    const char  *name;
    const char  *value;
    unsigned int side;
} want_text;

static const want_text want_texts[] = {
    { "journal.dir",  "journal", AOTX_SETTING_SIDE_BOOT },
    { "models.dir",   "models",  AOTX_SETTING_SIDE_BOOT },
    { "models.roles", "",        AOTX_SETTING_SIDE_BOOT },
    { "modules.dir",  "modules", AOTX_SETTING_SIDE_BOOT },
    { "tools.root",   "",        AOTX_SETTING_SIDE_BOOT },
    { "derive.list",  "",        AOTX_SETTING_SIDE_BOOT },
    { "tui.color",   "none",    AOTX_SETTING_SIDE_TERMINAL },
    { "tui.box",      "ascii",   AOTX_SETTING_SIDE_TERMINAL },
    { "tui.splash",   "auto",    AOTX_SETTING_SIDE_TERMINAL }
};

static aotx_settings table;
static char file_text[AOTX_FILE_MAX];

/* ---- files ---- */

static void put_file(const char *path, const char *content, size_t bytes)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    CHECK(fd >= 0, "the file %s does not open for a write", path);
    if (fd < 0) {
        return;
    }
    CHECK(write(fd, content, bytes) == (ssize_t)bytes, "the file %s does not take its bytes",
          path);
    close(fd);
}

/* Reads a whole file and gives the count of bytes, or -1. */
static int get_file(const char *path, char *out, size_t bytes)
{
    int fd = open(path, O_RDONLY);
    ssize_t got;
    if (fd < 0) {
        return -1;
    }
    got = read(fd, out, bytes - 1);
    close(fd);
    if (got < 0) {
        return -1;
    }
    out[got] = '\0';
    return (int)got;
}

/* Counts the files of the directory whose name starts with the temporary name of a write.
 * A write that leaves one behind is a write that did not clean up after itself. */
static int leftovers(const char *dir)
{
    DIR *at = opendir(dir);
    struct dirent *entry;
    int found = 0;
    if (at == NULL) {
        return -1;
    }
    while ((entry = readdir(at)) != NULL) {
        if (strncmp(entry->d_name, "aotx-settings-", 14) == 0) {
            found++;
        }
    }
    closedir(at);
    return found;
}

/* ---- the defaults and the key list ---- */

static void defaults(void)
{
    unsigned int i;
    unsigned int index = 0;
    CHECK(AOTX_SETTING_NUMBER_COUNT == sizeof(want_numbers) / sizeof(want_numbers[0]),
          "the key list holds %u number keys and %u are named here",
          (unsigned)AOTX_SETTING_NUMBER_COUNT,
          (unsigned)(sizeof(want_numbers) / sizeof(want_numbers[0])));
    CHECK(AOTX_SETTING_TEXT_COUNT == sizeof(want_texts) / sizeof(want_texts[0]),
          "the key list holds %u text keys and %u are named here",
          (unsigned)AOTX_SETTING_TEXT_COUNT,
          (unsigned)(sizeof(want_texts) / sizeof(want_texts[0])));
    aotx_settings_defaults(&table);
    for (i = 0; i < AOTX_SETTING_NUMBER_COUNT; i++) {
        const want_number *w = &want_numbers[i];
        CHECK(strcmp(aotx_settings_number_name(i), w->name) == 0,
              "number key %u is %s and %s was asked for", i, aotx_settings_number_name(i),
              w->name);
        CHECK(table.number[i] == w->value, "the default of %s is %lld and %lld was asked for",
              w->name, (long long)table.number[i], (long long)w->value);
        CHECK(aotx_settings_number_least(i) == w->least, "the least value of %s is wrong",
              w->name);
        CHECK(aotx_settings_number_most(i) == w->most, "the most value of %s is wrong", w->name);
        CHECK(aotx_settings_number_scale(i) == w->scale, "the scale of %s is wrong", w->name);
        CHECK(aotx_settings_number_side(i) == w->side, "the side of %s is wrong", w->name);
        CHECK(aotx_settings_number_effect(i) == w->effect, "the effect of %s is wrong", w->name);
        CHECK(table.number_given[i] == 0u, "the default of %s must not count as given", w->name);
        CHECK(aotx_settings_find(w->name, strlen(w->name), &index) == 0 && index == i,
              "the key %s is not found as number key %u", w->name, i);
    }
    for (i = 0; i < AOTX_SETTING_TEXT_COUNT; i++) {
        const want_text *w = &want_texts[i];
        CHECK(strcmp(aotx_settings_text_name(i), w->name) == 0, "text key %u is %s and %s"
              " was asked for", i, aotx_settings_text_name(i), w->name);
        CHECK(strcmp(table.text[i], w->value) == 0, "the default of %s is %s and %s was"
              " asked for", w->name, table.text[i], w->value);
        CHECK(aotx_settings_text_side(i) == w->side, "the side of %s is wrong", w->name);
        CHECK(table.text_given[i] == 0u, "the default of %s must not count as given", w->name);
        CHECK(aotx_settings_find(w->name, strlen(w->name), &index) == 1 && index == i,
              "the key %s is not found as text key %u", w->name, i);
    }
    CHECK(aotx_settings_find("no.such.key", 11, &index) == 2, "an unknown key must give 2");
    CHECK(table.refused_count == 0u, "the defaults must refuse nothing");
}

/* ---- the value of one key of one line of a file ---- */

/* Gives the value that line number i writes for key k, inside the range of the key. Each
 * time a key comes again the value changes, so a file where the last line does not win
 * cannot pass. */
static int64_t pick(unsigned int k, int i)
{
    int64_t least = aotx_settings_number_least(k);
    int64_t span = aotx_settings_number_most(k) - least + 1;
    int64_t occurrence = (int64_t)(i / (int)AOTX_SETTING_NUMBER_COUNT);
    return least + (occurrence * 3 + (int64_t)k) % span;
}

/* A file of n lines. Line i names number key i modulo the count of number keys. A file of
 * more lines than keys names a key again, and the last line of a key must win. Spaces go
 * around the key and the value, and the reader must drop them. */
static void batch(int n)
{
    char dir[256];
    char path[320];
    char line[128];
    char value[32];
    unsigned int k;
    size_t used = 0;
    int i;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(path, sizeof(path), "%s/aotx.settings", dir);
    used += (size_t)snprintf(file_text + used, sizeof(file_text) - used,
                             "# the settings of a run of %d lines\n\n", n);
    for (i = 0; i < n; i++) {
        k = (unsigned int)(i % (int)AOTX_SETTING_NUMBER_COUNT);
        aotx_settings_format(pick(k, i), aotx_settings_number_scale(k), value, sizeof(value));
        snprintf(line, sizeof(line), "   %s   =   %s   \n", aotx_settings_number_name(k), value);
        used += (size_t)snprintf(file_text + used, sizeof(file_text) - used, "%s", line);
    }
    for (i = 0; i < n && i < (int)AOTX_SETTING_TEXT_COUNT; i++) {
        used += (size_t)snprintf(file_text + used, sizeof(file_text) - used,
                                 "%s = text %d of %d\n", aotx_settings_text_name(i), i, n);
    }
    put_file(path, file_text, used);

    CHECK(aotx_settings_read(path, &table) == 0, "the file of %d lines is not read clean", n);
    CHECK(table.refused_count == 0u, "the file of %d lines refused %u lines", n,
          table.refused_count);
    for (k = 0; k < AOTX_SETTING_NUMBER_COUNT; k++) {
        int last = -1;
        for (i = 0; i < n; i++) {
            if ((unsigned int)(i % (int)AOTX_SETTING_NUMBER_COUNT) == k) {
                last = i;
            }
        }
        if (last < 0) {
            CHECK(table.number_given[k] == 0u, "the key %s is given and no line names it",
                  aotx_settings_number_name(k));
            CHECK(table.number[k] == want_numbers[k].value, "the key %s does not hold its"
                  " default", aotx_settings_number_name(k));
            continue;
        }
        CHECK(table.number_given[k] == 1u, "the key %s is not given and a line names it",
              aotx_settings_number_name(k));
        CHECK(table.number[k] == pick(k, last), "the key %s holds %lld and the last line of"
              " it gives %lld", aotx_settings_number_name(k), (long long)table.number[k],
              (long long)pick(k, last));
    }
    for (i = 0; i < (int)AOTX_SETTING_TEXT_COUNT; i++) {
        char want[320];
        if (i >= n) {
            CHECK(table.text_given[i] == 0u, "the text key %s is given and no line names it",
                  aotx_settings_text_name(i));
            continue;
        }
        if (i == AOTX_SET_JOURNAL_DIR || i == AOTX_SET_MODELS_DIR
            || i == AOTX_SET_MODULES_DIR || i == AOTX_SET_TOOLS_ROOT) {
            snprintf(want, sizeof(want), "%s/text %d of %d", dir, i, n);
        } else {
            snprintf(want, sizeof(want), "text %d of %d", i, n);
        }
        CHECK(table.text_given[i] == 1u, "the text key %s is not given",
              aotx_settings_text_name(i));
        CHECK(strcmp(table.text[i], want) == 0, "the text key %s holds %s and %s was asked for",
              aotx_settings_text_name(i), table.text[i], want);
    }
    printf("batch %d: lines %d, keys given %d\n", n, n,
           (n < (int)AOTX_SETTING_NUMBER_COUNT) ? n : (int)AOTX_SETTING_NUMBER_COUNT);
    aotx_remove_tree(dir);
}

/* ---- the text of a value, and the value of that text ---- */

/* Every number key takes values through the text that the format gives. The least value,
 * the most value and the values between them must come back the same. */
static void round_trip(int n)
{
    unsigned int k;
    int trips = 0;
    for (k = 0; k < AOTX_SETTING_NUMBER_COUNT; k++) {
        int64_t least = aotx_settings_number_least(k);
        int64_t most = aotx_settings_number_most(k);
        int scale = aotx_settings_number_scale(k);
        int step;
        for (step = 0; step < n; step++) {
            char value[32];
            char line[128];
            char reason[AOTX_SETTINGS_REASON_BYTES];
            int64_t want = least + ((most - least) * (int64_t)step) / (int64_t)((n > 1) ? n - 1
                                                                                        : 1);
            size_t bytes = aotx_settings_format(want, scale, value, sizeof(value));
            CHECK(bytes > 0 && bytes < sizeof(value), "the text of %lld of key %s is empty",
                  (long long)want, aotx_settings_number_name(k));
            snprintf(line, sizeof(line), "%s = %s", aotx_settings_number_name(k), value);
            aotx_settings_defaults(&table);
            reason[0] = '\0';
            CHECK(aotx_settings_line(line, strlen(line), &table, reason) == 0,
                  "the line %s is refused: %s", line, reason);
            CHECK(table.number[k] == want, "the key %s gives %lld from the text %s and %lld"
                  " was asked for", aotx_settings_number_name(k), (long long)table.number[k],
                  value, (long long)want);
            trips++;
        }
    }
    /* The scale of four decimals must print the decimals it needs and no more. */
    {
        char value[32];
        CHECK(aotx_settings_format(7000, AOTX_SETTING_SCALE_FIXED, value, sizeof(value)) == 3 &&
              strcmp(value, "0.7") == 0, "7000 at the scale of four decimals gives %s", value);
        aotx_settings_format(10000, AOTX_SETTING_SCALE_FIXED, value, sizeof(value));
        CHECK(strcmp(value, "1") == 0, "10000 at the scale of four decimals gives %s", value);
        aotx_settings_format(12345, AOTX_SETTING_SCALE_FIXED, value, sizeof(value));
        CHECK(strcmp(value, "1.2345") == 0, "12345 at the scale of four decimals gives %s",
              value);
        aotx_settings_format(1, AOTX_SETTING_SCALE_FIXED, value, sizeof(value));
        CHECK(strcmp(value, "0.0001") == 0, "1 at the scale of four decimals gives %s", value);
        aotx_settings_format(120, AOTX_SETTING_SCALE_ONE, value, sizeof(value));
        CHECK(strcmp(value, "120") == 0, "120 at the scale of one gives %s", value);
        aotx_settings_format(-7000, AOTX_SETTING_SCALE_FIXED, value, sizeof(value));
        CHECK(strcmp(value, "-0.7") == 0, "a value below zero gives %s", value);
    }
    /* A whole number for a key of four decimals is the whole number in the scaled unit. */
    {
        char reason[AOTX_SETTINGS_REASON_BYTES];
        aotx_settings_defaults(&table);
        CHECK(aotx_settings_line("sample.top_p = 1", 16, &table, reason) == 0,
              "a whole number for a key of four decimals is refused: %s", reason);
        CHECK(table.number[AOTX_SET_TOP_P] == 10000, "a whole number for a key of four"
              " decimals gives %lld", (long long)table.number[AOTX_SET_TOP_P]);
        CHECK(aotx_settings_line("sample.temperature = 2.0000", 27, &table, reason) == 0,
              "a whole number with decimals of zero is refused: %s", reason);
        CHECK(table.number[AOTX_SET_TEMPERATURE] == 20000, "2.0000 gives %lld",
              (long long)table.number[AOTX_SET_TEMPERATURE]);
        CHECK(aotx_settings_line("mirror.hz = 60.00", 17, &table, reason) == 0,
              "decimals of zero for a whole number key are refused: %s", reason);
        CHECK(table.number[AOTX_SET_MIRROR_HZ] == 60, "60.00 gives %lld",
              (long long)table.number[AOTX_SET_MIRROR_HZ]);
    }
    printf("round trip %d: values %d\n", n, trips);
}

/* ---- comments, blank lines and the lines the reader refuses ---- */

typedef struct refusal_case {
    const char *line;
    const char *reason;
} refusal_case;

static const refusal_case refusal_cases[] = {
    { "no.such.key = 1",                      "the key is not known" },
    { "tick.period_ms = 0",                   "the value 0 is not in the range 1 to 1000" },
    { "tick.period_ms = 2000",                "the value 2000 is not in the range 1 to 1000" },
    { "sample.temperature = 3",
      "the value 3 is not in the range 0 to 2" },
    { "tick.period_ms",                       "the line has no equals sign" },
    { "= 5",                                  "the line names no key" },
    { "TICK.period_ms = 5",
      "the key takes lower case letters, digits, dots and underscores only" },
    { "tick.period-ms = 5",
      "the key takes lower case letters, digits, dots and underscores only" },
    { "sample.temperature = 0.12345",         "the value has more than four decimals" },
    { "sample.temperature = 0.10000",         "the value has more than four decimals" },
    { "sample.temperature = abc",             "the value is not a number" },
    { "sample.temperature =",                 "the value is not a number" },
    { "sample.temperature = 0.",              "the value is not a number" },
    { "sample.temperature = 0.5x",            "the value is not a number" },
    { "tick.period_ms = 10.5",                "the key takes a whole number" },
    { "sample.temperature = 99999999999999999999", "the value is too large" }
};

#define AOTX_REFUSAL_COUNT (sizeof(refusal_cases) / sizeof(refusal_cases[0]))

static void refusals(void)
{
    char dir[256];
    char path[320];
    char reason[AOTX_SETTINGS_REASON_BYTES];
    char long_key[128];
    char long_text[AOTX_SETTING_TEXT_BYTES + 64];
    size_t used = 0;
    unsigned int i;

    /* Every kind, one line at a time, with the reason that the line must give. */
    for (i = 0; i < AOTX_REFUSAL_COUNT; i++) {
        aotx_settings_defaults(&table);
        reason[0] = '\0';
        CHECK(aotx_settings_line(refusal_cases[i].line, strlen(refusal_cases[i].line), &table,
                                 reason) == 1,
              "the line %s must be refused", refusal_cases[i].line);
        CHECK(strcmp(reason, refusal_cases[i].reason) == 0, "the line %s gives the reason %s"
              " and %s was asked for", refusal_cases[i].line, reason,
              refusal_cases[i].reason);
        CHECK(table.number_given[AOTX_SET_TICK_PERIOD_MS] == 0u &&
              table.number_given[AOTX_SET_TEMPERATURE] == 0u,
              "the refused line %s changed the table", refusal_cases[i].line);
    }
    /* A key of 64 bytes is one byte over the bound. */
    memset(long_key, 'a', 64);
    snprintf(long_key + 64, sizeof(long_key) - 64, " = 5");
    aotx_settings_defaults(&table);
    reason[0] = '\0';
    CHECK(aotx_settings_line(long_key, strlen(long_key), &table, reason) == 1,
          "a key of 64 bytes must be refused");
    CHECK(strcmp(reason, "the key is longer than 63 bytes") == 0,
          "a key of 64 bytes gives the reason %s", reason);
    /* A text of 256 bytes is one byte over the bound. */
    used = (size_t)snprintf(long_text, sizeof(long_text), "journal.dir = ");
    memset(long_text + used, 'd', AOTX_SETTING_TEXT_BYTES);
    long_text[used + AOTX_SETTING_TEXT_BYTES] = '\0';
    aotx_settings_defaults(&table);
    reason[0] = '\0';
    CHECK(aotx_settings_line(long_text, strlen(long_text), &table, reason) == 1,
          "a text of 256 bytes must be refused");
    CHECK(strcmp(reason, "the text is longer than 255 bytes") == 0,
          "a text of 256 bytes gives the reason %s", reason);
    /* A text of 255 bytes is inside the bound. */
    long_text[used + AOTX_SETTING_TEXT_BYTES - 1] = '\0';
    reason[0] = '\0';
    CHECK(aotx_settings_line(long_text, strlen(long_text), &table, reason) == 0,
          "a text of 255 bytes is refused: %s", reason);

    /* A comment line, a blank line and a line of spaces set no key and are not refused. */
    aotx_settings_defaults(&table);
    CHECK(aotx_settings_line("# a comment", 11, &table, reason) == 0, "a comment is refused");
    CHECK(aotx_settings_line("", 0, &table, reason) == 0, "a blank line is refused");
    CHECK(aotx_settings_line("   \t  ", 6, &table, reason) == 0, "a line of spaces is refused");
    CHECK(aotx_settings_line("   # a comment with spaces before it", 36, &table, reason) == 0,
          "a comment that starts after a space is refused");
    CHECK(table.refused_count == 0u, "a comment or a blank line must refuse nothing");
    /* A number sign inside a value is part of the value, because a text value runs to the
     * end of the line. */
    CHECK(aotx_settings_line("tui.box = a#b", 13, &table, reason) == 0,
          "a number sign inside a text value is refused");
    CHECK(strcmp(table.text[AOTX_SET_TUI_BOX], "a#b") == 0, "the text value is %s",
          table.text[AOTX_SET_TUI_BOX]);

    /* The same lines go in a file. Every refused line is counted with its line number.
     * The lines that follow it are read all the same. */
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(path, sizeof(path), "%s/aotx.settings", dir);
    used = (size_t)snprintf(file_text, sizeof(file_text), "# the head of the file\n\n");
    for (i = 0; i < AOTX_REFUSAL_COUNT; i++) {
        used += (size_t)snprintf(file_text + used, sizeof(file_text) - used, "%s\n",
                                 refusal_cases[i].line);
    }
    used += (size_t)snprintf(file_text + used, sizeof(file_text) - used, "mirror.hz = 45\n");
    put_file(path, file_text, used);
    CHECK(aotx_settings_read(path, &table) == 1, "a file with refused lines must give 1");
    CHECK(table.refused_count == AOTX_REFUSAL_COUNT, "the file refused %u lines and %u were"
          " written", table.refused_count, (unsigned)AOTX_REFUSAL_COUNT);
    for (i = 0; i < AOTX_REFUSAL_COUNT && i < AOTX_SETTINGS_REFUSALS; i++) {
        CHECK(table.refused[i].line == i + 3u, "refused line %u names line %u and %u was"
              " asked for", i, table.refused[i].line, i + 3u);
        CHECK(strcmp(table.refused[i].reason, refusal_cases[i].reason) == 0,
              "refused line %u gives the reason %s", i, table.refused[i].reason);
    }
    CHECK(table.number_given[AOTX_SET_MIRROR_HZ] == 1u && table.number[AOTX_SET_MIRROR_HZ] == 45,
          "the line after the refused lines is not read");
    aotx_remove_tree(dir);
    printf("refusals: kinds %u\n", (unsigned)AOTX_REFUSAL_COUNT + 3u);
}

/* A file that refuses more lines than the table of reasons holds counts every one of them
 * and keeps the reasons of the first ones. */
static void many_refusals(void)
{
    char dir[256];
    char path[320];
    size_t used = 0;
    unsigned int want = AOTX_SETTINGS_REFUSALS + 8u;
    unsigned int i;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(path, sizeof(path), "%s/aotx.settings", dir);
    for (i = 0; i < want; i++) {
        used += (size_t)snprintf(file_text + used, sizeof(file_text) - used,
                                 "no.such.key.%u = %u\n", i, i);
    }
    put_file(path, file_text, used);
    CHECK(aotx_settings_read(path, &table) == 1, "a file of refused lines must give 1");
    CHECK(table.refused_count == want, "the file refused %u lines and %u were written",
          table.refused_count, want);
    CHECK(table.refused[AOTX_SETTINGS_REFUSALS - 1u].line == AOTX_SETTINGS_REFUSALS,
          "the last kept refusal names line %u",
          table.refused[AOTX_SETTINGS_REFUSALS - 1u].line);
    CHECK(strcmp(table.refused[AOTX_SETTINGS_REFUSALS - 1u].reason, "the key is not known") == 0,
          "the last kept refusal gives the reason %s",
          table.refused[AOTX_SETTINGS_REFUSALS - 1u].reason);
    aotx_remove_tree(dir);
    printf("many refusals: counted %u, kept %u\n", table.refused_count,
           (unsigned)AOTX_SETTINGS_REFUSALS);
}

/* A line of more than 512 bytes is refused, and the file goes on after it. */
static void long_line(void)
{
    char dir[256];
    char path[320];
    size_t used;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(path, sizeof(path), "%s/aotx.settings", dir);
    used = (size_t)snprintf(file_text, sizeof(file_text), "tui.box = ");
    memset(file_text + used, 'b', 600);
    used += 600;
    used += (size_t)snprintf(file_text + used, sizeof(file_text) - used, "\nmirror.hz = 44\n");
    put_file(path, file_text, used);
    CHECK(aotx_settings_read(path, &table) == 1, "a file with a long line must give 1");
    CHECK(table.refused_count == 1u, "the file refused %u lines and one was written",
          table.refused_count);
    CHECK(table.refused[0].line == 1u, "the refused line is line %u", table.refused[0].line);
    CHECK(strcmp(table.refused[0].reason, "the line is longer than 512 bytes") == 0,
          "the long line gives the reason %s", table.refused[0].reason);
    CHECK(table.number[AOTX_SET_MIRROR_HZ] == 44, "the line after the long line is not read");
    aotx_remove_tree(dir);
    printf("long line: refused 1\n");
}

/* A file that is not there gives every default and 0. A path that names a directory gives
 * 2 with a reason, because the reader read nothing. */
static void missing_and_directory(void)
{
    char dir[256];
    char path[320];
    unsigned int i;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(path, sizeof(path), "%s/no-such-file.settings", dir);
    memset(&table, 0xff, sizeof(table));
    CHECK(aotx_settings_read(path, &table) == 0, "a file that is not there must give 0");
    CHECK(table.refused_count == 0u, "a file that is not there must refuse nothing");
    for (i = 0; i < AOTX_SETTING_NUMBER_COUNT; i++) {
        CHECK(table.number[i] == want_numbers[i].value, "the key %s does not hold its default",
              want_numbers[i].name);
        CHECK(table.number_given[i] == 0u, "the key %s counts as given", want_numbers[i].name);
    }
    for (i = 0; i < AOTX_SETTING_TEXT_COUNT; i++) {
        CHECK(strcmp(table.text[i], want_texts[i].value) == 0, "the key %s does not hold its"
              " default", want_texts[i].name);
    }
    memset(&table, 0xff, sizeof(table));
    CHECK(aotx_settings_read(dir, &table) == 2, "a directory must give 2");
    CHECK(table.refused_count == 1u, "a directory must give one reason and gives %u",
          table.refused_count);
    CHECK(strcmp(table.refused[0].reason, "the path names a directory") == 0,
          "a directory gives the reason %s", table.refused[0].reason);
    CHECK(table.number[AOTX_SET_MIRROR_HZ] == 30, "a directory must still give the defaults");
    aotx_remove_tree(dir);
    printf("missing file: 0, directory: 2\n");
}

/* Directory values in the file stand beside that file, independent of the directory from
 * which a program starts. Absolute values and non-directory text stay as written. */
static void relative_directories(void)
{
    char dir[256];
    char settings_dir[320];
    char path[384];
    char want[512];
    const char *text =
        "journal.dir = run\n"
        "models.dir = store/models\n"
        "modules.dir = roles\n"
        "tools.root = work\n"
        "models.roles = talker\n";
    const unsigned int indexes[] = {
        AOTX_SET_JOURNAL_DIR, AOTX_SET_MODELS_DIR,
        AOTX_SET_MODULES_DIR, AOTX_SET_TOOLS_ROOT
    };
    const char *tails[] = { "run", "store/models", "roles", "work" };
    unsigned int i;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(settings_dir, sizeof(settings_dir), "%s/config", dir);
    CHECK(mkdir(settings_dir, 0700) == 0, "the settings directory does not open");
    snprintf(path, sizeof(path), "%s/aotx.settings", settings_dir);
    put_file(path, text, strlen(text));
    CHECK(aotx_settings_read(path, &table) == 0, "the relative directories do not read");
    for (i = 0u; i < sizeof(indexes) / sizeof(indexes[0]); i++) {
        snprintf(want, sizeof(want), "%s/%s", settings_dir, tails[i]);
        CHECK(strcmp(table.text[indexes[i]], want) == 0,
              "the setting %s resolved as %s, not %s",
              aotx_settings_text_name(indexes[i]), table.text[indexes[i]], want);
    }
    CHECK(strcmp(table.text[AOTX_SET_MODELS_ROLES], "talker") == 0,
          "a text that is not a directory was changed");
    aotx_remove_tree(dir);
    printf("relative settings directories: 4\n");
}

/* ---- the writer ---- */

/* The head that every write of the batch must keep, byte for byte. */
static const char write_head[] = "# the head of a file of writes\n\n";

static const char fixture[] =
    "# the head of the file\n"
    "\n"
    "journal.dir = journal\n"
    "tick.period_ms = 10\n"
    "\n"
    "# the sampling of the console\n"
    "sample.temperature = 0.7\n";

/* One key of the file is written again, and every other byte of the file stays. */
static void write_middle(void)
{
    char dir[256];
    char path[320];
    char reason[AOTX_SETTINGS_REASON_BYTES];
    const char *want =
        "# the head of the file\n"
        "\n"
        "journal.dir = journal\n"
        "tick.period_ms = 25\n"
        "\n"
        "# the sampling of the console\n"
        "sample.temperature = 0.7\n";
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(path, sizeof(path), "%s/aotx.settings", dir);
    put_file(path, fixture, sizeof(fixture) - 1);
    reason[0] = '\0';
    CHECK(aotx_settings_write_key(path, "tick.period_ms", "25", reason) == 0,
          "the write of a key of the file is refused: %s", reason);
    CHECK(get_file(path, file_text, sizeof(file_text)) == (int)strlen(want),
          "the file holds %d bytes and %d were asked for",
          get_file(path, file_text, sizeof(file_text)), (int)strlen(want));
    CHECK(strcmp(file_text, want) == 0, "the file holds:\n%s", file_text);
    CHECK(leftovers(dir) == 0, "the write left %d temporary files", leftovers(dir));

    /* A file that names one key twice takes the new line in the place of each of them.
     * The line that wins holds the value of the write, and the file keeps its shape. */
    {
        const char *twice =
            "mirror.hz = 20\n"
            "# a comment between the two lines\n"
            "mirror.hz = 40\n";
        const char *after =
            "mirror.hz = 75\n"
            "# a comment between the two lines\n"
            "mirror.hz = 75\n";
        put_file(path, twice, strlen(twice));
        CHECK(aotx_settings_write_key(path, "mirror.hz", "75", reason) == 0,
              "the write of a key that the file names twice is refused: %s", reason);
        get_file(path, file_text, sizeof(file_text));
        CHECK(strcmp(file_text, after) == 0, "the file holds:\n%s", file_text);
        CHECK(aotx_settings_read(path, &table) == 0, "the file is not read clean");
        CHECK(table.number[AOTX_SET_MIRROR_HZ] == 75, "the key holds %lld after the write",
              (long long)table.number[AOTX_SET_MIRROR_HZ]);
    }
    aotx_remove_tree(dir);
    printf("write in the middle: bytes %d\n", (int)strlen(want));
}

/* A key that the file does not name goes to the end, and every other byte stays. */
static void write_end(void)
{
    char dir[256];
    char path[320];
    char reason[AOTX_SETTINGS_REASON_BYTES];
    const char *want =
        "# the head of the file\n"
        "\n"
        "journal.dir = journal\n"
        "tick.period_ms = 10\n"
        "\n"
        "# the sampling of the console\n"
        "sample.temperature = 0.7\n"
        "mirror.hz = 60\n";
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(path, sizeof(path), "%s/aotx.settings", dir);
    put_file(path, fixture, sizeof(fixture) - 1);
    reason[0] = '\0';
    CHECK(aotx_settings_write_key(path, "mirror.hz", "60", reason) == 0,
          "the write of a key the file does not name is refused: %s", reason);
    get_file(path, file_text, sizeof(file_text));
    CHECK(strcmp(file_text, want) == 0, "the file holds:\n%s", file_text);
    CHECK(leftovers(dir) == 0, "the write left %d temporary files", leftovers(dir));

    /* A file whose last line has no end byte gets one, so the new line stands alone. */
    put_file(path, "tui.box = ascii", 15);
    CHECK(aotx_settings_write_key(path, "mirror.hz", "90", reason) == 0,
          "the write to a file with no end byte is refused: %s", reason);
    get_file(path, file_text, sizeof(file_text));
    CHECK(strcmp(file_text, "tui.box = ascii\nmirror.hz = 90\n") == 0, "the file holds:\n%s",
          file_text);

    /* A file that is not there is made, and holds the one line. */
    snprintf(path, sizeof(path), "%s/new.settings", dir);
    CHECK(aotx_settings_write_key(path, "agent.budget", "12", reason) == 0,
          "the write to a file that is not there is refused: %s", reason);
    get_file(path, file_text, sizeof(file_text));
    CHECK(strcmp(file_text, "agent.budget = 12\n") == 0, "the new file holds:\n%s", file_text);
    CHECK(leftovers(dir) == 0, "the write left %d temporary files", leftovers(dir));
    aotx_remove_tree(dir);
    printf("write at the end: files 3\n");
}

/* The writer refuses a key and a value that the reader would refuse, and leaves the file
 * as it was. */
static void write_refusals(void)
{
    char dir[256];
    char path[320];
    char reason[AOTX_SETTINGS_REASON_BYTES];
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(path, sizeof(path), "%s/aotx.settings", dir);
    put_file(path, fixture, sizeof(fixture) - 1);
    reason[0] = '\0';
    CHECK(aotx_settings_write_key(path, "no.such.key", "1", reason) == 1,
          "the write of an unknown key must be refused");
    CHECK(strcmp(reason, "the key is not known") == 0, "the unknown key gives the reason %s",
          reason);
    reason[0] = '\0';
    CHECK(aotx_settings_write_key(path, "mirror.hz", "500", reason) == 1,
          "the write of a value out of range must be refused");
    CHECK(strcmp(reason, "the value 500 is not in the range 1 to 120") == 0,
          "the value out of range gives the reason %s", reason);
    reason[0] = '\0';
    CHECK(aotx_settings_write_key(path, "tui.box", "a\nb", reason) == 1,
          "a value with an end of line byte must be refused");
    CHECK(strcmp(reason, "the value holds an end of line byte") == 0,
          "the value with an end of line byte gives the reason %s", reason);
    get_file(path, file_text, sizeof(file_text));
    CHECK(strcmp(file_text, fixture) == 0, "a refused write changed the file");
    CHECK(leftovers(dir) == 0, "a refused write left %d temporary files", leftovers(dir));
    aotx_remove_tree(dir);
    printf("write refusals: 3\n");
}

/* n keys go to one file, one write at a time, and the file reads back with every one of
 * them. The file starts with a comment and a blank line, which every write keeps. */
static void write_batch(int n)
{
    char dir[256];
    char path[320];
    char reason[AOTX_SETTINGS_REASON_BYTES];
    char value[32];
    unsigned int k;
    int i;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(path, sizeof(path), "%s/aotx.settings", dir);
    put_file(path, write_head, sizeof(write_head) - 1);
    for (i = 0; i < n; i++) {
        k = (unsigned int)(i % (int)AOTX_SETTING_NUMBER_COUNT);
        aotx_settings_format(pick(k, i), aotx_settings_number_scale(k), value, sizeof(value));
        reason[0] = '\0';
        CHECK(aotx_settings_write_key(path, aotx_settings_number_name(k), value, reason) == 0,
              "the write of %s is refused: %s", aotx_settings_number_name(k), reason);
    }
    CHECK(leftovers(dir) == 0, "the writes left %d temporary files", leftovers(dir));
    get_file(path, file_text, sizeof(file_text));
    CHECK(strncmp(file_text, write_head, sizeof(write_head) - 1) == 0,
          "the writes did not keep the comment and the blank line");
    CHECK(aotx_settings_read(path, &table) == 0, "the file of %d writes is not read clean", n);
    for (k = 0; k < AOTX_SETTING_NUMBER_COUNT; k++) {
        int last = -1;
        for (i = 0; i < n; i++) {
            if ((unsigned int)(i % (int)AOTX_SETTING_NUMBER_COUNT) == k) {
                last = i;
            }
        }
        if (last < 0) {
            continue;
        }
        CHECK(table.number[k] == pick(k, last), "the key %s holds %lld after the writes and"
              " %lld was asked for", aotx_settings_number_name(k), (long long)table.number[k],
              (long long)pick(k, last));
    }
    aotx_remove_tree(dir);
    printf("write batch %d: keys %d\n", n,
           (n < (int)AOTX_SETTING_NUMBER_COUNT) ? n : (int)AOTX_SETTING_NUMBER_COUNT);
}

int main(void)
{
    defaults();
    batch(1);
    batch(64);
    round_trip(1);
    round_trip(64);
    refusals();
    many_refusals();
    long_line();
    missing_and_directory();
    relative_directories();
    write_middle();
    write_end();
    write_refusals();
    write_batch(1);
    write_batch(64);
    return aotx_report("settings_test", 900);
}
