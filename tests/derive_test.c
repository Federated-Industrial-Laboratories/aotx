/* Purpose: Check the message lines that the drain derives, and the per type derive switch.
 * Owns: One temporary journal and one host ring for each case.
 * Threading: Two processes; the test writes the ring while the drain reads it.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include <fcntl.h>
#include <time.h>

#define AOTX_TEXT_MAX_BYTES 1048576
#define AOTX_RING_BYTES     262144u

static int argument_count;
static char **arguments;
static char text[AOTX_TEXT_MAX_BYTES];

typedef struct run_ctx {
    aotx_map map;
    aotx_host_ring ring;
    aotx_fake_device device;
    char dir[256];
    char boot_dir[320];
    int echo_fd;
    int child;
} run_ctx;

static int slurp(const char *path, char *out, size_t bytes)
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

static int count_of(const char *hay, const char *needle)
{
    int found = 0;
    const char *at = hay;
    size_t len = strlen(needle);
    while ((at = strstr(at, needle)) != NULL) {
        found++;
        at += len;
    }
    return found;
}

/* Runs the validator that the command line names, with the message file as the last word. */
static void validate(const char *path)
{
    char *args[8];
    int i;
    int child;
    if (argument_count < 3) {
        printf("no validator was given, so the message file check is not applied\n");
        return;
    }
    for (i = 2; i < argument_count && i < 6; i++) {
        args[i - 2] = arguments[i];
    }
    args[argument_count - 2] = (char *)path;
    args[argument_count - 1] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the validator does not start");
    CHECK(aotx_wait(child) == 0, "the validator refuses the message file");
}

static void bus_path(const run_ctx *c, char *out, size_t bytes)
{
    char day[16];
    time_t now = (time_t)(aotx_wall_ns() / 1000000000u);
    struct tm parts;
    localtime_r(&now, &parts);
    strftime(day, sizeof(day), "%Y-%m-%d", &parts);
    snprintf(out, bytes, "%s/bus/%s-aotx.jsonl", c->dir, day);
}

/* Counts the records that the journal holds, so a filtered type is still on the disk. */
static int journal_records(const char *boot_dir)
{
    char names[64][AOTX_NAME_BYTES];
    static unsigned char block[262144];
    int count = aotx_segment_list(boot_dir, names, 64);
    int records = 0;
    int i;
    CHECK(count > 0, "the journal holds no segment");
    for (i = 0; i < count; i++) {
        char path[1024];
        aotx_segment_reader reader;
        snprintf(path, sizeof(path), "%.900s/%.*s", boot_dir, AOTX_NAME_BYTES - 1, names[i]);
        CHECK(aotx_segment_reader_open(&reader, path) == 0, "the segment does not read");
        for (;;) {
            uint32_t got = 0;
            int frame = aotx_segment_get(&reader, block, sizeof(block), &got);
            if (frame != AOTX_FRAME_OK) {
                break;
            }
            records += (int)((const aotx_block_header *)block)->record_count;
        }
        aotx_segment_reader_close(&reader);
    }
    return records;
}

static void start(run_ctx *c, uint64_t boot_id, const char *derive)
{
    char fd_text[16];
    char path[512];
    char *args[8];
    int n = 0;
    memset(c, 0, sizeof(*c));
    CHECK(aotx_temp_dir(c->dir, sizeof(c->dir)) == 0, "the temporary directory does not open");
    CHECK(aotx_host_ring_create(AOTX_RING_BYTES, boot_id, &c->map, &c->ring) == 0,
          "the ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", c->map.fd);
    snprintf(path, sizeof(path), "%s/echo.txt", c->dir);
    c->echo_fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    CHECK(c->echo_fd >= 0, "the echo file does not open");
    args[n++] = arguments[1];
    args[n++] = (char *)"--ring-fd";
    args[n++] = fd_text;
    args[n++] = (char *)"--journal";
    args[n++] = c->dir;
    if (derive != NULL) {
        args[n++] = (char *)"--derive";
        args[n++] = (char *)derive;
    }
    args[n] = NULL;
    c->child = aotx_spawn(args, -1, c->echo_fd);
    CHECK(c->child > 0, "the drain does not start");
    snprintf(c->boot_dir, sizeof(c->boot_dir), "%s/%016llx", c->dir, (unsigned long long)boot_id);
    aotx_fake_start(&c->device, &c->ring, boot_id);
}

static void finish(run_ctx *c)
{
    aotx_store_release16(&c->ring.pre->closed, 1);
    CHECK(aotx_wait(c->child) == 0, "the drain does not end with a clean status");
    CHECK(aotx_host_ring_cursor(&c->ring) == aotx_host_ring_head(&c->ring),
          "the drain did not reach the head of the ring");
    close(c->echo_fd);
    aotx_map_release(&c->map);
}

/* Writes one group of messages of every kind, with content that no other group holds. */
static void group(run_ctx *c, int i, uint32_t *writer_seq)
{
    char body[192];
    uint64_t finding_seq;
    uint64_t question_seq;
    aotx_commit_body commit;
    aotx_clock_body clock;

    clock.wall_ns = aotx_wall_ns();
    c->device.writer = AOTX_WRITER_FEEDER;
    aotx_fake_record(&c->device, AOTX_CLASS_A, AOTX_REC_TICK_START, &clock, sizeof(clock));

    c->device.writer = AOTX_WRITER_AGENT_BASE;
    snprintf(body, sizeof(body), "claim %d of the run", i);
    finding_seq = aotx_fake_bus(&c->device, AOTX_BUS_FINDING, (uint8_t)(1 + (i % 4)),
                                writer_seq[0]++, 0, 0, 0.0f, body);
    snprintf(body, sizeof(body), "question %d of the run", i);
    question_seq = aotx_fake_bus(&c->device, AOTX_BUS_QUESTION, 0, writer_seq[0]++, 0, 0, 0.0f,
                                 body);
    snprintf(body, sizeof(body), "path/of/%d\nready", i);
    aotx_fake_bus(&c->device, AOTX_BUS_HANDOFF, 0, writer_seq[0]++, 0, 0, 0.0f, body);

    c->device.writer = AOTX_WRITER_AGENT_BASE + 1u;
    snprintf(body, sizeof(body), "basis %d of the run", i);
    aotx_fake_bus(&c->device, AOTX_BUS_RANK, 0, writer_seq[1]++, finding_seq, 0,
                  (float)(i % 100) / 100.0f, body);
    snprintf(body, sizeof(body), "answer %d of the run", i);
    aotx_fake_bus(&c->device, AOTX_BUS_ANSWER, 0, writer_seq[1]++, question_seq, 0, 0.0f, body);
    snprintf(body, sizeof(body), "consumed %d\nproduced %d", i, i);
    aotx_fake_bus(&c->device, AOTX_BUS_COST, 0, writer_seq[1]++, 0, 0, 0.0f, body);
    snprintf(body, sizeof(body), "note %d of the run", i);
    aotx_fake_bus(&c->device, AOTX_BUS_NOTE, 0, writer_seq[1]++, 0, 0, 0.0f, body);

    if (i == 0) {
        c->device.writer = AOTX_WRITER_AGENT_BASE + 2u;
        /* A correction names a message that the map holds, and states a reason. */
        aotx_fake_bus(&c->device, AOTX_BUS_FINDING, AOTX_PROV_COMPUTED, writer_seq[2]++, 0,
                      finding_seq, 0.0f, "the claim of the first message is not right");
        /* A reference that the map does not hold must leave a gap that a reader can see. */
        aotx_fake_bus(&c->device, AOTX_BUS_RANK, 0, writer_seq[2]++, 999999ull, 0, 0.5f,
                      "a rank of a record that no line names");
        aotx_fake_bus(&c->device, AOTX_BUS_ANSWER, 0, writer_seq[2]++, 999998ull, 0, 0.0f,
                      "an answer to a record that no line names");
        /* A rank names a finding or a handoff only, so a rank of a question is a gap. */
        aotx_fake_bus(&c->device, AOTX_BUS_RANK, 0, writer_seq[2]++, question_seq, 0, 0.25f,
                      "a rank of a question");
        aotx_fake_bus(&c->device, AOTX_BUS_FINDING, AOTX_PROV_FETCHED, writer_seq[2]++, 0,
                      999997ull, 0.0f, "a correction of a record that no line names");
        aotx_fake_bus(&c->device, AOTX_BUS_COST, 0, writer_seq[2]++, 0, 0, 0.0f,
                      "the cost with no second part");
    }
    memset(&commit, 0, sizeof(commit));
    commit.state_hash = 0x0123456789abcdefull + (uint64_t)i;
    c->device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c->device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
    aotx_fake_commit(&c->device, 0);
}

static void batch(int n)
{
    run_ctx c;
    char path[1024];
    uint32_t writer_seq[3] = { 1u, 1u, 1u };
    int want = 7 * n + 6;
    int i;

    start(&c, 0x00b05e0000000001ull + (uint64_t)n, NULL);
    for (i = 0; i < n; i++) {
        group(&c, i, writer_seq);
    }
    finish(&c);

    bus_path(&c, path, sizeof(path));
    CHECK(slurp(path, text, sizeof(text)) > 0, "the message file does not read");
    CHECK(count_of(text, "\n") == want, "the file holds %d lines and %d were derived",
          count_of(text, "\n"), want);
    CHECK(count_of(text, "\"type\":\"finding\"") == n + 2,
          "the file holds %d findings", count_of(text, "\"type\":\"finding\""));
    CHECK(count_of(text, "\"type\":\"rank\"") == n, "the file holds %d ranks",
          count_of(text, "\"type\":\"rank\""));
    CHECK(count_of(text, "\"type\":\"question\"") == n, "the file holds %d questions",
          count_of(text, "\"type\":\"question\""));
    CHECK(count_of(text, "\"type\":\"answer\"") == n, "the file holds %d answers",
          count_of(text, "\"type\":\"answer\""));
    CHECK(count_of(text, "\"type\":\"handoff\"") == n, "the file holds %d handoffs",
          count_of(text, "\"type\":\"handoff\""));
    CHECK(count_of(text, "\"type\":\"cost\"") == n + 1, "the file holds %d costs",
          count_of(text, "\"type\":\"cost\""));
    /* Three messages take the note kind, because the record that each one names is a gap. */
    CHECK(count_of(text, "\"type\":\"note\"") == n + 3, "the file holds %d notes",
          count_of(text, "\"type\":\"note\""));
    CHECK(count_of(text, "\"unresolved\":\"") == 4, "the file names %d gaps and four are there",
          count_of(text, "\"unresolved\":\""));
    CHECK(count_of(text, "\"req\":[\"msg-relations\"]") == 1, "the correction has no request");
    CHECK(count_of(text, "\"corrects\":[\"agent-0-1\"]") == 1, "the correction names no message");
    CHECK(strstr(text, "\"id\":\"agent-0-1\",\"claim\":\"claim 0 of the run\"") != NULL,
          "the first finding does not carry its own id");
    CHECK(strstr(text, "\"provenance\":\"computed\"") != NULL, "no finding carries a provenance");
    CHECK(strstr(text, "\"re\":\"agent-0-1\",\"score\":0.000000,\"basis\":\"basis 0 of the run\"")
              != NULL, "the first rank does not name the first finding");
    CHECK(strstr(text, "\"path\":\"path/of/0\",\"status\":\"ready\"") != NULL,
          "the handoff does not carry its path and its state");
    CHECK(strstr(text, "\"consumed\":\"consumed 0\",\"produced\":\"produced 0\"") != NULL,
          "the cost does not carry both parts");
    CHECK(strstr(text, "\"produced\":\"not stated\"") != NULL,
          "a cost with one part must say that the second is not there");
    CHECK(strstr(text, "\"text\":\"a rank of a question\",\"kind\":\"rank\"") != NULL,
          "a rank of a message that is not rankable must take the note kind");
    /* Every group holds content that no other group holds, so a wrong record or a wrong
     * reference cannot hide behind a count. Agent 0 writes three messages for each group. */
    for (i = 0; i < n; i++) {
        char want_text[256];
        snprintf(want_text, sizeof(want_text),
                 "\"id\":\"agent-0-%d\",\"claim\":\"claim %d of the run\"", 3 * i + 1, i);
        CHECK(strstr(text, want_text) != NULL, "finding %d is not in the file", i);
        snprintf(want_text, sizeof(want_text),
                 "\"re\":\"agent-0-%d\",\"score\":%.6f,\"basis\":\"basis %d of the run\"",
                 3 * i + 1, (double)(i % 100) / 100.0, i);
        CHECK(strstr(text, want_text) != NULL, "rank %d does not name finding %d", i, i);
        snprintf(want_text, sizeof(want_text), "\"text\":\"note %d of the run\"", i);
        CHECK(strstr(text, want_text) != NULL, "note %d is not in the file", i);
        snprintf(want_text, sizeof(want_text), "\"path\":\"path/of/%d\",\"status\":\"ready\"", i);
        CHECK(strstr(text, want_text) != NULL, "handoff %d is not in the file", i);
    }
    validate(path);
    printf("batch %d: lines %d, gaps %d\n", n, count_of(text, "\n"),
           count_of(text, "\"unresolved\":\""));
    aotx_remove_tree(c.dir);
}

/* A type that the switch leaves out gives no line, and the journal still holds the record. */
static void filter(const char *derive, int want_lines, int want_console, int number)
{
    run_ctx c;
    char path[1024];
    char body[64];
    uint32_t writer_seq[3] = { 1u, 1u, 1u };
    aotx_commit_body commit;
    int records;
    int i;

    start(&c, 0x00b05e0000000100ull + (uint64_t)number, derive);
    for (i = 0; i < 4; i++) {
        c.device.writer = AOTX_WRITER_CONSOLE;
        snprintf(body, sizeof(body), "console %d", i);
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_CONSOLE, body, (uint32_t)strlen(body));
        snprintf(body, sizeof(body), "note %d", i);
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_NOTE, body, (uint32_t)strlen(body));
        c.device.writer = AOTX_WRITER_AGENT_BASE;
        snprintf(body, sizeof(body), "claim %d", i);
        aotx_fake_bus(&c.device, AOTX_BUS_FINDING, AOTX_PROV_COMPUTED, writer_seq[0]++, 0, 0,
                      0.0f, body);
    }
    memset(&commit, 0, sizeof(commit));
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);

    bus_path(&c, path, sizeof(path));
    text[0] = '\0';
    slurp(path, text, sizeof(text));
    CHECK(count_of(text, "\n") == want_lines, "the switch %s gives %d lines and %d were asked for",
          (derive == NULL) ? "of every type" : derive, count_of(text, "\n"), want_lines);
    if (want_lines > 0) {
        validate(path);
    }
    snprintf(path, sizeof(path), "%s/console.log", c.boot_dir);
    text[0] = '\0';
    slurp(path, text, sizeof(text));
    CHECK(count_of(text, "\n") == want_console, "the console log holds %d lines and %d were asked for",
          count_of(text, "\n"), want_console);
    records = journal_records(c.boot_dir);
    CHECK(records == 13, "the journal holds %d records and 13 were written", records);
    printf("switch %s: lines %d, console %d, records %d\n",
           (derive == NULL) ? "console,note,bus,bulk" : derive, want_lines, want_console, records);
    aotx_remove_tree(c.dir);
}

int main(int argc, char **argv)
{
    argument_count = argc;
    arguments = argv;
    if (argc < 2) {
        printf("usage: derive_test <drain program> [validator ...]\n");
        return 1;
    }
    batch(1);
    batch(64);
    filter(NULL, 12, 4, 1);
    filter("console", 4, 4, 2);
    filter("note,bus", 8, 0, 3);
    filter("none", 0, 0, 4);
    return aotx_report("derive_test", 300);
}
