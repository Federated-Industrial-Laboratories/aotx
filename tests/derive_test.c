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

/* The end of a reply gives one line, and the tokens of the reply give none. The line comes
 * from the system writer, whatever writer the record carries, because the event belongs to
 * the run. An open event and a release event are not an end and give no line. */
static void sequences(int n)
{
    run_ctx c;
    char path[1024];
    char want_text[256];
    aotx_commit_body commit;
    aotx_clock_body clock;
    int i;

    start(&c, 0x00c05e0000000001ull + (uint64_t)n, NULL);
    clock.wall_ns = aotx_wall_ns();
    c.device.writer = AOTX_WRITER_FEEDER;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_START, &clock, sizeof(clock));
    for (i = 0; i < n; i++) {
        aotx_sequence_body event;
        aotx_token_body token;
        /* The record carries an agent writer, and the line must still name the system. */
        c.device.writer = AOTX_WRITER_AGENT_BASE + (uint32_t)(i % 3);
        aotx_fake_token(i, &token);
        aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TOKEN, &token, sizeof(token));
        aotx_fake_sequence(i, AOTX_SEQ_DONE, &event);
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_SEQUENCE, &event, sizeof(event));
        if (i == 0) {
            aotx_fake_sequence(i, AOTX_SEQ_OPENED, &event);
            aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_SEQUENCE, &event, sizeof(event));
            aotx_fake_sequence(i, AOTX_SEQ_RELEASED, &event);
            aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_SEQUENCE, &event, sizeof(event));
            aotx_fake_sequence(i, AOTX_SEQ_STOPPED, &event);
            aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_SEQUENCE, &event, sizeof(event));
            /* A body that is shorter than the layout holds no counts and gives no line. */
            aotx_fake_sequence(i, AOTX_SEQ_DONE, &event);
            aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_SEQUENCE, &event, 8u);
        }
    }
    memset(&commit, 0, sizeof(commit));
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);

    bus_path(&c, path, sizeof(path));
    CHECK(slurp(path, text, sizeof(text)) > 0, "the message file does not read");
    CHECK(count_of(text, "\n") == n + 1, "the file holds %d lines and %d were asked for",
          count_of(text, "\n"), n + 1);
    CHECK(count_of(text, "\"agent\":\"system\"") == n + 1,
          "the file holds %d lines of the system writer", count_of(text, "\"agent\":\"system\""));
    CHECK(count_of(text, "sequence done") == n, "the file holds %d end lines and %d were asked for",
          count_of(text, "sequence done"), n);
    CHECK(count_of(text, "sequence stopped") == 1, "the file holds %d stopped lines",
          count_of(text, "sequence stopped"));
    for (i = 0; i < n; i++) {
        aotx_sequence_body event;
        aotx_fake_sequence(i, AOTX_SEQ_DONE, &event);
        snprintf(want_text, sizeof(want_text),
                 "\"text\":\"sequence done slot %u role %u prompt %u sampled %u ticks %llu\"",
                 event.slot, event.role, event.prompt_tokens, event.sampled_tokens,
                 (unsigned long long)event.ticks);
        CHECK(strstr(text, want_text) != NULL, "the end line of sequence %d is not in the file", i);
    }
    validate(path);
    snprintf(path, sizeof(path), "%s/console.log", c.boot_dir);
    text[0] = '\0';
    slurp(path, text, sizeof(text));
    CHECK(count_of(text, "\n") == 0, "a token record must not reach the console log");
    printf("sequences %d: lines %d, tokens written %d\n", n, n + 1, n);
    aotx_remove_tree(c.dir);
}

/* Marks the record that went in last as one that continues the line before it. */
static void mark_fragment(aotx_fake_device *d)
{
    aotx_record_header *h = (aotx_record_header *)(d->stage + AOTX_BLOCK_HEADER_BYTES +
                                                   (size_t)(d->count - 1) * AOTX_SLOT_BYTES);
    h->flags = (uint16_t)(h->flags | AOTX_FLAG_FRAGMENT);
}

/* A console record with the fragment flag continues the line before it. A reply that comes
 * in one record for each tick therefore reads as one line. A record whose text is white
 * space only reaches the console log and makes no message line. The line schema refuses a
 * text field that holds nothing. */
static void console_lines(int n)
{
    run_ctx c;
    char path[1024];
    char want_line[AOTX_TEXT_MAX_BYTES / 64];
    char part[32];
    aotx_commit_body commit;
    int notes;
    int i;

    start(&c, 0x00d05e0000000001ull + (uint64_t)n, NULL);
    c.device.writer = AOTX_WRITER_CONSOLE;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_CONSOLE, "conductor: ", 11u);
    snprintf(want_line, sizeof(want_line), "conductor: ");
    for (i = 0; i < n; i++) {
        snprintf(part, sizeof(part), " p%d", i);
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_CONSOLE, part,
                         (uint32_t)strlen(part));
        mark_fragment(&c.device);
        strncat(want_line, part, sizeof(want_line) - strlen(want_line) - 1);
    }
    /* A fragment of white space only still reaches the line and makes no message. */
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_CONSOLE, " ", 1u);
    mark_fragment(&c.device);
    strncat(want_line, " ", sizeof(want_line) - strlen(want_line) - 1);
    /* A record with no flag ends that line and starts one of its own. */
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_CONSOLE, "   ", 3u);
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_CONSOLE, "done", 4u);
    memset(&commit, 0, sizeof(commit));
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);

    snprintf(path, sizeof(path), "%s/console.log", c.boot_dir);
    text[0] = '\0';
    slurp(path, text, sizeof(text));
    CHECK(count_of(text, "\n") == 3, "the console log holds %d lines and 3 were asked for",
          count_of(text, "\n"));
    strncat(want_line, "\n", sizeof(want_line) - strlen(want_line) - 1);
    CHECK(strstr(text, want_line) != NULL, "the fragments do not read as one line");
    CHECK(strstr(text, "\n   \ndone\n") != NULL,
          "a record with no flag does not start a line of its own");

    bus_path(&c, path, sizeof(path));
    CHECK(slurp(path, text, sizeof(text)) > 0, "the message file does not read");
    notes = count_of(text, "\"type\":\"note\"");
    CHECK(notes == n + 2, "the file holds %d notes and %d were asked for", notes, n + 2);
    CHECK(count_of(text, "\"text\":\"   \"") == 0, "a text of white space only made a line");
    CHECK(count_of(text, "\"text\":\" \"") == 0, "a fragment of one space made a line");
    for (i = 0; i < n; i++) {
        snprintf(part, sizeof(part), "\"text\":\" p%d\"", i);
        CHECK(strstr(text, part) != NULL, "fragment %d has no message line", i);
    }
    validate(path);
    printf("console %d: lines 3, fragments %d, notes %d\n", n, n + 1, notes);
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
        {
            aotx_sequence_body event;
            aotx_token_body token;
            aotx_fake_sequence(i, AOTX_SEQ_DONE, &event);
            aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_SEQUENCE, &event, sizeof(event));
            /* A token record reaches the journal and makes no line, whatever the switch
             * holds, because the text of a reply comes from the console records. */
            aotx_fake_token(i, &token);
            aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TOKEN, &token, sizeof(token));
        }
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
    CHECK(records == 21, "the journal holds %d records and 21 were written", records);
    CHECK(count_of(text, "sequence done") == 0,
          "the console log must not hold a sequence line");
    printf("switch %s: lines %d, console %d, records %d\n",
           (derive == NULL) ? "console,note,bus,bulk,sequence" : derive, want_lines,
           want_console, records);
    aotx_remove_tree(c.dir);
}

/* A request that needs no authorization makes its line at once. A request that waits for
 * the operator makes its line when the record that grants it comes. That line carries the
 * fields of the request and not of the record that grants it. A request the operator
 * refuses makes no line at all. */
static void requests(int n)
{
    run_ctx c;
    char path[1024];
    char want[512];
    aotx_commit_body commit;
    aotx_clock_body clock;
    aotx_tool_request_body r;
    int i;

    start(&c, 0x00e05e0000000001ull + (uint64_t)n, NULL);
    clock.wall_ns = aotx_wall_ns();
    c.device.writer = AOTX_WRITER_FEEDER;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_START, &clock, sizeof(clock));
    /* The requests are made in one tick and the operator answers in the next. A request
     * that waits for the operator takes its deadline at the grant. The line of a granted
     * request therefore states the deadline and the tick of the record that grants it. It
     * states every other field of the request itself. */
    for (i = 0; i < n; i++) {
        c.device.writer = AOTX_WRITER_AGENT_BASE + (uint32_t)(i % 3);
        aotx_fake_request(i, AOTX_AUTH_NONE, &r);
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
        aotx_fake_request(1000 + i, AOTX_AUTH_PENDING, &r);
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
        /* The table is direct mapped by the identity. The identities of the two groups
         * that wait together are 64 apart, so no request takes the place of another. */
        aotx_fake_request(1064 + i, AOTX_AUTH_PENDING, &r);
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
    }
    aotx_fake_commit(&c.device, 0);
    for (i = 0; i < n; i++) {
        c.device.writer = AOTX_WRITER_AGENT_BASE + (uint32_t)(i % 3);
        /* Every field of the record that grants the request is another value. The line
         * takes its deadline and its tick from the grant and no other field of it. */
        aotx_fake_request(5000 + i, AOTX_AUTH_GRANTED, &r);
        r.request = (uint32_t)(2000 + i);
        r.arg_len = (uint32_t)snprintf(r.arg, AOTX_TOOL_ARG_BYTES, "not-the-path.txt");
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
        aotx_fake_request(5000 + i, AOTX_AUTH_REFUSED, &r);
        r.request = (uint32_t)(2064 + i);
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
    }
    /* A grant for a request the table does not hold takes the fields of the grant. */
    aotx_fake_request(8000, AOTX_AUTH_GRANTED, &r);
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
    /* A grant that carries no path names no file, so it makes no line. */
    aotx_fake_request(8001, AOTX_AUTH_GRANTED, &r);
    r.arg_len = 0;
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
    /* A body that is shorter than the fixed fields names no request. */
    aotx_fake_request(8002, AOTX_AUTH_NONE, &r);
    aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, 8u);
    memset(&commit, 0, sizeof(commit));
    c.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&c.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
    aotx_fake_commit(&c.device, 0);
    finish(&c);

    snprintf(path, sizeof(path), "%s/requests.jsonl", c.dir);
    CHECK(slurp(path, text, sizeof(text)) > 0, "the requests file does not read");
    CHECK(count_of(text, "\n") == 2 * n + 1, "the requests file holds %d lines and %d were"
          " asked for", count_of(text, "\n"), 2 * n + 1);
    CHECK(count_of(text, "\"auth\":\"none\"") == n, "the file holds %d requests that need no"
          " authorization", count_of(text, "\"auth\":\"none\""));
    CHECK(count_of(text, "\"auth\":\"granted\"") == n + 1, "the file holds %d requests the"
          " operator granted", count_of(text, "\"auth\":\"granted\""));
    CHECK(count_of(text, "\"auth\":\"pending\"") == 0, "a request that waits for the operator"
          " must make no line");
    CHECK(count_of(text, "not-the-path.txt") == 0,
          "the line of a granted request must carry the path of the request");
    CHECK(count_of(text, "\"request\":9001") == 0, "a grant with no path made a line");
    CHECK(count_of(text, "\"request\":9002") == 0, "a body that is too short made a line");
    CHECK(count_of(text, "\"request\":9000") == 1, "a grant the table lost made no line");
    /* A line states the tick its deadline counts from. A tool that needs no authorization
     * takes the tick of the request. A tool the operator granted takes the tick of the
     * grant, because the deadline of such a request starts there. */
    CHECK(count_of(text, "\"tick\":1}") == n,
          "%d lines state the tick the request was made at and %d were asked for",
          count_of(text, "\"tick\":1}"), n);
    CHECK(count_of(text, "\"tick\":2}") == n + 1,
          "%d lines state the tick the operator answered at and %d were asked for",
          count_of(text, "\"tick\":2}"), n + 1);
    for (i = 0; i < n; i++) {
        snprintf(want, sizeof(want),
                 "{\"request\":%d,\"agent\":%d,\"turn\":%d,\"tool\":\"fs_read\","
                 "\"arg\":\"file-%d.txt\",\"deadline\":%d,\"auth\":\"none\",\"tick\":1}",
                 1000 + i, i % 64, i % 8, i, 500 + i);
        CHECK(strstr(text, want) != NULL, "request %d of the first group is not in the file", i);
        snprintf(want, sizeof(want),
                 "{\"request\":%d,\"agent\":%d,\"turn\":%d,\"tool\":\"fs_read\","
                 "\"arg\":\"file-%d.txt\",\"deadline\":%d,\"auth\":\"granted\","
                 "\"tick\":2}",
                 2000 + i, (1000 + i) % 64, (1000 + i) % 8, 1000 + i, 5500 + i);
        CHECK(strstr(text, want) != NULL, "granted request %d is not in the file", i);
        snprintf(want, sizeof(want), "\"arg\":\"file-%d.txt\"", 1064 + i);
        CHECK(strstr(text, want) == NULL, "refused request %d is in the file", i);
    }
    printf("requests %d: lines %d, granted %d\n", n, count_of(text, "\n"),
           count_of(text, "\"auth\":\"granted\""));
    aotx_remove_tree(c.dir);
}

/* Every line of the chain carries the digest of the line before it, so a reader can prove
 * that no line was taken out. The first line has no line before it and carries 64 zeros. */
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
        aotx_fake_record(&c.device, AOTX_CLASS_B, AOTX_REC_MANIFEST, &m, sizeof(m));
    }
    /* A body that is shorter than the layout holds no hash, so it makes no line. */
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
    /* The first line has no line before it, so the digest it names is 64 zeros. */
    memset(digest_text, '0', 64);
    digest_text[64] = '\0';
    at = text;
    while (*at != '\0' && line < n) {
        char *end = strchr(at, '\n');
        size_t bytes;
        if (end == NULL) {
            break;
        }
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

/* A task that is done gives an artifact to the run, so its line is a handoff. Every other
 * task state and every agent event gives a note. */
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
    /* A body that is shorter than the layout makes no line. */
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
    sequences(1);
    sequences(64);
    console_lines(1);
    console_lines(64);
    filter(NULL, 16, 4, 1);
    filter("console", 4, 4, 2);
    filter("note,bus", 8, 0, 3);
    filter("sequence", 4, 0, 4);
    filter("none", 0, 0, 5);
    requests(1);
    requests(64);
    turns(1);
    turns(64);
    events(1);
    events(64);
    return aotx_report("derive_test", 800);
}
