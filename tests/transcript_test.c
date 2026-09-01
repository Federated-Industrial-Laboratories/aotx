/* Purpose: Check every transcript element and the journal comparison at two agent counts.
 * Owns: One fake journal and its derived transcript files for each count.
 * Threading: Two processes while the drain reads the fake device ring.
 * Lifetime: The run of the test. */
#include "tests/disk_fake.h"

#include <fcntl.h>

#define AOTX_TRANSCRIPT_RING (1024u * 1024u)

static const char *drain_program;
static const char *journal_program;

typedef struct transcript_run {
    aotx_map map;
    aotx_host_ring ring;
    aotx_fake_device device;
    char dir[256];
    char boot_dir[320];
    int child;
    int output;
} transcript_run;

static void start_run(transcript_run *r, uint64_t boot)
{
    char fd_text[16];
    char output_path[320];
    char *argv[8];
    int at = 0;
    memset(r, 0, sizeof(*r));
    CHECK(aotx_temp_dir(r->dir, sizeof(r->dir)) == 0, "the temporary directory does not open");
    CHECK(aotx_host_ring_create(AOTX_TRANSCRIPT_RING, boot, &r->map, &r->ring) == 0,
          "the host ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", r->map.fd);
    snprintf(output_path, sizeof(output_path), "%s/drain.out", r->dir);
    r->output = open(output_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    argv[at++] = (char *)drain_program;
    argv[at++] = (char *)"--ring-fd";
    argv[at++] = fd_text;
    argv[at++] = (char *)"--journal";
    argv[at++] = r->dir;
    argv[at++] = (char *)"--derive";
    argv[at++] = (char *)"transcript";
    argv[at] = NULL;
    r->child = aotx_spawn(argv, -1, r->output);
    CHECK(r->child > 0, "the drain does not start");
    snprintf(r->boot_dir, sizeof(r->boot_dir), "%s/%016llx", r->dir,
             (unsigned long long)boot);
    aotx_fake_start(&r->device, &r->ring, boot);
}

static void finish_run(transcript_run *r)
{
    aotx_store_release16(&r->ring.pre->closed, 1u);
    CHECK(aotx_wait(r->child) == 0, "the drain does not end cleanly");
    close(r->output);
    aotx_map_release(&r->map);
}

static void add_text_record(aotx_fake_device *d, uint8_t type, uint32_t writer,
                            uint16_t flags, const unsigned char *text, uint32_t length)
{
    d->writer = writer;
    d->flags = flags;
    aotx_fake_record(d, (type == AOTX_REC_INPUT_LINE) ? AOTX_CLASS_A : AOTX_CLASS_B,
                     type, text, length);
    d->flags = 0u;
}

static void add_request(aotx_fake_device *d, int number, uint32_t auth, uint32_t request)
{
    aotx_tool_request_body body;
    aotx_fake_request(number, auth, &body);
    body.request = request;
    body.turn = (uint32_t)(number + 1);
    d->writer = AOTX_WRITER_AGENT_BASE + (uint32_t)number;
    aotx_fake_record(d, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &body, sizeof(body));
}

static void add_result(aotx_fake_device *d, int number, uint32_t request)
{
    aotx_tool_reply_body body;
    unsigned int part;
    for (part = 0u; part < 2u; part++) {
        memset(&body, 0, sizeof(body));
        body.agent = (uint32_t)number;
        body.request = request;
        body.status = AOTX_TOOL_OK;
        body.part = part;
        body.parts = 2u;
        body.len = (uint32_t)snprintf(body.bytes, AOTX_TOOL_REPLY_BYTES,
                                     "result part %u of agent %d%s", part, number,
                                     (part == 0u) ? "\n" : "");
        d->writer = AOTX_WRITER_FEEDER;
        aotx_fake_record(d, AOTX_CLASS_A, AOTX_REC_TOOL_REPLY, &body, sizeof(body));
    }
}

static void add_reply_token(aotx_fake_device *d, int number, uint32_t position,
                            const unsigned char *text, uint32_t length, int last)
{
    aotx_token_body body;
    memset(&body, 0, sizeof(body));
    body.slot = (uint32_t)number;
    body.token = 9000u + position;
    body.position = position;
    body.flags = AOTX_TOKEN_SAMPLED | (last ? AOTX_TOKEN_LAST : 0u);
    body.seed = 0x51eed00000000000ull + (uint64_t)number;
    body.draw = position;
    body.role = 1u;
    body.text_len = length;
    memcpy(body.text, text, length);
    d->writer = AOTX_WRITER_AGENT_BASE + (uint32_t)number;
    aotx_fake_record(d, AOTX_CLASS_A, AOTX_REC_TOKEN, &body, sizeof(body));
}

static void add_reply(aotx_fake_device *d, int number, const unsigned char *text,
                      uint32_t length)
{
    uint32_t at = 0u;
    while (at < length) {
        uint32_t span = length - at;
        if (span > sizeof(((aotx_token_body *)0)->text)) {
            span = sizeof(((aotx_token_body *)0)->text);
        }
        add_reply_token(d, number, at, text + at, span, at + span == length);
        at += span;
    }
}

static void add_summary(aotx_fake_device *d, int number)
{
    aotx_bus_body body;
    memset(&body, 0, sizeof(body));
    body.kind = AOTX_BUS_FINDING;
    body.provenance = AOTX_PROV_COMPUTED;
    body.writer_seq = 1u;
    body.text_len = (uint32_t)snprintf(body.text, AOTX_BUS_TEXT_BYTES,
                                      "summary of agent %d", number);
    d->writer = AOTX_WRITER_AGENT_BASE + (uint32_t)number;
    aotx_fake_record(d, AOTX_CLASS_B, AOTX_REC_BUS, &body, sizeof(body));
}

static void add_task_end(aotx_fake_device *d, int number, int verdict)
{
    aotx_task_body body;
    aotx_fake_task(number, AOTX_TASK_DONE, &body);
    body.verify = verdict ? AOTX_VERIFY_SIBLING : AOTX_VERIFY_NONE;
    body.agent = verdict ? (uint32_t)(number + 1) : (uint32_t)number;
    d->writer = AOTX_WRITER_AGENT_BASE + (uint32_t)number;
    aotx_fake_record(d, AOTX_CLASS_B, AOTX_REC_TASK, &body, sizeof(body));
}

static void add_group(transcript_run *r, int number)
{
    unsigned char line[280];
    unsigned char reply[260];
    aotx_manifest_body manifest;
    aotx_selection_body selection;
    aotx_commit_body commit;
    uint32_t request = (uint32_t)(5000 + number * 2);
    uint32_t refused = request + 1u;
    uint32_t line_len;
    uint32_t reply_len;
    unsigned int i;

    line_len = (uint32_t)snprintf((char *)line, sizeof(line),
                                 (number == 0) ? "say prompt of agent %d "
                                               : "task %d prompt of agent %d ",
                                 number, number);
    while (line_len < 250u) line[line_len++] = (unsigned char)('a' + number % 26);
    add_text_record(&r->device, AOTX_REC_INPUT_LINE, AOTX_WRITER_FEEDER, 0u,
                    line, AOTX_BODY_BYTES);
    add_text_record(&r->device, AOTX_REC_INPUT_LINE, AOTX_WRITER_FEEDER,
                    AOTX_FLAG_FRAGMENT, line + AOTX_BODY_BYTES,
                    line_len - AOTX_BODY_BYTES);

    reply_len = (uint32_t)snprintf((char *)reply, sizeof(reply), "reply of agent %d ", number);
    while (reply_len < 230u) reply[reply_len++] = (unsigned char)('A' + number % 26);
    add_text_record(&r->device, AOTX_REC_CONSOLE, AOTX_WRITER_CONSOLE, 0u,
                    (const unsigned char *)"echo: input accepted", 20u);
    add_reply(&r->device, number, reply, reply_len);
    add_text_record(&r->device, AOTX_REC_CONSOLE, AOTX_WRITER_CONSOLE, 0u,
                    (const unsigned char *)"import: module ready", 20u);
    add_text_record(&r->device, AOTX_REC_CONSOLE, AOTX_WRITER_CONSOLE, 0u,
                    (const unsigned char *)"spawn: worker ready", 19u);

    add_request(&r->device, number, AOTX_AUTH_PENDING, request);
    aotx_fake_manifest(number, &manifest);
    manifest.agent = (uint32_t)number;
    manifest.turn = (uint32_t)(number + 1);
    manifest.finish = AOTX_TURN_TOOL;
    manifest.tool = AOTX_TOOL_FS_READ;
    manifest.request = request;
    r->device.writer = AOTX_WRITER_AGENT_BASE + (uint32_t)number;
    aotx_fake_record(&r->device, AOTX_CLASS_B, AOTX_REC_MANIFEST,
                     &manifest, sizeof(manifest));
    add_request(&r->device, number, AOTX_AUTH_GRANTED, request);
    add_result(&r->device, number, request);

    memset(&selection, 0, sizeof(selection));
    selection.agent = (uint32_t)number;
    selection.turn = (uint32_t)(number + 1);
    selection.count = 3u;
    selection.pages = 32u + (uint32_t)number;
    selection.summary_seq = 7000u + (uint64_t)number;
    for (i = 0u; i < selection.count; i++) selection.seq[i] = 8000u + number * 4u + i;
    r->device.writer = AOTX_WRITER_AGENT_BASE + (uint32_t)number;
    aotx_fake_record(&r->device, AOTX_CLASS_A, AOTX_REC_SELECTION,
                     &selection, sizeof(selection));
    add_summary(&r->device, number);
    add_task_end(&r->device, number, 0);
    add_task_end(&r->device, number, 1);
    add_request(&r->device, number, AOTX_AUTH_REFUSED, refused);

    memset(&commit, 0, sizeof(commit));
    commit.state_hash = 0x1234000000000000ull + (uint64_t)number;
    r->device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&r->device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT,
                     &commit, sizeof(commit));
    aotx_fake_commit(&r->device, 0);
}

/* Compares the complete lines of the two new element kinds. A changed kind proves that
 * each whole-line comparison rejects a broken line. */
static void line_types(void)
{
    transcript_run run;
    aotx_manifest_body manifest;
    aotx_commit_body commit;
    char path[512];
    char got[2048];
    char changed[2048];
    const char *part;
    const char *bound;
    static const char want[] =
        "{\"tick\":1,\"kind\":\"line\",\"text\":\"hello\",\"request\":0,"
        "\"status\":\"\",\"turn\":1}\n"
        "{\"tick\":1,\"kind\":\"part\",\"text\":\"first \",\"request\":0,"
        "\"status\":\"open\",\"turn\":1}\n"
        "{\"tick\":1,\"kind\":\"part\",\"text\":\"second\",\"request\":0,"
        "\"status\":\"open\",\"turn\":1}\n"
        "{\"tick\":1,\"kind\":\"reply\",\"text\":\"first second\",\"request\":0,"
        "\"status\":\"\",\"turn\":1}\n"
        "{\"tick\":1,\"kind\":\"bound\",\"text\":\"\",\"request\":0,"
        "\"status\":\"limit\",\"turn\":1}\n";

    start_run(&run, 0x00c01200000000a1ull);
    add_text_record(&run.device, AOTX_REC_INPUT_LINE, AOTX_WRITER_FEEDER, 0u,
                    (const unsigned char *)"say hello", 9u);
    add_reply_token(&run.device, 0, 0u, (const unsigned char *)"first ", 6u, 0);
    add_reply_token(&run.device, 0, 1u, (const unsigned char *)"second", 6u, 1);
    memset(&manifest, 0, sizeof(manifest));
    manifest.agent = 0u;
    manifest.turn = 1u;
    manifest.finish = AOTX_TURN_LIMIT;
    run.device.writer = AOTX_WRITER_AGENT_BASE;
    aotx_fake_record(&run.device, AOTX_CLASS_B, AOTX_REC_MANIFEST,
                     &manifest, sizeof(manifest));
    memset(&commit, 0, sizeof(commit));
    run.device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&run.device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT,
                     &commit, sizeof(commit));
    aotx_fake_commit(&run.device, 0);
    finish_run(&run);
    snprintf(path, sizeof(path), "%s/transcript/0.jsonl", run.boot_dir);
    {
        int fd = open(path, O_RDONLY);
        ssize_t bytes = (fd >= 0) ? read(fd, got, sizeof(got) - 1u) : -1;
        CHECK(bytes >= 0, "the line type transcript does not read");
        if (fd >= 0) close(fd);
        got[(bytes >= 0) ? (size_t)bytes : 0u] = '\0';
    }
    CHECK(strcmp(got, want) == 0, "the new transcript lines differ as a whole");
    snprintf(changed, sizeof(changed), "%s", got);
    part = strstr(changed, "\"kind\":\"part\"");
    CHECK(part != NULL, "the part line is not present");
    if (part != NULL) changed[(size_t)(part - changed) + 8u] = 'x';
    CHECK(strcmp(changed, want) != 0, "the part line check accepted a changed kind");
    snprintf(changed, sizeof(changed), "%s", got);
    bound = strstr(changed, "\"kind\":\"bound\"");
    CHECK(bound != NULL, "the bound line is not present");
    if (bound != NULL) changed[(size_t)(bound - changed) + 8u] = 'x';
    CHECK(strcmp(changed, want) != 0, "the bound line check accepted a changed kind");
    printf("line types: 2 exact lines, 2 changed lines refused\n");
    aotx_remove_tree(run.dir);
}

static int run_journal(const char *dir, int agent, const char *output)
{
    char agent_text[16];
    char *argv[7];
    int fd = open(output, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    int child;
    snprintf(agent_text, sizeof(agent_text), "%d", agent);
    argv[0] = (char *)journal_program;
    argv[1] = (char *)"transcript";
    argv[2] = (char *)dir;
    argv[3] = (char *)"--agent";
    argv[4] = agent_text;
    argv[5] = NULL;
    child = aotx_spawn(argv, -1, fd);
    close(fd);
    return (child > 0) ? aotx_wait(child) : -1;
}

static void read_back(const char *path, int want, int agent)
{
    char *line = NULL;
    size_t cap = 0;
    FILE *file = fopen(path, "r");
    int count = 0;
    int replies = 0;
    CHECK(file != NULL, "the transcript %s does not open", path);
    if (file == NULL) return;
    while (getline(&line, &cap, file) > 0) {
        char kind[24];
        char status[32];
        uint64_t tick = 0;
        uint64_t turn = 0;
        uint64_t request = 0;
        CHECK(aotx_json_number(line, "\"tick\":", &tick), "line %d has no tick", count);
        CHECK(aotx_json_text(line, "\"kind\":\"", kind, sizeof(kind)),
              "line %d has no kind", count);
        CHECK(aotx_json_number(line, "\"request\":", &request),
              "line %d has no request", count);
        CHECK(aotx_json_text(line, "\"status\":\"", status, sizeof(status)),
              "line %d has no status", count);
        CHECK(aotx_json_number(line, "\"turn\":", &turn), "line %d has no turn", count);
        CHECK(strstr(line, "\"text\":") != NULL || strstr(line, "\"tool\":") != NULL,
              "line %d has no text or tool", count);
        CHECK(tick > 0u && turn > 0u, "line %d has no ordering values", count);
        CHECK(strstr(line, "echo: input accepted") == NULL
              && strstr(line, "import: module ready") == NULL
              && strstr(line, "spawn: worker ready") == NULL,
              "line %d contains console output", count);
        if (strcmp(kind, "reply") == 0) {
            char prefix[64];
            snprintf(prefix, sizeof(prefix), "reply of agent %d ", agent);
            CHECK(strstr(line, prefix) != NULL, "the reply does not contain token bytes");
            replies++;
        }
        count++;
    }
    CHECK(count == want, "the transcript has %d lines, not %d", count, want);
    CHECK(replies == 1, "the transcript has %d replies, not one", replies);
    free(line);
    fclose(file);
}

static void batch(int agents)
{
    transcript_run run;
    char path[512];
    char output[512];
    int i;
    start_run(&run, 0x00c0120000000000ull + (uint64_t)agents);
    for (i = 0; i < agents; i++) add_group(&run, i);
    finish_run(&run);
    for (i = 0; i < agents; i++) {
        snprintf(path, sizeof(path), "%s/transcript/%d.jsonl", run.boot_dir, i);
        read_back(path, 12, i);
        snprintf(output, sizeof(output), "%s/compare-%d.out", run.dir, i);
        CHECK(run_journal(run.dir, i, output) == 0,
              "the journal comparison failed for agent %d", i);
    }
    snprintf(path, sizeof(path), "%s/transcript/0.jsonl", run.boot_dir);
    {
        int fd = open(path, O_RDWR);
        unsigned char byte;
        CHECK(fd >= 0, "the transcript to damage does not open");
        CHECK(pread(fd, &byte, 1u, 20) == 1, "the transcript byte does not read");
        byte = (byte == 'x') ? 'y' : 'x';
        CHECK(pwrite(fd, &byte, 1u, 20) == 1, "the transcript byte does not change");
        close(fd);
    }
    snprintf(output, sizeof(output), "%s/compare-bad.out", run.dir);
    CHECK(run_journal(run.dir, 0, output) == 1,
          "the journal comparison accepted a changed transcript line");
    aotx_remove_tree(run.dir);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("usage: transcript_test <drain program> <journal program>\n");
        return 2;
    }
    drain_program = argv[1];
    journal_program = argv[2];
    line_types();
    batch(1);
    batch(64);
    return aotx_report("transcript_test", 3000);
}
