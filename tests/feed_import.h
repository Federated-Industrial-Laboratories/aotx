/* Purpose: Give the feeder check the two arms that publish module directories as imports.
 * Owns: Nothing; the check that includes this file holds the ring and the child process.
 * Threading: Two processes; the check reads the ring while the feeder writes it.
 * Lifetime: One case of the check.
 *
 * The file is included by tests/feed_test.c after the parts that every arm shares. */
#ifndef AOTX_TESTS_FEED_IMPORT_H
#define AOTX_TESTS_FEED_IMPORT_H

#include "disk/feed/import.h"

/* Module directories one arm makes. */
#define AOTX_FEED_MODULES 64

typedef struct import_taken {
    int settings;              /* setting records that came */
    int lines;                 /* input line records that came */
    int heads;                 /* import heads that came */
    int parts;                 /* import parts that came */
    uint64_t last_setting_seq; /* the sequence of the last setting record */
    uint64_t first_head_seq;   /* the sequence of the first import head */
    uint64_t last_import_seq;  /* the sequence of the last import record */
    uint64_t first_line_seq;   /* the sequence of the first input line record */
    char text[AOTX_LINES_MAX][AOTX_BODY_BYTES + 1];
    char name[AOTX_FEED_MODULES][AOTX_IMPORT_NAME_BYTES];
    uint32_t number[AOTX_FEED_MODULES];
    uint32_t kind[AOTX_FEED_MODULES];
    uint32_t bytes[AOTX_FEED_MODULES]; /* the body bytes that the head states */
} import_taken;

/* Consumes the slots that the feeder published and keeps the imports in order. */
static void take_imports(aotx_inbound_ring *ring, import_taken *t)
{
    uint64_t consumed = aotx_inbound_consumed(ring);
    uint64_t head = aotx_inbound_head(ring);
    while (consumed < head) {
        const unsigned char *slot = ring->slots + (consumed & ring->mask) * AOTX_SLOT_BYTES;
        const aotx_record_header *h = (const aotx_record_header *)slot;
        CHECK(aotx_record_valid(h) == 1, "a slot does not validate");
        if (h->type == AOTX_REC_IMPORT) {
            aotx_import_head body;
            CHECK(h->cls == AOTX_CLASS_A, "an import record is not authoritative");
            CHECK(h->writer == AOTX_WRITER_FEEDER, "an import record holds the writer %u",
                  h->writer);
            memcpy(&body, aotx_record_body(h), sizeof(body));
            t->last_import_seq = h->seq;
            if (body.part != 0u) {
                t->parts++;
            } else {
                if (t->heads == 0) {
                    t->first_head_seq = h->seq;
                }
                if (t->heads < AOTX_FEED_MODULES) {
                    snprintf(t->name[t->heads], AOTX_IMPORT_NAME_BYTES, "%s", body.name);
                    t->number[t->heads] = body.import;
                    t->kind[t->heads] = body.kind;
                    t->bytes[t->heads] = body.file_bytes[1];
                }
                t->heads++;
            }
        } else if (h->type == AOTX_REC_SETTING) {
            t->last_setting_seq = h->seq;
            t->settings++;
        } else if (h->type == AOTX_REC_INPUT_LINE) {
            if (t->lines == 0) {
                t->first_line_seq = h->seq;
            }
            if (t->lines < AOTX_LINES_MAX) {
                memcpy(t->text[t->lines], aotx_record_body(h), h->body_len);
                t->text[t->lines][h->body_len] = '\0';
            }
            t->lines++;
        }
        consumed++;
        aotx_store_release(&ring->pre->consumed, consumed);
    }
}

/* Makes one directory and refuses a failure that is not a directory that is already
 * there. */
static void module_dir(const char *path)
{
    CHECK(mkdir(path, 0755) == 0 || errno == EEXIST, "the directory %s does not open", path);
}

/* Writes one file of a module directory. */
static void module_file(const char *path, const void *bytes, size_t len)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    CHECK(fd >= 0, "the file %s does not open", path);
    if (fd >= 0) {
        CHECK(write(fd, bytes, len) == (ssize_t)len, "the file %s does not write", path);
        close(fd);
    }
}

/* Writes one text file of a module directory. */
static void module_text(const char *path, const char *text)
{
    module_file(path, text, strlen(text));
}

/* Gives the body byte of one module at one place. Each module gives another byte run. */
static unsigned char module_byte(int module, uint32_t at)
{
    return (unsigned char)(0x20u + ((uint32_t)module * 11u + at * 5u) % 90u);
}

/* Builds n skill directories under the root, with distinct names and distinct bodies.
 * Gives back the count of records that the imports of all of them must publish. */
static int build_modules(const char *root, int n)
{
    unsigned char body[2048];
    char path[512];
    char file[768];
    char manifest[256];
    char name[32];
    int records = 0;
    int i;
    module_dir(root);
    for (i = 0; i < n; i++) {
        uint32_t len = 100u + (uint32_t)i * 11u;
        uint32_t manifest_len;
        uint32_t j;
        snprintf(name, sizeof(name), "m%02d", i);
        snprintf(path, sizeof(path), "%s/%s", root, name);
        module_dir(path);
        manifest_len = (uint32_t)snprintf(manifest, sizeof(manifest),
                                          "kind: skill\nname: %s\nbody: skill.txt\n", name);
        snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
        module_file(file, manifest, manifest_len);
        for (j = 0; j < len; j++) {
            body[j] = module_byte(i, j);
        }
        snprintf(file, sizeof(file), "%s/skill.txt", path);
        module_file(file, body, len);
        records += 1 + (int)((manifest_len + AOTX_IMPORT_TEXT_BYTES - 1u) /
                             AOTX_IMPORT_TEXT_BYTES) +
                   (int)((len + AOTX_IMPORT_TEXT_BYTES - 1u) / AOTX_IMPORT_TEXT_BYTES);
    }
    return records;
}

/* Checks that the heads name the modules in name order, with contiguous numbers. */
static void check_order(const import_taken *t, int n)
{
    char name[32];
    int i;
    for (i = 0; i < n && i < t->heads && i < AOTX_FEED_MODULES; i++) {
        snprintf(name, sizeof(name), "m%02d", i);
        CHECK(strcmp(t->name[i], name) == 0, "head %d names %s and %s was asked for", i,
              t->name[i], name);
        CHECK(t->number[i] == (uint32_t)(i + 1), "head %d holds the number %u", i,
              t->number[i]);
        CHECK(t->kind[i] == AOTX_MODULE_SKILL, "head %d gives the kind %u", i, t->kind[i]);
        CHECK(t->bytes[i] == 100u + (uint32_t)i * 11u, "head %d gives %u body bytes", i,
              t->bytes[i]);
    }
}

/* Reads the report file of the child into the buffer. */
static void read_report(const char *path, char *out, size_t bytes)
{
    int fd = open(path, O_RDONLY);
    ssize_t got = 0;
    out[0] = '\0';
    CHECK(fd >= 0, "the report file does not read");
    if (fd >= 0) {
        got = read(fd, out, bytes - 1);
        close(fd);
    }
    out[(got > 0) ? (size_t)got : 0] = '\0';
}

/* Runs the feeder with a settings file and a directory of module directories. The setting
 * records must come first, the imports after them, and the first line of the standard
 * input after every import. */
static void modules_arm(int n)
{
    aotx_map map;
    aotx_inbound_ring ring;
    import_taken got;
    char dir[256];
    char root[320];
    char settings_path[400];
    char err_path[400];
    char report[8192];
    char line[128];
    char fd_text[16];
    char *args[8];
    unsigned int k;
    unsigned int key = 0;
    int want_records;
    int pipe_fds[2];
    int err_fd;
    int saved;
    int child;
    uint64_t deadline;

    memset(&got, 0, sizeof(got));
    /* The settings file names one key that the device applies, so the arm proves the order
     * of two record types and not only of one. */
    for (k = 0; k < AOTX_SETTING_NUMBER_COUNT; k++) {
        if (aotx_settings_number_side(k) == AOTX_SETTING_SIDE_DEVICE) {
            key = k;
            break;
        }
    }
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(root, sizeof(root), "%s/modules", dir);
    want_records = build_modules(root, n);
    snprintf(settings_path, sizeof(settings_path), "%s/aotx.settings", dir);
    snprintf(err_path, sizeof(err_path), "%s/report.txt", dir);
    {
        char value[32];
        aotx_settings_format(aotx_settings_number_least(key),
                             aotx_settings_number_scale(key), value, sizeof(value));
        snprintf(line, sizeof(line), "%s = %s\n", aotx_settings_number_name(key), value);
        module_file(settings_path, line, strlen(line));
    }

    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, &map, &ring) == 0, "the ring does not open");
    CHECK(pipe(pipe_fds) == 0, "the line pipe does not open");
    /* The line waits in the pipe before the feeder starts. A feeder that reads the
     * standard input before it imports the modules thus cannot pass this case. */
    CHECK(write(pipe_fds[1], "the first operator line\n", 24) == 24,
          "the operator line does not write");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = (char *)"--settings";
    args[4] = settings_path;
    args[5] = (char *)"--modules";
    args[6] = root;
    args[7] = NULL;
    err_fd = open(err_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    saved = dup(2);
    CHECK(err_fd >= 0 && saved >= 0, "the report file does not open");
    fflush(stderr);
    dup2(err_fd, 2);
    child = aotx_spawn(args, pipe_fds[0], -1);
    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    close(err_fd);
    CHECK(child > 0, "the feeder does not start");
    close(pipe_fds[0]);
    close(pipe_fds[1]);

    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while ((got.settings < 1 || got.heads < n || got.lines < 1) &&
           aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        take_imports(&ring, &got);
        aotx_pause(&backoff);
    }
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(child) == 0, "the feeder does not end with a clean status");
    take_imports(&ring, &got);

    CHECK(got.heads == n, "the walk gave %d imports and %d directories were written",
          got.heads, n);
    CHECK(got.heads + got.parts == want_records, "the imports gave %d records and %d were"
          " asked for", got.heads + got.parts, want_records);
    CHECK(got.settings >= 1, "the run gave %d setting records", got.settings);
    CHECK(got.lines >= 1, "the run gave %d input line records", got.lines);
    CHECK(got.last_setting_seq < got.first_head_seq,
          "an import came before the last setting record");
    CHECK(got.last_import_seq < got.first_line_seq,
          "the first line of the standard input came before the last import record");
    check_order(&got, n);
    read_report(err_path, report, sizeof(report));
    CHECK(strstr(report, "import: ") == NULL, "the feeder refused a module: %s", report);
    printf("modules %d: imports %d, records %d, settings %d, lines %d\n", n, got.heads,
           got.heads + got.parts, got.settings, got.lines);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

/* Runs the feeder over the standard input only. A line that asks for an import makes the
 * import and no input line record. Every other line reaches the device as before. */
static void import_line_arm(int n)
{
    aotx_map map;
    aotx_inbound_ring ring;
    import_taken got;
    char dir[256];
    char root[320];
    char err_path[400];
    char report[8192];
    char line[512];
    char fd_text[16];
    char *args[4];
    int want_records;
    int pipe_fds[2];
    int err_fd;
    int saved;
    int child;
    int i;
    uint64_t deadline;

    memset(&got, 0, sizeof(got));
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(root, sizeof(root), "%s/modules", dir);
    want_records = build_modules(root, n);
    snprintf(err_path, sizeof(err_path), "%s/report.txt", dir);

    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, &map, &ring) == 0, "the ring does not open");
    CHECK(pipe(pipe_fds) == 0, "the line pipe does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = NULL;
    err_fd = open(err_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    saved = dup(2);
    CHECK(err_fd >= 0 && saved >= 0, "the report file does not open");
    fflush(stderr);
    dup2(err_fd, 2);
    child = aotx_spawn(args, pipe_fds[0], -1);
    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    close(err_fd);
    CHECK(child > 0, "the feeder does not start");
    close(pipe_fds[0]);

    for (i = 0; i < n; i++) {
        int used = snprintf(line, sizeof(line), "import %s/m%02d\n", root, i);
        CHECK(write(pipe_fds[1], line, (size_t)used) == used, "the import line does not write");
    }
    /* A path that names no directory gives one line of the standard error and one line
     * that states the refusal to the operator. */
    {
        int used = snprintf(line, sizeof(line), "import %s/absent\n", root);
        CHECK(write(pipe_fds[1], line, (size_t)used) == used, "the line does not write");
    }
    /* A line that does not ask for an import reaches the device as before. */
    CHECK(write(pipe_fds[1], "spawn worker\n", 13) == 13, "the line does not write");

    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while ((got.heads < n || got.lines < 2) && aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        take_imports(&ring, &got);
        aotx_pause(&backoff);
    }
    close(pipe_fds[1]);
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(child) == 0, "the feeder does not end with a clean status");
    take_imports(&ring, &got);

    CHECK(got.heads == n, "the lines gave %d imports and %d were asked for", got.heads, n);
    CHECK(got.heads + got.parts == want_records, "the imports gave %d records and %d were"
          " asked for", got.heads + got.parts, want_records);
    /* The refused path gives one line that states the refusal, and the line that asks for
     * no import gives one of its own. */
    CHECK(got.lines == 2, "the run gave %d input line records and two were asked for",
          got.lines);
    snprintf(line, sizeof(line), "import %s/absent refused: the file is not there", root);
    CHECK(strcmp(got.text[0], line) == 0, "the refusal line holds [%s] and [%s] was asked"
          " for", got.text[0], line);
    CHECK(strcmp(got.text[1], "spawn worker") == 0, "the input line holds [%s]", got.text[1]);
    check_order(&got, n);
    read_report(err_path, report, sizeof(report));
    snprintf(line, sizeof(line), "import: %s/absent: ", root);
    CHECK(strstr(report, line) != NULL, "the refused path gives no line of the standard"
          " error: %s", report);
    CHECK(strstr(report, "the file is not there") != NULL,
          "the refused path states no reason: %s", report);
    printf("import lines %d: imports %d, records %d, input lines %d\n", n, got.heads,
           got.heads + got.parts, got.lines);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}


/* Counts the times one text is in another. */
static int import_count(const char *hay, const char *needle)
{
    size_t len = strlen(needle);
    const char *at = hay;
    int count = 0;
    while ((at = strstr(at, needle)) != NULL) {
        count++;
        at += len;
    }
    return count;
}

/* Reads a whole file into the buffer. Returns the byte count, or -1. */
static int import_slurp(const char *path, char *out, size_t bytes)
{
    int fd = open(path, O_RDONLY);
    ssize_t got;
    if (fd < 0) {
        return -1;
    }
    got = read(fd, out, bytes - 1);
    close(fd);
    out[(got > 0) ? (size_t)got : 0] = '\0';
    return (int)got;
}

/* ---- an import that a surface the feeder does not read asks for ---- */

/* Fills one request that names the import tool. No agent made it: the console writes it
 * for a line that the operator typed in a surface the feeder does not read. */
static void import_request(aotx_tool_request_body *r, uint32_t request, const char *path)
{
    memset(r, 0, sizeof(*r));
    r->agent = AOTX_REQUEST_NO_AGENT;
    r->turn = 0u;
    r->tool = AOTX_TOOL_IMPORT;
    r->request = request;
    r->deadline = 0u;
    r->auth = AOTX_AUTH_NONE;
    r->arg_len = (uint32_t)snprintf(r->arg, AOTX_TOOL_ARG_BYTES, "%s", path);
}

/* The real drain writes the requests line and the real feeder reads it. A drift between
 * the name the drain gives the tool and the name the feeder matches thus shows in a check.
 * A request that names the import tool reads the module directory and publishes no reply.
 * A request that names a directory which is not there gives the line of a refused import.
 * One identity that comes twice imports one time. */
static void import_request_arm(int n)
{
    aotx_map map;
    aotx_host_ring ring;
    aotx_fake_device device;
    aotx_map imap;
    aotx_inbound_ring iring;
    import_taken got;
    aotx_tool_request_body r;
    char dir[256];
    char root[320];
    char requests[400];
    char path[512];
    char want[512];
    char report[16384];
    char fd_text[16];
    char *args[8];
    uint64_t boot_id = 0x00100d0000000001ull + (uint64_t)n;
    uint64_t deadline;
    int feeder;
    int drain;
    int i;

    memset(&got, 0, sizeof(got));
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(root, sizeof(root), "%.300s/modules", dir);
    build_modules(root, n);
    snprintf(requests, sizeof(requests), "%.300s/requests.jsonl", dir);

    /* The feeder starts before the drain makes the requests file, so it reads the file
     * from its first byte. */
    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, &imap, &iring) == 0,
          "the inbound ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", imap.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = (char *)"--root";
    args[4] = root;
    args[5] = (char *)"--requests";
    args[6] = requests;
    args[7] = NULL;
    feeder = aotx_spawn(args, -1, -1);
    CHECK(feeder > 0, "the feeder does not start");
    wait_for_start(&iring);

    CHECK(aotx_host_ring_create(262144u, boot_id, &map, &ring) == 0,
          "the host ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[2];
    args[1] = (char *)"--ring-fd";
    args[2] = fd_text;
    args[3] = (char *)"--journal";
    args[4] = dir;
    args[5] = NULL;
    drain = aotx_spawn(args, -1, -1);
    CHECK(drain > 0, "the drain does not start");

    aotx_fake_start(&device, &ring, boot_id);
    device.writer = AOTX_WRITER_CONSOLE;
    for (i = 0; i < n; i++) {
        snprintf(path, sizeof(path), "%.400s/m%02d", root, i);
        import_request(&r, (uint32_t)(i + 1), path);
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
        if ((i + 1) % 32 == 0) {
            aotx_fake_commit(&device, 0);
        }
    }
    /* One identity that the feeder already took must import no second time. */
    snprintf(path, sizeof(path), "%.400s/m00", root);
    import_request(&r, 1u, path);
    aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
    /* One path that names no directory gives the line of a refused import. */
    snprintf(path, sizeof(path), "%.400s/absent", root);
    import_request(&r, (uint32_t)(n + 1), path);
    aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
    aotx_fake_commit(&device, 0);

    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while ((got.heads < n || got.lines < 1) && aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        take_imports(&iring, &got);
        aotx_pause(&backoff);
    }
    take_imports(&iring, &got);
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(drain) == 0, "the drain does not end with a clean status");
    aotx_store_release16(&iring.pre->closed, 1);
    CHECK(aotx_wait(feeder) == 0, "the feeder does not end with a clean status");
    take_imports(&iring, &got);

    CHECK(got.heads == n, "the requests gave %d imports and %d were asked for", got.heads, n);
    CHECK(got.lines == 1, "the run gave %d refusal lines and one was asked for", got.lines);
    snprintf(want, sizeof(want), "import %.400s/absent refused: the file is not there", root);
    CHECK(strcmp(got.text[0], want) == 0, "the refusal line holds [%s] and [%s] was asked"
          " for", got.text[0], want);
    check_order(&got, n);
    /* The drain names the tool in the line, and the feeder matched that name. */
    CHECK(import_slurp(requests, report, sizeof(report)) > 0,
          "the requests file does not read");
    CHECK(import_count(report, "\"tool\":\"import\"") == n + 2,
          "the requests file names the import tool %d times and %d were written",
          import_count(report, "\"tool\":\"import\""), n + 2);
    printf("import requests %d: imports %d, refusal lines %d\n", n, got.heads, got.lines);
    aotx_map_release(&map);
    aotx_map_release(&imap);
    aotx_remove_tree(dir);
}

/* ---- the feeder, the drain and the journal reader over one import ---- */

/* Bytes of the message file that this arm reads back. */
#define AOTX_IMPORT_TEXT_MAX 262144

static char import_file[AOTX_IMPORT_TEXT_MAX];

/* Writes the path of the message file that the drain wrote today. */
static void import_bus_path(const char *dir, char *out, size_t bytes)
{
    char day[16];
    time_t now = (time_t)(aotx_wall_ns() / 1000000000u);
    struct tm parts;
    localtime_r(&now, &parts);
    strftime(day, sizeof(day), "%Y-%m-%d", &parts);
    snprintf(out, bytes, "%s/bus/%s-aotx.jsonl", dir, day);
}

/* Runs the line validator over the message file, when the check was given one. */
static void import_validate(const char *path)
{
    char *args[3];
    int child;
    if (lint_program == NULL) {
        printf("no validator was given, so the message file check is not applied\n");
        return;
    }
    args[0] = (char *)lint_program;
    args[1] = (char *)path;
    args[2] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the validator does not start");
    CHECK(aotx_wait(child) == 0, "the validator refuses the message file");
}

/* Closes one tick of the block under construction. The last record of a tick is the tick
 * commit, and a journal with no complete tick holds nothing that a reader can take. */
static void close_tick(aotx_fake_device *device)
{
    aotx_commit_body commit;
    uint32_t writer = device->writer;
    memset(&commit, 0, sizeof(commit));
    device->writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
    aotx_fake_commit(device, 0);
    device->writer = writer;
}

/* Takes the records the feeder published. Each one goes into a block of the host ring,
 * the way the device writes the records that it applied. */
static void mirror(aotx_inbound_ring *iring, aotx_fake_device *device, import_taken *t)
{
    uint64_t consumed = aotx_inbound_consumed(iring);
    uint64_t head = aotx_inbound_head(iring);
    while (consumed < head) {
        const unsigned char *slot = iring->slots + (consumed & iring->mask) * AOTX_SLOT_BYTES;
        const aotx_record_header *h = (const aotx_record_header *)slot;
        if (h->type == AOTX_REC_IMPORT) {
            aotx_import_head body;
            memcpy(&body, aotx_record_body(h), sizeof(body));
            if (body.part == 0u) {
                if (t->heads < AOTX_FEED_MODULES) {
                    snprintf(t->name[t->heads], AOTX_IMPORT_NAME_BYTES, "%s", body.name);
                    t->number[t->heads] = body.import;
                }
                t->heads++;
            } else {
                t->parts++;
            }
            device->writer = h->writer;
            aotx_fake_record(device, h->cls, h->type, aotx_record_body(h), h->body_len);
            if (device->count >= AOTX_FAKE_RECORDS - 2u) {
                close_tick(device);
            }
        }
        consumed++;
        aotx_store_release(&iring->pre->consumed, consumed);
    }
}

/* Runs the journal reader over the journal and gives back the count of its lines. */
static int import_reader(const char *dir, const char *out_path)
{
    char *args[4];
    int out_fd = open(out_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    int child;
    CHECK(out_fd >= 0, "the output file does not open");
    args[0] = (char *)reader_program;
    args[1] = (char *)"modules";
    args[2] = (char *)dir;
    args[3] = NULL;
    child = aotx_spawn(args, -1, out_fd);
    CHECK(child > 0, "the journal reader does not start");
    CHECK(aotx_wait(child) == 0, "the journal reader refuses the journal");
    close(out_fd);
    return import_slurp(out_path, import_file, sizeof(import_file));
}

/* The real feeder writes the imports and the real drain reads them. A drift between the
 * writer of the record and its reader then shows in a check and not in a run. */
static void import_loop(int n)
{
    aotx_map map;
    aotx_host_ring ring;
    aotx_fake_device device;
    aotx_map imap;
    aotx_inbound_ring iring;
    import_taken got;
    char dir[256];
    char root[320];
    char path[512];
    char want[512];
    char fd_text[16];
    char *args[6];
    uint64_t boot_id = 0x00100c0000000001ull + (uint64_t)n;
    uint64_t deadline;
    int feeder;
    int drain;
    int i;

    memset(&got, 0, sizeof(got));
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(root, sizeof(root), "%s/modules", dir);
    build_modules(root, n);

    CHECK(aotx_inbound_create(256u, &imap, &iring) == 0, "the inbound ring does not open");
    CHECK(aotx_host_ring_create(262144u, boot_id, &map, &ring) == 0,
          "the host ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[2];
    args[1] = (char *)"--ring-fd";
    args[2] = fd_text;
    args[3] = (char *)"--journal";
    args[4] = dir;
    args[5] = NULL;
    drain = aotx_spawn(args, -1, -1);
    CHECK(drain > 0, "the drain does not start");
    aotx_fake_start(&device, &ring, boot_id);

    snprintf(fd_text, sizeof(fd_text), "%d", imap.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = (char *)"--modules";
    args[4] = root;
    args[5] = NULL;
    feeder = aotx_spawn(args, -1, -1);
    CHECK(feeder > 0, "the feeder does not start");

    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while (got.heads < n && aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        mirror(&iring, &device, &got);
        aotx_pause(&backoff);
    }
    mirror(&iring, &device, &got);
    close_tick(&device);
    aotx_store_release16(&iring.pre->closed, 1);
    CHECK(aotx_wait(feeder) == 0, "the feeder does not end with a clean status");
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(drain) == 0, "the drain does not end with a clean status");

    CHECK(got.heads == n, "the feeder gave %d imports and %d were asked for", got.heads, n);
    import_bus_path(dir, path, sizeof(path));
    CHECK(import_slurp(path, import_file, sizeof(import_file)) > 0,
          "the message file does not read");
    CHECK(import_count(import_file, "\n") == n, "the message file holds %d lines and %d"
          " were asked for", import_count(import_file, "\n"), n);
    for (i = 0; i < n; i++) {
        snprintf(want, sizeof(want), "\"text\":\"module m%02d skill import %d from ", i, i + 1);
        CHECK(strstr(import_file, want) != NULL, "the note of module %d is not in the"
              " message file: %s", i, want);
    }
    CHECK(import_count(import_file, "\"agent\":\"feeder\"") == n,
          "the message file holds %d feeder lines",
          import_count(import_file, "\"agent\":\"feeder\""));
    import_validate(path);

    /* The journal reader takes the same records back from the segments. */
    snprintf(path, sizeof(path), "%s/modules.txt", dir);
    CHECK(import_reader(dir, path) > 0, "the journal reader wrote no line");
    CHECK(import_count(import_file, "\n") == n, "the reader printed %d lines and %d were"
          " asked for", import_count(import_file, "\n"), n);
    for (i = 0; i < n; i++) {
        snprintf(want, sizeof(want), "import=%d kind=skill name=m%02d files=2", i + 1, i);
        CHECK(strstr(import_file, want) != NULL, "the reader line of module %d is not there:"
              " %s", i, want);
    }
    printf("import loop %d: imports %d, parts %d\n", n, got.heads, got.parts);
    aotx_map_release(&map);
    aotx_map_release(&imap);
    aotx_remove_tree(dir);
}

#endif
