/* Purpose: Check the text that the journal reader prints for the token records of a run.
 * Owns: One temporary journal for each case.
 * Threading: Two processes; the test reads the file that the reader writes.
 * Lifetime: The run of the program. */
#include "disk/settings/settings.h"
#include "tests/disk_fake.h"

#include <fcntl.h>

#define AOTX_RING_BYTES 262144u

#define AOTX_LINES_MAX 4096
#define AOTX_TEXT_MAX  262144

static char **arguments;
static char text[AOTX_TEXT_MAX];
static char *lines[AOTX_LINES_MAX];

/* Marks the record that went in last as one that a restore applied again. */
static void mark_replayed(aotx_fake_device *d)
{
    aotx_record_header *h = (aotx_record_header *)(d->stage + AOTX_BLOCK_HEADER_BYTES +
                                                   (size_t)(d->count - 1) * AOTX_SLOT_BYTES);
    h->flags = (uint16_t)(h->flags | AOTX_FLAG_REPLAYED);
}

/* Writes one boot directory with n ticks. Each tick holds one token record. The last tick
 * adds a token record whose body is too short, which the reader must not print. */
static void build_journal(const char *dir, uint64_t boot_id, int n, int replayed)
{
    aotx_fake_device device;
    aotx_segment_writer writer;
    char boot_dir[320];
    char body[64];
    int i;
    snprintf(boot_dir, sizeof(boot_dir), "%s/%016llx", dir, (unsigned long long)boot_id);
    CHECK(aotx_make_dir(boot_dir) == 0, "the boot directory does not open");
    CHECK(aotx_segment_open(&writer, boot_dir, 4096) == 0, "the segment does not open");
    aotx_fake_start(&device, NULL, boot_id);
    for (i = 0; i < n; i++) {
        aotx_boot_body boot;
        aotx_clock_body clock;
        aotx_commit_body commit;
        aotx_token_body token;
        const aotx_block_header *h;
        if (i == 0) {
            memset(&boot, 0, sizeof(boot));
            boot.boot_id = boot_id;
            boot.wall_ns = 1700000000000000000ull + boot_id;
            aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_BOOT, &boot, sizeof(boot));
        }
        clock.wall_ns = 1700000000000000000ull + (uint64_t)i;
        aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TICK_START, &clock, sizeof(clock));
        snprintf(body, sizeof(body), "line %d", i);
        aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_INPUT_LINE, body, (uint32_t)strlen(body));
        aotx_fake_token(i, &token);
        aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TOKEN, &token, sizeof(token));
        if (replayed) {
            mark_replayed(&device);
        }
        {
            /* One setting for each tick, from the feeder and from the console in turn, so
             * a line that states the wrong writer cannot pass. */
            aotx_setting_body setting;
            aotx_fake_setting(i, &setting);
            device.writer = ((i % 2) != 0) ? AOTX_WRITER_CONSOLE : AOTX_WRITER_FEEDER;
            aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_SETTING, &setting,
                             sizeof(setting));
            if (replayed) {
                mark_replayed(&device);
            }
            device.writer = AOTX_WRITER_SYSTEM;
        }
        if (i == n - 1) {
            /* A body that is shorter than the layout holds no token, so the reader counts
             * it and prints nothing for it. */
            aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TOKEN, &token, 8u);
        }
        memset(&commit, 0, sizeof(commit));
        commit.state_hash = 0x00bb000000000000ull + (uint64_t)i + 1;
        aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
        aotx_fake_commit(&device, 0);
        h = (const aotx_block_header *)device.stage;
        CHECK(aotx_segment_put(&writer, device.stage, h->byte_len) == 0, "a frame does not write");
    }
    CHECK(aotx_segment_close(&writer) == 0, "the segment does not close");
}

/* Runs the reader and returns its exit status. The lines go to the file that out_path
 * names, and the reader writes its own report to the standard error. */
static int run_reader(const char *dir, const char *boot, const char *out_path)
{
    char *args[6];
    int out_fd = open(out_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    int child;
    CHECK(out_fd >= 0, "the output file does not open");
    args[0] = arguments[1];
    args[1] = (char *)"tokens";
    args[2] = (char *)dir;
    if (boot != NULL) {
        args[3] = (char *)"--boot";
        args[4] = (char *)boot;
        args[5] = NULL;
    } else {
        args[3] = NULL;
    }
    child = aotx_spawn(args, -1, out_fd);
    CHECK(child > 0, "the reader does not start");
    close(out_fd);
    return aotx_wait(child);
}

/* Reads the file and gives the count of lines. Each line loses its end byte. */
static int split(const char *path)
{
    int fd = open(path, O_RDONLY);
    ssize_t got;
    int count = 0;
    char *at;
    CHECK(fd >= 0, "the output file does not read");
    if (fd < 0) {
        return 0;
    }
    got = read(fd, text, sizeof(text) - 1);
    close(fd);
    CHECK(got >= 0, "the output file gives no bytes");
    if (got < 0) {
        return 0;
    }
    text[got] = '\0';
    at = text;
    while (*at != '\0' && count < AOTX_LINES_MAX) {
        char *end = strchr(at, '\n');
        lines[count++] = at;
        if (end == NULL) {
            break;
        }
        *end = '\0';
        at = end + 1;
    }
    return count;
}

/* Gives the count of lines of a text. */
static int count_lines(const char *at)
{
    int count = 0;
    while (*at != '\0') {
        if (*at == '\n') {
            count++;
        }
        at++;
    }
    return count;
}

/* Checks that each line names the token that the number gives, in order. */
static void compare(int count, int n, int replayed)
{
    int i;
    CHECK(count == n, "the reader printed %d lines and %d tokens are in the journal", count, n);
    for (i = 0; i < count && i < n; i++) {
        aotx_token_body token;
        char want[256];
        char tail[32];
        aotx_fake_token(i, &token);
        snprintf(want, sizeof(want),
                 "slot=%u position=%u token=%u flags=0x%04x seed=%016llx draw=%llu role=%u",
                 token.slot, token.position, token.token, token.flags,
                 (unsigned long long)token.seed, (unsigned long long)token.draw, token.role);
        CHECK(strncmp(lines[i], want, strlen(want)) == 0,
              "line %d is [%s] and [%s] was asked for", i, lines[i], want);
        snprintf(tail, sizeof(tail), " sampled=%d replayed=%d",
                 ((token.flags & AOTX_TOKEN_SAMPLED) != 0) ? 1 : 0, replayed);
        CHECK(strstr(lines[i], tail) != NULL, "line %d does not state both flags", i);
    }
}

static void batch(int n)
{
    char dir[256];
    char boot_dir[320];
    char out_path[1024];
    uint64_t boot_id = 0x00000000face0001ull;
    int count;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    build_journal(dir, boot_id, n, 0);
    snprintf(boot_dir, sizeof(boot_dir), "%s/%016llx", dir, (unsigned long long)boot_id);
    snprintf(out_path, sizeof(out_path), "%s/tokens.txt", dir);

    /* A boot directory is read as it stands. */
    CHECK(run_reader(boot_dir, NULL, out_path) == 0, "the reader does not end with a clean status");
    count = split(out_path);
    compare(count, n, 0);

    /* A journal directory with no boot named gives the newest boot. */
    CHECK(run_reader(dir, NULL, out_path) == 0, "the reader refuses a journal directory");
    count = split(out_path);
    compare(count, n, 0);

    /* A journal directory with a boot named gives that boot. */
    {
        char name[32];
        snprintf(name, sizeof(name), "%016llx", (unsigned long long)boot_id);
        CHECK(run_reader(dir, name, out_path) == 0, "the reader refuses a named boot");
        count = split(out_path);
        compare(count, n, 0);
    }
    printf("batch %d: token lines %d\n", n, count);
    aotx_remove_tree(dir);
}

/* Checks the choice between two boots, the replayed field, and the refusals. */
static void choices(void)
{
    char dir[256];
    char out_path[1024];
    char name[32];
    int count;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(out_path, sizeof(out_path), "%s/tokens.txt", dir);

    /* A directory with no boot in it holds no journal. */
    CHECK(run_reader(dir, NULL, out_path) == AOTX_EXIT_NOJOURNAL,
          "an empty directory must give the no journal status");
    CHECK(run_reader(dir, "0000000000000009", out_path) == AOTX_EXIT_NOJOURNAL,
          "a boot that is not there must give the no journal status");

    build_journal(dir, 0x00000000face0001ull, 3, 0);
    build_journal(dir, 0x00000000face0002ull, 5, 1);

    /* The boot record wall clock of the second journal is the later one. */
    CHECK(run_reader(dir, NULL, out_path) == 0, "the reader does not end with a clean status");
    count = split(out_path);
    compare(count, 5, 1);

    snprintf(name, sizeof(name), "%016llx", 0x00000000face0001ull);
    CHECK(run_reader(dir, name, out_path) == 0, "the reader refuses the first boot");
    count = split(out_path);
    compare(count, 3, 0);
    printf("choices: newest boot 5 lines, named boot %d lines\n", count);
    aotx_remove_tree(dir);
}

/* A command that the reader does not know must give a fault status and no line. */
static void refusals(void)
{
    char dir[256];
    char out_path[1024];
    char *args[4];
    int child;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(out_path, sizeof(out_path), "%s/tokens.txt", dir);
    args[0] = arguments[1];
    args[1] = (char *)"records";
    args[2] = dir;
    args[3] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the reader does not start");
    CHECK(aotx_wait(child) == AOTX_EXIT_FAULT, "an unknown command must give the fault status");
    args[1] = (char *)"tokens";
    args[2] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the reader does not start");
    CHECK(aotx_wait(child) == AOTX_EXIT_FAULT, "a command with no directory must be refused");
    aotx_remove_tree(dir);
}

/* ---- the chain of turns and the requests file ---- */

/* Runs the reader with the standard error of the test sent to a file. A case can then
 * read the report that the reader writes there. Returns the exit status. */
static int run_report(const char *command, const char *dir, const char *boot,
                      const char *out_path, const char *err_path)
{
    char *args[6];
    int out_fd = open(out_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    int err_fd = open(err_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    int saved = dup(2);
    int child;
    int status;
    CHECK(out_fd >= 0 && err_fd >= 0 && saved >= 0, "the report files do not open");
    args[0] = arguments[1];
    args[1] = (char *)command;
    args[2] = (char *)dir;
    if (boot != NULL) {
        args[3] = (char *)"--boot";
        args[4] = (char *)boot;
        args[5] = NULL;
    } else {
        args[3] = NULL;
    }
    fflush(stderr);
    dup2(err_fd, 2);
    child = aotx_spawn(args, -1, out_fd);
    CHECK(child > 0, "the reader does not start");
    status = aotx_wait(child);
    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    close(out_fd);
    close(err_fd);
    return status;
}

/* Reads a whole file into the text buffer. Returns the byte count, or -1. */
static int read_all(const char *path, char *out, size_t bytes)
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

/* Puts one byte into a file and gives back the byte that was there, so a case can put the
 * file back as it was. */
static unsigned char swap_byte(const char *path, long at, unsigned char to)
{
    unsigned char was = 0;
    int fd = open(path, O_RDWR);
    CHECK(fd >= 0, "the file does not open for a change");
    if (fd < 0) {
        return 0;
    }
    CHECK(pread(fd, &was, 1, at) == 1, "the byte does not read");
    if (was == to) {
        to = (unsigned char)((to == '0') ? '1' : '0');
    }
    CHECK(pwrite(fd, &to, 1, at) == 1, "the byte does not write");
    close(fd);
    return was;
}

/* Runs the drain over a ring that carries n turns and n requests. The chain file and the
 * requests file therefore come from the program that writes them in a run. */
static void build_derived(const char *dir, uint64_t boot_id, int n)
{
    aotx_map map;
    aotx_host_ring ring;
    aotx_fake_device device;
    char fd_text[16];
    char *args[6];
    int child;
    int i;
    CHECK(aotx_host_ring_create(AOTX_RING_BYTES, boot_id, &map, &ring) == 0,
          "the ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[2];
    args[1] = (char *)"--ring-fd";
    args[2] = fd_text;
    args[3] = (char *)"--journal";
    args[4] = (char *)dir;
    args[5] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the drain does not start");
    aotx_fake_start(&device, &ring, boot_id);
    for (i = 0; i < n; i++) {
        aotx_manifest_body turn;
        aotx_tool_request_body request;
        device.writer = AOTX_WRITER_AGENT_BASE + (uint32_t)(i % 3);
        aotx_fake_manifest(i, &turn);
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_MANIFEST, &turn, sizeof(turn));
        aotx_fake_request(i, AOTX_AUTH_NONE, &request);
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &request,
                         sizeof(request));
        if ((i + 1) % 64 == 0) {
            aotx_fake_commit(&device, 0);
        }
    }
    aotx_fake_commit(&device, 0);
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(child) == 0, "the drain does not end with a clean status");
    aotx_map_release(&map);
}

/* The chain holds when every line names the digest of the line before it. A changed byte
 * breaks the chain at the line that follows the change. A changed digest field breaks the
 * chain at the line that carries the field. */
static void chain(int n)
{
    char dir[256];
    char path[1024];
    char out_path[1024];
    char err_path[1024];
    char name[32];
    char want[128];
    uint64_t boot_id = 0x0000000ceed00001ull + (uint64_t)n;
    const char *at;
    unsigned char was;
    long prev_at;
    long tokens_at;
    long bytes;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    build_derived(dir, boot_id, n);
    snprintf(path, sizeof(path), "%s/manifest/%016llx.jsonl", dir,
             (unsigned long long)boot_id);
    snprintf(out_path, sizeof(out_path), "%s/turns.txt", dir);
    snprintf(err_path, sizeof(err_path), "%s/report.txt", dir);
    snprintf(name, sizeof(name), "%016llx", (unsigned long long)boot_id);

    CHECK(run_report("manifest", dir, NULL, out_path, err_path) == 0,
          "the reader refuses a chain that holds");
    CHECK(read_all(err_path, text, sizeof(text)) > 0, "the report does not read");
    CHECK(strstr(text, "chain holds") != NULL, "the report does not state that the chain holds");
    snprintf(want, sizeof(want), "turns %d chain holds", n);
    CHECK(strstr(text, want) != NULL, "the report does not count %d turns", n);
    CHECK(read_all(out_path, text, sizeof(text)) > 0, "the turns do not read");
    CHECK(count_lines(text) == n, "the reader printed %d turns and %d were written",
          count_lines(text), n);
    {
        aotx_manifest_body turn;
        aotx_fake_manifest(0, &turn);
        snprintf(want, sizeof(want), "line=1 agent=%u turn=%u input=%016llx output=%016llx",
                 turn.agent, turn.turn, (unsigned long long)turn.input_hash,
                 (unsigned long long)turn.output_hash);
        CHECK(strstr(text, want) != NULL, "the first turn is not in the printed lines");
    }
    CHECK(run_report("manifest", dir, name, out_path, err_path) == 0,
          "the reader refuses the boot that the command names");

    /* The digest field of the first line, changed. The break is at the first line. */
    bytes = read_all(path, text, sizeof(text));
    CHECK(bytes > 0, "the chain file does not read");
    at = strstr(text, "\"prev\":\"");
    CHECK(at != NULL, "the first line names no digest");
    prev_at = (long)(at - text) + 8;
    was = swap_byte(path, prev_at, '0');
    CHECK(run_report("manifest", dir, NULL, out_path, err_path) == 1,
          "a changed digest must give the fault status");
    CHECK(read_all(err_path, text, sizeof(text)) > 0, "the report does not read");
    CHECK(strstr(text, "chain breaks at line 1") != NULL,
          "the report does not name the first line");
    swap_byte(path, prev_at, was);
    CHECK(run_report("manifest", dir, NULL, out_path, err_path) == 0,
          "the chain must hold again when the byte goes back");

    /* A byte of the content of the first line, changed. The break is at the second line,
     * because that line names the digest of the first. */
    if (n >= 2) {
        bytes = read_all(path, text, sizeof(text));
        CHECK(bytes > 0, "the chain file does not read");
        at = strstr(text, "\"tokens\":");
        CHECK(at != NULL, "the first line names no token count");
        tokens_at = (long)(at - text) + 9;
        was = swap_byte(path, tokens_at, '0');
        CHECK(run_report("manifest", dir, NULL, out_path, err_path) == 1,
              "a changed line must give the fault status");
        CHECK(read_all(err_path, text, sizeof(text)) > 0, "the report does not read");
        CHECK(strstr(text, "chain breaks at line 2") != NULL,
              "the report does not name the second line");
        swap_byte(path, tokens_at, was);
        CHECK(run_report("manifest", dir, NULL, out_path, err_path) == 0,
              "the chain must hold again when the byte goes back");
    }

    /* A last line that the disk cut carries no end byte, so the chain cannot go on. */
    bytes = read_all(path, text, sizeof(text));
    CHECK(truncate(path, bytes - 1) == 0, "the file does not lose its last byte");
    CHECK(run_report("manifest", dir, NULL, out_path, err_path) == 1,
          "a line with no end byte must give the fault status");
    CHECK(read_all(err_path, text, sizeof(text)) > 0, "the report does not read");
    snprintf(want, sizeof(want), "chain breaks at line %d", n);
    CHECK(strstr(text, want) != NULL, "the report does not name the last line");

    /* A directory that holds no chain file gives the status of a journal that is not there. */
    snprintf(path, sizeof(path), "%s/none", dir);
    CHECK(aotx_make_dir(path) == 0, "the empty directory does not open");
    CHECK(run_report("manifest", path, NULL, out_path, err_path) == AOTX_EXIT_NOJOURNAL,
          "a directory with no chain file must give the no journal status");

    printf("chain %d: turns %d, breaks found 3\n", n, n);
    aotx_remove_tree(dir);
}

/* The reader prints the requests that the drain derived. */
static void requests(int n)
{
    char dir[256];
    char out_path[1024];
    char err_path[1024];
    char want[256];
    uint64_t boot_id = 0x0000000feed00001ull + (uint64_t)n;
    int i;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    build_derived(dir, boot_id, n);
    snprintf(out_path, sizeof(out_path), "%s/requests.txt", dir);
    snprintf(err_path, sizeof(err_path), "%s/report.txt", dir);
    CHECK(run_report("requests", dir, NULL, out_path, err_path) == 0,
          "the reader refuses the requests file");
    CHECK(read_all(out_path, text, sizeof(text)) > 0, "the requests do not read");
    CHECK(count_lines(text) == n, "the reader printed %d requests and %d were written",
          count_lines(text), n);
    for (i = 0; i < n; i++) {
        aotx_tool_request_body request;
        aotx_fake_request(i, AOTX_AUTH_NONE, &request);
        snprintf(want, sizeof(want),
                 "request=%u agent=%u turn=%u tool=fs_read auth=none deadline=%llu tick=%llu"
                 " arg=file-%d.txt",
                 request.request, request.agent, request.turn,
                 (unsigned long long)request.deadline, (unsigned long long)(i / 64 + 1), i);
        CHECK(strstr(text, want) != NULL, "request %d is not in the printed lines", i);
    }
    CHECK(read_all(err_path, text, sizeof(text)) > 0, "the report does not read");
    CHECK(strstr(text, "lines not read 0") != NULL, "the report names a line it cannot read");
    printf("requests %d: lines %d\n", n, n);
    aotx_remove_tree(dir);
}

/* The reader prints one line for each setting record. The line gives the value in the
 * unit the operator writes. It states the flag of a record that a restore applied again. */
static void settings(int n, int replayed)
{
    char dir[256];
    char out_path[1024];
    char err_path[1024];
    char want[256];
    uint64_t boot_id = 0x0000000005e70001ull + (uint64_t)replayed;
    int count;
    int i;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    build_journal(dir, boot_id, n, replayed);
    snprintf(out_path, sizeof(out_path), "%s/settings.txt", dir);
    snprintf(err_path, sizeof(err_path), "%s/report.txt", dir);
    CHECK(run_report("settings", dir, NULL, out_path, err_path) == 0,
          "the reader refuses the settings of a journal");
    count = split(out_path);
    CHECK(count == n, "the reader printed %d settings and %d are in the journal", count, n);
    for (i = 0; i < count && i < n; i++) {
        aotx_setting_body body;
        char value[32];
        aotx_fake_setting(i, &body);
        aotx_settings_format(body.value, (int)body.scale, value, sizeof(value));
        snprintf(want, sizeof(want), "tick=%d writer=%u key=%s value=%s", i + 1,
                 ((i % 2) != 0) ? AOTX_WRITER_CONSOLE : AOTX_WRITER_FEEDER, body.key, value);
        CHECK(strncmp(lines[i], want, strlen(want)) == 0,
              "line %d is [%s] and [%s] was asked for", i, lines[i], want);
        snprintf(want, sizeof(want), " replayed=%d", replayed);
        CHECK(strstr(lines[i], want) != NULL, "line %d does not state the replayed flag", i);
    }
    CHECK(read_all(err_path, text, sizeof(text)) > 0, "the report does not read");
    snprintf(want, sizeof(want), "settings %d", n);
    CHECK(strstr(text, want) != NULL, "the report does not count the settings");
    printf("settings %d: lines %d, replayed %d\n", n, count, replayed);
    aotx_remove_tree(dir);
}

int main(int argc, char **argv)
{
    arguments = argv;
    if (argc < 3) {
        printf("usage: journal_test <journal reader program> <drain program>\n");
        return 1;
    }
    batch(1);
    batch(64);
    choices();
    refusals();
    chain(1);
    chain(64);
    requests(1);
    requests(64);
    settings(1, 0);
    settings(64, 1);
    return aotx_report("journal_test", 400);
}
