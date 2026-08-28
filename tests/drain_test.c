/* Purpose: Run the drain against a host ring and check the segments and the derived files.
 * Owns: One temporary journal and one host ring for each case.
 * Threading: Two processes; the test writes the ring while the drain reads it.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include <fcntl.h>
#include <signal.h>
#include <time.h>

#define AOTX_LINE_BYTES_MAX 4096

static int argument_count;
static char **arguments;

/* Counts the lines of a file and gives back the whole content. */
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

static int count_lines(const char *text)
{
    int lines = 0;
    const char *at = text;
    while (*at != '\0') {
        if (*at == '\n') {
            lines++;
        }
        at++;
    }
    return lines;
}

/* Runs the validator that the command line names, with the bus file as the last word. */
static void validate(const char *path)
{
    char *args[8];
    int i;
    int child;
    if (argument_count < 3) {
        printf("no validator was given, so the bus file check is not applied\n");
        return;
    }
    for (i = 2; i < argument_count && i < 6; i++) {
        args[i - 2] = arguments[i];
    }
    args[argument_count - 2] = (char *)path;
    args[argument_count - 1] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the validator does not start");
    CHECK(aotx_wait(child) == 0, "the validator refuses the bus file");
}

/* Reads back every frame of the journal and counts the blocks and the records. */
static void read_journal(const char *boot_dir, int *blocks, int *records)
{
    char names[64][AOTX_NAME_BYTES];
    static unsigned char block[16384];
    int count = aotx_segment_list(boot_dir, names, 64);
    int i;
    *blocks = 0;
    *records = 0;
    CHECK(count > 0, "the journal holds no segment");
    for (i = 0; i < count; i++) {
        char path[1024];
        aotx_segment_reader reader;
        snprintf(path, sizeof(path), "%.900s/%.*s", boot_dir, AOTX_NAME_BYTES - 1, names[i]);
        CHECK(aotx_segment_reader_open(&reader, path) == 0, "the segment does not read");
        for (;;) {
            uint32_t got = 0;
            const char *reason = "";
            int frame = aotx_segment_get(&reader, block, sizeof(block), &got);
            if (frame == AOTX_FRAME_END) {
                break;
            }
            CHECK(frame == AOTX_FRAME_OK, "a frame of the journal is not whole");
            if (frame != AOTX_FRAME_OK) {
                break;
            }
            CHECK(aotx_block_valid(block, got, &reason) == 0, "a block is wrong: %s", reason);
            *records += (int)((const aotx_block_header *)block)->record_count;
            (*blocks)++;
        }
        aotx_segment_reader_close(&reader);
    }
}

static void batch(int n, uint64_t ring_bytes)
{
    aotx_map map;
    aotx_host_ring ring;
    aotx_fake_device device;
    char dir[256];
    char boot_dir[320];
    char path[1024];
    char fd_text[16];
    char *args[6];
    static char text[262144];
    uint64_t boot_id = 0x00c0ffee00000001ull;
    int consoles = 0;
    int notes = 0;
    int blocks = 0;
    int records = 0;
    int out_fd;
    int child;
    int i;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    CHECK(aotx_host_ring_create(ring_bytes, boot_id, &map, &ring) == 0,
          "the ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    snprintf(path, sizeof(path), "%s/echo.txt", dir);
    out_fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    CHECK(out_fd >= 0, "the echo file does not open");
    args[0] = arguments[1];
    args[1] = (char *)"--ring-fd";
    args[2] = fd_text;
    args[3] = (char *)"--journal";
    args[4] = dir;
    args[5] = NULL;
    child = aotx_spawn(args, -1, out_fd);
    CHECK(child > 0, "the drain does not start");

    aotx_fake_start(&device, &ring, boot_id);
    for (i = 0; i < n; i++) {
        aotx_boot_body boot;
        aotx_clock_body clock;
        aotx_commit_body commit;
        char body[64];
        int r;
        if (i == 0) {
            /* The first console record comes before any tick start, so its lag is not
             * known and the bus line must carry a null. */
            memset(&boot, 0, sizeof(boot));
            boot.boot_id = boot_id;
            boot.wall_ns = aotx_wall_ns();
            aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_BOOT, &boot, sizeof(boot));
            snprintf(body, sizeof(body), "console line %d", i);
            aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_CONSOLE, body, (uint32_t)strlen(body));
            consoles++;
            /* A body can hold any byte. The bus line must escape the special ones, and
             * put a question mark where the bytes are not valid UTF-8. */
            {
                const char *odd = "odd \"quote\" \\ slash\ttab\nfeed \xff bad";
                aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_CONSOLE, odd,
                                 (uint32_t)strlen(odd));
            }
            consoles++;
        } else {
            clock.wall_ns = aotx_wall_ns();
            aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TICK_START, &clock, sizeof(clock));
            for (r = 0; r < 2; r++) {
                snprintf(body, sizeof(body), "console line %d part %d", i, r);
                aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_CONSOLE, body,
                                 (uint32_t)strlen(body));
                consoles++;
            }
            snprintf(body, sizeof(body), "note line %d", i);
            aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_NOTE, body, (uint32_t)strlen(body));
            notes++;
        }
        memset(&commit, 0, sizeof(commit));
        commit.state_hash = 0x0123456789abcdefull + (uint64_t)i;
        commit.applied_count = (uint64_t)i + 1;
        aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
        aotx_fake_commit(&device, 0);
    }
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(child) == 0, "the drain does not end with a clean status");
    CHECK(aotx_host_ring_cursor(&ring) == aotx_host_ring_head(&ring),
          "the drain did not reach the head of the ring");

    snprintf(boot_dir, sizeof(boot_dir), "%s/%016llx", dir, (unsigned long long)boot_id);
    read_journal(boot_dir, &blocks, &records);
    CHECK(blocks == n, "the journal holds %d blocks and %d were written", blocks, n);

    snprintf(path, sizeof(path), "%s/console.log", boot_dir);
    CHECK(slurp(path, text, sizeof(text)) > 0, "the console log does not read");
    CHECK(count_lines(text) == consoles, "the console log holds %d lines and %d were written",
          count_lines(text), consoles);
    CHECK(strstr(text, "console line 0") != NULL, "the first console line is not in the log");

    snprintf(path, sizeof(path), "%s/echo.txt", dir);
    CHECK(slurp(path, text, sizeof(text)) > 0, "the echo file does not read");
    CHECK(count_lines(text) == consoles, "the terminal echo holds %d lines and %d were written",
          count_lines(text), consoles);

    {
        char day[16];
        time_t now = (time_t)(aotx_wall_ns() / 1000000000u);
        struct tm parts;
        localtime_r(&now, &parts);
        strftime(day, sizeof(day), "%Y-%m-%d", &parts);
        snprintf(path, sizeof(path), "%s/bus/%s-aotx.jsonl", dir, day);
    }
    CHECK(slurp(path, text, sizeof(text)) > 0, "the bus file does not read");
    CHECK(count_lines(text) == consoles + notes, "the bus file holds %d lines and %d were derived",
          count_lines(text), consoles + notes);
    CHECK(strstr(text, "\"agent\":\"console\"") != NULL, "the bus line has no agent");
    CHECK(strstr(text, "\"seq\":1,") != NULL, "the bus file does not start at sequence one");
    CHECK(strstr(text, "\"lag_ms\":null") != NULL, "a line without a tick start must carry null");
    CHECK(strstr(text, "odd \\\"quote\\\" \\\\ slash\\ttab\\nfeed ? bad") != NULL,
          "the bus line does not escape the special bytes");
    if (n > 1) {
        CHECK(strstr(text, "note line 1") != NULL, "the note record is not derived");
    }
    validate(path);

    aotx_map_release(&map);
    close(out_fd);
    aotx_remove_tree(dir);
}

/* Sends the end signal instead of the closed flag. Every block that the device published
 * before the signal must still reach the journal. */
static void on_end_signal(int n)
{
    aotx_map map;
    aotx_host_ring ring;
    aotx_fake_device device;
    char dir[256];
    char boot_dir[320];
    char fd_text[16];
    char *args[6];
    uint64_t boot_id = 0x00c0ffee00000002ull;
    int blocks = 0;
    int records = 0;
    int child;
    int i;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    CHECK(aotx_host_ring_create(262144u, boot_id, &map, &ring) == 0, "the ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--ring-fd";
    args[2] = fd_text;
    args[3] = (char *)"--journal";
    args[4] = dir;
    args[5] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the drain does not start");
    /* The drain makes the boot directory before it reads the ring. The signal must come
     * after that, or it reaches a program that has not yet set its signal handler. */
    snprintf(boot_dir, sizeof(boot_dir), "%s/%016llx", dir, (unsigned long long)boot_id);
    {
        uint64_t deadline = aotx_wall_ns() + 15000000000ull;
        struct stat st;
        uint64_t backoff = 0;
        while (stat(boot_dir, &st) != 0 && aotx_wall_ns() < deadline) {
            aotx_pause(&backoff);
        }
        CHECK(stat(boot_dir, &st) == 0, "the drain does not make the boot directory");
    }
    aotx_fake_start(&device, &ring, boot_id);
    for (i = 0; i < n; i++) {
        aotx_commit_body commit;
        char body[64];
        snprintf(body, sizeof(body), "signal line %d", i);
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_CONSOLE, body, (uint32_t)strlen(body));
        memset(&commit, 0, sizeof(commit));
        commit.state_hash = (uint64_t)i;
        aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
        aotx_fake_commit(&device, 0);
    }
    CHECK(kill((pid_t)child, SIGTERM) == 0, "the end signal does not reach the drain");
    CHECK(aotx_wait(child) == 0, "the drain does not end with a clean status");
    read_journal(boot_dir, &blocks, &records);
    CHECK(blocks == n, "the journal holds %d blocks and %d were published", blocks, n);
    CHECK(aotx_host_ring_cursor(&ring) == aotx_host_ring_head(&ring),
          "the drain did not reach the head of the ring");
    printf("end signal %d: blocks %d, records %d\n", n, blocks, records);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

/* A preamble that names another layout version must stop the drain with the layout status. */
static void refuse_layout(void)
{
    aotx_map map;
    aotx_host_ring ring;
    char dir[256];
    char fd_text[16];
    char *args[6];
    int child;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    CHECK(aotx_host_ring_create(65536u, 5, &map, &ring) == 0, "the ring does not open");
    ring.pre->layout = AOTX_WIRE_LAYOUT + 1;
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--ring-fd";
    args[2] = fd_text;
    args[3] = (char *)"--journal";
    args[4] = dir;
    args[5] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the drain does not start");
    CHECK(aotx_wait(child) == AOTX_EXIT_LAYOUT, "a wrong layout version must give status 2");
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

int main(int argc, char **argv)
{
    argument_count = argc;
    arguments = argv;
    if (argc < 2) {
        printf("usage: drain_test <drain program> [validator ...]\n");
        return 1;
    }
    batch(1, 262144u);
    batch(64, 262144u);
    /* A ring this small makes the device write pad blocks, which the journal leaves out. */
    batch(64, 8192u);
    on_end_signal(1);
    on_end_signal(64);
    refuse_layout();
    return aotx_report("drain_test", 30);
}
