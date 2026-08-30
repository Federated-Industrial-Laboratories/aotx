/* Purpose: Check model fetch lines through the feeder input and attach socket.
 * Owns: One copied feeder, one local child program, and one inbound ring for each case.
 * Threading: Three processes; the check consumes records while two children run.
 * Lifetime: The run of each case. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "tests/disk_fake.h"

#include "disk/feed/attach.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>

#define AOTX_FETCH_RING 256u
#define AOTX_FETCH_LINES 96u

static const char *test_program;
static const char *feed_program;

typedef struct fetch_lines {
    unsigned int count;
    char line[AOTX_FETCH_LINES][AOTX_INPUT_LINE_BYTES + 1u];
} fetch_lines;

/* This program also acts as the model child when the copied feeder starts it. */
static int fake_model(int argc, char **argv)
{
    if (argc != 5 || strcmp(argv[1], "--dir") != 0 ||
        strcmp(argv[3], "fetch") != 0) {
        return 2;
    }
    printf("host origin.example\n");
    fflush(stdout);
    usleep(250000u);
    printf("bytes 7 total 19 rate 3\n");
    fflush(stdout);
    usleep(100000u);
    return 0;
}

/* Copies one executable, so its sibling lookup stays inside the fixture. */
static int copy_program(const char *from, const char *to)
{
    unsigned char room[65536];
    int in = open(from, O_RDONLY | O_CLOEXEC);
    int out = open(to, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0700);
    int state = 0;
    if (in < 0 || out < 0) {
        state = -1;
    }
    while (state == 0) {
        ssize_t got = read(in, room, sizeof(room));
        size_t at = 0u;
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got < 0) {
            state = -1;
            break;
        }
        if (got == 0) {
            break;
        }
        while (at < (size_t)got) {
            ssize_t wrote = write(out, room + at, (size_t)got - at);
            if (wrote < 0 && errno == EINTR) {
                continue;
            }
            if (wrote <= 0) {
                state = -1;
                break;
            }
            at += (size_t)wrote;
        }
    }
    if (in >= 0) {
        close(in);
    }
    if (out >= 0 && close(out) != 0) {
        state = -1;
    }
    return state;
}

/* Makes the feeder and model program as siblings in the fixture directory. */
static int fixture_programs(const char *dir, char *feed, size_t feed_bytes)
{
    char models[256];
    snprintf(feed, feed_bytes, "%s/aotx_feed", dir);
    snprintf(models, sizeof(models), "%s/aotx_models", dir);
    return copy_program(feed_program, feed) == 0 &&
           copy_program(test_program, models) == 0 ? 0 : -1;
}

/* Takes all complete input lines from the ring and releases their slots. */
static void collect(aotx_inbound_ring *ring, fetch_lines *lines)
{
    uint64_t at = aotx_inbound_consumed(ring);
    uint64_t head = aotx_inbound_head(ring);
    while (at < head) {
        const unsigned char *slot = ring->slots + (at & ring->mask) * AOTX_SLOT_BYTES;
        const aotx_record_header *record = (const aotx_record_header *)slot;
        CHECK(aotx_record_valid(record) == 1, "a fetch record does not validate");
        if (record->type == AOTX_REC_INPUT_LINE && lines->count < AOTX_FETCH_LINES) {
            uint32_t bytes = record->body_len;
            if (bytes > AOTX_INPUT_LINE_BYTES) {
                bytes = AOTX_INPUT_LINE_BYTES;
            }
            memcpy(lines->line[lines->count], aotx_record_body(record), bytes);
            lines->line[lines->count][bytes] = '\0';
            lines->count++;
        }
        at++;
        aotx_store_release(&ring->pre->consumed, at);
    }
}

static int holds(const fetch_lines *lines, const char *text)
{
    unsigned int i;
    for (i = 0u; i < lines->count; i++) {
        if (strcmp(lines->line[i], text) == 0) {
            return 1;
        }
    }
    return 0;
}

static int complete(const fetch_lines *lines)
{
    return holds(lines, "note fetch model-00 on disk");
}

/* Waits for the local model child to end and its final note to arrive. */
static void wait_fetch(aotx_inbound_ring *ring, fetch_lines *lines)
{
    uint64_t end = aotx_wall_ns() + 5000000000ull;
    while (!complete(lines) && aotx_wall_ns() < end) {
        uint64_t backoff = 0u;
        collect(ring, lines);
        aotx_pause(&backoff);
    }
    collect(ring, lines);
    CHECK(complete(lines), "the final model fetch note did not arrive");
}

/* Checks the notes that one accepted line and the other refused lines made. */
static void check_lines(const fetch_lines *lines, int n)
{
    int i;
    char want[192];
    CHECK(lines->count == (unsigned int)n + 3u,
          "the batch %d made %u fetch notes", n, lines->count);
    CHECK(holds(lines, "note fetch model-00 started"), "the start note is absent");
    CHECK(holds(lines, "note fetch model-00 host origin.example"),
          "the host note is absent");
    CHECK(holds(lines, "note fetch model-00 7 of 19"), "the progress note is absent");
    CHECK(holds(lines, "note fetch model-00 on disk"), "the final state is absent");
    for (i = 1; i < n; i++) {
        snprintf(want, sizeof(want),
                 "note fetch model-%02d refused because fetch model-00 runs", i);
        CHECK(holds(lines, want), "the refusal note for model-%02d is absent", i);
    }
}

/* Gives n model fetch lines to the standard input of the feeder. */
static void input_batch(int n)
{
    aotx_map map;
    aotx_inbound_ring ring;
    fetch_lines lines;
    char dir[128];
    char feeder[256];
    char fd_text[16];
    char *args[4];
    int input[2];
    int child;
    int i;
    memset(&lines, 0, sizeof(lines));
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the input fixture does not open");
    CHECK(fixture_programs(dir, feeder, sizeof(feeder)) == 0,
          "the input fixture programs do not copy");
    CHECK(aotx_inbound_create(AOTX_FETCH_RING, &map, &ring) == 0,
          "the input ring does not open");
    CHECK(pipe(input) == 0, "the feeder input pipe does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = feeder; args[1] = (char *)"--inbound-fd"; args[2] = fd_text; args[3] = NULL;
    child = aotx_spawn(args, input[0], -1);
    CHECK(child > 0, "the input feeder does not start");
    close(input[0]);
    for (i = 0; i < n; i++) {
        char line[64];
        int bytes = snprintf(line, sizeof(line), "model fetch model-%02d\n", i);
        CHECK(write(input[1], line, (size_t)bytes) == bytes,
              "input fetch line %d does not write", i);
    }
    close(input[1]);
    wait_fetch(&ring, &lines);
    check_lines(&lines, n);
    aotx_store_release16(&ring.pre->closed, 1u);
    CHECK(aotx_wait(child) == 0, "the input feeder does not end cleanly");
    aotx_map_release(&map);
    aotx_remove_tree(dir);
    printf("feeder input batch %d: notes %u\n", n, lines.count);
}

/* Makes a mirror preamble that permits a terminal to attach. */
static int make_mirror(void)
{
    size_t bytes = sizeof(aotx_mirror_preamble)
                 + (size_t)AOTX_MIRROR_SLOTS * sizeof(aotx_mirror_snapshot);
    aotx_mirror_preamble *pre;
    void *base;
    int fd = memfd_create("aotx_fetch_mirror", 0u);
    if (fd < 0 || ftruncate(fd, (off_t)bytes) != 0) {
        return -1;
    }
    base = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        close(fd);
        return -1;
    }
    pre = (aotx_mirror_preamble *)base;
    memset(pre, 0, sizeof(*pre));
    pre->magic = AOTX_MIRROR_MAGIC;
    pre->layout = AOTX_MIRROR_LAYOUT;
    pre->slots = AOTX_MIRROR_SLOTS;
    pre->slot_bytes = (uint32_t)sizeof(aotx_mirror_snapshot);
    pre->cols = AOTX_MIRROR_COLS;
    pre->rows = AOTX_MIRROR_ROWS;
    munmap(base, bytes);
    return fd;
}

static int connect_terminal(const char *dir)
{
    struct sockaddr_un address;
    size_t dir_bytes = strlen(dir);
    size_t name_bytes = strlen(AOTX_ATTACH_NAME);
    uint64_t end = aotx_wall_ns() + 2000000000ull;
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (dir_bytes + name_bytes + 2u > sizeof(address.sun_path)) {
        close(fd);
        return -1;
    }
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, dir, dir_bytes);
    address.sun_path[dir_bytes] = '/';
    memcpy(address.sun_path + dir_bytes + 1u, AOTX_ATTACH_NAME, name_bytes);
    while (fd >= 0 && connect(fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
        if (errno != ENOENT && errno != ECONNREFUSED) {
            close(fd);
            return -1;
        }
        if (aotx_wall_ns() >= end) {
            close(fd);
            return -1;
        }
        usleep(10000u);
    }
    return fd;
}

static void send_fetch(int fd, int index)
{
    unsigned char frame[5u + 64u];
    char line[64];
    int bytes = snprintf(line, sizeof(line), "model fetch model-%02d", index);
    frame[0] = (unsigned char)AOTX_ATTACH_LINE;
    frame[1] = (unsigned char)(bytes & 0xff);
    frame[2] = (unsigned char)((bytes >> 8) & 0xff);
    frame[3] = 0u;
    frame[4] = 0u;
    memcpy(frame + 5, line, (size_t)bytes);
    CHECK(write(fd, frame, (size_t)bytes + 5u) == bytes + 5,
          "attached fetch line %d does not write", index);
}

/* Gives n model fetch lines to one terminal attached to the feeder. */
static void attach_batch(int n)
{
    aotx_map map;
    aotx_inbound_ring ring;
    fetch_lines lines;
    char dir[128];
    char feeder[256];
    char fd_text[16];
    char mirror_text[16];
    char *args[8];
    int input[2];
    int mirror;
    int terminal;
    int child;
    int i;
    memset(&lines, 0, sizeof(lines));
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the attach fixture does not open");
    CHECK(fixture_programs(dir, feeder, sizeof(feeder)) == 0,
          "the attach fixture programs do not copy");
    CHECK(aotx_inbound_create(AOTX_FETCH_RING, &map, &ring) == 0,
          "the attach ring does not open");
    mirror = make_mirror();
    CHECK(mirror >= 0, "the attach mirror does not open");
    CHECK(pipe(input) == 0, "the attached feeder input does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    snprintf(mirror_text, sizeof(mirror_text), "%d", mirror);
    args[0] = feeder; args[1] = (char *)"--inbound-fd"; args[2] = fd_text;
    args[3] = (char *)"--attach"; args[4] = dir; args[5] = (char *)"--mirror-fd";
    args[6] = mirror_text; args[7] = NULL;
    child = aotx_spawn(args, input[0], -1);
    CHECK(child > 0, "the attached feeder does not start");
    close(input[0]);
    close(input[1]);
    terminal = connect_terminal(dir);
    CHECK(terminal >= 0, "the terminal does not attach to the feeder");
    for (i = 0; i < n && terminal >= 0; i++) {
        send_fetch(terminal, i);
    }
    wait_fetch(&ring, &lines);
    check_lines(&lines, n);
    if (terminal >= 0) {
        close(terminal);
    }
    aotx_store_release16(&ring.pre->closed, 1u);
    CHECK(aotx_wait(child) == 0, "the attached feeder does not end cleanly");
    close(mirror);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
    printf("feeder attach batch %d: notes %u\n", n, lines.count);
}

int main(int argc, char **argv)
{
    const char *base = strrchr(argv[0], '/');
    if (base != NULL && strcmp(base + 1, "aotx_models") == 0) {
        return fake_model(argc, argv);
    }
    if (argc != 2) {
        return 2;
    }
    test_program = argv[0];
    feed_program = argv[1];
    input_batch(1);
    input_batch(64);
    attach_batch(1);
    attach_batch(64);
    return aotx_report("feed_models_test", 160);
}
