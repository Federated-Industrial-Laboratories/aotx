/* Purpose: Check the socket that terminal programs attach to and the records it publishes.
 * Owns: One inbound ring, one mirror and one socket for each case.
 * Threading: One thread; the test drives the socket and reads the ring as the device would.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include "disk/feed/attach.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>

#define AOTX_RING_SLOTS 256u

/* The key code of the enter key, as the window sends it. The case counts up from it, so
 * each frame carries a code of its own. */
#define AOTX_TEST_KEY_FIRST 257u

/* Makes a mirror of one preamble and two slots, as the seam glue makes it. */
static int make_mirror(void)
{
    size_t bytes = sizeof(aotx_mirror_preamble)
                 + (size_t)AOTX_MIRROR_SLOTS * sizeof(aotx_mirror_snapshot);
    aotx_mirror_preamble *pre;
    void *base;
    int fd = memfd_create("aotx_mirror", MFD_CLOEXEC);
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

/* The count of terminals that the mirror preamble holds. */
static unsigned int attached_count(int mirror_fd)
{
    aotx_mirror_preamble *pre;
    unsigned int count;
    void *base = mmap(NULL, sizeof(aotx_mirror_preamble), PROT_READ, MAP_SHARED,
                      mirror_fd, 0);
    if (base == MAP_FAILED) {
        return ~0u;
    }
    pre = (aotx_mirror_preamble *)base;
    count = __atomic_load_n(&pre->attached, __ATOMIC_ACQUIRE);
    munmap(base, sizeof(aotx_mirror_preamble));
    return count;
}

/* Runs the poll of the feeder over the socket until it has nothing more to take. */
static void drive(aotx_attach *a, const aotx_inbound_ring *ring)
{
    int turn;
    for (turn = 0; turn < 8; turn++) {
        struct pollfd fds[AOTX_ATTACH_MAX + 1u];
        unsigned int count = aotx_attach_poll_set(a, fds, AOTX_ATTACH_MAX + 1u);
        if (count == 0 || poll(fds, (nfds_t)count, 20) <= 0) {
            return;
        }
        if (aotx_attach_take(a, fds, count, ring, NULL) != 0) {
            return;
        }
    }
}

/* Connects one terminal to the socket. Returns the descriptor, or -1. */
static int connect_one(const char dir[128])
{
    struct sockaddr_un address;
    char path[256];
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    snprintf(path, sizeof(path), "%s/%s", dir, AOTX_ATTACH_NAME);
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, path, strlen(path));
    if (connect(fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

/* Takes the mirror descriptor that the feeder sends. Returns it, or -1. */
static int take_mirror(int fd)
{
    struct msghdr message;
    struct iovec io;
    struct cmsghdr *control;
    union {
        char bytes[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } room;
    char payload = 0;
    int mirror = -1;
    memset(&message, 0, sizeof(message));
    memset(&room, 0, sizeof(room));
    io.iov_base = &payload;
    io.iov_len = 1;
    message.msg_iov = &io;
    message.msg_iovlen = 1;
    message.msg_control = room.bytes;
    message.msg_controllen = sizeof(room.bytes);
    if (recvmsg(fd, &message, 0) != 1) {
        return -1;
    }
    control = CMSG_FIRSTHDR(&message);
    if (control == NULL || control->cmsg_type != SCM_RIGHTS) {
        return -1;
    }
    memcpy(&mirror, CMSG_DATA(control), sizeof(int));
    return mirror;
}

static void send_key(int fd, unsigned int code)
{
    unsigned char frame[1u + sizeof(aotx_key_body)];
    aotx_key_body body;
    memset(&body, 0, sizeof(body));
    body.key = code;
    body.action = 1u;
    frame[0] = (unsigned char)AOTX_ATTACH_KEY;
    memcpy(frame + 1, &body, sizeof(body));
    CHECK(write(fd, frame, sizeof(frame)) == (ssize_t)sizeof(frame),
          "the key frame does not go out");
}

static void send_line(int fd, const char *line)
{
    unsigned char frame[5u + AOTX_BODY_BYTES];
    size_t bytes = strlen(line);
    frame[0] = (unsigned char)AOTX_ATTACH_LINE;
    frame[1] = (unsigned char)(bytes & 0xffu);
    frame[2] = (unsigned char)((bytes >> 8) & 0xffu);
    frame[3] = 0;
    frame[4] = 0;
    memcpy(frame + 5, line, bytes);
    CHECK(write(fd, frame, bytes + 5u) == (ssize_t)(bytes + 5u),
          "the line frame does not go out");
}

static const aotx_record_header *slot_of(const aotx_inbound_ring *ring, uint64_t index)
{
    return (const aotx_record_header *)(ring->slots + (index & ring->mask) * AOTX_SLOT_BYTES);
}

/* One case: n key frames and n lines, one after the other, land as records in order. */
static void batch(int n)
{
    char dir[128];
    aotx_map map;
    aotx_inbound_ring ring;
    static aotx_attach state;
    int mirror_fd;
    int client;
    int taken;
    int i;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, &map, &ring) == 0, "the ring does not open");
    mirror_fd = make_mirror();
    CHECK(mirror_fd >= 0, "the mirror does not open");
    CHECK(aotx_attach_open(&state, dir, mirror_fd) == 0, "the socket does not open");
    CHECK(attached_count(mirror_fd) == 0u, "the attached count is not zero at the start");

    client = connect_one(dir);
    CHECK(client >= 0, "the terminal does not connect");
    drive(&state, &ring);
    CHECK(state.clients == 1u, "the socket did not accept the terminal");
    CHECK(attached_count(mirror_fd) == 1u, "the attached count did not go to one");

    taken = take_mirror(client);
    CHECK(taken >= 0, "the mirror descriptor did not arrive");
    if (taken >= 0) {
        aotx_mirror_preamble *pre;
        void *base = mmap(NULL, sizeof(aotx_mirror_preamble), PROT_READ, MAP_SHARED,
                          taken, 0);
        CHECK(base != MAP_FAILED, "the descriptor that arrived does not map");
        if (base != MAP_FAILED) {
            pre = (aotx_mirror_preamble *)base;
            CHECK(pre->magic == AOTX_MIRROR_MAGIC, "the descriptor is not the mirror");
            CHECK(pre->cols == AOTX_MIRROR_COLS, "the mirror has the wrong width");
            munmap(base, sizeof(aotx_mirror_preamble));
        }
        close(taken);
    }

    for (i = 0; i < n; i++) {
        char line[64];
        snprintf(line, sizeof(line), "note frame %d of %d", i, n);
        send_key(client, AOTX_TEST_KEY_FIRST + (unsigned int)i);
        send_line(client, line);
    }
    drive(&state, &ring);
    CHECK(state.keys == (uint64_t)n, "the socket published %llu key frames, not %d",
          (unsigned long long)state.keys, n);
    CHECK(state.lines == (uint64_t)n, "the socket published %llu lines, not %d",
          (unsigned long long)state.lines, n);
    CHECK(aotx_inbound_head(&ring) == (uint64_t)(2 * n), "the ring holds %llu records",
          (unsigned long long)aotx_inbound_head(&ring));
    for (i = 0; i < n; i++) {
        const aotx_record_header *key = slot_of(&ring, (uint64_t)(2 * i));
        const aotx_record_header *line = slot_of(&ring, (uint64_t)(2 * i + 1));
        const aotx_key_body *body = (const aotx_key_body *)aotx_record_body(key);
        char want[64];
        snprintf(want, sizeof(want), "note frame %d of %d", i, n);
        CHECK(key->type == AOTX_REC_KEY, "record %d is not a key", 2 * i);
        CHECK(key->cls == AOTX_CLASS_A, "the key record %d is not class A", 2 * i);
        CHECK(body->key == AOTX_TEST_KEY_FIRST + (unsigned int)i,
              "the key record %d holds the wrong code", 2 * i);
        CHECK(line->type == AOTX_REC_INPUT_LINE, "record %d is not a line", 2 * i + 1);
        CHECK(line->body_len == (uint32_t)strlen(want), "line %d has the wrong length", i);
        CHECK(memcmp(aotx_record_body(line), want, strlen(want)) == 0,
              "line %d holds the wrong bytes", i);
    }

    close(client);
    drive(&state, &ring);
    CHECK(state.clients == 0u, "the socket did not drop the terminal that left");
    CHECK(attached_count(mirror_fd) == 0u, "the attached count did not go back to zero");
    aotx_attach_close(&state);
    close(mirror_fd);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

/* A line longer than one record body is refused with a reason, and no record goes out. */
static void long_line(void)
{
    char dir[128];
    aotx_map map;
    aotx_inbound_ring ring;
    static aotx_attach state;
    unsigned char frame[5u + AOTX_BODY_BYTES + 16u];
    unsigned char answer[256];
    size_t bytes = AOTX_BODY_BYTES + 16u;
    int mirror_fd;
    int client;
    ssize_t got;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, &map, &ring) == 0, "the ring does not open");
    mirror_fd = make_mirror();
    CHECK(aotx_attach_open(&state, dir, mirror_fd) == 0, "the socket does not open");
    client = connect_one(dir);
    CHECK(client >= 0, "the terminal does not connect");
    drive(&state, &ring);
    CHECK(take_mirror(client) >= 0, "the mirror descriptor did not arrive");

    memset(frame, 'a', sizeof(frame));
    frame[0] = (unsigned char)AOTX_ATTACH_LINE;
    frame[1] = (unsigned char)(bytes & 0xffu);
    frame[2] = (unsigned char)((bytes >> 8) & 0xffu);
    frame[3] = 0;
    frame[4] = 0;
    CHECK(write(client, frame, bytes + 5u) == (ssize_t)(bytes + 5u),
          "the long line does not go out");
    drive(&state, &ring);
    CHECK(aotx_inbound_head(&ring) == 0, "a line over one body must make no record");
    CHECK(state.refused == 1u, "the socket did not count the refusal");
    got = recv(client, answer, sizeof(answer), MSG_DONTWAIT);
    CHECK(got > 5 && answer[0] == (unsigned char)AOTX_ATTACH_REASON,
          "no reason frame came back");
    CHECK(state.clients == 1u, "the terminal must stay after a refused line");

    /* A line that follows the refused one is taken, so the frame boundary was kept. */
    send_line(client, "note after the refusal");
    drive(&state, &ring);
    CHECK(aotx_inbound_head(&ring) == 1, "the line after the refusal did not go out");

    close(client);
    aotx_attach_close(&state);
    close(mirror_fd);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

/* Several terminals attach at one time, and the count of the mirror follows them. */
static void several(void)
{
    char dir[128];
    aotx_map map;
    aotx_inbound_ring ring;
    static aotx_attach state;
    int mirror_fd;
    int client[AOTX_ATTACH_MAX];
    unsigned int i;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, &map, &ring) == 0, "the ring does not open");
    mirror_fd = make_mirror();
    CHECK(aotx_attach_open(&state, dir, mirror_fd) == 0, "the socket does not open");
    for (i = 0; i < AOTX_ATTACH_MAX; i++) {
        client[i] = connect_one(dir);
        drive(&state, &ring);
        CHECK(take_mirror(client[i]) >= 0, "the mirror did not reach terminal %u", i);
    }
    CHECK(state.clients == AOTX_ATTACH_MAX, "the socket holds %u terminals, not %u",
          state.clients, AOTX_ATTACH_MAX);
    CHECK(attached_count(mirror_fd) == AOTX_ATTACH_MAX,
          "the attached count does not hold every terminal");
    for (i = 0; i < AOTX_ATTACH_MAX; i++) {
        close(client[i]);
    }
    drive(&state, &ring);
    CHECK(state.clients == 0u, "the socket did not drop the terminals that left");
    CHECK(attached_count(mirror_fd) == 0u, "the attached count did not clear");
    aotx_attach_close(&state);
    close(mirror_fd);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

/* A socket with no mirror refuses every terminal with a reason and holds none. */
static void no_mirror(void)
{
    char dir[128];
    aotx_map map;
    aotx_inbound_ring ring;
    static aotx_attach state;
    unsigned char answer[256];
    int client;
    ssize_t got;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, &map, &ring) == 0, "the ring does not open");
    CHECK(aotx_attach_open(&state, dir, -1) == 0, "the socket does not open");
    client = connect_one(dir);
    CHECK(client >= 0, "the terminal does not connect");
    drive(&state, &ring);
    CHECK(state.clients == 0u, "a socket with no mirror must hold no terminal");
    CHECK(state.refused == 1u, "the refusal is not counted");
    got = recv(client, answer, sizeof(answer), MSG_DONTWAIT);
    CHECK(got > 5 && answer[0] == (unsigned char)AOTX_ATTACH_REASON,
          "no reason frame came back");
    close(client);
    aotx_attach_close(&state);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

/* The peer credential rule: the user of the terminal is the user of the feeder. */
static void peer(void)
{
    unsigned int own = (unsigned int)getuid();
    CHECK(aotx_attach_peer_allowed(own, own) == 0, "the own user must be allowed");
    CHECK(aotx_attach_peer_allowed(own + 1u, own) == 1, "another user must be refused");
    CHECK(aotx_attach_peer_allowed(0u, own + 1u) == 1, "the root user must be refused");
    CHECK(aotx_attach_peer_allowed(0u, 0u) == 0, "one user must be allowed by itself");
}

/* The socket file has the mode 0600 and goes away at the close. */
static void mode(void)
{
    char dir[128];
    char path[256];
    static aotx_attach state;
    struct stat file;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    CHECK(aotx_attach_open(&state, dir, -1) == 0, "the socket does not open");
    snprintf(path, sizeof(path), "%s/%s", dir, AOTX_ATTACH_NAME);
    CHECK(stat(path, &file) == 0, "the socket file is not there");
    CHECK((file.st_mode & 0777) == 0600, "the socket mode is %o, not 600",
          file.st_mode & 0777);
    aotx_attach_close(&state);
    CHECK(stat(path, &file) != 0, "the socket file stays after the close");
    aotx_remove_tree(dir);
}

int main(void)
{
    peer();
    mode();
    batch(1);
    batch(64);
    long_line();
    several();
    no_mirror();
    return aotx_report("attach_test", 300);
}
