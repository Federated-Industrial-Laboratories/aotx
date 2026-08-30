/* Purpose: Hold the socket that terminal programs attach to and read their frames.
 * Owns: The listening socket, the connection of each terminal, and the mapped preamble.
 * Threading: One thread, the feeder's loop; every descriptor is not blocking.
 * Lifetime: From the open of the socket to the close of the feeder. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/feed/attach.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

/* The terminals that wait to be accepted. One is enough for a normal run; the bound keeps
 * a burst of connections from filling the kernel queue. */
#define AOTX_ATTACH_BACKLOG 8

/* The bytes one read takes from a terminal. */
#define AOTX_ATTACH_READ    1024

/* The line that goes to the console when a terminal of another user asks to attach. The
 * feeder sends no bus record of its own, so the note takes the path every action takes:
 * one command line that the parser reads. */
#define AOTX_ATTACH_NOTE "note a terminal of another user asked to attach and was refused"

int aotx_attach_peer_allowed(unsigned int peer_uid, unsigned int own_uid)
{
    return (peer_uid == own_uid) ? 0 : 1;
}

/* Writes the count of terminals into the mirror, so the raster thread knows that somebody
 * reads. The store is a release, because the count follows the state it stands for. */
static void publish_count(aotx_attach *a)
{
    if (a->preamble != NULL) {
        __atomic_store_n(&a->preamble->attached, a->clients, __ATOMIC_RELEASE);
    }
}

/* Sends one reason frame. The write is best effort: a terminal that went away gets no
 * reason, and the connection closes on the next read. */
static void send_reason(int fd, const char *reason)
{
    unsigned char frame[5 + 128];
    size_t bytes = strlen(reason);
    if (bytes > sizeof(frame) - 5) {
        bytes = sizeof(frame) - 5;
    }
    frame[0] = (unsigned char)AOTX_ATTACH_REASON;
    frame[1] = (unsigned char)(bytes & 0xffu);
    frame[2] = (unsigned char)((bytes >> 8) & 0xffu);
    frame[3] = 0;
    frame[4] = 0;
    memcpy(frame + 5, reason, bytes);
    if (write(fd, frame, bytes + 5) < 0) {
        return;
    }
}

/* Sends the mirror descriptor with one byte of payload. Returns 0 or -1. */
static int send_mirror(int fd, int mirror_fd)
{
    struct msghdr message;
    struct iovec io;
    struct cmsghdr *control;
    union {
        char bytes[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } room;
    char payload = (char)AOTX_ATTACH_MIRROR;

    memset(&message, 0, sizeof(message));
    memset(&room, 0, sizeof(room));
    io.iov_base = &payload;
    io.iov_len = 1;
    message.msg_iov = &io;
    message.msg_iovlen = 1;
    message.msg_control = room.bytes;
    message.msg_controllen = sizeof(room.bytes);
    control = CMSG_FIRSTHDR(&message);
    control->cmsg_level = SOL_SOCKET;
    control->cmsg_type = SCM_RIGHTS;
    control->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(control), &mirror_fd, sizeof(int));
    return (sendmsg(fd, &message, 0) == 1) ? 0 : -1;
}

/* Reopens the mirror for the terminal without write access. Descriptor passing preserves
 * the access mode of the descriptor, so the writable feeder descriptor must not cross. */
static int read_mirror(int mirror_fd)
{
    char path[64];
    if (snprintf(path, sizeof(path), "/proc/self/fd/%d", mirror_fd) >= (int)sizeof(path)) {
        return -1;
    }
    return open(path, O_RDONLY | O_CLOEXEC);
}

/* Maps the head of the mirror and checks that it is the layout this build reads. */
static int map_mirror(aotx_attach *a, int mirror_fd)
{
    long page = sysconf(_SC_PAGESIZE);
    size_t bytes = (page > 0) ? (size_t)page : 4096u;
    void *base;
    aotx_mirror_preamble *pre;
    if (bytes < sizeof(aotx_mirror_preamble)) {
        bytes = sizeof(aotx_mirror_preamble);
    }
    base = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, mirror_fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "feed: the mirror descriptor does not map\n");
        return -1;
    }
    pre = (aotx_mirror_preamble *)base;
    if (pre->magic != AOTX_MIRROR_MAGIC || pre->layout != AOTX_MIRROR_LAYOUT) {
        fprintf(stderr, "feed: the mirror preamble does not match this layout version\n");
        munmap(base, bytes);
        return -1;
    }
    a->preamble = pre;
    a->map_bytes = bytes;
    return 0;
}

int aotx_attach_open(aotx_attach *a, const char *dir, int mirror_fd)
{
    struct sockaddr_un address;
    mode_t was;
    char socket_path[sizeof(address.sun_path)];
    int fd;

    memset(a, 0, sizeof(*a));
    a->listen_fd = -1;
    a->dir_fd = -1;
    a->mirror_fd = mirror_fd;
    if (dir == NULL) {
        return 0;
    }
    snprintf(a->path, sizeof(a->path), "%s/%s", dir, AOTX_ATTACH_NAME);
    if (mirror_fd >= 0 && map_mirror(a, mirror_fd) != 0) {
        return -1;
    }
    a->dir_fd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (a->dir_fd < 0) {
        fprintf(stderr, "feed: the socket directory does not open at %s\n", dir);
        if (a->preamble != NULL) {
            munmap(a->preamble, a->map_bytes);
            a->preamble = NULL;
        }
        return -1;
    }
    if (snprintf(socket_path, sizeof(socket_path), "/proc/self/fd/%d/%s",
                 a->dir_fd, AOTX_ATTACH_NAME) >= (int)sizeof(socket_path)) {
        fprintf(stderr, "feed: the socket name is too long\n");
        close(a->dir_fd);
        a->dir_fd = -1;
        if (a->preamble != NULL) {
            munmap(a->preamble, a->map_bytes);
            a->preamble = NULL;
        }
        return -1;
    }
    fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        fprintf(stderr, "feed: the socket does not open\n");
        close(a->dir_fd);
        a->dir_fd = -1;
        if (a->preamble != NULL) {
            munmap(a->preamble, a->map_bytes);
            a->preamble = NULL;
        }
        return -1;
    }
    /* A socket file of a run that ended holds the name. The name is in the journal
     * directory of this run, so no other run owns it. */
    unlinkat(a->dir_fd, AOTX_ATTACH_NAME, 0);
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, socket_path, strlen(socket_path));
    /* The mask makes the socket file 0600 at the moment it appears, so no window exists
     * in which another user may connect. */
    was = umask(0177);
    if (bind(fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
        umask(was);
        fprintf(stderr, "feed: the socket does not bind at %s\n", a->path);
        close(fd);
        close(a->dir_fd);
        a->dir_fd = -1;
        if (a->preamble != NULL) {
            munmap(a->preamble, a->map_bytes);
            a->preamble = NULL;
        }
        return -1;
    }
    umask(was);
    if (fchmodat(a->dir_fd, AOTX_ATTACH_NAME, S_IRUSR | S_IWUSR, 0) != 0
        || listen(fd, AOTX_ATTACH_BACKLOG) != 0) {
        fprintf(stderr, "feed: the socket does not listen at %s\n", a->path);
        close(fd);
        unlinkat(a->dir_fd, AOTX_ATTACH_NAME, 0);
        close(a->dir_fd);
        a->dir_fd = -1;
        if (a->preamble != NULL) {
            munmap(a->preamble, a->map_bytes);
            a->preamble = NULL;
        }
        return -1;
    }
    a->listen_fd = fd;
    publish_count(a);
    return 0;
}

void aotx_attach_close(aotx_attach *a)
{
    unsigned int i;
    for (i = 0; i < a->clients; i++) {
        close(a->client[i].fd);
    }
    a->clients = 0;
    publish_count(a);
    if (a->listen_fd >= 0) {
        close(a->listen_fd);
        a->listen_fd = -1;
    }
    if (a->dir_fd >= 0) {
        unlinkat(a->dir_fd, AOTX_ATTACH_NAME, 0);
        close(a->dir_fd);
        a->dir_fd = -1;
    }
    if (a->preamble != NULL) {
        munmap(a->preamble, a->map_bytes);
        a->preamble = NULL;
    }
}

unsigned int aotx_attach_poll_set(const aotx_attach *a, struct pollfd *fds, unsigned int most)
{
    unsigned int count = 0;
    unsigned int i;
    if (a->listen_fd < 0 || most == 0) {
        return 0;
    }
    fds[count].fd = a->listen_fd;
    fds[count].events = POLLIN;
    fds[count].revents = 0;
    count++;
    for (i = 0; i < a->clients && count < most; i++) {
        fds[count].fd = a->client[i].fd;
        fds[count].events = POLLIN;
        fds[count].revents = 0;
        count++;
    }
    return count;
}

/* Takes one terminal out of the table and closes it. The last row moves into the gap, so
 * the table holds no hole and the poll set is built from the front. */
static void drop_client(aotx_attach *a, unsigned int index)
{
    close(a->client[index].fd);
    a->clients--;
    if (index != a->clients) {
        a->client[index] = a->client[a->clients];
    }
    memset(&a->client[a->clients], 0, sizeof(a->client[a->clients]));
    a->left++;
    publish_count(a);
}

/* Publishes one record of the terminal. Returns 0, or -1 when the ring closed. */
static int put(const aotx_inbound_ring *ring, const volatile sig_atomic_t *stop,
               uint8_t type, const void *body, uint32_t len)
{
    aotx_record_header h;
    memset(&h, 0, sizeof(h));
    h.writer = AOTX_WRITER_FEEDER;
    h.cls = AOTX_CLASS_A;
    h.type = type;
    h.body_len = len;
    if (aotx_inbound_wait(ring, stop) != 0) {
        return -1;
    }
    aotx_inbound_put(ring, &h, body);
    return 0;
}

/* Accepts one terminal. The peer must be the user that runs the feeder, a mirror must
 * exist, and the table must have room. Every refusal states its reason to the terminal
 * and closes the connection. Returns 0, or -1 when the ring closed. */
static int accept_one(aotx_attach *a, const aotx_inbound_ring *ring,
                      const volatile sig_atomic_t *stop)
{
    struct ucred peer;
    socklen_t bytes = (socklen_t)sizeof(peer);
    aotx_attach_client *client;
    int fd = accept4(a->listen_fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (fd < 0) {
        return 0;
    }
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &bytes) != 0
        || aotx_attach_peer_allowed((unsigned int)peer.uid, (unsigned int)getuid()) != 0) {
        send_reason(fd, "this system runs for another user");
        close(fd);
        a->refused++;
        return put(ring, stop, AOTX_REC_INPUT_LINE, AOTX_ATTACH_NOTE,
                   (uint32_t)strlen(AOTX_ATTACH_NOTE));
    }
    if (a->mirror_fd < 0) {
        send_reason(fd, "this system holds no mirror to read");
        close(fd);
        a->refused++;
        return 0;
    }
    if (a->clients == AOTX_ATTACH_MAX) {
        send_reason(fd, "the count of terminals is at its bound");
        close(fd);
        a->refused++;
        return 0;
    }
    {
        int mirror = read_mirror(a->mirror_fd);
        int sent = (mirror >= 0) ? send_mirror(fd, mirror) : -1;
        if (mirror >= 0) {
            close(mirror);
        }
        if (sent != 0) {
            close(fd);
            a->refused++;
            return 0;
        }
    }
    client = &a->client[a->clients];
    memset(client, 0, sizeof(*client));
    client->fd = fd;
    a->clients++;
    a->joined++;
    publish_count(a);
    return 0;
}

/* Takes the bytes of one message that is complete. Returns 0, or -1 when the ring
 * closed. */
static int take_message(aotx_attach *a, aotx_attach_client *client,
                        const aotx_inbound_ring *ring, const volatile sig_atomic_t *stop)
{
    if (client->part[0] == (unsigned char)AOTX_ATTACH_KEY) {
        a->keys++;
        return put(ring, stop, AOTX_REC_KEY, client->part + 1,
                   (uint32_t)sizeof(aotx_key_body));
    }
    a->lines++;
    return aotx_line_publish(ring, stop, client->part + 5, client->fill - 5u);
}

/* The state of the message at the head of a terminal's bytes. */
#define AOTX_ATTACH_GOOD   0  /* the whole length is known and `need` gives it */
#define AOTX_ATTACH_HEAD   1  /* more bytes are needed before the length is known */
#define AOTX_ATTACH_LONG   2  /* the line is longer than the input bound */
#define AOTX_ATTACH_BROKEN 3  /* the kind byte or the length is not one this build takes */

static int message_state(const aotx_attach_client *client, uint32_t *need, uint32_t *length)
{
    *length = 0;
    *need = 0;
    if (client->fill == 0) {
        *need = 1u;
        return AOTX_ATTACH_HEAD;
    }
    if (client->part[0] == (unsigned char)AOTX_ATTACH_KEY) {
        *need = (uint32_t)(1u + sizeof(aotx_key_body));
        return AOTX_ATTACH_GOOD;
    }
    if (client->part[0] != (unsigned char)AOTX_ATTACH_LINE) {
        return AOTX_ATTACH_BROKEN;
    }
    if (client->fill < 5u) {
        *need = 5u;
        return AOTX_ATTACH_HEAD;
    }
    *length = (uint32_t)client->part[1] | ((uint32_t)client->part[2] << 8)
              | ((uint32_t)client->part[3] << 16) | ((uint32_t)client->part[4] << 24);
    if (*length > AOTX_INPUT_LINE_BYTES) {
        return AOTX_ATTACH_LONG;
    }
    *need = 5u + *length;
    return AOTX_ATTACH_GOOD;
}

/* Reads one terminal and publishes what it sent. Returns 0 when the terminal stays, 1 when
 * the caller must drop it, and -1 when the ring closed. */
static int read_client(aotx_attach *a, aotx_attach_client *client,
                       const aotx_inbound_ring *ring, const volatile sig_atomic_t *stop)
{
    unsigned char buffer[AOTX_ATTACH_READ];
    ssize_t got = read(client->fd, buffer, sizeof(buffer));
    size_t at = 0;
    uint32_t length = 0;
    uint32_t need = 0;
    int state;
    if (got == 0) {
        return 1;
    }
    if (got < 0) {
        return (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) ? 0 : 1;
    }
    while (at < (size_t)got) {
        uint32_t take;
        size_t left;
        state = message_state(client, &need, &length);
        if (state == AOTX_ATTACH_BROKEN) {
            send_reason(client->fd, "the frame of this message is not known");
            a->refused++;
            return 1;
        }
        if (state == AOTX_ATTACH_LONG) {
            send_reason(client->fd, AOTX_INPUT_LINE_REASON);
            a->refused++;
            return 1;
        }
        left = (size_t)got - at;
        take = need - client->fill;
        if (take > left) {
            take = (uint32_t)left;
        }
        memcpy(client->part + client->fill, buffer + at, take);
        client->fill += take;
        at += take;
        if (state == AOTX_ATTACH_GOOD && client->fill == need) {
            if (take_message(a, client, ring, stop) != 0) {
                return -1;
            }
            client->fill = 0;
        }
    }
    /* A complete invalid head can end at the read boundary. Refuse it without waiting for
     * another byte that the terminal has no reason to send. */
    if (client->fill > 0) {
        state = message_state(client, &need, &length);
        if (state == AOTX_ATTACH_BROKEN || state == AOTX_ATTACH_LONG) {
            send_reason(client->fd, (state == AOTX_ATTACH_LONG) ? AOTX_INPUT_LINE_REASON
                                                                : "the frame is not known");
            a->refused++;
            return 1;
        }
    }
    return 0;
}

int aotx_attach_take(aotx_attach *a, const struct pollfd *fds, unsigned int count,
                     const aotx_inbound_ring *ring, const volatile sig_atomic_t *stop)
{
    unsigned int i;
    if (a->listen_fd < 0 || count == 0) {
        return 0;
    }
    if ((fds[0].revents & POLLIN) != 0 && accept_one(a, ring, stop) != 0) {
        return -1;
    }
    /* The rows after the first stand for the terminals in the order the poll set was
     * built. A terminal that goes away moves the last row into its place, so the walk runs
     * from the end and no row is missed. */
    for (i = count; i > 1; i--) {
        unsigned int index = i - 2u;
        short events = fds[i - 1u].revents;
        int rc;
        if (index >= a->clients || fds[i - 1u].fd != a->client[index].fd) {
            continue;
        }
        if ((events & (POLLIN | POLLHUP | POLLERR)) == 0) {
            continue;
        }
        rc = read_client(a, &a->client[index], ring, stop);
        if (rc < 0) {
            return -1;
        }
        if (rc > 0) {
            drop_client(a, index);
        }
    }
    return 0;
}
