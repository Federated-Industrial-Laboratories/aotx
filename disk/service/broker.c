/* Purpose: Exchange bounded service packets without exposing operator resources.
 * Owns: Local peer sockets and mailbox transfers; CUDA owns grants and request state.
 * Threading: One nonblocking poll loop with bounded peers and packets.
 * Lifetime: One runtime; parent exit stops the broker. */
#define _GNU_SOURCE
#include "disk/service/broker.h"
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
static volatile sig_atomic_t aotx_service_stopped, aotx_service_reload;
static void aotx_service_signal(int signal)
{
    if (signal == SIGHUP) aotx_service_reload = 1;
    else aotx_service_stopped = 1;
}
static void aotx_service_drop(aotx_service_peer *p)
{ if (p->fd >= 0) close(p->fd); p->fd = -1; }
static void aotx_service_accept(int listener, aotx_service_peer *peers, aotx_service_mailbox *mailboxes)
{
    for (unsigned take = 0; take < 16; ++take) {
        int fd = accept4(listener, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (fd < 0) return;
        struct ucred who; socklen_t bytes = sizeof who;
        unsigned at = AOTX_SERVICE_CHANNELS;
        if (!getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &who, &bytes) && who.uid == geteuid())
            for (unsigned i = 1; i < AOTX_SERVICE_CHANNELS; ++i)
                if (peers[i].fd < 0 && !peers[i].pending &&
                    __atomic_load_n(&mailboxes[i].state, __ATOMIC_ACQUIRE) == 0) { at = i; break; }
        if (at == AOTX_SERVICE_CHANNELS) close(fd);
        else { peers[at].fd = fd; peers[at].touched = aotx_service_now(); }
    }
}
static void aotx_service_peer_step(aotx_service_peer *p, aotx_service_mailbox *m, short events)
{
    uint64_t state = __atomic_load_n(&m->state, __ATOMIC_ACQUIRE);
    if (p->fd < 0) {
        if (p->pending && state == 2) { __atomic_store_n(&m->state, 0, __ATOMIC_RELEASE); p->pending = 0; }
        return;
    }
    if ((events & (POLLHUP | POLLERR | POLLNVAL)) || aotx_service_now() - p->touched > 30000000000ull) {
        aotx_service_drop(p); return;
    }
    if (p->pending && state == 2) {
        uint64_t length = m->length;
        if (length < AOTX_SERVICE_HEAD || length > AOTX_SERVICE_FRAME) { aotx_service_drop(p); return; }
        ssize_t sent = send(p->fd, m->bytes, (size_t)length, MSG_DONTWAIT | MSG_NOSIGNAL);
        if (sent == (ssize_t)length) {
            __atomic_store_n(&m->state, 0, __ATOMIC_RELEASE); p->pending = 0; p->touched = aotx_service_now();
        } else if (sent >= 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) aotx_service_drop(p);
        return;
    }
    if (!p->pending && (events & POLLIN)) {
        struct iovec part = {m->bytes, AOTX_SERVICE_FRAME};
        struct msghdr message; memset(&message, 0, sizeof message);
        message.msg_iov = &part; message.msg_iovlen = 1;
        ssize_t got = recvmsg(p->fd, &message, MSG_DONTWAIT);
        if (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return;
        if (got < AOTX_SERVICE_HEAD || (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) ||
            aotx_service_get(m->bytes + 8, 4) == AOTX_SERVICE_GRANTS) { aotx_service_drop(p); return; }
        m->length = (uint64_t)got; ++m->generation;
        __atomic_store_n(&m->state, 1, __ATOMIC_RELEASE); p->pending = 1; p->touched = aotx_service_now();
    }
}
int main(int argc, char **argv)
{
    int fd = -1, listener = -1, lock = -1, bad = 1;
    const char *journal = NULL, *grants = NULL;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 == argc) return 2;
        if (!strcmp(argv[i], "--ring-fd")) fd = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--journal")) journal = argv[++i];
        else if (!strcmp(argv[i], "--grants")) grants = argv[++i];
        else return 2;
    }
    if (fd < 0 || !journal || !grants) return 2;
    signal(SIGTERM, aotx_service_signal); signal(SIGINT, aotx_service_signal);
    signal(SIGHUP, aotx_service_signal); signal(SIGPIPE, SIG_IGN);
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) || getppid() == 1) return 2;
    size_t bytes = (sizeof(aotx_service_ring) + (size_t)AOTX_SERVICE_CHANNELS * sizeof(aotx_service_mailbox) + 4095) & ~(size_t)4095;
    struct stat st;
    if (fstat(fd, &st) || st.st_size != (off_t)bytes) return 2;
    aotx_service_ring *ring = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ring == MAP_FAILED) return 2;
    if (ring->schema != AOTX_SERVICE_SCHEMA || ring->channels != AOTX_SERVICE_CHANNELS ||
        ring->frame_bytes != AOTX_SERVICE_FRAME) { munmap(ring, bytes); return 2; }
    aotx_service_mailbox *m = (aotx_service_mailbox *)((unsigned char *)ring + sizeof *ring);
    aotx_service_peer *peers = calloc(AOTX_SERVICE_CHANNELS, sizeof *peers);
    struct pollfd *polls = calloc(AOTX_SERVICE_CHANNELS, sizeof *polls);
    char path[4096] = {0};
    if (peers) for (unsigned i = 0; i < AOTX_SERVICE_CHANNELS; ++i) peers[i].fd = -1;
    if (!peers || !polls) goto done;
    if (aotx_service_grant_file(grants, m)) goto done;
    uint64_t control_at = aotx_service_now();
    int control = 1;
    while (!aotx_service_stopped && !__atomic_load_n(&ring->closed, __ATOMIC_ACQUIRE)) {
        if (control && __atomic_load_n(&m[0].state, __ATOMIC_ACQUIRE) == 2) {
            if (aotx_service_get(m[0].bytes + 8, 4) != 200) goto done;
            __atomic_store_n(&m[0].state, 0, __ATOMIC_RELEASE); control = 0;
            if (listener < 0) {
                listener = aotx_service_listen(journal, path, sizeof path, &lock);
                if (listener < 0) goto done;
                puts("service: ready"); fflush(stdout);
            }
        }
        if (control && aotx_service_now() - control_at > 30000000000ull) goto done;
        if (aotx_service_reload && !control) {
            aotx_service_reload = 0;
            for (unsigned i = 1; i < AOTX_SERVICE_CHANNELS; ++i) aotx_service_drop(peers + i);
            if (aotx_service_grant_file(grants, m)) goto done;
            control = 1; control_at = aotx_service_now();
        }
        polls[0] = (struct pollfd){control ? -1 : listener, POLLIN, 0};
        for (unsigned i = 1; i < AOTX_SERVICE_CHANNELS; ++i) {
            short events = peers[i].pending ?
                (__atomic_load_n(&m[i].state, __ATOMIC_ACQUIRE) == 2 ? POLLOUT : 0) : POLLIN;
            polls[i] = (struct pollfd){peers[i].fd, events, 0};
        }
        int ready = poll(polls, AOTX_SERVICE_CHANNELS, 5);
        if (ready < 0 && errno != EINTR) goto done;
        if (polls[0].revents & POLLIN) aotx_service_accept(listener, peers, m);
        for (unsigned i = 1; i < AOTX_SERVICE_CHANNELS; ++i)
            aotx_service_peer_step(peers + i, m + i, polls[i].revents);
    }
    bad = 0;
done:
    if (bad) fputs("service: the channel stopped with an error\n", stderr);
    if (peers) for (unsigned i = 1; i < AOTX_SERVICE_CHANNELS; ++i) aotx_service_drop(peers + i);
    if (listener >= 0) { close(listener); unlink(path); }
    if (lock >= 0) close(lock);
    free(peers); free(polls); munmap(ring, bytes); close(fd);
    return bad;
}
