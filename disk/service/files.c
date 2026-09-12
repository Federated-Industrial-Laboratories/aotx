/* Purpose: Read an operator grant table and hold the local socket path lease.
 * Owns: Bounded regular-file reads and the socket lock descriptor.
 * Threading: One broker transfer owner.
 * Lifetime: Grant reload or socket lifetime. */
#define _GNU_SOURCE
#include "disk/service/broker.h"
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
uint64_t aotx_service_now(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) return 0;
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
int aotx_service_grant_file(const char *path, aotx_service_mailbox *m)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return 1;
    struct stat st;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_uid != geteuid() ||
        (st.st_mode & 0022) || st.st_size < AOTX_SERVICE_HEAD || st.st_size > AOTX_SERVICE_FRAME ||
        flock(fd, LOCK_SH | LOCK_NB)) { close(fd); return 1; }
    size_t at = 0, n = (size_t)st.st_size;
    while (at < n) {
        ssize_t got = read(fd, m->bytes + at, n - at);
        if (got <= 0) { close(fd); return 1; }
        at += (size_t)got;
    }
    unsigned char extra;
    int bad = read(fd, &extra, 1) != 0 || memcmp(m->bytes, AOTX_SERVICE_MAGIC, 8) ||
        aotx_service_get(m->bytes + 8, 4) != AOTX_SERVICE_GRANTS ||
        aotx_service_get(m->bytes + 88, 4) != n - AOTX_SERVICE_HEAD;
    close(fd);
    if (bad) return 1;
    m->length = n; ++m->generation;
    __atomic_store_n(&m->state, 1, __ATOMIC_RELEASE);
    return 0;
}
int aotx_service_listen(const char *journal, char *path, size_t bytes, int *lock)
{
    struct sockaddr_un address;
    memset(&address, 0, sizeof address); address.sun_family = AF_UNIX;
    char lease[4096];
    if (snprintf(path, bytes, "%s/service.sock", journal) >= (int)bytes ||
        strlen(path) >= sizeof address.sun_path ||
        snprintf(lease, sizeof lease, "%s/service.lock", journal) >= (int)sizeof lease) return -1;
    *lock = open(lease, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (*lock < 0) return -1;
    struct stat st;
    if (fstat(*lock, &st) || !S_ISREG(st.st_mode) || st.st_uid != geteuid() ||
        (st.st_mode & 0077) || flock(*lock, LOCK_EX | LOCK_NB)) return -1;
    if (lstat(path, &st) == 0 && (!S_ISSOCK(st.st_mode) || st.st_uid != geteuid())) return -1;
    unlink(path); strcpy(address.sun_path, path);
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    mode_t previous = umask(0077);
    int bad = bind(fd, (struct sockaddr *)&address, sizeof address);
    umask(previous);
    if (bad || listen(fd, AOTX_SERVICE_CHANNELS)) { close(fd); return -1; }
    return fd;
}
