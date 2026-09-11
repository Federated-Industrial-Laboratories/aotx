/* Purpose: Perform bounded file IO and checksum operations.
 * Owns: File leases and fixed-size streaming buffers.
 * Threading: One caller for each file operation.
 * Lifetime: One CCIR call. */
#include "disk/ccir/internal.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

uint16_t aotx_ccir_u16(const unsigned char *p)
{ return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)); }
uint32_t aotx_ccir_u32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
uint64_t aotx_ccir_u64(const unsigned char *p)
{ return (uint64_t)aotx_ccir_u32(p) | ((uint64_t)aotx_ccir_u32(p + 4) << 32); }
void aotx_ccir_put(unsigned char *p, uint64_t n, unsigned int bytes)
{
    unsigned int i;
    for (i = 0; i < bytes; i++) p[i] = (unsigned char)(n >> (8u * i));
}
int aotx_ccir_zero(const unsigned char *p, size_t bytes)
{
    size_t i;
    for (i = 0; i < bytes; i++) if (p[i]) return 0;
    return 1;
}
void aotx_ccir_hash(const void *data, size_t bytes, unsigned char out[32])
{
    aotx_sha256 state;
    aotx_sha256_init(&state);
    aotx_sha256_update(&state, data, bytes);
    aotx_sha256_final(&state, out);
}
int aotx_ccir_pread(int fd, void *data, size_t bytes, uint64_t offset)
{
    unsigned char *p = data;
    if (offset > INT64_MAX || bytes > (uint64_t)INT64_MAX - offset)
        return AOTX_CCIR_LIMIT;
    while (bytes) {
        size_t take = bytes < AOTX_CCIR_CHUNK ? bytes : AOTX_CCIR_CHUNK;
        ssize_t n = pread(fd, p, take, (off_t)offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return n == 0 ? AOTX_CCIR_INVALID : AOTX_CCIR_IO;
        p += n; bytes -= (size_t)n; offset += (uint64_t)n;
    }
    return AOTX_CCIR_OK;
}
int aotx_ccir_pwrite(int fd, const void *data, size_t bytes, uint64_t offset)
{
    const unsigned char *p = data;
    if (offset > INT64_MAX || bytes > (uint64_t)INT64_MAX - offset)
        return AOTX_CCIR_LIMIT;
    while (bytes) {
        size_t take = bytes < AOTX_CCIR_CHUNK ? bytes : AOTX_CCIR_CHUNK;
        ssize_t n = pwrite(fd, p, take, (off_t)offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return AOTX_CCIR_IO;
        p += n; bytes -= (size_t)n; offset += (uint64_t)n;
    }
    return AOTX_CCIR_OK;
}
int aotx_ccir_hash_fd(int fd, uint64_t offset, uint64_t bytes, unsigned char out[32])
{
    unsigned char block[AOTX_CCIR_CHUNK];
    aotx_sha256 state;
    aotx_sha256_init(&state);
    while (bytes) {
        size_t take = bytes < sizeof(block) ? (size_t)bytes : sizeof(block);
        int rc = aotx_ccir_pread(fd, block, take, offset);
        if (rc) return rc;
        aotx_sha256_update(&state, block, take);
        offset += take; bytes -= take;
    }
    aotx_sha256_final(&state, out);
    return AOTX_CCIR_OK;
}
int aotx_ccir_lock(const char *path, int write, int create, int *fd)
{
    struct stat st;
    int flags = (write ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK;
    if (create) flags |= O_CREAT | O_EXCL;
    *fd = open(path, flags, 0600);
    if (*fd < 0) return errno == EEXIST ? AOTX_CCIR_EXISTS : AOTX_CCIR_IO;
    if (fstat(*fd, &st) || !S_ISREG(st.st_mode)) {
        close(*fd); *fd = -1; return AOTX_CCIR_INVALID;
    }
    if (write) {
        struct flock owner = {0};
        owner.l_type = F_WRLCK; owner.l_whence = SEEK_SET; owner.l_len = 1;
        if (fcntl(*fd, F_OFD_SETLK, &owner)) {
            int rc = errno == EAGAIN || errno == EACCES ? AOTX_CCIR_BUSY : AOTX_CCIR_IO;
            close(*fd); *fd = -1; return rc;
        }
    }
    if (flock(*fd, (write ? LOCK_EX : LOCK_SH) | LOCK_NB)) {
        int rc = errno == EWOULDBLOCK ? AOTX_CCIR_BUSY : AOTX_CCIR_IO;
        close(*fd); *fd = -1; return rc;
    }
    return AOTX_CCIR_OK;
}
int aotx_ccir_parent_sync(const char *path)
{
    char parent[PATH_MAX];
    char *slash;
    int fd, rc;
    size_t bytes = strlen(path);
    if (!bytes || bytes >= sizeof(parent)) return AOTX_CCIR_LIMIT;
    memcpy(parent, path, bytes + 1u);
    slash = strrchr(parent, '/');
    if (!slash) strcpy(parent, ".");
    else if (slash == parent) slash[1] = 0;
    else *slash = 0;
    fd = open(parent, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return AOTX_CCIR_IO;
    rc = fsync(fd) ? AOTX_CCIR_IO : AOTX_CCIR_OK;
    close(fd);
    return rc;
}
void aotx_ccir_default_limits(aotx_ccir_limits *limits)
{
    limits->file_bytes = AOTX_CCIR_FILE_BYTES ? AOTX_CCIR_FILE_BYTES : INT64_MAX;
    limits->section_bytes = limits->file_bytes;
    limits->sections = AOTX_CCIR_SECTIONS;
}
int aotx_ccir_limits_get(const aotx_ccir_limits *in, aotx_ccir_limits *out)
{
    if (in) *out = *in;
    else aotx_ccir_default_limits(out);
    return out->file_bytes >= AOTX_CCIR_DATA && out->file_bytes <= INT64_MAX &&
           out->section_bytes && out->section_bytes <= out->file_bytes &&
           out->sections >= 2u && out->sections <= AOTX_CCIR_SECTIONS ?
           AOTX_CCIR_OK : AOTX_CCIR_LIMIT;
}
const char *aotx_ccir_status_text(int status)
{
    static const char *const names[] = {"ok", "file IO failed", "invalid file or input",
        "required format is not supported", "size limit exceeded", "file is in use",
        "destination exists", "file changed; read it again"};
    return status >= 0 && status <= AOTX_CCIR_CHANGED ? names[status] : "unknown error";
}
