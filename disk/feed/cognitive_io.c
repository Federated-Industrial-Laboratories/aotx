/* Purpose: Read bounded live memory files and publish their exact bytes.
 * Owns: File leases, framing checks and temporary transfer buffers.
 * Threading: One feeder thread publishes groups of at most 32 records.
 * Lifetime: One file transfer; no source path enters cognitive state. */
#include "disk/feed/cognitive_io.h"
#include "cognitive/io.h"
#include "cognitive/live.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

static uint64_t aotx_live_get(const unsigned char *p, unsigned n) {
    uint64_t value = 0;
    for (unsigned i = 0; i < n; ++i) value |= (uint64_t)p[i] << (i * 8);
    return value;
}
static void aotx_live_put(unsigned char *p, uint64_t value, unsigned n) {
    for (unsigned i = 0; i < n; ++i) p[i] = (unsigned char)(value >> (i * 8));
}
static int aotx_live_word(const unsigned char *line, uint32_t length,
                           uint32_t *at, const char *word) {
    uint32_t n = (uint32_t)strlen(word);
    if (*at > length || n > length - *at || memcmp(line + *at, word, n)) return 0;
    if (*at + n < length && line[*at + n] != ' ' && line[*at + n] != '\t') return 0;
    *at += n;
    while (*at < length && (line[*at] == ' ' || line[*at] == '\t')) ++*at;
    return 1;
}
static unsigned aotx_live_command(const unsigned char *line, uint32_t length,
                                   char path[PATH_MAX], int *valid) {
    static const char *const names[] = {"load", "apply", "bind", "query", "text", "retain"};
    static const unsigned operations[] = {AOTX_LIVE_LOAD, AOTX_LIVE_UPDATE,
        AOTX_LIVE_BIND, AOTX_LIVE_QUERY, AOTX_LIVE_TEXT, AOTX_LIVE_RETAIN};
    uint32_t at = 0;
    path[0] = 0; *valid = 0;
    while (at < length && (line[at] == ' ' || line[at] == '\t')) ++at;
    if (!aotx_live_word(line, length, &at, "memory")) return 0;
    unsigned op = 0;
    for (unsigned i = 0; i < 6; ++i) {
        uint32_t end = at;
        if (aotx_live_word(line, length, &end, names[i])) { op = operations[i]; at = end; break; }
    }
    if (!op || at == length || length - at >= PATH_MAX) return op;
    for (uint32_t i = at; i < length; ++i) if (line[i] < 32 || line[i] == 127) return op;
    memcpy(path, line + at, length - at); path[length - at] = 0;
    *valid = 1;
    return op;
}
static uint32_t aotx_live_row(unsigned op) {
    return op == AOTX_LIVE_BIND ? AOTX_LIVE_BIND_ROW :
        op == AOTX_LIVE_RETAIN ? AOTX_LIVE_RETAIN_ROW : AOTX_LIVE_QUERY_ROW;
}
static int aotx_live_framing(unsigned op, const unsigned char *data, uint32_t bytes) {
    if (op == AOTX_LIVE_UPDATE) {
        if (bytes < AOTX_COG_HEADER || memcmp(data, "AOTXLOG1", 8) ||
            aotx_live_get(data + 8, 4) != AOTX_COG_SCHEMA ||
            aotx_live_get(data + 12, 4) != AOTX_COG_HEADER ||
            aotx_live_get(data + 16, 4) != AOTX_COG_OBJECT ||
            aotx_live_get(data + 80, 8) != bytes) return AOTX_CCIR_INVALID;
        uint64_t count = aotx_live_get(data + 20, 4), payload = aotx_live_get(data + 24, 8);
        return count <= AOTX_COG_OBJECTS && payload <= AOTX_COG_PAYLOAD &&
               bytes == AOTX_COG_HEADER + count * AOTX_COG_OBJECT + payload
            ? AOTX_CCIR_OK : AOTX_CCIR_INVALID;
    }
    uint32_t row = aotx_live_row(op);
    const char *magic = op == AOTX_LIVE_BIND ? "AOTXBND1" : op == AOTX_LIVE_TEXT ? "AOTXTXT1" :
        op == AOTX_LIVE_RETAIN ? "AOTXRTN1" : "AOTXLIV1";
    if (bytes < AOTX_LIVE_HEADER || memcmp(data, magic, 8) ||
        aotx_live_get(data + 12, 4) != AOTX_LIVE_SCHEMA ||
        aotx_live_get(data + 40, 4) != row) return AOTX_CCIR_INVALID;
    uint64_t count = aotx_live_get(data + 8, 4);
    return count && count <= AOTX_RECALL_BATCH && bytes == AOTX_LIVE_HEADER + count * row
        ? AOTX_CCIR_OK : AOTX_CCIR_INVALID;
}
static int aotx_live_read(unsigned op, const char *path, unsigned char **out, uint32_t *bytes) {
    uint32_t cap = op == AOTX_LIVE_UPDATE ? AOTX_COG_IMAGE : AOTX_LIVE_HEADER +
        AOTX_RECALL_BATCH * aotx_live_row(op);
    uint32_t minimum = op == AOTX_LIVE_UPDATE ? AOTX_COG_HEADER : AOTX_LIVE_HEADER;
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) return AOTX_CCIR_IO;
    struct stat st;
    int status = AOTX_CCIR_OK;
    if (fstat(fd, &st)) status = AOTX_CCIR_IO;
    else if (!S_ISREG(st.st_mode) || st.st_size < minimum || st.st_size > cap) status = AOTX_CCIR_INVALID;
    else if (flock(fd, LOCK_SH | LOCK_NB)) status = AOTX_CCIR_BUSY;
    unsigned char *data = status ? NULL : malloc((size_t)st.st_size);
    if (!status && !data) status = AOTX_CCIR_IO;
    size_t done = 0;
    while (!status && done < (size_t)st.st_size) {
        ssize_t got = read(fd, data + done, (size_t)st.st_size - done);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) { status = AOTX_CCIR_IO; break; }
        done += (size_t)got;
    }
    if (!status) {
        unsigned char extra;
        ssize_t got;
        do { got = read(fd, &extra, 1); } while (got < 0 && errno == EINTR);
        if (got != 0) status = got < 0 ? AOTX_CCIR_IO : AOTX_CCIR_INVALID;
    }
    close(fd);
    if (!status) status = aotx_live_framing(op, data, (uint32_t)done);
    if (status) { free(data); return status; }
    *out = data; *bytes = (uint32_t)done;
    return AOTX_CCIR_OK;
}
static int aotx_live_id(unsigned char id[16]) {
    size_t at = 0;
    while (at < 16) {
        ssize_t got = getrandom(id + at, 16 - at, 0);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return AOTX_CCIR_IO;
        at += (size_t)got;
    }
    unsigned any = 0;
    for (unsigned i = 0; i < 16; ++i) any |= id[i];
    return any ? AOTX_CCIR_OK : AOTX_CCIR_IO;
}
static int aotx_live_publish(unsigned op, const unsigned char id[16],
                              const unsigned char *data, uint32_t bytes,
                              const aotx_inbound_ring *ring, const volatile sig_atomic_t *stop) {
    aotx_record_header headers[AOTX_LINE_PARTS_MAX];
    unsigned char parts[AOTX_LINE_PARTS_MAX][AOTX_BODY_BYTES];
    const void *bodies[AOTX_LINE_PARTS_MAX];
    uint32_t at = 0;
    while (at < bytes) {
        uint32_t count = 0;
        while (count < AOTX_LINE_PARTS_MAX && at < bytes) {
            uint32_t n = bytes - at;
            if (n > AOTX_LIVE_DATA) n = AOTX_LIVE_DATA;
            aotx_record_header *h = &headers[count];
            unsigned char *part = parts[count];
            memset(h, 0, sizeof(*h)); memset(part, 0, AOTX_BODY_BYTES);
            h->writer = AOTX_WRITER_FEEDER; h->cls = AOTX_CLASS_A;
            h->type = AOTX_LIVE_RECORD; h->body_len = AOTX_LIVE_PART + n;
            aotx_live_put(part, AOTX_LIVE_SCHEMA, 4); aotx_live_put(part + 4, op, 4);
            memcpy(part + 8, id, 16);
            aotx_live_put(part + 24, bytes, 4); aotx_live_put(part + 28, at, 4);
            memcpy(part + AOTX_LIVE_PART, data + at, n);
            bodies[count++] = part; at += n;
        }
        if (aotx_line_publish_records(ring, stop, headers, bodies, count)) return -1;
    }
    return 1;
}
static int aotx_live_refuse(unsigned op, int status, const aotx_inbound_ring *ring,
                             const volatile sig_atomic_t *stop) {
    static const char *const names[] = {"", "load", "apply", "bind", "query", "", "text", "", "retain"};
    char line[AOTX_BODY_BYTES];
    const char *reason = aotx_ccir_status_text(status);
    fprintf(stderr, "memory %s refused: %s\n", names[op], reason);
    int n = snprintf(line, sizeof(line), "note memory %s refused: %s", names[op], reason);
    if (n < 0 || (size_t)n >= sizeof(line)) return -1;
    return aotx_line_publish(ring, stop, (const unsigned char *)line, (uint32_t)n) ? -1 : 1;
}
int aotx_live_feed_line(const unsigned char *line, uint32_t length,
                         const aotx_inbound_ring *ring, const volatile sig_atomic_t *stop) {
    char path[PATH_MAX];
    int valid;
    unsigned op = aotx_live_command(line, length, path, &valid);
    if (!op) return 0;
    if (!valid) return aotx_live_refuse(op, AOTX_CCIR_INVALID, ring, stop);
    unsigned char *data = NULL, id[16];
    uint32_t bytes = 0;
    int status;
    aotx_cognitive_file file;
    if (op == AOTX_LIVE_LOAD) {
        status = aotx_cognitive_file_open(path, &file);
        if (!status) {
            bytes = (uint32_t)(16 + file.checkpoint_bytes + file.tail_bytes);
            data = malloc(bytes);
            if (!data) status = AOTX_CCIR_IO;
            else {
                aotx_live_put(data, file.checkpoint_bytes, 8); aotx_live_put(data + 8, file.tail_bytes, 8);
                memcpy(data + 16, file.checkpoint, (size_t)file.checkpoint_bytes);
                if (file.tail_bytes) memcpy(data + 16 + file.checkpoint_bytes, file.tail, (size_t)file.tail_bytes);
            }
        }
    } else status = aotx_live_read(op, path, &data, &bytes);
    if (!status) status = aotx_live_id(id);
    int result = status ? aotx_live_refuse(op, status, ring, stop)
                        : aotx_live_publish(op, id, data, bytes, ring, stop);
    free(data);
    if (op == AOTX_LIVE_LOAD) aotx_cognitive_file_close(&file);
    return result;
}
