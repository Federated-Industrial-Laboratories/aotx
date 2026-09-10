/* Purpose: Transfer bounded recall files and encode committed context rows.
 * Owns: Request framing, output file metadata and JSON byte escaping.
 * Threading: One caller retains the source lease until output completes.
 * Lifetime: One explicit select or replay command. */
#include "cognitive/recall_io.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

static uint64_t aotx_recall_le(const unsigned char *p, unsigned bytes) {
    uint64_t value = 0;
    for (unsigned i = 0; i < bytes; ++i) value |= (uint64_t)p[i] << (8 * i);
    return value;
}
static int aotx_recall_read(const char *path, aotx_recall_file *file) {
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) return AOTX_CCIR_IO;
    struct stat st;
    int status = AOTX_CCIR_OK;
    if (flock(fd, LOCK_SH | LOCK_NB)) status = AOTX_CCIR_BUSY;
    else if (fstat(fd, &st)) status = AOTX_CCIR_IO;
    else if (!S_ISREG(st.st_mode) || st.st_size < AOTX_RECALL_HEADER ||
             st.st_size > AOTX_RECALL_REQUESTS) status = AOTX_CCIR_INVALID;
    size_t done = 0;
    while (!status && done < (size_t)st.st_size) {
        ssize_t got = read(fd, file->requests + done, (size_t)st.st_size - done);
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
    if (status) return status;
    const unsigned char *p = file->requests;
    uint32_t count = (uint32_t)aotx_recall_le(p + 8, 4);
    if (memcmp(p, "AOTXREQ1", 8) || !count || count > AOTX_RECALL_BATCH ||
        aotx_recall_le(p + 12, 4) != 1 || aotx_recall_le(p + 40, 4) != AOTX_RECALL_QUERY ||
        done != AOTX_RECALL_HEADER + (uint64_t)count * AOTX_RECALL_QUERY) return AOTX_CCIR_INVALID;
    file->count = count; file->request_bytes = done;
    return AOTX_CCIR_OK;
}

int aotx_recall_file_open(const char *path, const char *requests, aotx_recall_file *file) {
    memset(file, 0, sizeof(*file));
    int status = aotx_cognitive_file_open(path, &file->source);
    if (status) return status;
    file->requests = calloc(1, AOTX_RECALL_REQUESTS);
    file->checkpoint = malloc(AOTX_COG_IMAGE);
    file->rows = calloc(AOTX_RECALL_BATCH, sizeof(*file->rows));
    if (!file->requests || !file->checkpoint || !file->rows) status = AOTX_CCIR_IO;
    if (!status && requests) status = aotx_recall_read(requests, file);
    if (status) aotx_recall_file_close(file);
    return status;
}

int aotx_recall_file_write(aotx_recall_file *file, const char *path, uint64_t bytes) {
    const unsigned char *checkpoint = file->checkpoint;
    const aotx_cognitive_file *source = &file->source;
    if (bytes < AOTX_COG_HEADER || bytes > AOTX_COG_IMAGE ||
        memcmp(checkpoint, "AOTXOBJ1", 8) || memcmp(checkpoint + 48, source->view.lineage, 16) ||
        aotx_recall_le(checkpoint + 80, 8) != bytes) return AOTX_CCIR_INVALID;
    aotx_ccir_meta meta;
    meta.checkpoint_sequence = aotx_recall_le(checkpoint + 32, 8);
    meta.durable_sequence = meta.checkpoint_sequence;
    meta.source_tick = aotx_recall_le(checkpoint + 40, 8);
    if (meta.durable_sequence <= source->view.meta.durable_sequence ||
        meta.source_tick <= source->view.meta.source_tick) return AOTX_CCIR_INVALID;
    aotx_ccir_input inputs[AOTX_CCIR_SECTIONS];
    memset(inputs, 0, sizeof(inputs));
    unsigned char manifest[AOTX_CCIR_MANIFEST_BYTES];
    aotx_ccir_manifest(manifest, source->view.sections[source->checkpoint_index].id, NULL);
    manifest[20] = checkpoint[8];
    uint32_t count = 0;
    for (uint32_t i = 0; i < source->view.count; ++i) {
        if (i == source->tail_index) continue;
        aotx_ccir_input *in = &inputs[count++];
        in->section = source->view.sections[i];
        in->source = AOTX_CCIR_FILE; in->fd = source->view.fd;
        in->source_offset = in->section.offset;
        if (i == source->checkpoint_index) {
            in->source = AOTX_CCIR_MEMORY; in->data = checkpoint; in->section.bytes = bytes;
            in->section.schema = (uint16_t)aotx_recall_le(checkpoint + 8, 4);
        } else if (i == source->manifest_index) {
            in->source = AOTX_CCIR_MEMORY; in->data = manifest;
        }
    }
    return aotx_ccir_create(path, source->view.lineage, inputs, count, &meta, NULL);
}

static void aotx_recall_hex(const unsigned char *bytes, unsigned count) {
    for (unsigned i = 0; i < count; ++i) printf("%02x", bytes[i]);
}
static void aotx_recall_json(const unsigned char *bytes, uint32_t count) {
    putchar('"');
    for (uint32_t i = 0; i < count; ++i) {
        unsigned char value = bytes[i];
        if (value == '"' || value == '\\') { putchar('\\'); putchar(value); }
        else if (value < 32) printf("\\u%04x", value);
        else putchar(value);
    }
    putchar('"');
}
int aotx_recall_file_rows(const aotx_recall_file *file) {
    if (!file->count || file->count > AOTX_RECALL_BATCH) return AOTX_CCIR_INVALID;
    for (uint32_t i = 0; i < file->count; ++i) {
        const aotx_recall_result *r = &file->rows[i];
        if (r->status) return 200 + (int)r->status;
        if (r->count > AOTX_RECALL_LIMIT || r->context_bytes > AOTX_RECALL_CONTEXT)
            return AOTX_CCIR_INVALID;
    }
    for (uint32_t i = 0; i < file->count; ++i) {
        const aotx_recall_result *r = &file->rows[i];
        fputs("{\"request\":\"", stdout); aotx_recall_hex(r->request_id, 16);
        fputs("\",\"selection\":\"", stdout); aotx_recall_hex(r->selection_id, 16);
        printf("\",\"cut\":%llu,\"count\":%u,\"searches\":%u,\"context\":",
               (unsigned long long)r->cut, r->count, r->searches);
        aotx_recall_json(r->context, r->context_bytes); fputs("}\n", stdout);
    }
    return fflush(stdout) || ferror(stdout) ? AOTX_CCIR_IO : AOTX_CCIR_OK;
}
void aotx_recall_file_close(aotx_recall_file *file) {
    aotx_cognitive_file_close(&file->source);
    free(file->requests); free(file->checkpoint); free(file->rows);
    file->requests = file->checkpoint = NULL; file->rows = NULL;
}
int aotx_recall_file_options(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        puts("Use: aotx_ccir_recall select INPUT REQUESTS OUTPUT\n"
             "     aotx_ccir_recall replay INPUT\n"
             "Select prepared GPU memory and write a new CCIR before context output.\n"
             "Replay uses the recorded choices without vector search.\n"
             "The output must not exist. This command does not load a language model.");
        return 2;
    }
    if (argc == 5 && !strcmp(argv[1], "select")) return 0;
    if (argc == 3 && !strcmp(argv[1], "replay")) return 1;
    fputs("Use: aotx_ccir_recall select INPUT REQUESTS OUTPUT\n"
          "     aotx_ccir_recall replay INPUT\n", stderr);
    return -1;
}
void aotx_recall_file_report(int status) {
    static const char *const state[] = {
        "valid", "invalid format", "capacity limit", "invalid reference", "scope conflict",
        "version conflict", "sequence conflict", "source conflict", "invalid media layout",
        "missing object", "stale object", "access denied"};
    if (!status) return;
    const char *reason = status >= 200 && status < 212 ? state[status - 200] :
        status == 100 ? "device transfer or launch failed" : aotx_ccir_status_text(status);
    fprintf(stderr, "recall failed: %s (%d)\n", reason, status);
}
