/* Purpose: Validate exact policy bundle bytes before native code admission.
 * Owns: Bounded read buffers and full-file trust comparison.
 * Threading: One caller processes the complete image and metadata extent batch.
 * Lifetime: Owned buffers end at file close; no code is loaded here. */
#include "disk/policy/file.h"
#include "disk/ccir/internal.h"
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static uint64_t capacity(void) {
    return AOTX_POLICY_FILE_HEADER + (uint64_t)AOTX_POLICY_IMAGE_BYTES +
        2ull * AOTX_POLICY_METADATA_BYTES;
}
static int entry_valid(const unsigned char *name) {
    const unsigned char *end = memchr(name, 0, 64);
    if (!end || end == name || !aotx_ccir_zero(end, 64 - (size_t)(end - name))) return 0;
    for (const unsigned char *p = name; p != end; ++p)
        if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || *p == '_' ||
            (p != name && *p >= '0' && *p <= '9'))) return 0;
    return 1;
}
static int framing(const unsigned char *h, uint64_t bytes, aotx_policy_config *c) {
    if (bytes < AOTX_POLICY_FILE_HEADER || memcmp(h, "AOTXPL01", 8) ||
        !aotx_ccir_zero(h + 176, 80)) return AOTX_CCIR_INVALID;
    c->abi = aotx_ccir_u32(h + 16);
    if (aotx_ccir_u32(h + 8) != 1 ||
        (c->abi != AOTX_POLICY_ABI && c->abi != AOTX_POLICY_APPRAISAL_ABI && c->abi != AOTX_POLICY_REVIEW_ABI))
        return AOTX_CCIR_UNSUPPORTED;
    c->mode = aotx_ccir_u32(h + 12); c->state_schema = aotx_ccir_u32(h + 20);
    c->state_bytes = aotx_ccir_u32(h + 24); c->architecture = aotx_ccir_u32(h + 28);
    c->threads = aotx_ccir_u32(h + 32); c->registers = aotx_ccir_u32(h + 36);
    c->shared_bytes = aotx_ccir_u32(h + 40); c->local_bytes = aotx_ccir_u32(h + 44);
    c->pressure = aotx_ccir_u32(h + 48); c->minimum_move = aotx_ccir_u32(h + 52);
    c->backoff = aotx_ccir_u32(h + 56); c->format = aotx_ccir_u32(h + 60);
    uint64_t image = aotx_ccir_u64(h + 64);
    uint32_t provenance = aotx_ccir_u32(h + 72), license = aotx_ccir_u32(h + 76);
    if (!c->state_schema || !c->state_bytes || !c->threads || c->threads > 1024 ||
        c->pressure > 100 || !c->minimum_move || !c->backoff || !provenance || !license)
        return AOTX_CCIR_INVALID;
    if (c->state_bytes > AOTX_POLICY_STATE_BYTES || image > AOTX_POLICY_IMAGE_BYTES ||
        provenance > AOTX_POLICY_METADATA_BYTES || license > AOTX_POLICY_METADATA_BYTES ||
        bytes > SIZE_MAX - 1 || bytes > capacity()) return AOTX_CCIR_LIMIT;
    if (bytes != AOTX_POLICY_FILE_HEADER + image + provenance + license) return AOTX_CCIR_INVALID;
    if (c->mode == AOTX_POLICY_SUPPLIED || c->mode == AOTX_POLICY_RULES) {
        if (c->state_schema != 1 || c->state_bytes != 16) return AOTX_CCIR_UNSUPPORTED;
        if (c->architecture || c->threads != 64 || c->registers || c->shared_bytes ||
            c->local_bytes || c->format || image || !aotx_ccir_zero(h + 80, 96))
            return AOTX_CCIR_INVALID;
    } else if (c->mode == AOTX_POLICY_NATIVE) {
        if (c->format != 1 && c->format != 2) return AOTX_CCIR_UNSUPPORTED;
        if (!c->architecture || !c->registers || !image || !entry_valid(h + 80))
            return AOTX_CCIR_INVALID;
    } else return AOTX_CCIR_UNSUPPORTED;
    return 0;
}
static int content(aotx_policy_file *out) {
    const unsigned char *h = out->buffer;
    int rc = framing(h, out->buffer_bytes, &out->config);
    if (rc) return rc;
    out->image_bytes = (size_t)aotx_ccir_u64(h + 64);
    out->provenance_bytes = aotx_ccir_u32(h + 72); out->license_bytes = aotx_ccir_u32(h + 76);
    out->provenance = h + AOTX_POLICY_FILE_HEADER + out->image_bytes;
    out->license = out->provenance + out->provenance_bytes;
    if (memchr(out->provenance, 0, out->provenance_bytes) || memchr(out->license, 0, out->license_bytes))
        return AOTX_CCIR_INVALID;
    memcpy(out->entry, h + 80, sizeof(out->entry));
    if (out->image_bytes) {
        unsigned char digest[32];
        aotx_ccir_hash(h + AOTX_POLICY_FILE_HEADER, out->image_bytes, digest);
        if (memcmp(digest, h + 144, 32)) return AOTX_POLICY_FILE_DIGEST;
        if (out->config.format == 1 && memchr(h + AOTX_POLICY_FILE_HEADER, 0, out->image_bytes - 1))
            return AOTX_CCIR_INVALID;
        out->image = malloc(out->image_bytes + 1);
        if (!out->image) return AOTX_CCIR_IO;
        memcpy(out->image, h + AOTX_POLICY_FILE_HEADER, out->image_bytes);
        out->image[out->image_bytes] = 0;
    }
    aotx_ccir_hash(h, out->buffer_bytes, out->digest);
    return 0;
}
void aotx_policy_file_close(aotx_policy_file *file) {
    if (!file) return;
    free(file->image); free(file->buffer); memset(file, 0, sizeof(*file));
}
int aotx_policy_file_decode(const void *bytes, size_t length, aotx_policy_file *out) {
    if (!out) return AOTX_CCIR_INVALID;
    memset(out, 0, sizeof(*out));
    if (!bytes || length < AOTX_POLICY_FILE_HEADER) return AOTX_CCIR_INVALID;
    int rc = framing(bytes, length, &out->config);
    if (rc) { memset(out, 0, sizeof(*out)); return rc; }
    out->buffer = malloc(length);
    if (!out->buffer) return AOTX_CCIR_IO;
    memcpy(out->buffer, bytes, length); out->buffer_bytes = length;
    rc = content(out);
    if (rc) aotx_policy_file_close(out);
    return rc;
}
int aotx_policy_file_extent(int fd, uint64_t offset, uint64_t bytes, aotx_policy_file *out) {
    if (!out) return AOTX_CCIR_INVALID;
    memset(out, 0, sizeof(*out));
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) return AOTX_CCIR_IO;
    if (!S_ISREG(st.st_mode) || st.st_size < 0 || offset > (uint64_t)st.st_size ||
        bytes > (uint64_t)st.st_size - offset || bytes < AOTX_POLICY_FILE_HEADER)
        return AOTX_CCIR_INVALID;
    if (bytes > capacity() || bytes > SIZE_MAX - 1) return AOTX_CCIR_LIMIT;
    unsigned char h[AOTX_POLICY_FILE_HEADER];
    int rc = aotx_ccir_pread(fd, h, sizeof(h), offset);
    if (!rc) rc = framing(h, bytes, &out->config);
    if (rc) { memset(out, 0, sizeof(*out)); return rc; }
    out->buffer = malloc((size_t)bytes);
    if (!out->buffer) return AOTX_CCIR_IO;
    out->buffer_bytes = (size_t)bytes;
    rc = aotx_ccir_pread(fd, out->buffer, (size_t)bytes, offset);
    if (!rc) rc = content(out);
    if (rc) aotx_policy_file_close(out);
    return rc;
}
static int hex(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static int trusted(const char *text, const unsigned char digest[32]) {
    if (!text || strlen(text) != 64) return 0;
    unsigned difference = 0;
    for (unsigned i = 0; i < 32; ++i) {
        int a = hex((unsigned char)text[2 * i]), b = hex((unsigned char)text[2 * i + 1]);
        if (a < 0 || b < 0) return 0;
        difference |= ((unsigned)a * 16u + (unsigned)b) ^ digest[i];
    }
    return !difference;
}
int aotx_policy_file_read(const char *path, const char *trust, int require_trust,
    aotx_policy_file *out) {
    if (!out) return AOTX_CCIR_INVALID;
    memset(out, 0, sizeof(*out));
    if (!path || !*path || strlen(path) >= PATH_MAX) return AOTX_CCIR_INVALID;
    int fd = -1, rc = aotx_ccir_lock(path, 0, 0, &fd);
    if (rc) return rc;
    struct stat st;
    rc = fstat(fd, &st) || st.st_size < 0 ? AOTX_CCIR_IO :
        aotx_policy_file_extent(fd, 0, (uint64_t)st.st_size, out);
    close(fd);
    if (!rc && (trust || (require_trust && out->config.mode == AOTX_POLICY_NATIVE)) &&
        !trusted(trust, out->digest)) rc = AOTX_POLICY_FILE_TRUST;
    if (rc) aotx_policy_file_close(out);
    return rc;
}
const char *aotx_policy_status_text(int status) {
    if (status == AOTX_POLICY_FILE_TRUST) return "exact policy digest is not trusted";
    if (status == AOTX_POLICY_FILE_DIGEST) return "policy image digest does not match";
    return aotx_ccir_status_text(status);
}
