/* Purpose: Create and inspect policy bundles without loading native code.
 * Owns: Command options and bounded image and metadata input buffers.
 * Threading: One process handles the complete input file batch.
 * Lifetime: Input buffers end before process exit. */
#include "disk/policy/file.h"
#include "disk/ccir/internal.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *const options[] = {
    "--output", "--mode", "--image", "--format", "--kernel", "--architecture",
    "--state-schema", "--state-bytes", "--threads", "--registers", "--shared-bytes",
    "--local-bytes", "--pressure", "--minimum-move", "--backoff", "--provenance", "--license", "--abi"
};
static int number(const char *text, uint32_t *out) {
    if (!text || *text < '0' || *text > '9') return AOTX_CCIR_INVALID;
    char *end = NULL; errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || *end || value > UINT32_MAX) return AOTX_CCIR_INVALID;
    *out = (uint32_t)value;
    return 0;
}
static int load(const char *path, size_t capacity, void **out, size_t *bytes) {
    *out = NULL; *bytes = 0;
    int fd = -1, rc = aotx_ccir_lock(path, 0, 0, &fd);
    if (rc) return rc;
    struct stat st;
    if (fstat(fd, &st) || st.st_size <= 0) rc = AOTX_CCIR_INVALID;
    if (!rc && (uint64_t)st.st_size > capacity) rc = AOTX_CCIR_LIMIT;
    if (!rc) {
        *bytes = (size_t)st.st_size; *out = malloc(*bytes);
        rc = *out ? aotx_ccir_pread(fd, *out, *bytes, 0) : AOTX_CCIR_IO;
    }
    close(fd);
    if (rc) { free(*out); *out = NULL; *bytes = 0; }
    return rc;
}
static int inspect(const char *path) {
    aotx_policy_file file;
    int rc = aotx_policy_file_read(path, NULL, 0, &file);
    if (rc) return rc;
    const aotx_policy_config *c = &file.config;
    char digest[65], image[65];
    aotx_sha256_text(file.digest, digest); aotx_sha256_text(file.buffer + 144, image);
    printf("policy_digest=%s\nmode=%u\nabi=%u\nstate_schema=%u\nstate_bytes=%u\n"
           "architecture=%u\nthreads=%u\nregisters=%u\nshared_bytes=%u\nlocal_bytes=%u\n"
           "pressure=%u\nminimum_move=%u\nbackoff=%u\nformat=%u\nentry=%s\n"
           "image_bytes=%zu\nimage_digest=%s\nprovenance_bytes=%u\nlicense_bytes=%u\n",
           digest, c->mode, c->abi, c->state_schema, c->state_bytes,
           c->architecture, c->threads, c->registers, c->shared_bytes, c->local_bytes,
           c->pressure, c->minimum_move, c->backoff, c->format, file.entry,
           file.image_bytes, image, file.provenance_bytes, file.license_bytes);
    aotx_policy_file_close(&file);
    return ferror(stdout) ? AOTX_CCIR_IO : 0;
}
static int create(const char *const *value) {
    aotx_policy_source s = {0};
    aotx_policy_config *c = &s.config;
    c->state_schema = 1; c->state_bytes = 16; c->threads = 64;
    c->minimum_move = 1; c->backoff = 1; c->abi = AOTX_POLICY_ABI;
    if (value[17] && number(value[17], &c->abi)) return AOTX_CCIR_INVALID;
    if (c->abi != AOTX_POLICY_ABI && c->abi != AOTX_POLICY_APPRAISAL_ABI) return AOTX_CCIR_UNSUPPORTED;
    if (!strcmp(value[1], "supplied")) c->mode = AOTX_POLICY_SUPPLIED;
    else if (!strcmp(value[1], "rules")) c->mode = AOTX_POLICY_RULES;
    else if (!strcmp(value[1], "native")) c->mode = AOTX_POLICY_NATIVE;
    else return AOTX_CCIR_INVALID;
    if (value[3]) {
        if (!strcmp(value[3], "ptx")) c->format = 1;
        else if (!strcmp(value[3], "cubin")) c->format = 2;
        else return AOTX_CCIR_INVALID;
    }
    uint32_t *fields[] = {&c->architecture, &c->state_schema, &c->state_bytes,
        &c->threads, &c->registers, &c->shared_bytes, &c->local_bytes, &c->pressure,
        &c->minimum_move, &c->backoff};
    for (unsigned i = 0; i < 10; ++i)
        if (value[i + 5] && number(value[i + 5], fields[i])) return AOTX_CCIR_INVALID;
    if (c->mode == AOTX_POLICY_NATIVE) {
        for (unsigned i = 2; i <= 11; ++i) if (!value[i]) return AOTX_CCIR_INVALID;
    } else if (value[2] || value[3] || value[4] || value[5] || value[9] || value[10] || value[11])
        return AOTX_CCIR_INVALID;
    s.entry = value[4];
    void *image = NULL, *provenance = NULL, *license = NULL;
    int rc = value[2] ? load(value[2], AOTX_POLICY_IMAGE_BYTES, &image, &s.image_bytes) : 0;
    if (!rc) rc = load(value[15], AOTX_POLICY_METADATA_BYTES, &provenance, &s.provenance_bytes);
    if (!rc) rc = load(value[16], AOTX_POLICY_METADATA_BYTES, &license, &s.license_bytes);
    s.image = image; s.provenance = provenance; s.license = license;
    if (!rc) rc = aotx_policy_file_write(value[0], &s);
    free(image); free(provenance); free(license);
    if (!rc) rc = inspect(value[0]);
    return rc;
}
int main(int argc, char **argv) {
    int rc;
    if (argc == 3 && !strcmp(argv[1], "--inspect")) rc = inspect(argv[2]);
    else {
        const char *value[sizeof(options) / sizeof(options[0])] = {0};
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc) goto usage;
            unsigned at = 0;
            while (at < sizeof(value) / sizeof(value[0]) && strcmp(argv[i], options[at])) ++at;
            if (at == sizeof(value) / sizeof(value[0]) || value[at]) goto usage;
            value[at] = argv[i + 1];
        }
        if (!value[0] || !value[1] || !value[15] || !value[16]) goto usage;
        rc = create(value);
    }
    if (rc) fprintf(stderr, "policy file operation refused: %s (%d)\n", aotx_policy_status_text(rc), rc);
    return rc ? 1 : 0;
usage:
    fputs("Use: aotx_policy_pack --inspect FILE\n"
          "     aotx_policy_pack --output FILE --mode supplied|rules|native\n"
          "                      --provenance FILE --license FILE\n"
          "                      [--abi 1|2] [--pressure N] [--minimum-move N] [--backoff N]\n"
          "Native mode also requires all options below.\n"
          "  --image FILE --format ptx|cubin --kernel NAME --architecture N\n"
          "  --state-schema N --state-bytes N --threads N --registers N\n"
          "  --shared-bytes N --local-bytes N\n"
          "Data modes use state schema 1, 16 state bytes, and 64 threads.\n"
          "Inspection does not execute code or grant trust.\n", stderr);
    return 2;
}
