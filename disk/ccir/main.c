/* Purpose: Pack, inspect, verify, append and compact CCIR data-state files.
 * Owns: Command arguments and bounded input file descriptors.
 * Threading: One command operates on section or file batches.
 * Lifetime: Input descriptors close before the program returns. */
#include "disk/ccir/ccir.h"
#include "disk/runtime/runtime.h"
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void usage(void)
{
    puts("aotx_ccir pack FILE CHECKPOINT LINEAGE CHECKPOINT_SEQ DURABLE_SEQ TICK [TAIL]\n"
         "aotx_ccir append FILE CHECKPOINT CHECKPOINT_SEQ DURABLE_SEQ TICK [TAIL]\n"
         "aotx_ccir inspect FILE...\n"
         "aotx_ccir verify FILE...\n"
         "aotx_ccir compact SOURCE DESTINATION\n"
         "LINEAGE is a nonzero 32-digit hexadecimal ID.\n"
         "Inputs contain encoded state bytes. Pack does not validate GPU state.\n"
         "Required sections use the data-state profile. No code is loaded.");
}
static int number(const char *text, uint64_t *out)
{
    uint64_t value = 0u;
    if (!*text) return 0;
    for (; *text; text++) {
        unsigned int digit = (unsigned int)(*text - '0');
        if (digit > 9u || value > (UINT64_MAX - digit) / 10u) return 0;
        value = value * 10u + digit;
    }
    *out = value;
    return 1;
}
static int hex(unsigned char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static int identity(const char *text, unsigned char id[16])
{
    unsigned int i, any = 0u;
    if (strlen(text) != 32u) return 0;
    for (i = 0; i < 16u; i++) {
        int high = hex((unsigned char)text[2u * i]);
        int low = hex((unsigned char)text[2u * i + 1u]);
        if (high < 0 || low < 0) return 0;
        id[i] = (unsigned char)(high * 16 + low); any |= id[i];
    }
    return any != 0u;
}
static void id_text(const unsigned char id[16], char out[33])
{
    unsigned int i;
    static const char digits[] = "0123456789abcdef";
    for (i = 0; i < 16u; i++) {
        out[2u * i] = digits[id[i] >> 4]; out[2u * i + 1u] = digits[id[i] & 15u];
    }
    out[32] = 0;
}
static int show(char **paths, int count, int inspect)
{
    int n, result = 0;
    for (n = 0; n < count; n++) {
        aotx_ccir_view view;
        char lineage[33], incarnation[33];
        uint32_t i;
        int rc = aotx_ccir_open(paths[n], NULL, &view);
        if (rc) {
            fprintf(stderr, "file %d: %s\n", n, aotx_ccir_status_text(rc));
            result = rc; continue;
        }
        for (uint32_t j = 0; j < view.count && !rc; ++j)
            if (view.sections[j].type == AOTX_CCIR_RUNTIME &&
                (view.sections[j].flags & AOTX_CCIR_REQUIRED)) rc = aotx_runtime_dependencies(&view);
        if (rc) {
            fprintf(stderr, "file %d: runtime dependencies refused: %s\n", n, aotx_ccir_status_text(rc));
            result = rc; aotx_ccir_close(&view); continue;
        }
        id_text(view.lineage, lineage); id_text(view.incarnation, incarnation);
        printf("file %d: verified generation %llu sections %u checkpoint %llu durable %llu "
               "tick %llu fallback %u trailing %llu\n", n,
               (unsigned long long)view.generation, view.count,
               (unsigned long long)view.meta.checkpoint_sequence,
               (unsigned long long)view.meta.durable_sequence,
               (unsigned long long)view.meta.source_tick, view.fallback,
               (unsigned long long)view.trailing_bytes);
        if (inspect) {
            printf("lineage %s incarnation %s\n", lineage, incarnation);
            for (i = 0; i < view.count; i++) {
                const aotx_ccir_section *s = &view.sections[i];
                char id[33]; id_text(s->id, id);
                printf("section %u type %u schema %u required %u id %s offset %llu bytes %llu\n",
                       i, s->type, s->schema, s->flags, id,
                       (unsigned long long)s->offset, (unsigned long long)s->bytes);
            }
        }
        aotx_ccir_close(&view);
    }
    return result;
}
static int file_input(const char *path, aotx_ccir_input *input)
{
    struct stat st;
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) return AOTX_CCIR_IO;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size <= 0) {
        close(fd); return AOTX_CCIR_INVALID;
    }
    input->source = AOTX_CCIR_FILE; input->fd = fd;
    input->section.bytes = (uint64_t)st.st_size;
    return AOTX_CCIR_OK;
}

static void tail_id(const aotx_ccir_view *old, unsigned char id[16])
{
    uint32_t candidate, i;
    memset(id, 0, 16u);
    for (candidate = 1u; candidate <= old->count + 1u; candidate++) {
        id[0] = (unsigned char)candidate; id[1] = (unsigned char)(candidate >> 8u);
        for (i = 0; i < old->count; i++)
            if (!memcmp(id, old->sections[i].id, 16u)) break;
        if (i == old->count) return;
    }
}

static int retained(const char *path, int has_tail, aotx_ccir_input *inputs,
                     uint32_t *count, aotx_ccir_revision *expected)
{
    aotx_ccir_view old;
    uint32_t i;
    int old_tail = 0, rc = aotx_ccir_open(path, NULL, &old);
    if (rc) return rc;
    *count = has_tail ? 3u : 2u;
    memcpy(expected->prologue_digest, old.prologue_digest, 32u);
    memcpy(expected->commit_digest, old.commit_digest, 32u);
    for (i = 0; i < old.count; i++) {
        const aotx_ccir_section *s = &old.sections[i];
        if (s->type <= AOTX_CCIR_TAIL) {
            if (s->type == AOTX_CCIR_TAIL) {
                old_tail = 1;
                if (!has_tail) continue;
            }
            memcpy(inputs[s->type - 1u].section.id, s->id, 16u);
        } else {
            if (*count == AOTX_CCIR_SECTIONS) { rc = AOTX_CCIR_LIMIT; break; }
            inputs[*count].section = *s;
            inputs[(*count)++].source = AOTX_CCIR_REUSE;
        }
    }
    if (!rc && has_tail && !old_tail) tail_id(&old, inputs[2].section.id);
    aotx_ccir_close(&old);
    return rc;
}

static int pack(int argc, char **argv, int append)
{
    aotx_ccir_input inputs[AOTX_CCIR_SECTIONS];
    aotx_ccir_meta meta;
    aotx_ccir_revision expected;
    unsigned char manifest[AOTX_CCIR_MANIFEST_BYTES], lineage[16] = {0};
    uint32_t count = 2u, i;
    int rc, checkpoint_fd = -1, tail_fd = -1;
    int offset = append ? 4 : 5;
    int has_tail = argc == offset + 4;
    if (argc != offset + 3 && !has_tail) return AOTX_CCIR_INVALID;
    if (!number(argv[offset], &meta.checkpoint_sequence) ||
        !number(argv[offset + 1], &meta.durable_sequence) ||
        !number(argv[offset + 2], &meta.source_tick) ||
        (!append && !identity(argv[4], lineage))) return AOTX_CCIR_INVALID;
    memset(inputs, 0, sizeof(inputs));
    for (i = 0; i < 3u; i++) {
        inputs[i].section.type = i + 1u; inputs[i].section.schema = 1u;
        inputs[i].section.flags = AOTX_CCIR_REQUIRED;
        inputs[i].section.id[0] = (unsigned char)(i + 1u);
        inputs[i].section.alignment = 128u;
    }
    if (append) {
        rc = retained(argv[2], has_tail, inputs, &count, &expected);
        if (rc) return rc;
    } else count = has_tail ? 3u : 2u;
    aotx_ccir_manifest(manifest, inputs[1].section.id,
                        has_tail ? inputs[2].section.id : NULL);
    inputs[0].data = manifest; inputs[0].section.bytes = sizeof(manifest);
    rc = file_input(argv[3], &inputs[1]);
    if (rc) return rc;
    checkpoint_fd = inputs[1].fd;
    if (has_tail) {
        rc = file_input(argv[offset + 3], &inputs[2]);
        if (!rc) tail_fd = inputs[2].fd;
    }
    if (!rc) {
        if (append) rc = aotx_ccir_append_if(argv[2], &expected, inputs, count, &meta, NULL);
        else rc = aotx_ccir_create(argv[2], lineage, inputs, count, &meta, NULL);
    }
    close(checkpoint_fd);
    if (tail_fd >= 0) close(tail_fd);
    return rc;
}
int main(int argc, char **argv)
{
    int rc;
    if (argc == 2 && !strcmp(argv[1], "--help")) { usage(); return 0; }
    if (argc >= 3 && (!strcmp(argv[1], "inspect") || !strcmp(argv[1], "verify")))
        return show(argv + 2, argc - 2, !strcmp(argv[1], "inspect"));
    if (argc >= 2 && !strcmp(argv[1], "pack")) rc = pack(argc, argv, 0);
    else if (argc >= 2 && !strcmp(argv[1], "append")) rc = pack(argc, argv, 1);
    else if (argc == 4 && !strcmp(argv[1], "compact"))
        rc = aotx_ccir_compact(argv[2], argv[3], NULL);
    else { usage(); return AOTX_CCIR_INVALID; }
    if (rc) fprintf(stderr, "ccir: %s\n", aotx_ccir_status_text(rc));
    else puts("ccir: complete");
    return rc;
}
