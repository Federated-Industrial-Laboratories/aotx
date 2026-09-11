/* Purpose: Collect selected regular component files and verify their exact bytes.
 * Owns: Read descriptors and required asset index rows.
 * Threading: One packager processes a bounded directory batch.
 * Lifetime: Source descriptors stay open until publication or refusal. */
#include "disk/runtime/pack.h"
#include <stdio.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int aotx_runtime_pack_asset(aotx_runtime_pack *p, const char *path,
                             const char *name, uint32_t kind) {
    if (p->count == AOTX_CCIR_SECTIONS || !aotx_runtime_name(name)) return AOTX_CCIR_LIMIT;
    for (uint32_t i = 0; i < p->index.count; ++i)
        if (!strcmp(name, (char *)p->index.rows[i] + 64)) return AOTX_CCIR_INVALID;
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    struct stat st;
    if (fd < 0) return AOTX_CCIR_IO;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size <= 0) {
        close(fd); return AOTX_CCIR_INVALID;
    }
    aotx_ccir_limits limits; aotx_ccir_default_limits(&limits);
    if ((uint64_t)st.st_size > limits.section_bytes) { close(fd); return AOTX_CCIR_LIMIT; }
    unsigned char digest[32];
    int rc = aotx_ccir_hash_fd(fd, 0, (uint64_t)st.st_size, digest);
    if (rc) { close(fd); return rc; }
    aotx_ccir_input *in = &p->inputs[p->count];
    memset(in, 0, sizeof(*in));
    in->section.type = AOTX_CCIR_ASSET; in->section.schema = 1;
    in->section.flags = AOTX_CCIR_REQUIRED; in->section.alignment = 4096;
    in->section.id[0] = AOTX_CCIR_ASSET;
    aotx_ccir_put(in->section.id + 8, p->index.count + 1, 8);
    in->section.bytes = (uint64_t)st.st_size;
    memcpy(in->section.digest, digest, 32);
    in->source = AOTX_CCIR_FILE; in->fd = fd;
    unsigned char *row = p->index.rows[p->index.count++];
    memset(row, 0, AOTX_RUNTIME_ROW);
    memcpy(row, in->section.id, 16); aotx_ccir_put(row + 16, kind, 4);
    aotx_ccir_put(row + 24, in->section.bytes, 8);
    memcpy(row + 32, digest, 32); strcpy((char *)row + 64, name);
    ++p->count;
    return 0;
}
static int module_kind(const char *path) {
    FILE *in = fopen(path, "r");
    if (!in) return AOTX_CCIR_IO;
    char line[2048]; int found = 0, rc = 0;
    while (fgets(line, sizeof(line), in)) {
        char key[64], value[64], extra;
        if (sscanf(line, " %63[^:]: %63s %c", key, value, &extra) < 2) continue;
        size_t n = strlen(key);
        while (n && (key[n - 1] == ' ' || key[n - 1] == '\t')) key[--n] = 0;
        if (strcmp(key, "kind")) continue;
        if (found++ || (strcmp(value, "role") && strcmp(value, "skill"))) rc = AOTX_CCIR_UNSUPPORTED;
    }
    if (ferror(in)) rc = AOTX_CCIR_IO;
    fclose(in);
    return rc ? rc : found == 1 ? 0 : AOTX_CCIR_INVALID;
}
int aotx_runtime_pack_tree(aotx_runtime_pack *p, const char *root,
                            const char *path, const char *prefix, uint32_t kind) {
    char full[2048]; struct dirent **list = NULL;
    int n = snprintf(full, sizeof(full), "%s/%s", root, path);
    if (n < 0 || (size_t)n >= sizeof(full)) return AOTX_CCIR_LIMIT;
    int count = scandir(full, &list, NULL, alphasort), rc = 0;
    if (count < 0) return AOTX_CCIR_IO;
    for (int i = 0; i < count; ++i) {
        const char *name = list[i]->d_name;
        if (name[0] == '.') { free(list[i]); continue; }
        char child[256], file[2048], key[256]; struct stat st;
        int a = snprintf(child, sizeof(child), "%s%s%s", path, *path ? "/" : "", name);
        int b = snprintf(file, sizeof(file), "%s/%s", root, child);
        int c = snprintf(key, sizeof(key), "%s%s", prefix, child);
        if (!rc && (a < 0 || (size_t)a >= sizeof(child) || b < 0 || (size_t)b >= sizeof(file) ||
            c < 0 || (size_t)c >= sizeof(key))) rc = AOTX_CCIR_LIMIT;
        if (!rc && lstat(file, &st)) rc = AOTX_CCIR_IO;
        if (!rc && S_ISDIR(st.st_mode)) rc = aotx_runtime_pack_tree(p, root, child, prefix, kind);
        else if (!rc) {
            size_t bytes = strlen(name);
            int weight = kind == 1 && bytes >= 5 && !strcmp(name + bytes - 5, ".gguf");
            int manifest = kind == 1 && (!strcmp(child, "manifest.jsonl") ||
                !strcmp(child, "vision.jsonl") || !strcmp(child, "media.profile"));
            if (!weight && !manifest) {
                if (!S_ISREG(st.st_mode)) rc = AOTX_CCIR_INVALID;
                if (!rc && kind == 2 && !strcmp(name, "module.manifest")) rc = module_kind(file);
                if (!rc) rc = aotx_runtime_pack_asset(p, file, key, kind);
            }
        }
        free(list[i]);
    }
    free(list);
    return rc;
}
void aotx_runtime_pack_close(aotx_runtime_pack *p) {
    for (uint32_t i = 0; i < p->count; ++i)
        if (p->inputs[i].source == AOTX_CCIR_FILE && p->inputs[i].fd >= 0) close(p->inputs[i].fd);
    aotx_ccir_close(&p->source); free(p->memory);
}
