/* Purpose: Discover sampling controls declared by model metadata.
 * Owns: The local parameters.jsonl catalog during one replacement.
 * Threading: One process scans one store at a time.
 * Lifetime: One store scan. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/models/models.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define AOTX_PARAMETERS_LINE 4096u

static int is_gguf(const char *path)
{
    unsigned char magic[4];
    int fd = open(path, O_RDONLY);
    ssize_t got = fd >= 0 ? read(fd, magic, sizeof magic) : -1;
    if (fd >= 0) close(fd);
    return got == (ssize_t)sizeof magic && memcmp(magic, "GGUF", sizeof magic) == 0;
}

static int add_real(char *out, size_t bytes, int *used, const char *name,
                    const aotx_modelfile *file, int *first)
{
    char key[128]; float value[3];
    static const char *part[3] = { "default", "min", "max" };
    for (unsigned int i = 0u; i < 3u; ++i) {
        snprintf(key, sizeof key, "sampling.%s.%s", name, part[i]);
        if (aotx_modelfile_f32(file, key, &value[i]) != 0) return 0;
    }
    int wrote = snprintf(out + *used, bytes - (size_t)*used,
        "%s\"%s\":{\"default\":%.9g,\"min\":%.9g,\"max\":%.9g}",
        *first ? "" : ",", name, (double)value[0], (double)value[1], (double)value[2]);
    if (wrote < 0 || (size_t)wrote >= bytes - (size_t)*used) return -1;
    *used += wrote; *first = 0; return 1;
}

static int add_whole(char *out, size_t bytes, int *used, const char *name,
                     const aotx_modelfile *file, int *first)
{
    char key[128]; uint32_t value[3];
    static const char *part[3] = { "default", "min", "max" };
    for (unsigned int i = 0u; i < 3u; ++i) {
        snprintf(key, sizeof key, "sampling.%s.%s", name, part[i]);
        if (aotx_modelfile_u32(file, key, &value[i]) != 0) return 0;
    }
    int wrote = snprintf(out + *used, bytes - (size_t)*used,
        "%s\"%s\":{\"default\":%u,\"min\":%u,\"max\":%u}",
        *first ? "" : ",", name, value[0], value[1], value[2]);
    if (wrote < 0 || (size_t)wrote >= bytes - (size_t)*used) return -1;
    *used += wrote; *first = 0; return 1;
}

int aotx_model_parameters_line(const char *path, const char *name, char *out, size_t bytes)
{
    static const char *real[] = { "temperature", "top_p", "min_p", "repeat_penalty",
                                  "presence_penalty", "frequency_penalty" };
    static const char *whole[] = { "top_k", "repeat_window" };
    aotx_modelfile *file = NULL;
    if (aotx_modelfile_open(path, &file) != 0) return -1;
    int used = snprintf(out, bytes, "{\"name\":\"%s\",\"parameters\":{", name);
    int first = 1;
    if (used < 0 || (size_t)used >= bytes) { aotx_modelfile_close(file); return -1; }
    for (unsigned int i = 0u; i < sizeof real / sizeof real[0]; ++i)
        if (add_real(out, bytes, &used, real[i], file, &first) < 0) used = -1;
    for (unsigned int i = 0u; used >= 0 && i < sizeof whole / sizeof whole[0]; ++i)
        if (add_whole(out, bytes, &used, whole[i], file, &first) < 0) used = -1;
    aotx_modelfile_close(file);
    if (used < 0 || first) return 0;
    int wrote = snprintf(out + used, bytes - (size_t)used, "}}\n");
    return wrote < 0 || (size_t)wrote >= bytes - (size_t)used ? -1 : used + wrote;
}

int aotx_model_parameters_scan(const char *dir, const aotx_model_catalog *catalog,
                               char *reason, size_t reason_bytes)
{
    char temp[AOTX_MODEL_PATH], target[AOTX_MODEL_PATH], path[AOTX_MODEL_PATH];
    snprintf(temp, sizeof temp, "%s/.parameters-XXXXXX", dir);
    snprintf(target, sizeof target, "%s/parameters.jsonl", dir);
    int fd = mkstemp(temp);
    if (fd < 0) return -1;
    for (unsigned int i = 0u; i < catalog->count; ++i) {
        struct stat info; char line[AOTX_PARAMETERS_LINE];
        snprintf(path, sizeof path, "%s/%s", dir, catalog->entry[i].file);
        if (stat(path, &info) != 0 || !S_ISREG(info.st_mode) || !is_gguf(path)) continue;
        int count = aotx_model_parameters_line(path, catalog->entry[i].name,
                                                line, sizeof line);
        if (count < 0 || (count > 0 && write(fd, line, (size_t)count) != count)) {
            close(fd); unlink(temp);
            if (reason && reason_bytes) snprintf(reason, reason_bytes,
                "the model parameter catalog does not write");
            return -1;
        }
    }
    if (fsync(fd) != 0 || close(fd) != 0 || rename(temp, target) != 0) {
        unlink(temp); return -1;
    }
    return 0;
}
