/* Purpose: Check runtime dependencies and materialize only required disk metadata.
 * Owns: Exclusive temporary files for the settings and data module readers.
 * Threading: One boot process; model bytes remain bounded CCIR extents.
 * Lifetime: Temporary metadata ends with the boot process. */
#include "disk/runtime/activate.h"
#include "disk/runtime/replay.h"
#include "disk/ccir/internal.h"
#include "cognitive/format.h"
#include "profile/profile.cuh"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int parents(const char *path) {
    char name[2048];
    if (strlen(path) >= sizeof(name)) return AOTX_CCIR_LIMIT;
    strcpy(name, path);
    for (char *at = name + 1; *at; ++at) if (*at == '/') {
        *at = 0; struct stat st;
        if (mkdir(name, 0700) && (errno != EEXIST || lstat(name, &st) || !S_ISDIR(st.st_mode)))
            return AOTX_CCIR_IO;
        *at = '/';
    }
    return 0;
}
static void remove_tree(const char *path) {
    DIR *dir = opendir(path);
    if (!dir) return;
    struct dirent *at;
    while ((at = readdir(dir))) {
        if (!strcmp(at->d_name, ".") || !strcmp(at->d_name, "..")) continue;
        char file[2048]; struct stat st;
        int n = snprintf(file, sizeof(file), "%s/%s", path, at->d_name);
        if (n < 0 || (size_t)n >= sizeof(file) || lstat(file, &st)) continue;
        if (S_ISDIR(st.st_mode)) remove_tree(file); else unlink(file);
    }
    closedir(dir); rmdir(path);
}
void aotx_runtime_release(aotx_runtime_boot *boot) {
    if (boot && boot->owned && boot->root[0]) { remove_tree(boot->root); boot->root[0] = 0; boot->owned = 0; }
}
static int materialize(const aotx_ccir_view *view, const aotx_runtime_index *index,
                        aotx_runtime_boot *boot) {
    unsigned char buffer[65536];
    unsigned settings = 0, modules = 0;
    for (uint32_t i = 0; i < index->count; ++i) {
        const unsigned char *row = index->rows[i];
        const char *name = (const char *)row + 64;
        if (strcmp(name, "settings") && aotx_ccir_u32(row + 16) != 2) continue;
        if (aotx_ccir_u32(row + 16) == 2 && strncmp(name, "modules/", 8)) return AOTX_CCIR_INVALID;
        int at = aotx_runtime_section(view, row);
        const aotx_ccir_section *s = view->sections + at;
        char file[2048]; int n = snprintf(file, sizeof(file), "%s/%s", boot->root, name);
        if (n < 0 || (size_t)n >= sizeof(file)) return AOTX_CCIR_LIMIT;
        int rc = parents(file);
        if (rc) return rc;
        int fd = open(file, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) return AOTX_CCIR_IO;
        for (uint64_t cursor = 0; !rc && cursor < s->bytes;) {
            size_t bytes = s->bytes - cursor < sizeof(buffer) ? (size_t)(s->bytes - cursor) : sizeof(buffer);
            rc = aotx_ccir_pread(view->fd, buffer, bytes, s->offset + cursor);
            if (!rc) rc = aotx_ccir_pwrite(fd, buffer, bytes, cursor);
            cursor += bytes;
        }
        close(fd);
        if (rc) return rc;
        settings += !strcmp(name, "settings"); modules += aotx_ccir_u32(row + 16) == 2;
    }
    return settings == 1 && modules ? 0 : AOTX_CCIR_INVALID;
}
int aotx_runtime_prepare(const char *path, const char *journal, unsigned architecture,
                          aotx_runtime_boot *boot) {
    if (!path || !journal || !boot) return AOTX_CCIR_INVALID;
    memset(boot, 0, sizeof(*boot));
    aotx_ccir_view view;
    int rc = aotx_ccir_open(path, NULL, &view);
    if (rc) return rc;
    aotx_runtime_index *index = malloc(sizeof(*index));
    if (!index) { aotx_ccir_close(&view); return AOTX_CCIR_IO; }
    rc = aotx_runtime_index_read(view.fd, &view, index);
    if (!rc) rc = aotx_runtime_dependencies(&view);
    unsigned features = 0;
#ifdef AOTX_AFFECT
    features = AOTX_RUNTIME_AFFECT;
#endif
    unsigned char *h = index->header;
    if (!rc && (aotx_ccir_u32(h + 20) != features || aotx_ccir_u32(h + 24) != AOTX_WIRE_LAYOUT ||
        aotx_ccir_u32(h + 28) > AOTX_SLOTS || aotx_ccir_u32(h + 32) > AOTX_COG_OBJECTS ||
        aotx_ccir_u64(h + 40) > AOTX_COG_PAYLOAD || aotx_ccir_u32(h + 36) > architecture)) rc = AOTX_CCIR_UNSUPPORTED;
    unsigned char replay[128];
    if (!rc) {
        int at = aotx_runtime_section(&view, h + 128);
        rc = aotx_runtime_replay_header(view.fd, view.sections + at, replay);
    }
    if (!rc) {
        boot->mode = aotx_ccir_u32(replay + 12);
        aotx_runtime_revision(&view, boot->revision);
    }
    if (!rc) {
        strcpy(boot->roles, (char *)h + 64);
        int n = snprintf(boot->root, sizeof(boot->root), "%s/.ccir-XXXXXX", journal);
        if (n < 0 || (size_t)n >= sizeof(boot->root)) rc = AOTX_CCIR_LIMIT;
        if (!rc) rc = parents(boot->root);
        if (!rc) {
            if (!mkdtemp(boot->root)) rc = AOTX_CCIR_IO;
            else boot->owned = 1;
        }
        if (!rc) rc = materialize(&view, index, boot);
        if (!rc) {
            n = snprintf(boot->modules, sizeof(boot->modules), "%s/modules", boot->root);
            if (n < 0 || (size_t)n >= sizeof(boot->modules)) rc = AOTX_CCIR_LIMIT;
            n = snprintf(boot->settings, sizeof(boot->settings), "%s/settings", boot->root);
            if (n < 0 || (size_t)n >= sizeof(boot->settings)) rc = AOTX_CCIR_LIMIT;
        }
    }
    aotx_ccir_close(&view); free(index);
    if (rc) aotx_runtime_release(boot);
    return rc;
}
