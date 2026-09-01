/* Purpose: Read and change the local model store and its two record files.
 * Owns: Nothing; each operation opens and closes its files.
 * Threading: One process changes a store at a time.
 * Lifetime: One call. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/models/models.h"

#include "disk/modelfile/manifest.h"
#include "disk/wire/diskwire.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define AOTX_STORE_NAME "store.jsonl"
#define AOTX_HASH_ROOM  (1024u * 1024u)

static void say(char *reason, size_t bytes, const char *text)
{
    if (reason != NULL && bytes != 0u) {
        snprintf(reason, bytes, "%s", text);
    }
}

static int join(char *out, size_t bytes, const char *dir, const char *name)
{
    int wrote = snprintf(out, bytes, "%s/%s", dir, name);
    return (wrote < 0 || (size_t)wrote >= bytes) ? -1 : 0;
}

static int text_field(const char *line, const char *key, char *out, size_t bytes)
{
    char mark[64];
    int wrote = snprintf(mark, sizeof(mark), "\"%s\":\"", key);
    return (wrote > 0 && (size_t)wrote < sizeof(mark))
         ? !aotx_json_text(line, mark, out, bytes) : 1;
}

static int digest_ok(const char *text)
{
    unsigned int i;
    for (i = 0u; i < 64u; i++) {
        if (!((text[i] >= '0' && text[i] <= '9') ||
              (text[i] >= 'a' && text[i] <= 'f'))) {
            return 0;
        }
    }
    return text[64] == '\0';
}

int aotx_model_store_line(const char *line, aotx_model_store_record *record)
{
    uint64_t bytes = 0u;
    memset(record, 0, sizeof(*record));
    if (text_field(line, "name", record->name, sizeof(record->name)) ||
        text_field(line, "file", record->file, sizeof(record->file)) ||
        !aotx_json_number(line, "\"bytes\":", &bytes) ||
        text_field(line, "sha256", record->sha256, sizeof(record->sha256)) ||
        text_field(line, "source", record->source, sizeof(record->source)) ||
        text_field(line, "date", record->date, sizeof(record->date)) ||
        text_field(line, "revision", record->revision, sizeof(record->revision)) ||
        strstr(line, "\"verified\":true") == NULL || !digest_ok(record->sha256)) {
        return -1;
    }
    record->bytes = bytes;
    record->verified = 1;
    return (record->name[0] != '\0' && record->file[0] != '\0' && bytes != 0u) ? 0 : -1;
}

int aotx_model_store_write_line(char *out, size_t bytes,
                                const aotx_model_store_record *record)
{
    int wrote;
    if (record == NULL || !digest_ok(record->sha256) || record->name[0] == '\0' ||
        record->file[0] == '\0' || record->bytes == 0u || record->source[0] == '\0' ||
        record->date[0] == '\0' || record->revision[0] == '\0') {
        return -1;
    }
    wrote = snprintf(out, bytes,
                     "{\"name\":\"%s\",\"file\":\"%s\",\"bytes\":%llu,"
                     "\"sha256\":\"%s\",\"source\":\"%s\",\"date\":\"%s\","
                     "\"revision\":\"%s\",\"verified\":true}\n",
                     record->name, record->file, (unsigned long long)record->bytes,
                     record->sha256, record->source, record->date, record->revision);
    return (wrote < 0 || (size_t)wrote >= bytes) ? -1 : 0;
}

int aotx_model_store_read(const char *dir, aotx_model_store_record *records,
                          unsigned int most)
{
    char path[AOTX_MODEL_PATH];
    char line[AOTX_MODEL_LINE];
    FILE *file;
    unsigned int count = 0u;
    if (join(path, sizeof(path), dir, AOTX_STORE_NAME) != 0) {
        return -1;
    }
    file = fopen(path, "r");
    if (file == NULL) {
        return (errno == ENOENT) ? 0 : -1;
    }
    while (fgets(line, (int)sizeof(line), file) != NULL) {
        aotx_model_store_record hold;
        unsigned int i;
        if (strlen(line) + 1u == sizeof(line) || aotx_model_store_line(line, &hold) != 0) {
            fclose(file);
            return -1;
        }
        for (i = 0u; i < count; i++) {
            if (strcmp(records[i].name, hold.name) == 0) {
                records[i] = hold;
                break;
            }
        }
        if (i == count) {
            if (count >= most) {
                fclose(file);
                return -1;
            }
            records[count++] = hold;
        }
    }
    fclose(file);
    return (int)count;
}

int aotx_model_store_append(const char *dir, const aotx_model_store_record *record,
                            char *reason, size_t reason_bytes)
{
    char path[AOTX_MODEL_PATH];
    char line[AOTX_MODEL_LINE];
    int fd;
    if (join(path, sizeof(path), dir, AOTX_STORE_NAME) != 0 ||
        aotx_model_store_write_line(line, sizeof(line), record) != 0) {
        say(reason, reason_bytes, "the local store line does not fit");
        return -1;
    }
    fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0 || write(fd, line, strlen(line)) != (ssize_t)strlen(line) || fsync(fd) != 0) {
        if (fd >= 0) {
            close(fd);
        }
        say(reason, reason_bytes, "the local store record does not write");
        return -1;
    }
    close(fd);
    return 0;
}

static int manifest_read_optional(const char *dir, aotx_manifest_entry *entry)
{
    char path[AOTX_MODEL_PATH];
    if (join(path, sizeof(path), dir, AOTX_MANIFEST_NAME) != 0) {
        return -1;
    }
    if (access(path, F_OK) != 0 && errno == ENOENT) {
        return 0;
    }
    return aotx_manifest_read(dir, entry, AOTX_MANIFEST_MAX);
}

static const aotx_manifest_entry *manifest_file(const aotx_manifest_entry *entry, int count,
                                                 const char *name, const char *role,
                                                 const char *file)
{
    int i;
    for (i = 0; i < count; i++) {
        if (strcmp(entry[i].name, name) == 0 && strcmp(entry[i].role, role) == 0
            && strcmp(entry[i].path, file) == 0) {
            return &entry[i];
        }
    }
    return NULL;
}

static const aotx_model_store_record *local_name(const aotx_model_store_record *record,
                                                  int count, const char *name)
{
    int i;
    for (i = 0; i < count; i++) {
        if (strcmp(record[i].name, name) == 0) {
            return &record[i];
        }
    }
    return NULL;
}

static int known_file(const aotx_model_catalog *catalog, const char *file)
{
    unsigned int i;
    for (i = 0u; i < catalog->count; i++) {
        if (strcmp(catalog->entry[i].file, file) == 0) {
            return 1;
        }
    }
    return 0;
}

static int known_disk_name(const aotx_model_catalog *catalog, const char *file)
{
    char name[AOTX_MODEL_TEXT];
    size_t bytes = strlen(file);
    if (known_file(catalog, file)) {
        return 1;
    }
    if (bytes > 5u && strcmp(file + bytes - 5u, ".part") == 0 &&
        bytes - 5u < sizeof(name)) {
        memcpy(name, file, bytes - 5u);
        name[bytes - 5u] = '\0';
        return known_file(catalog, name);
    }
    return 0;
}

static int suffix(const char *text, const char *end)
{
    size_t a = strlen(text);
    size_t b = strlen(end);
    return a >= b && strcmp(text + a - b, end) == 0;
}

int aotx_model_store_scan(const char *dir, const aotx_model_catalog *catalog,
                          aotx_model_view *views, unsigned int most,
                          char *reason, size_t reason_bytes)
{
    aotx_manifest_entry manifest[AOTX_MANIFEST_MAX];
    aotx_model_store_record local[AOTX_MODEL_CATALOG_MAX];
    int manifest_count = manifest_read_optional(dir, manifest);
    int local_count = aotx_model_store_read(dir, local, AOTX_MODEL_CATALOG_MAX);
    unsigned int count = 0u;
    unsigned int i;
    DIR *open_dir;
    struct dirent *at;
    if (manifest_count < 0 || local_count < 0) {
        say(reason, reason_bytes, "a store record does not read");
        return -1;
    }
    for (i = 0u; i < catalog->count && count < most; i++) {
        const aotx_model_catalog_entry *known = &catalog->entry[i];
        const aotx_manifest_entry *active = manifest_file(manifest, manifest_count,
                                                           known->name, known->role,
                                                           known->file);
        const aotx_model_store_record *saved = local_name(local, local_count, known->name);
        char path[AOTX_MODEL_PATH];
        char part[AOTX_MODEL_PATH];
        struct stat info;
        aotx_model_view *view = &views[count++];
        memset(view, 0, sizeof(*view));
        view->catalog = *known;
        view->verified = known->verified;
        if (join(path, sizeof(path), dir, known->file) != 0 ||
            snprintf(part, sizeof(part), "%s.part", path) >= (int)sizeof(part)) {
            view->state = AOTX_MODEL_NOT_FETCHED;
            continue;
        }
        if (stat(path, &info) == 0 && S_ISREG(info.st_mode)) {
            int meta_ok = (uint64_t)info.st_size == known->bytes;
            view->bytes_on_disk = (uint64_t)info.st_size;
            if (active != NULL) {
                meta_ok = meta_ok && active->bytes == known->bytes &&
                          strcmp(active->sha256, known->sha256) == 0;
                view->state = meta_ok ? AOTX_MODEL_ON_DISK : AOTX_MODEL_DIGEST_DIFFERS;
            } else {
                if (saved != NULL) {
                    meta_ok = meta_ok && saved->bytes == known->bytes &&
                              strcmp(saved->sha256, known->sha256) == 0;
                    view->verified = meta_ok && saved->verified;
                }
                view->state = meta_ok ? AOTX_MODEL_NOT_ACTIVE : AOTX_MODEL_DIGEST_DIFFERS;
            }
        } else if (stat(part, &info) == 0 && S_ISREG(info.st_mode)) {
            view->state = AOTX_MODEL_FETCHING;
            view->bytes_on_disk = (uint64_t)info.st_size;
        } else {
            view->state = AOTX_MODEL_NOT_FETCHED;
        }
    }
    open_dir = opendir(dir);
    if (open_dir == NULL) {
        say(reason, reason_bytes, "the models directory does not open");
        return -1;
    }
    while ((at = readdir(open_dir)) != NULL && count < most) {
        char path[AOTX_MODEL_PATH];
        struct stat info;
        aotx_model_view *view;
        if ((!suffix(at->d_name, ".gguf") && !suffix(at->d_name, ".gguf.part")) ||
            known_disk_name(catalog, at->d_name) ||
            join(path, sizeof(path), dir, at->d_name) != 0 ||
            stat(path, &info) != 0 || !S_ISREG(info.st_mode)) {
            continue;
        }
        view = &views[count++];
        memset(view, 0, sizeof(*view));
        snprintf(view->catalog.name, sizeof(view->catalog.name), "%.*s",
                 (int)sizeof(view->catalog.name) - 1, at->d_name);
        snprintf(view->catalog.file, sizeof(view->catalog.file), "%s", at->d_name);
        snprintf(view->catalog.role, sizeof(view->catalog.role), "-");
        snprintf(view->catalog.quant, sizeof(view->catalog.quant), "-");
        snprintf(view->catalog.source, sizeof(view->catalog.source), "-");
        view->bytes_on_disk = (uint64_t)info.st_size;
        view->state = suffix(at->d_name, ".part") ? AOTX_MODEL_FETCHING
                                                   : AOTX_MODEL_NOT_ACTIVE;
    }
    closedir(open_dir);
    if (aotx_model_parameters_scan(dir, catalog, reason, reason_bytes) != 0) return -1;
    return (int)count;
}

const char *aotx_model_state_text(enum aotx_model_state state)
{
    switch (state) {
    case AOTX_MODEL_ON_DISK:        return "on disk";
    case AOTX_MODEL_NOT_ACTIVE:     return "on disk, not in the manifest";
    case AOTX_MODEL_FETCHING:       return "fetching";
    case AOTX_MODEL_DIGEST_DIFFERS: return "digest differs";
    default:                        return "not fetched";
    }
}

int aotx_model_store_check(const char *dir, char *reason, size_t reason_bytes)
{
    aotx_manifest_entry entry[AOTX_MANIFEST_MAX];
    int count = aotx_manifest_read(dir, entry, AOTX_MANIFEST_MAX);
    int i;
    if (count < 0) {
        say(reason, reason_bytes, "the models manifest does not read");
        return -1;
    }
    for (i = 0; i < count; i++) {
        int state = aotx_manifest_check(dir, &entry[i]);
        if (state != 0) {
            if (reason != NULL && reason_bytes != 0u) {
                snprintf(reason, reason_bytes, "the digest of %s differs", entry[i].path);
            }
            return -1;
        }
    }
    return count;
}

static int hash_entry(const char *dir, const aotx_model_catalog_entry *entry,
                      char *reason, size_t reason_bytes)
{
    char path[AOTX_MODEL_PATH];
    char digest[AOTX_SHA256_HEX];
    uint64_t bytes = 0u;
    void *room = malloc(AOTX_HASH_ROOM);
    int state;
    if (room == NULL || join(path, sizeof(path), dir, entry->file) != 0) {
        free(room);
        say(reason, reason_bytes, "the memory or path for the digest is not available");
        return -1;
    }
    state = aotx_sha256_file(path, digest, &bytes, room, AOTX_HASH_ROOM);
    free(room);
    if (state != 0) {
        say(reason, reason_bytes, "the model file does not read");
        return -1;
    }
    if (bytes != entry->bytes || strcmp(digest, entry->sha256) != 0) {
        say(reason, reason_bytes, "the model digest differs from the catalog");
        return -1;
    }
    return 0;
}

int aotx_model_store_activate(const char *dir, const aotx_model_catalog_entry *entry,
                              const char *role, char *reason, size_t reason_bytes)
{
    aotx_manifest_entry old[AOTX_MANIFEST_MAX];
    aotx_manifest_entry fresh;
    char target[AOTX_MODEL_PATH];
    char temp[AOTX_MODEL_PATH];
    int count = manifest_read_optional(dir, old);
    int fd;
    int i;
    int placed = 0;
    if (entry == NULL || strcmp(role, entry->role) != 0) {
        say(reason, reason_bytes, "the role does not match the catalog entry");
        return -1;
    }
    if (strlen(entry->name) >= sizeof(fresh.name) ||
        strlen(entry->source) >= sizeof(fresh.source) ||
        strlen(entry->license) >= sizeof(fresh.license)) {
        say(reason, reason_bytes, "a catalog field does not fit the models manifest");
        return -1;
    }
    if (count < 0 || hash_entry(dir, entry, reason, reason_bytes) != 0 ||
        join(target, sizeof(target), dir, AOTX_MANIFEST_NAME) != 0 ||
        snprintf(temp, sizeof(temp), "%s/.manifest-XXXXXX", dir) >= (int)sizeof(temp)) {
        return -1;
    }
    memset(&fresh, 0, sizeof(fresh));
    memcpy(fresh.name, entry->name, strlen(entry->name) + 1u);
    snprintf(fresh.role, sizeof(fresh.role), "%s", role);
    snprintf(fresh.path, sizeof(fresh.path), "%s", entry->file);
    memcpy(fresh.source, entry->source, strlen(entry->source) + 1u);
    snprintf(fresh.revision, sizeof(fresh.revision), "%s", entry->revision);
    memcpy(fresh.license, entry->license, strlen(entry->license) + 1u);
    fresh.bytes = entry->bytes;
    snprintf(fresh.sha256, sizeof(fresh.sha256), "%s", entry->sha256);
    fd = mkstemp(temp);
    if (fd < 0) {
        say(reason, reason_bytes, "the temporary manifest does not open");
        return -1;
    }
    /* One manifest line for each role: an old entry with this role or this name leaves. */
    for (i = 0; i < count + (placed == 0); i++) {
        char line[AOTX_MANIFEST_LINE];
        const aotx_manifest_entry *write_entry;
        if (i < count && strcmp(old[i].name, entry->name) != 0 &&
            strcmp(old[i].role, role) != 0) {
            write_entry = &old[i];
        } else if (!placed) {
            write_entry = &fresh;
            placed = 1;
        } else {
            continue;
        }
        if (aotx_manifest_write_line(line, sizeof(line), write_entry) != 0 ||
            write(fd, line, strlen(line)) != (ssize_t)strlen(line)) {
            close(fd);
            unlink(temp);
            say(reason, reason_bytes, "the models manifest does not write");
            return -1;
        }
    }
    if (fsync(fd) != 0 || close(fd) != 0 || rename(temp, target) != 0) {
        unlink(temp);
        say(reason, reason_bytes, "the models manifest does not replace");
        return -1;
    }
    return 0;
}

int aotx_model_store_remove(const char *dir, const aotx_model_catalog_entry *entry,
                            char *reason, size_t reason_bytes)
{
    aotx_manifest_entry manifest[AOTX_MANIFEST_MAX];
    char path[AOTX_MODEL_PATH];
    int count = manifest_read_optional(dir, manifest);
    int i;
    if (entry == NULL || count < 0 || join(path, sizeof(path), dir, entry->file) != 0) {
        say(reason, reason_bytes, "the model name or store does not read");
        return -1;
    }
    for (i = 0; i < count; i++) {
        if (strcmp(manifest[i].path, entry->file) == 0) {
            say(reason, reason_bytes, "the models manifest names this file");
            return -1;
        }
    }
    if (unlink(path) != 0) {
        say(reason, reason_bytes, (errno == ENOENT) ? "the model file is not there"
                                                    : "the model file does not remove");
        return -1;
    }
    return 0;
}
