/* Purpose: Read the repository catalog of known model files.
 * Owns: Nothing; the caller owns the catalog table.
 * Threading: One thread for each read.
 * Lifetime: One call. */
#include "disk/models/models.h"

#include "disk/wire/diskwire.h"

#include <stdio.h>
#include <string.h>

static void say(char *reason, size_t bytes, const char *text)
{
    if (reason != NULL && bytes != 0u) {
        snprintf(reason, bytes, "%s", text);
    }
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

static int text_field(const char *line, const char *key, char *out, size_t bytes)
{
    char mark[64];
    int wrote = snprintf(mark, sizeof(mark), "\"%s\":\"", key);
    if (wrote < 0 || (size_t)wrote >= sizeof(mark)) {
        return 0;
    }
    return aotx_json_text(line, mark, out, bytes);
}

static int role_ok(const char *role)
{
    return strcmp(role, "language") == 0 || strcmp(role, "language-q4") == 0 ||
           strcmp(role, "embedding") == 0 || strcmp(role, "reranker") == 0;
}

static int base_name(const char *name)
{
    return name[0] != '\0' && name[0] != '.' && strchr(name, '/') == NULL &&
           strchr(name, '\\') == NULL && strstr(name, "..") == NULL;
}

static int read_line(const char *line, aotx_model_catalog_entry *entry)
{
    uint64_t bytes = 0u;
    const char *tail = line + strlen(line);
    memset(entry, 0, sizeof(*entry));
    while (tail > line && (tail[-1] == '\n' || tail[-1] == '\r' || tail[-1] == ' ')) {
        tail--;
    }
    if (line[0] != '{' || tail == line || tail[-1] != '}' || !aotx_json_whole(line) ||
        !text_field(line, "name", entry->name, sizeof(entry->name)) ||
        !text_field(line, "role", entry->role, sizeof(entry->role)) ||
        !text_field(line, "repository", entry->repository, sizeof(entry->repository)) ||
        !text_field(line, "file", entry->file, sizeof(entry->file)) ||
        !text_field(line, "revision", entry->revision, sizeof(entry->revision)) ||
        !aotx_json_number(line, "\"bytes\":", &bytes) ||
        !text_field(line, "sha256", entry->sha256, sizeof(entry->sha256)) ||
        !text_field(line, "license", entry->license, sizeof(entry->license)) ||
        !text_field(line, "quant", entry->quant, sizeof(entry->quant)) ||
        !text_field(line, "profiles", entry->profiles, sizeof(entry->profiles)) ||
        !text_field(line, "source", entry->source, sizeof(entry->source)) ||
        !text_field(line, "note", entry->note, sizeof(entry->note))) {
        return -1;
    }
    if (strstr(line, "\"verified\":true") != NULL) {
        entry->verified = 1;
    } else if (strstr(line, "\"verified\":false") != NULL) {
        entry->verified = 0;
    } else {
        return -1;
    }
    entry->bytes = bytes;
    if (!base_name(entry->name) || !base_name(entry->file) || !role_ok(entry->role) ||
        entry->repository[0] == '\0' || entry->revision[0] == '\0' || bytes == 0u ||
        !digest_ok(entry->sha256) || entry->license[0] == '\0' ||
        entry->quant[0] == '\0' || entry->profiles[0] == '\0' ||
        entry->source[0] == '\0') {
        return -1;
    }
    return 0;
}

int aotx_model_catalog_read(const char *path, aotx_model_catalog *catalog,
                            char *reason, size_t reason_bytes)
{
    char line[AOTX_MODEL_LINE];
    FILE *file;
    unsigned int number = 0u;
    if (path == NULL || catalog == NULL) {
        say(reason, reason_bytes, "the catalog read has no path or table");
        return -1;
    }
    memset(catalog, 0, sizeof(*catalog));
    file = fopen(path, "r");
    if (file == NULL) {
        say(reason, reason_bytes, "the catalog file does not open");
        return -1;
    }
    while (fgets(line, (int)sizeof(line), file) != NULL) {
        size_t length = strlen(line);
        unsigned int i;
        number++;
        if (length + 1u == sizeof(line) || catalog->count >= AOTX_MODEL_CATALOG_MAX) {
            say(reason, reason_bytes, "the catalog has a line beyond its bound");
            fclose(file);
            return -1;
        }
        if (length == 0u || (length == 1u && line[0] == '\n')) {
            continue;
        }
        if (read_line(line, &catalog->entry[catalog->count]) != 0) {
            if (reason != NULL && reason_bytes != 0u) {
                snprintf(reason, reason_bytes, "catalog line %u does not read", number);
            }
            fclose(file);
            return -1;
        }
        for (i = 0u; i < catalog->count; i++) {
            if (strcmp(catalog->entry[i].name,
                       catalog->entry[catalog->count].name) == 0) {
                say(reason, reason_bytes, "the catalog repeats a name");
                fclose(file);
                return -1;
            }
        }
        catalog->count++;
    }
    if (ferror(file) != 0 || catalog->count == 0u) {
        say(reason, reason_bytes, "the catalog has no readable entry");
        fclose(file);
        return -1;
    }
    fclose(file);
    return (int)catalog->count;
}

const aotx_model_catalog_entry *aotx_model_catalog_find(const aotx_model_catalog *catalog,
                                                        const char *name)
{
    unsigned int i;
    if (catalog == NULL || name == NULL) {
        return NULL;
    }
    for (i = 0u; i < catalog->count; i++) {
        if (strcmp(catalog->entry[i].name, name) == 0) {
            return &catalog->entry[i];
        }
    }
    return NULL;
}
