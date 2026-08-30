/* Purpose: Read the models manifest and compare each model file with its line.
 * Owns: Nothing; the caller owns the array of entries that the read fills.
 * Threading: One thread; the caller makes the calls one at a time.
 * Lifetime: The call.
 *
 * One line holds one JSON object with the keys name, role, path, source, revision,
 * license, bytes, and sha256. A field value holds no control byte, no quotation mark, and no
 * backslash. The line therefore needs no escape, and the reader needs no escape rule. */
#include "disk/modelfile/manifest.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The buffer that hashes a file. A larger buffer does not make the disk faster here. */
#define AOTX_HASH_BUFFER (1024u * 1024u)

int aotx_manifest_path(char *out, size_t out_bytes, const char *dir, const char *name)
{
    int written;
    if (dir == NULL || name == NULL) {
        return -1;
    }
    /* A path that starts at the root does not join with the directory. */
    written = (name[0] == '/') ? snprintf(out, out_bytes, "%s", name)
                               : snprintf(out, out_bytes, "%s/%s", dir, name);
    if (written < 0 || (size_t)written >= out_bytes) {
        return -1;
    }
    return 0;
}

int aotx_manifest_field(const char *value)
{
    size_t i;
    if (value == NULL) {
        return -1;
    }
    for (i = 0; value[i] != '\0'; i++) {
        unsigned char b = (unsigned char)value[i];
        if (b < 0x20u || b == 0x7fu || b == '"' || b == '\\') {
            return -1;
        }
    }
    return 0;
}

/* Gives the first byte after the colon of a key, or null when the key is not in the line.
 * A field value holds no quotation mark, so this text can only be a key. */
static const char *after_key(const char *line, const char *key)
{
    char want[64];
    const char *at;
    int written = snprintf(want, sizeof(want), "\"%s\":", key);
    if (written < 0 || (size_t)written >= sizeof(want)) {
        return NULL;
    }
    at = strstr(line, want);
    return (at != NULL) ? (at + written) : NULL;
}

/* Copies the string value of a key into a field. Returns 0, or -1 on a bad line. */
static int read_text(const char *line, const char *key, char *out, size_t out_bytes)
{
    const char *at = after_key(line, key);
    const char *end;
    size_t length;
    if (at == NULL || *at != '"') {
        return -1;
    }
    at++;
    end = strchr(at, '"');
    if (end == NULL) {
        return -1;
    }
    length = (size_t)(end - at);
    if (length + 1u > out_bytes) {
        return -1;
    }
    memcpy(out, at, length);
    out[length] = '\0';
    /* A control byte or a backslash in a line means the writer was not this program. */
    return aotx_manifest_field(out);
}

/* Reads the whole number value of a key. Returns 0, or -1 on a bad line. */
static int read_number(const char *line, const char *key, uint64_t *out)
{
    const char *at = after_key(line, key);
    uint64_t value = 0;
    int digits = 0;
    if (at == NULL) {
        return -1;
    }
    while (*at >= '0' && *at <= '9') {
        if (value > (UINT64_MAX - (uint64_t)(*at - '0')) / 10u) {
            return -1;
        }
        value = value * 10u + (uint64_t)(*at - '0');
        at++;
        digits++;
    }
    if (digits == 0 || digits > 20) {
        return -1;
    }
    *out = value;
    return 0;
}

/* Checks that the text of a digest is 64 hexadecimal characters in the low case. */
static int digest_text(const char *text)
{
    int i;
    for (i = 0; i < 64; i++) {
        char c = text[i];
        int ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        if (!ok) {
            return -1;
        }
    }
    return (text[64] == '\0') ? 0 : -1;
}

int aotx_manifest_line(const char *line, aotx_manifest_entry *entry)
{
    memset(entry, 0, sizeof(*entry));
    if (read_text(line, "name", entry->name, sizeof(entry->name)) != 0 ||
        read_text(line, "path", entry->path, sizeof(entry->path)) != 0 ||
        read_text(line, "source", entry->source, sizeof(entry->source)) != 0 ||
        read_text(line, "revision", entry->revision, sizeof(entry->revision)) != 0 ||
        read_text(line, "license", entry->license, sizeof(entry->license)) != 0 ||
        read_text(line, "sha256", entry->sha256, sizeof(entry->sha256)) != 0 ||
        read_number(line, "bytes", &entry->bytes) != 0) {
        return -1;
    }
    /* An old line used its name as its role. The next write gives it the separate field. */
    if (read_text(line, "role", entry->role, sizeof(entry->role)) != 0) {
        if (strlen(entry->name) >= sizeof(entry->role)) {
            return -1;
        }
        snprintf(entry->role, sizeof(entry->role), "%s", entry->name);
    }
    if (entry->name[0] == '\0' || entry->role[0] == '\0' || entry->path[0] == '\0') {
        return -1;
    }
    return digest_text(entry->sha256);
}

int aotx_manifest_write_line(char *out, size_t out_bytes, const aotx_manifest_entry *entry)
{
    int written;
    if (aotx_manifest_field(entry->name) != 0 || aotx_manifest_field(entry->role) != 0 ||
        aotx_manifest_field(entry->path) != 0 ||
        aotx_manifest_field(entry->source) != 0 || aotx_manifest_field(entry->revision) != 0 ||
        aotx_manifest_field(entry->license) != 0 || digest_text(entry->sha256) != 0) {
        return -1;
    }
    written = snprintf(out, out_bytes,
                       "{\"name\":\"%s\",\"role\":\"%s\",\"path\":\"%s\","
                       "\"source\":\"%s\","
                       "\"revision\":\"%s\",\"license\":\"%s\",\"bytes\":%llu,"
                       "\"sha256\":\"%s\"}\n",
                       entry->name, entry->role, entry->path, entry->source, entry->revision,
                       entry->license, (unsigned long long)entry->bytes, entry->sha256);
    if (written < 0 || (size_t)written >= out_bytes) {
        return -1;
    }
    return 0;
}

int aotx_manifest_read(const char *dir, aotx_manifest_entry *entries, int max_entries)
{
    char path[AOTX_MANIFEST_PATH];
    char line[AOTX_MANIFEST_LINE];
    FILE *file;
    int count = 0;
    if (entries == NULL || max_entries <= 0) {
        return -1;
    }
    if (aotx_manifest_path(path, sizeof(path), dir, AOTX_MANIFEST_NAME) != 0) {
        return -1;
    }
    file = fopen(path, "r");
    if (file == NULL) {
        fprintf(stderr, "aotx_manifest: %s: the manifest does not open\n", path);
        return -1;
    }
    while (fgets(line, (int)sizeof(line), file) != NULL) {
        size_t length = strlen(line);
        if (length + 1u == sizeof(line)) {
            fprintf(stderr, "aotx_manifest: %s: a line is too long\n", path);
            fclose(file);
            return -1;
        }
        while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r')) {
            line[--length] = '\0';
        }
        if (length == 0) {
            continue;
        }
        if (count >= max_entries) {
            fprintf(stderr, "aotx_manifest: %s: the manifest holds more lines than the room\n",
                    path);
            fclose(file);
            return -1;
        }
        if (aotx_manifest_line(line, &entries[count]) != 0) {
            fprintf(stderr, "aotx_manifest: %s: line %d does not read\n", path, count + 1);
            fclose(file);
            return -1;
        }
        count++;
    }
    fclose(file);
    return count;
}

int aotx_manifest_check(const char *dir, const aotx_manifest_entry *entry)
{
    char path[AOTX_MANIFEST_PATH];
    char text[AOTX_SHA256_HEX];
    void *buffer;
    uint64_t bytes = 0;
    int rc;
    if (entry == NULL) {
        return 2;
    }
    if (aotx_manifest_path(path, sizeof(path), dir, entry->path) != 0) {
        return 2;
    }
    buffer = malloc(AOTX_HASH_BUFFER);
    if (buffer == NULL) {
        fprintf(stderr, "aotx_manifest: the memory for the hash buffer is not there\n");
        return 2;
    }
    rc = aotx_sha256_file(path, text, &bytes, buffer, AOTX_HASH_BUFFER);
    free(buffer);
    if (rc != 0) {
        return 2;
    }
    if (bytes != entry->bytes || strcmp(text, entry->sha256) != 0) {
        return 1;
    }
    return 0;
}
