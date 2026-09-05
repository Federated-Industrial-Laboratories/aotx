/* Purpose: Read the models manifest and compare each model file with its line.
 * Owns: Nothing; the caller owns the array of entries that the read fills.
 * Threading: One thread; the caller makes the calls one at a time.
 * Lifetime: The call. */
#include "disk/modelfile/manifest.h"
#include "disk/modelfile/manifest_json.h"
#include <stdarg.h>

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

static int text(aotx_manifest_json *j, char *out, size_t room)
{
    size_t length;
    if (aotx_manifest_json_string(j, (unsigned char *)out, room - 1u, &length) != 0 ||
        memchr(out, 0, length) != NULL) return -1;
    out[length] = '\0';
    return 0;
}

static int key_index(const char *key, const char *const *names, unsigned count)
{
    for (unsigned i = 0; i < count; ++i)
        if (strcmp(key, names[i]) == 0) return (int)i;
    return -1;
}

static int wrap_read(aotx_manifest_json *j, aotx_wrap *wrap)
{
    unsigned seen = 0, used = 0;
    if (aotx_manifest_json_take(j, '{') != 0) return -1;
    for (;;) {
        char key[32];
        int index;
        uint64_t number;
        if (text(j, key, sizeof(key)) != 0 || aotx_manifest_json_take(j, ':') != 0) return -1;
        index = key_index(key, aotx_wrap_names, AOTX_WRAP_SPANS);
        if (index < 0) index = strcmp(key, "end_ids") == 0 ? 9 :
                               strcmp(key, "prefix_length") == 0 ? 10 : -1;
        if (index < 0 || (seen & (1u << index))) return -1;
        seen |= 1u << index;
        if (index < 9) {
            unsigned char span[AOTX_WRAP_SPAN_BYTES];
            size_t length;
            if (aotx_manifest_json_string(j, span, sizeof(span), &length) != 0 ||
                length > AOTX_WRAP_BYTES - used) return -1;
            wrap->offset[index] = (uint16_t)used;
            wrap->length[index] = (uint8_t)length;
            memcpy(wrap->bytes + used, span, length);
            used += (unsigned)length;
        } else if (index == 9) {
            if (aotx_manifest_json_take(j, '[') != 0) return -1;
            do {
                if (wrap->end_count == AOTX_WRAP_ENDS ||
                    aotx_manifest_json_number(j, &number) != 0 || number > UINT32_MAX) return -1;
                wrap->end_ids[wrap->end_count++] = (uint32_t)number;
                if (aotx_manifest_json_take(j, ']') == 0) break;
                if (aotx_manifest_json_take(j, ',') != 0) return -1;
            } while (1);
        } else {
            if (aotx_manifest_json_number(j, &number) != 0 || number > AOTX_WRAP_SPAN_BYTES) return -1;
            wrap->prefix_length = (uint8_t)number;
        }
        if (aotx_manifest_json_take(j, '}') == 0) break;
        if (aotx_manifest_json_take(j, ',') != 0) return -1;
    }
    return (seen & 1023u) == 1023u && aotx_wrap_valid(wrap) ? 0 : -1;
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
    static const char *const keys[] = {"name", "role", "path", "source", "revision",
        "license", "sha256", "bytes", "wrap", "probe_numerator", "probe_denominator"};
    aotx_manifest_json j;
    unsigned seen = 0;
    size_t length;
    if (line == NULL || entry == NULL) return -1;
    for (length = 0; length < AOTX_MANIFEST_LINE && line[length]; ++length) {}
    if (length == AOTX_MANIFEST_LINE) return -1;
    j.at = (const unsigned char *)line; j.end = j.at + length;
    memset(entry, 0, sizeof(*entry));
    entry->probe_numerator = 2; entry->probe_denominator = 3;
    entry->wrap.think_open_id = UINT32_MAX;
    entry->wrap.think_close_id = UINT32_MAX;
    if (aotx_manifest_json_take(&j, '{') != 0) return -1;
    for (;;) {
        char key[32];
        int index;
        uint64_t number;
        char *fields[] = {entry->name, entry->role, entry->path, entry->source,
                          entry->revision, entry->license, entry->sha256};
        const size_t rooms[] = {sizeof(entry->name), sizeof(entry->role), sizeof(entry->path),
            sizeof(entry->source), sizeof(entry->revision), sizeof(entry->license), sizeof(entry->sha256)};
        if (text(&j, key, sizeof(key)) != 0 || aotx_manifest_json_take(&j, ':') != 0) return -1;
        index = key_index(key, keys, sizeof(keys) / sizeof(keys[0]));
        if (index < 0 || (seen & (1u << index))) return -1;
        seen |= 1u << index;
        if (index < 7) {
            if (text(&j, fields[index], rooms[index]) != 0) return -1;
        } else if (index == 8) {
            if (wrap_read(&j, &entry->wrap) != 0) return -1;
            entry->wrap_present = 1;
        } else {
            if (aotx_manifest_json_number(&j, &number) != 0) return -1;
            if (index == 7) entry->bytes = number;
            else {
                if (number > UINT32_MAX) return -1;
                if (index == 9) entry->probe_numerator = (uint32_t)number;
                else entry->probe_denominator = (uint32_t)number;
            }
        }
        if (aotx_manifest_json_take(&j, '}') == 0) break;
        if (aotx_manifest_json_take(&j, ',') != 0) return -1;
    }
    if ((seen & 253u) != 253u || aotx_manifest_json_end(&j) != 0 ||
        entry->probe_numerator >= entry->probe_denominator) return -1;
    if (!(seen & 2u)) {
        if (strlen(entry->name) >= sizeof(entry->role)) return -1;
        memcpy(entry->role, entry->name, strlen(entry->name) + 1u);
    }
    if (!entry->name[0] || !entry->role[0] || !entry->path[0]) return -1;
    return digest_text(entry->sha256);
}

static int append(char **at, size_t *room, const char *format, ...)
{
    va_list args;
    int n;
    va_start(args, format);
    n = vsnprintf(*at, *room, format, args);
    va_end(args);
    if (n < 0 || (size_t)n >= *room) return -1;
    *at += n; *room -= (size_t)n;
    return 0;
}

int aotx_manifest_write_line(char *out, size_t out_bytes, const aotx_manifest_entry *entry)
{
    char *at = out;
    size_t room = out_bytes;
    uint32_t numerator, denominator;
    static const char *const keys[] = {"name", "role", "path", "source", "revision", "license", "sha256"};
    if (out == NULL || entry == NULL || !out_bytes) return -1;
    const char *fields[] = {entry->name, entry->role, entry->path, entry->source,
                           entry->revision, entry->license, entry->sha256};
    const size_t sizes[] = {sizeof(entry->name), sizeof(entry->role), sizeof(entry->path),
        sizeof(entry->source), sizeof(entry->revision), sizeof(entry->license), sizeof(entry->sha256)};
    numerator = entry->probe_numerator; denominator = entry->probe_denominator;
    if (!numerator && !denominator) { numerator = 2; denominator = 3; }
    if (numerator >= denominator || entry->wrap_present > 1u ||
        (entry->wrap_present && !aotx_wrap_valid(&entry->wrap))) return -1;
    if (append(&at, &room, "{")) return -1;
    for (unsigned i = 0; i < 7u; ++i) {
        const char *end = memchr(fields[i], 0, sizes[i]);
        if (!end || (i < 3u && end == fields[i]) ||
            append(&at, &room, "%s\"%s\":", i ? "," : "", keys[i]) ||
            aotx_manifest_json_quote(&at, &room, (const unsigned char *)fields[i],
                                    (size_t)(end - fields[i]))) return -1;
    }
    if (digest_text(entry->sha256) || append(&at, &room,
        ",\"bytes\":%llu,\"probe_numerator\":%u,\"probe_denominator\":%u",
        (unsigned long long)entry->bytes, numerator, denominator)) return -1;
    if (entry->wrap_present) {
        const aotx_wrap *w = &entry->wrap;
        if (append(&at, &room, ",\"wrap\":{")) return -1;
        for (unsigned i = 0; i < AOTX_WRAP_SPANS; ++i) {
            if (append(&at, &room, "%s\"%s\":", i ? "," : "", aotx_wrap_names[i]) ||
                aotx_manifest_json_quote(&at, &room, w->bytes + w->offset[i], w->length[i])) return -1;
        }
        if (append(&at, &room, ",\"prefix_length\":%u,\"end_ids\":[", w->prefix_length)) return -1;
        for (unsigned i = 0; i < w->end_count; ++i)
            if (append(&at, &room, "%s%u", i ? "," : "", w->end_ids[i])) return -1;
        if (append(&at, &room, "]}")) return -1;
    }
    return append(&at, &room, "}\n");
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
    for (;;) {
        size_t length = 0;
        int byte;
        while ((byte = fgetc(file)) != EOF && byte != '\n') {
            if (byte == 0 || length + 1u >= sizeof(line)) {
                fprintf(stderr, "aotx_manifest: %s: a line has invalid bytes or length\n", path);
                fclose(file);
                return -1;
            }
            line[length++] = (char)byte;
        }
        if (byte == EOF && ferror(file)) {
            fclose(file);
            return -1;
        }
        if (byte == EOF && length == 0) break;
        while (length > 0 && line[length - 1u] == '\r') --length;
        line[length] = '\0';
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
