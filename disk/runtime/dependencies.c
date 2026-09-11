/* Purpose: Verify text model and data module dependencies before activation.
 * Owns: Bounded metadata buffers and required logical name checks.
 * Threading: One caller holds a validated file view for the complete component batch.
 * Lifetime: One inspection; no component code is loaded or executed. */
#include "disk/runtime/runtime.h"
#include "disk/ccir/internal.h"
#include "disk/modelfile/manifest.h"
#include "disk/runtime/replay.h"
#include "disk/settings/settings.h"
#include <stdlib.h>
#include <string.h>

static int named(const aotx_runtime_index *index, const char *name, uint32_t kind) {
    for (uint32_t i = 0; i < index->count; ++i)
        if (!strcmp((const char *)index->rows[i] + 64, name) &&
            aotx_ccir_u32(index->rows[i] + 16) == kind) return (int)i;
    return -1;
}
static char *text_asset(const aotx_ccir_view *view, const unsigned char *row, size_t cap) {
    int at = aotx_runtime_section(view, row);
    if (at < 0 || view->sections[at].bytes > cap) return NULL;
    const aotx_ccir_section *s = view->sections + at;
    char *text = malloc((size_t)s->bytes + 1);
    if (!text) return NULL;
    if (aotx_ccir_pread(view->fd, text, (size_t)s->bytes, s->offset) || memchr(text, 0, (size_t)s->bytes)) {
        free(text); return NULL;
    }
    text[s->bytes] = 0;
    return text;
}
static int settings(const aotx_ccir_view *view, const aotx_runtime_index *index) {
    int at = named(index, "settings", 1);
    if (at < 0) return AOTX_CCIR_INVALID;
    char *text = text_asset(view, index->rows[at], 1048576);
    aotx_settings *table = malloc(sizeof(*table));
    if (!text || !table) { free(text); free(table); return AOTX_CCIR_INVALID; }
    aotx_settings_defaults(table);
    char *save = NULL, reason[AOTX_SETTINGS_REASON_BYTES]; int rc = 0;
    for (char *line = strtok_r(text, "\n", &save); line && !rc; line = strtok_r(NULL, "\n", &save))
        if (aotx_settings_line(line, strlen(line), table, reason)) rc = AOTX_CCIR_INVALID;
    for (unsigned i = 0; !rc && i < AOTX_SETTING_NUMBER_COUNT; ++i)
        if (table->number_given[i] && aotx_settings_number_side(i) != AOTX_SETTING_SIDE_DEVICE)
            rc = AOTX_CCIR_UNSUPPORTED;
    for (unsigned i = 0; !rc && i < AOTX_SETTING_TEXT_COUNT; ++i)
        if (table->text_given[i] && aotx_settings_text_side(i) != AOTX_SETTING_SIDE_DEVICE)
            rc = AOTX_CCIR_UNSUPPORTED;
    free(table); free(text);
    return rc;
}
static int models(const aotx_ccir_view *view, const aotx_runtime_index *index) {
    int at = named(index, "manifest.jsonl", 1);
    if (at < 0) return AOTX_CCIR_INVALID;
    char *text = text_asset(view, index->rows[at], 8 * AOTX_MANIFEST_LINE);
    if (!text) return AOTX_CCIR_INVALID;
    aotx_manifest_entry entries[8];
    uint32_t count = 0; int rc = 0;
    char *save = NULL;
    for (char *line = strtok_r(text, "\n", &save); line && !rc; line = strtok_r(NULL, "\n", &save)) {
        if (count == 8 || strlen(line) >= AOTX_MANIFEST_LINE || aotx_manifest_line(line, entries + count)) {
            rc = AOTX_CCIR_INVALID; break;
        }
        aotx_manifest_entry *e = entries + count++;
        int asset = named(index, e->path, 1); unsigned char digest[32];
        if (asset < 0 || !e->source[0] || !e->revision[0] || !e->license[0] ||
            aotx_manifest_digest(e->sha256, digest) ||
            e->bytes != aotx_ccir_u64(index->rows[asset] + 24) ||
            memcmp(digest, index->rows[asset] + 32, 32)) { rc = AOTX_CCIR_INVALID; break; }
        for (uint32_t i = 0; i + 1 < count; ++i)
            if (!strcmp(entries[i].name, e->name) || !strcmp(entries[i].role, e->role)) rc = AOTX_CCIR_INVALID;
        int section = aotx_runtime_section(view, index->rows[asset]);
        aotx_modelfile *file = NULL;
        if (!rc && aotx_modelfile_open_extent(e->name, view->fd, view->sections[section].offset,
                                              e->bytes, &file)) rc = AOTX_CCIR_INVALID;
        aotx_modelfile_close(file);
    }
    free(text);
    char roles[64]; strcpy(roles, (const char *)index->header + 64); save = NULL;
    uint32_t language = 0, selected = 0;
    if (roles[0] == ',' || roles[strlen(roles) - 1] == ',' || strstr(roles, ",,")) return AOTX_CCIR_INVALID;
    for (char *role = strtok_r(roles, ",", &save); role && !rc; role = strtok_r(NULL, ",", &save)) {
        uint32_t found = 0;
        for (uint32_t i = 0; i < count; ++i) if (!strcmp(entries[i].role, role)) {
            if (selected & (1u << i)) rc = AOTX_CCIR_INVALID;
            selected |= 1u << i; ++found;
        }
        if (strcmp(role, "language") && strcmp(role, "language-q4") &&
            strcmp(role, "embedding") && strcmp(role, "reranker")) rc = AOTX_CCIR_UNSUPPORTED;
        if (found != 1) rc = AOTX_CCIR_INVALID;
        language += !strcmp(role, "language") || !strcmp(role, "language-q4");
    }
    return rc ? rc : selected && language == 1 ? 0 : AOTX_CCIR_INVALID;
}
static int module(const aotx_ccir_view *view, const aotx_runtime_index *index, uint32_t at) {
    const char *name = (const char *)index->rows[at] + 64;
    size_t bytes = strlen(name), suffix = sizeof("module.manifest") - 1;
    if (strncmp(name, "modules/", 8)) return AOTX_CCIR_INVALID;
    if (bytes <= suffix || strcmp(name + bytes - suffix, "module.manifest")) return 0;
    char *text = text_asset(view, index->rows[at], 1048576);
    if (!text) return AOTX_CCIR_INVALID;
    char *save = NULL; uint32_t kind = 0; int rc = 0;
    for (char *line = strtok_r(text, "\n", &save); line && !rc; line = strtok_r(NULL, "\n", &save)) {
        while (*line == ' ' || *line == '\t') ++line;
        if (*line == '#') continue;
        char *colon = strchr(line, ':');
        if (!colon) continue;
        char *end = colon; while (end > line && (end[-1] == ' ' || end[-1] == '\t')) --end;
        *end = 0; char *value = colon + 1;
        while (*value == ' ' || *value == '\t') ++value;
        end = value + strlen(value);
        while (end > value && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r')) --end;
        *end = 0;
        if (!strcmp(line, "kind")) {
            if (kind++ || (strcmp(value, "role") && strcmp(value, "skill"))) rc = AOTX_CCIR_UNSUPPORTED;
        } else if (!strcmp(line, "body") && *value) {
            char key[AOTX_RUNTIME_NAME];
            if (!aotx_runtime_name(value) || bytes - suffix + strlen(value) >= sizeof(key)) {
                rc = AOTX_CCIR_INVALID; break;
            }
            memcpy(key, name, bytes - suffix); strcpy(key + bytes - suffix, value);
            if (named(index, key, 2) < 0) rc = AOTX_CCIR_INVALID;
        }
    }
    free(text);
    return rc ? rc : kind == 1 ? 0 : AOTX_CCIR_INVALID;
}
static int references(const aotx_ccir_view *view, const aotx_runtime_index *index) {
    const char *names[] = {"steer.jsonl", "probes.jsonl", "affect/calibration.jsonl"};
    for (unsigned i = 0; i < 3; ++i) {
        int at = named(index, names[i], 1);
        if (at < 0) continue;
        char *text = text_asset(view, index->rows[at], 1048576);
        if (!text) return AOTX_CCIR_INVALID;
        char *save = NULL, *last = NULL; int rc = 0;
        for (char *line = strtok_r(text, "\n", &save); line && !rc; line = strtok_r(NULL, "\n", &save)) {
            if (i == 2) { last = line; continue; }
            const char *key = strstr(line, "\"file\":\""); char file[AOTX_RUNTIME_NAME];
            if (!key || sscanf(key, "\"file\":\"%255[^\"]\"", file) != 1 ||
                !aotx_runtime_name(file) || named(index, file, 1) < 0) rc = AOTX_CCIR_INVALID;
        }
        if (!rc && i == 2) {
            const char *key = last ? strstr(last, "\"composite\":[") : NULL;
            char file[2][AOTX_RUNTIME_NAME];
            if (!key || sscanf(key, "\"composite\":[\"%255[^\"]\",\"%255[^\"]\"]", file[0], file[1]) != 2)
                rc = AOTX_CCIR_INVALID;
            for (unsigned j = 0; !rc && j < 2; ++j)
                if (!aotx_runtime_name(file[j]) || named(index, file[j], 1) < 0) rc = AOTX_CCIR_INVALID;
        }
        free(text);
        if (rc) return rc;
    }
    return 0;
}
int aotx_runtime_dependencies(const aotx_ccir_view *view) {
    aotx_runtime_index *index = malloc(sizeof(*index));
    if (!index) return AOTX_CCIR_IO;
    int rc = aotx_runtime_index_read(view->fd, view, index);
    if (!rc && (named(index, "settings", 1) < 0 ||
        named(index, "modules/conductor/module.manifest", 2) < 0 ||
        ((aotx_ccir_u32(index->header + 20) & AOTX_RUNTIME_AFFECT) &&
         named(index, "quality/refusal-phrases.txt", 1) < 0))) rc = AOTX_CCIR_INVALID;
    if (!rc) rc = settings(view, index);
    if (!rc) rc = models(view, index);
    if (!rc) rc = references(view, index);
    for (uint32_t i = 0; !rc && i < index->count; ++i)
        if (aotx_ccir_u32(index->rows[i] + 16) == 2) rc = module(view, index, i);
    unsigned char header[AOTX_RUNTIME_REPLAY_HEADER];
    if (!rc) {
        int at = aotx_runtime_section(view, index->header + 128);
        rc = aotx_runtime_replay_header(view->fd, view->sections + at, header);
        if (!rc && aotx_ccir_u32(header + 12) == 2) {
            unsigned char *buffer = malloc(16u * 1024u * 1024u); aotx_journal_scan scan;
            rc = buffer ? aotx_runtime_replay_walk(view, buffer, 16u * 1024u * 1024u,
                NULL, NULL, &scan) : AOTX_CCIR_IO;
            free(buffer);
        }
    }
    free(index);
    return rc;
}
