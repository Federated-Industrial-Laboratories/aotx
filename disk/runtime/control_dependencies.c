/* Purpose: Check fitted control bindings in a complete runtime before activation.
 * Owns: Bounded model metadata and exact component references.
 * Threading: One reader checks all fitted assets in the runtime index.
 * Lifetime: One inspection; no model code executes. */
#include "disk/runtime/control.h"
#include "disk/runtime/runtime.h"
#include "disk/ccir/internal.h"
#include "disk/modelfile/manifest.h"
#include <stdlib.h>
#include <string.h>

static int named(const aotx_runtime_index *index, const char *name) {
    for (unsigned i = 0; i < index->count; ++i)
        if (aotx_ccir_u32(index->rows[i] + 16) == 1 && !strcmp((const char *)index->rows[i] + 64, name)) return (int)i;
    return -1;
}
static int selected(const char *roles, const char *role) {
    size_t bytes = strlen(role);
    for (const char *at = roles; at; at = strchr(at, ',')) {
        if (*at == ',') ++at;
        if (!strncmp(at, role, bytes) && (!at[bytes] || at[bytes] == ',')) return 1;
    }
    return 0;
}
static int current(const aotx_ccir_view *view, const aotx_runtime_index *index,
    aotx_control_identity *identity) {
    int at = named(index, "manifest.jsonl");
    if (at < 0) return 1;
    int section = aotx_runtime_section(view, index->rows[at]);
    if (section < 0) return 1;
    const aotx_ccir_section *s = view->sections + section;
    if (s->bytes > 8u * AOTX_MANIFEST_LINE) return 1;
    char *text = malloc((size_t)s->bytes + 1);
    if (!text) return 1;
    int bad = aotx_ccir_pread(view->fd, text, (size_t)s->bytes, s->offset) || memchr(text, 0, (size_t)s->bytes);
    text[s->bytes] = 0;
    char *save = NULL; unsigned found = 0, best = 3;
    for (char *line = strtok_r(text, "\n", &save); line && !bad; line = strtok_r(NULL, "\n", &save)) {
        aotx_manifest_entry entry;
        if (aotx_manifest_line(line, &entry)) { bad = 1; break; }
        unsigned rank = !strcmp(entry.role, "language") ? 0 : !strcmp(entry.role, "language-q4") ? 1 :
            !strcmp(entry.role, "language-audio") ? 2 : 3;
        if (rank >= best || !selected((const char *)index->header + 64, entry.role)) continue;
        best = rank; found = 1;
        at = named(index, entry.path);
        if (at < 0 || aotx_manifest_digest(entry.sha256, identity->model)) { bad = 1; break; }
        section = aotx_runtime_section(view, index->rows[at]);
        if (section < 0) { bad = 1; break; }
        s = view->sections + section;
        aotx_modelfile *file = NULL;
        bad = aotx_modelfile_open_extent(entry.name, view->fd, s->offset, s->bytes, &file);
        if (!bad) bad = aotx_wrap_read(file, &entry, &identity->wrap);
        aotx_modelfile_close(file);
    }
    free(text);
    return bad || found != 1;
}
int aotx_control_reference(const aotx_ccir_view *view, const aotx_runtime_index *index,
    const char *name, unsigned kind, unsigned response) {
    aotx_control_identity expected;
    if (current(view, index, &expected)) return AOTX_CCIR_INVALID;
    int asset = named(index, name);
    if (asset < 0) return AOTX_CCIR_INVALID;
    char key[AOTX_RUNTIME_NAME]; int n = snprintf(key, sizeof(key), "%s.binding", name);
    if (n < 0 || (size_t)n >= sizeof(key)) return AOTX_CCIR_INVALID;
    int at = named(index, key);
    if (at < 0) return AOTX_CCIR_INVALID;
    int section = aotx_runtime_section(view, index->rows[at]);
    if (section < 0 || view->sections[section].bytes != AOTX_CONTROL_BYTES) return AOTX_CCIR_INVALID;
    unsigned char raw[AOTX_CONTROL_BYTES], digest[32]; aotx_control_identity identity; unsigned positions;
    if (aotx_ccir_pread(view->fd, raw, sizeof(raw), view->sections[section].offset) ||
        aotx_control_decode(raw, kind, &identity, digest, response && kind == AOTX_CONTROL_VECTOR ? &positions : NULL) ||
        memcmp(&identity, &expected, sizeof(identity)) ||
        memcmp(digest, index->rows[asset] + 32, 32)) return AOTX_CCIR_INVALID;
    return 0;
}
