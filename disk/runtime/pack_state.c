/* Purpose: Read prepared memory and write selected device settings for runtime creation.
 * Owns: The initial live header and canonical settings asset.
 * Threading: One packager reads complete source files.
 * Lifetime: Source data stays owned until publication. */
#include "disk/runtime/pack.h"
#include <stdio.h>
#include "cognitive/io.h"
#include "cognitive/checkpoint.h"
#include "disk/settings/settings.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int aotx_runtime_pack_memory(aotx_runtime_pack *p, const char *path) {
    aotx_cognitive_file source;
    int rc = aotx_cognitive_file_open(path, &source);
    if (rc) return rc;
    if (source.tail_bytes) { aotx_cognitive_file_close(&source); return AOTX_CCIR_UNSUPPORTED; }
    p->memory = source.checkpoint; p->memory_bytes = source.checkpoint_bytes;
    source.checkpoint = NULL;
    p->source = source.view; source.view.fd = -1;
    aotx_cognitive_file_close(&source);
    unsigned char *h = p->live;
    memcpy(h, "AOTXLCP1", 8); aotx_ccir_put(h + 8, 1, 4);
    aotx_ccir_put(h + 12, AOTX_CP_ROW, 4); aotx_ccir_put(h + 24, p->memory_bytes, 8);
    memcpy(h + 32, p->source.lineage, 16);
    aotx_ccir_put(h + 48, p->source.meta.durable_sequence, 8);
    aotx_ccir_put(h + 56, p->source.meta.source_tick, 8); aotx_ccir_put(h + 64, 1, 8);
    return 0;
}
int aotx_runtime_pack_settings(aotx_runtime_pack *p, const char *path) {
    aotx_settings *settings = malloc(sizeof(*settings));
    if (!settings) return AOTX_CCIR_IO;
    aotx_settings_defaults(settings);
    int rc = 0;
    if (path && (access(path, R_OK) || aotx_settings_read(path, settings))) rc = AOTX_CCIR_INVALID;
    FILE *file = rc ? NULL : tmpfile();
    if (!rc && !file) rc = AOTX_CCIR_IO;
    for (unsigned i = 0; !rc && i < AOTX_SETTING_NUMBER_COUNT; ++i) {
        if (aotx_settings_number_side(i) != AOTX_SETTING_SIDE_DEVICE) continue;
        char value[64];
        if (!aotx_settings_format(settings->number[i], aotx_settings_number_scale(i), value, sizeof(value)) ||
            fprintf(file, "%s = %s\n", aotx_settings_number_name(i), value) < 0) rc = AOTX_CCIR_IO;
    }
    for (unsigned i = 0; !rc && i < AOTX_SETTING_TEXT_COUNT; ++i) {
        if (aotx_settings_text_side(i) != AOTX_SETTING_SIDE_DEVICE) continue;
        if (fprintf(file, "%s = %s\n", aotx_settings_text_name(i), settings->text[i]) < 0) rc = AOTX_CCIR_IO;
    }
    if (!rc && fflush(file)) rc = AOTX_CCIR_IO;
    if (!rc) {
        char name[64]; snprintf(name, sizeof(name), "/proc/self/fd/%d", fileno(file));
        rc = aotx_runtime_pack_asset(p, name, "settings", 1);
    }
    if (file) fclose(file);
    free(settings);
    return rc;
}
