/* Purpose: Publish memory and complete runtime replay state in one file generation.
 * Owns: The replacement directory and temporary replay source.
 * Threading: One drain holds the persistent writer and retries without losing snapshots.
 * Lifetime: Each acknowledgement follows a synchronized complete generation. */
#include "disk/runtime/replay.h"
#include "cognitive/checkpoint_io.h"
#include "disk/ccir/internal.h"
#include <string.h>

int aotx_runtime_checkpoint_write(aotx_checkpoint_disk *d, const unsigned char *image,
    uint64_t bytes, uint32_t base, int same) {
    if (!d->runtime || !d->ring || !d->journal) return AOTX_CCIR_UNSUPPORTED;
    const aotx_ccir_section *saved = NULL;
    for (uint32_t i = 0; i < d->view.count; ++i)
        if (d->view.sections[i].type == AOTX_CCIR_REPLAY &&
            (d->view.sections[i].flags & AOTX_CCIR_REQUIRED)) saved = d->view.sections + i;
    unsigned char header[128];
    int rc = aotx_runtime_replay_header(d->view.fd, saved, header);
    if (rc) return rc;
    if (aotx_ccir_u32(header + 12) == 2 && aotx_ccir_u64(header + 16) == d->ring->boot) {
        uint64_t old = aotx_ccir_u64(header + 72);
        if (old > d->runtime_sequence) return AOTX_CCIR_CHANGED;
        if (old == d->runtime_sequence)
            return same ? aotx_ccir_writer_sync(&d->view, d->path) : AOTX_CCIR_CHANGED;
    }
    /* A new boot must reproduce its file checkpoint before it can replace the replay state. */
    if ((aotx_ccir_u32(header + 12) == 1 || aotx_ccir_u64(header + 16) != d->ring->boot) && !same)
        return AOTX_CCIR_CHANGED;
    FILE *replay = NULL; uint64_t replay_bytes = 0;
    aotx_ccir_limits limits; aotx_ccir_default_limits(&limits);
    rc = aotx_runtime_replay_collect(d->journal, d->ring->boot, aotx_cp_get(image + 72, 8),
        aotx_cp_get(image + 64, 8), d->runtime_sequence, limits.section_bytes, &replay, &replay_bytes);
    if (rc) return rc;
    aotx_ccir_input inputs[AOTX_CCIR_SECTIONS];
    memset(inputs, 0, sizeof(inputs));
    uint64_t packed = AOTX_CCIR_DATA + AOTX_CCIR_COMMIT + 256;
    int shrink = 0;
    for (uint32_t i = 0; i < d->view.count; ++i) {
        aotx_ccir_input *in = inputs + i;
        in->section = d->view.sections[i]; in->source = AOTX_CCIR_REUSE;
        if (in->section.type == AOTX_CCIR_CHECKPOINT) {
            shrink = bytes - base < in->section.bytes;
            in->source = AOTX_CCIR_MEMORY; in->data = image + base; in->section.bytes = bytes - base;
            in->section.schema = (uint16_t)aotx_cp_get(image + base + 8, 4);
        } else if (in->section.type == AOTX_CCIR_LIVE && (in->section.flags & AOTX_CCIR_REQUIRED)) {
            in->source = AOTX_CCIR_MEMORY; in->data = image; in->section.bytes = base;
        } else if (in->section.type == AOTX_CCIR_REPLAY && (in->section.flags & AOTX_CCIR_REQUIRED)) {
            in->source = AOTX_CCIR_FILE; in->fd = fileno(replay); in->section.bytes = replay_bytes;
        }
        uint64_t cost = AOTX_CCIR_ROW + in->section.alignment - 1 + in->section.bytes;
        if (cost > UINT64_MAX - packed) { fclose(replay); return AOTX_CCIR_LIMIT; }
        packed += cost;
    }
    unsigned char manifest[96];
    for (uint32_t i = 0; i < d->view.count; ++i) if (inputs[i].section.type == AOTX_CCIR_MANIFEST) {
        rc = aotx_ccir_pread(d->view.fd, manifest, sizeof(manifest), inputs[i].section.offset);
        if (rc) break;
        aotx_ccir_put(manifest + 20, aotx_cp_get(image + base + 8, 4), 4);
        inputs[i].source = AOTX_CCIR_MEMORY; inputs[i].data = manifest;
    }
    aotx_ccir_meta meta = {aotx_cp_get(image + 48, 8), aotx_cp_get(image + 48, 8), aotx_cp_get(image + 56, 8)};
    if (!rc) {
        if (shrink || (packed <= UINT64_MAX / 2 && d->view.end > 2 * packed))
            rc = aotx_ccir_writer_replace(&d->view, d->path, inputs, d->view.count, &meta, NULL);
        else rc = aotx_ccir_writer_append(&d->view, inputs, d->view.count, &meta, NULL);
        if (rc == AOTX_CCIR_LIMIT)
            rc = aotx_ccir_writer_replace(&d->view, d->path, inputs, d->view.count, &meta, NULL);
    }
    fclose(replay);
    return rc;
}
