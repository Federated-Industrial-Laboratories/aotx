/* Purpose: Include all cold extents before publishing a memory generation.
 * Owns: The temporary cold section and complete directory batch.
 * Threading: The checkpoint writer holds the exclusive file lease.
 * Lifetime: One atomic generation publication or compaction. */
#include "disk/cognitive/cold_io.h"
#include "disk/ccir/internal.h"

int aotx_cold_commit(aotx_ccir_view *v, const char *path, const unsigned char *image,
    uint64_t bytes, aotx_ccir_input *inputs, uint32_t count, const aotx_ccir_meta *meta, int replace) {
    uint32_t at = count;
    for (uint32_t i = 0; i < count; ++i) if (inputs[i].section.type == AOTX_CCIR_COLD) at = i;
    FILE *file = NULL;
    int rc = 0;
    if (at < count || (bytes >= AOTX_COG_HEADER && aotx_ccir_u32(image + 8) == 3)) {
        if (at == count && count == AOTX_CCIR_SECTIONS) return AOTX_CCIR_LIMIT;
        rc = aotx_cold_section_build(v, image, bytes, inputs + at, &file);
        if (!rc && at == count) ++count;
    }
    if (!rc) rc = replace ? aotx_ccir_writer_replace(v, path, inputs, count, meta, NULL) :
        aotx_ccir_writer_append(v, inputs, count, meta, NULL);
    if (rc == AOTX_CCIR_LIMIT && !replace)
        rc = aotx_ccir_writer_replace(v, path, inputs, count, meta, NULL);
    if (file) fclose(file);
    return rc;
}
