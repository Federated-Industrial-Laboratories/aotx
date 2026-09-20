/* Purpose: Read and preserve exact cold payload extents in CCIR files.
 * Owns: Bounded file catalogs and temporary extent files; no memory selection.
 * Threading: Each caller owns its catalog and file descriptor lease.
 * Lifetime: One file check, read batch or generation write. */
#ifndef AOTX_COLD_IO_H
#define AOTX_COLD_IO_H
#include "cuda/cognitive/cold.h"
#include "disk/ccir/ccir.h"
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct aotx_cold_catalog {
    int fd;
    uint32_t count;
    uint64_t payload, bytes;
    unsigned char *rows;
} aotx_cold_catalog;
int aotx_cold_catalog_open(const aotx_ccir_view *view, aotx_cold_catalog *catalog);
void aotx_cold_catalog_close(aotx_cold_catalog *catalog);
int aotx_cold_catalog_find(const aotx_cold_catalog *catalog, const unsigned char *row);
int aotx_cold_catalog_read(const aotx_cold_catalog *catalog, uint32_t index,
    uint64_t offset, size_t bytes, unsigned char *out);
int aotx_cold_section_build(const aotx_ccir_view *view, const unsigned char *image,
    uint64_t bytes, aotx_ccir_input *input, FILE **temporary);
int aotx_cold_profile(const aotx_ccir_view *view);
int aotx_cold_commit(aotx_ccir_view *view, const char *path, const unsigned char *image,
    uint64_t bytes, aotx_ccir_input *inputs, uint32_t count, const aotx_ccir_meta *meta, int replace);
struct aotx_checkpoint_disk;
int aotx_cold_disk_pass(struct aotx_checkpoint_disk *disk);
void aotx_cold_disk_close(struct aotx_checkpoint_disk *disk);
#ifdef __cplusplus
}
#endif
#endif
