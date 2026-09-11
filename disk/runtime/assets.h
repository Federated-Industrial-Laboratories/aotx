/* Purpose: Read assets from a model directory or a complete runtime file.
 * Owns: Short file leases and bounded source descriptors.
 * Threading: One loading thread; a batch may retain one validated runtime directory.
 * Lifetime: Each source ends at close; the batch ends before the device pump resumes. */
#ifndef AOTX_ASSETS_H
#define AOTX_ASSETS_H
#include "disk/runtime/runtime.h"
#include <stdio.h>
typedef struct aotx_asset {
    int fd;
    uint64_t offset, bytes;
    unsigned char digest[32];
} aotx_asset;
#ifdef __cplusplus
extern "C" {
#endif
int aotx_asset_is_runtime(const char *store);
int aotx_asset_begin(const char *store);
void aotx_asset_end(void);
int aotx_asset_index(const char *store, aotx_runtime_index *index);
int aotx_asset_open(const char *store, const char *name, aotx_asset *asset);
void aotx_asset_close(aotx_asset *asset);
int aotx_asset_read(const aotx_asset *asset, uint64_t offset, size_t bytes, void *out);
FILE *aotx_asset_stream(const char *store, const char *name);
int aotx_asset_names(const char *store, const char *prefix, const char *suffix,
                     char (*names)[AOTX_RUNTIME_NAME], unsigned capacity);
#ifdef __cplusplus
}
#endif
#endif
