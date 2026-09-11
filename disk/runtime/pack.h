/* Purpose: Declare complete runtime file creation from selected source components.
 * Owns: Input descriptors and byte tables until the generation is published.
 * Threading: One packager streams a complete component batch.
 * Lifetime: One exclusive file creation. */
#ifndef AOTX_RUNTIME_PACK_H
#define AOTX_RUNTIME_PACK_H
#include "disk/runtime/runtime.h"
#include "disk/modelfile/modelfile.h"
#include "disk/ccir/internal.h"
#define AOTX_RUNTIME_REPLAY_HEADER 128u
typedef struct aotx_runtime_pack {
    aotx_ccir_input inputs[AOTX_CCIR_SECTIONS];
    uint32_t count;
    aotx_runtime_index index;
    unsigned char manifest[96], live[128], replay[128];
    unsigned char *memory;
    uint64_t memory_bytes;
    aotx_ccir_view source;
} aotx_runtime_pack;
int aotx_runtime_pack_asset(aotx_runtime_pack *pack, const char *path,
                             const char *name, uint32_t kind);
int aotx_runtime_pack_tree(aotx_runtime_pack *pack, const char *root,
                            const char *path, const char *prefix, uint32_t kind);
int aotx_runtime_pack_models(aotx_runtime_pack *pack, const char *store);
int aotx_runtime_pack_memory(aotx_runtime_pack *pack, const char *path);
int aotx_runtime_pack_settings(aotx_runtime_pack *pack, const char *path);
void aotx_runtime_pack_close(aotx_runtime_pack *pack);
#endif
