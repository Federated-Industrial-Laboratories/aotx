/* Purpose: Open model metadata and tensors from one validated store asset.
 * Owns: A bounded source duplicate for the lifetime of each model file.
 * Threading: The loader reads each file within its model batch.
 * Lifetime: The model file closes its source before the loading batch ends. */
#include "disk/modelfile/modelfile.h"
#include "disk/runtime/assets.h"
#include <string.h>
int aotx_modelfile_open_entry(const char *store, const aotx_manifest_entry *entry,
                              aotx_modelfile **file) {
    if (!file) return 1;
    *file = NULL;
    if (!entry) return 1;
    aotx_asset asset;
    if (aotx_asset_open(store, entry->path, &asset)) return 1;
    int rc = aotx_modelfile_open_extent(entry->name, asset.fd, asset.offset, asset.bytes, file);
    aotx_asset_close(&asset);
    return rc;
}

int aotx_manifest_digest(const char *text, unsigned char digest[32])
{
    if (!text || !digest || strlen(text) != 64) return 1;
    for (unsigned int i = 0u; i < 32u; ++i) {
        unsigned int high = (text[2u * i] <= '9') ? (unsigned int)(text[2u * i] - '0')
                                                  : (unsigned int)(text[2u * i] - 'a') + 10u;
        unsigned int low = (text[2u * i + 1u] <= '9')
                         ? (unsigned int)(text[2u * i + 1u] - '0')
                         : (unsigned int)(text[2u * i + 1u] - 'a') + 10u;
        if (high > 15u || low > 15u) {
            return 1;
        }
        digest[i] = (unsigned char)((high << 4) | low);
    }
    return 0;
}
