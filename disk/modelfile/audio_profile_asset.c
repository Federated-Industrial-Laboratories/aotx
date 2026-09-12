/* Purpose: Read a sound profile from a store or validated runtime asset batch.
 * Owns: One bounded profile buffer and a short asset lease.
 * Threading: One disk loader; no device allocation occurs here.
 * Lifetime: One profile read. */
#include "disk/modelfile/audio_profile.h"
#include "disk/runtime/assets.h"
#include "disk/ccir/internal.h"
#include <errno.h>
#include <string.h>

int aotx_audio_profile_store(const char *store, aotx_audio_profile *profile)
{
    aotx_asset asset;
    unsigned char bytes[AOTX_AUDIO_PROFILE_BYTES];
    errno=0;
    if (aotx_asset_open(store,"audio.profile",&asset)) {
        if (errno != ENOENT || aotx_asset_is_runtime(store)) return 1;
        aotx_audio_profile_default(profile); return 0;
    }
    int rc=asset.bytes != sizeof(bytes) || aotx_asset_read(&asset,0,sizeof(bytes),bytes) ||
        aotx_audio_profile_read(bytes,sizeof(bytes),profile);
    aotx_asset_close(&asset); return rc;
}
int aotx_audio_profile_view(const aotx_ccir_view *view, const aotx_runtime_index *index,
                             aotx_audio_profile *profile)
{
    for (uint32_t i=0; i<index->count; ++i) {
        const unsigned char *row=index->rows[i];
        if (strcmp((const char *)row+64,"audio.profile") || aotx_ccir_u32(row+16) != 1) continue;
        int at=aotx_runtime_section(view,row);
        if (at < 0) return 1;
        const aotx_ccir_section *s=view->sections+at;
        unsigned char bytes[AOTX_AUDIO_PROFILE_BYTES];
        return s->bytes != sizeof(bytes) || aotx_ccir_pread(view->fd,bytes,sizeof(bytes),s->offset) ||
            aotx_audio_profile_read(bytes,sizeof(bytes),profile);
    }
    return 1;
}
