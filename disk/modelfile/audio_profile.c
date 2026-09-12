/* Purpose: Validate independently configurable sound capacities.
 * Owns: No allocation; output fields change only after complete validation.
 * Threading: One disk reader before device allocation.
 * Lifetime: One profile read or write. */
#include "disk/modelfile/audio_profile.h"
#include "cuda/media/wire.h"
#include <string.h>
void aotx_audio_profile_default(aotx_audio_profile *p)
{
    p->feature_rows=AOTX_AUDIO_FEATURE_ROWS; p->workers=AOTX_AUDIO_WORKERS;
    p->source_frames=AOTX_AUDIO_SOURCE_FRAMES;
}
int aotx_audio_profile_read(const unsigned char *p,uint64_t n,aotx_audio_profile *out)
{
    if (!p || !out || n!=AOTX_AUDIO_PROFILE_BYTES || memcmp(p,"AOTXAU01",8) || aotx_media_get(p+8,4)!=1) return 1;
    for (unsigned i=24;i<AOTX_AUDIO_PROFILE_BYTES;++i) if (p[i]) return 1;
    aotx_audio_profile v;
    v.feature_rows=(uint32_t)aotx_media_get(p+12,4); v.workers=(uint32_t)aotx_media_get(p+16,4);
    v.source_frames=(uint32_t)aotx_media_get(p+20,4);
    if (!v.feature_rows || !v.workers || v.workers>65535u || !v.source_frames || v.source_frames>1440000u) return 1;
    *out=v; return 0;
}
void aotx_audio_profile_write(const aotx_audio_profile *v,unsigned char p[AOTX_AUDIO_PROFILE_BYTES])
{
    memset(p,0,AOTX_AUDIO_PROFILE_BYTES); memcpy(p,"AOTXAU01",8); aotx_media_put(p+8,1,4);
    aotx_media_put(p+12,v->feature_rows,4); aotx_media_put(p+16,v->workers,4); aotx_media_put(p+20,v->source_frames,4);
}
int aotx_audio_profile_fits(const aotx_audio_profile *v)
{
    return v && v->feature_rows<=AOTX_AUDIO_FEATURE_ROWS && v->workers<=AOTX_AUDIO_WORKERS &&
        v->source_frames<=AOTX_AUDIO_SOURCE_FRAMES;
}
