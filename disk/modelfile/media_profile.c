/* Purpose: Validate independent image capacities from portable profile bytes.
 * Owns: No allocation; output fields change only after complete validation.
 * Threading: One disk reader, before any image allocation.
 * Lifetime: One profile read or write. */
#include "disk/modelfile/media_profile.h"
#include "cuda/media/wire.h"
#include <string.h>

void aotx_media_profile_default(aotx_media_profile *p)
{
    p->bytes=AOTX_MEDIA_BYTES; p->horizontal=AOTX_MEDIA_HORIZONTAL;
    p->objects=AOTX_MEDIA_OBJECTS; p->feature_rows=AOTX_MEDIA_FEATURE_ROWS;
    p->workers=AOTX_MEDIA_WORKERS; p->pixels=AOTX_MEDIA_PIXELS;
    p->dimension=AOTX_MEDIA_DIMENSION; p->patches=AOTX_MEDIA_PATCHES;
}
int aotx_media_profile_read(const unsigned char *p, uint64_t bytes, aotx_media_profile *out)
{
    aotx_media_profile v;
    if (!p || !out || bytes != AOTX_MEDIA_PROFILE_BYTES || memcmp(p,"AOTXIM01",8) ||
        aotx_media_get(p+8,4) != 1 || aotx_media_get(p+44,4)) return 1;
    for (unsigned i=56; i<AOTX_MEDIA_PROFILE_BYTES; ++i) if (p[i]) return 1;
    v.objects=(uint32_t)aotx_media_get(p+12,4); v.bytes=aotx_media_get(p+16,8);
    v.feature_rows=(uint32_t)aotx_media_get(p+24,4); v.workers=(uint32_t)aotx_media_get(p+28,4);
    v.pixels=(uint32_t)aotx_media_get(p+32,4); v.dimension=(uint32_t)aotx_media_get(p+36,4);
    v.patches=(uint32_t)aotx_media_get(p+40,4); v.horizontal=aotx_media_get(p+48,8);
    if (!v.objects || !v.bytes || v.bytes > UINT64_MAX/8u || !v.feature_rows || !v.workers ||
        v.workers > v.objects || v.workers > 65535u || !v.pixels || !v.dimension || v.dimension > 65535u ||
        v.patches < 256u || v.patches > 65536u || v.patches%4u || !v.horizontal ||
        v.horizontal > UINT64_MAX/8u) return 1;
    *out=v; return 0;
}
void aotx_media_profile_write(const aotx_media_profile *v, unsigned char *p)
{
    memset(p,0,AOTX_MEDIA_PROFILE_BYTES); memcpy(p,"AOTXIM01",8);
    aotx_media_put(p+8,1,4); aotx_media_put(p+12,v->objects,4); aotx_media_put(p+16,v->bytes,8);
    aotx_media_put(p+24,v->feature_rows,4); aotx_media_put(p+28,v->workers,4);
    aotx_media_put(p+32,v->pixels,4); aotx_media_put(p+36,v->dimension,4);
    aotx_media_put(p+40,v->patches,4); aotx_media_put(p+48,v->horizontal,8);
}
int aotx_media_profile_fits(const aotx_media_profile *v)
{
    return v && v->objects <= AOTX_MEDIA_OBJECTS && v->bytes <= AOTX_MEDIA_BYTES &&
        v->feature_rows <= AOTX_MEDIA_FEATURE_ROWS && v->workers <= AOTX_MEDIA_WORKERS &&
        v->pixels <= AOTX_MEDIA_PIXELS && v->dimension <= AOTX_MEDIA_DIMENSION &&
        v->patches <= AOTX_MEDIA_PATCHES && v->horizontal <= AOTX_MEDIA_HORIZONTAL;
}
