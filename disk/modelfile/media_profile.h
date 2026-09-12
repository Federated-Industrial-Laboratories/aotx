/* Purpose: Read and write image allocation requirements in runtime assets.
 * Owns: No storage; the caller owns profile bytes and decoded fields.
 * Threading: One disk reader before device allocation or file creation.
 * Lifetime: One profile validation. */
#ifndef AOTX_MEDIA_PROFILE_IO_H
#define AOTX_MEDIA_PROFILE_IO_H
#include "cuda/media/profile.h"
#include "disk/runtime/runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
void aotx_media_profile_default(aotx_media_profile *profile);
int aotx_media_profile_read(const unsigned char *bytes, uint64_t count, aotx_media_profile *profile);
void aotx_media_profile_write(const aotx_media_profile *profile, unsigned char bytes[AOTX_MEDIA_PROFILE_BYTES]);
int aotx_media_profile_fits(const aotx_media_profile *profile);
int aotx_media_profile_store(const char *store, aotx_media_profile *profile);
int aotx_media_profile_view(const aotx_ccir_view *view, const aotx_runtime_index *index,
                             aotx_media_profile *profile);
#ifdef __cplusplus
}
#endif
#endif
