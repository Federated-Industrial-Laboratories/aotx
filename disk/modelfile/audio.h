/* Purpose: Validate the trained sound component before its weights reach the device.
 * Owns: No allocation; the caller owns the open file and result descriptor.
 * Threading: The disk reader validates each file before load.
 * Lifetime: From file open to device weight installation. */
#ifndef AOTX_MODELFILE_AUDIO_H
#define AOTX_MODELFILE_AUDIO_H
#include "disk/modelfile/modelfile.h"
#include "cuda/audio/format.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Returns zero only for a complete supported tensor and processor contract. */
int aotx_audio_file(const aotx_modelfile *file, aotx_audio_desc *out);
/* Returns two entries, zero for an absent manifest, or -1 for invalid input. */
int aotx_audio_manifest(const char *store, aotx_manifest_entry entries[2]);
int aotx_audio_manifest_text(const char *text, size_t bytes, aotx_manifest_entry entries[2]);
/* Returns the exact paired language entry index, or -1. */
int aotx_audio_pair(const aotx_manifest_entry pair[2], const aotx_manifest_entry *models,
                      unsigned count);
#ifdef __cplusplus
}
#endif
#endif
