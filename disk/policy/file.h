/* Purpose: Read and write portable creator policy bundles without code execution.
 * Owns: Exact bundle bytes, native image bytes, and validated disk metadata.
 * Threading: One caller processes each complete file or extent.
 * Lifetime: Read buffers remain valid until file close. */
#ifndef AOTX_POLICY_FILE_H
#define AOTX_POLICY_FILE_H
#include "cuda/policy/abi.h"
#include <stddef.h>
#include <stdint.h>
#define AOTX_POLICY_FILE_HEADER 256u
#ifndef AOTX_POLICY_METADATA_BYTES
#define AOTX_POLICY_METADATA_BYTES 65536u
#endif
#define AOTX_POLICY_FILE_TRUST 8
#define AOTX_POLICY_FILE_DIGEST 9
/* Little-endian header, 256 bytes: AOTXPL01 at 0; schema 1 at 8; mode at 12; ABI at 16.
 * State schema/bytes at 20/24; architecture/threads/registers/shared bytes/local bytes at 28/32/36/40/44.
 * Pressure/minimum movement/backoff/format at 48/52/56/60; image bytes (u64) at 64.
 * Provenance/license bytes (u32) at 72/76; entry name at 80..143; image SHA256 at 144..175.
 * Bytes 176..255 are zero. The entry has a zero terminator and zero padding.
 *
 * Image, provenance, and license extents follow the header without padding or trailing bytes.
 * Modes are supplied 1, rules 2, and native 3; image formats are none 0, PTX 1, and cubin 2.
 * The full bundle SHA256 defines its revision; this digest must come from the local operator for native admission.
 * Image storage has an additional zero byte outside the exact image extent. */
typedef struct aotx_policy_file {
    aotx_policy_config config;
    unsigned char digest[32];
    char entry[64];
    unsigned char *image;
    size_t image_bytes;
    unsigned char *buffer;
    size_t buffer_bytes;
    const unsigned char *provenance, *license;
    uint32_t provenance_bytes, license_bytes;
} aotx_policy_file;
typedef struct aotx_policy_source {
    aotx_policy_config config;
    const char *entry;
    const void *image, *provenance, *license;
    size_t image_bytes, provenance_bytes, license_bytes;
} aotx_policy_source;
#ifdef __cplusplus
extern "C" {
#endif
int aotx_policy_file_read(const char *path, const char *trust, int require_trust,
    aotx_policy_file *out);
/* Extent reads validate only. They never grant trust or execute code. */
int aotx_policy_file_extent(int fd, uint64_t offset, uint64_t bytes, aotx_policy_file *out);
int aotx_policy_file_decode(const void *bytes, size_t length, aotx_policy_file *out);
int aotx_policy_file_write(const char *path, const aotx_policy_source *source);
void aotx_policy_file_close(aotx_policy_file *file);
const char *aotx_policy_status_text(int status);
#ifdef __cplusplus
}
#endif
#endif
