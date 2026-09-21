/* Purpose: Define complete source spans from pinned Unicode sentence boundaries.
 * Owns: The property classes, byte offsets and profile identity.
 * Launch shape: One device thread per bounded source row.
 * Lifetime: One source parse; property data remains resident. */
#ifndef AOTX_COGNITIVE_SOURCE_SPANS_CUH
#define AOTX_COGNITIVE_SOURCE_SPANS_CUH
#include <stdint.h>
#define AOTX_SOURCE_BYTES 2048u
#define AOTX_SOURCE_SPANS 512u
#include "cognitive/source_profile.h"
typedef struct aotx_source_span { uint32_t start, length; } aotx_source_span;
enum aotx_source_class {
    AOTX_SB_OTHER, AOTX_SB_CR, AOTX_SB_LF, AOTX_SB_EXTEND, AOTX_SB_SEP,
    AOTX_SB_FORMAT, AOTX_SB_SP, AOTX_SB_LOWER, AOTX_SB_UPPER, AOTX_SB_OLETTER,
    AOTX_SB_NUMERIC, AOTX_SB_ATERM, AOTX_SB_STERM, AOTX_SB_CLOSE, AOTX_SB_CONTINUE
};
extern __device__ const uint32_t aotx_source_properties[][2];
extern __device__ const uint32_t aotx_source_property_count;
extern __device__ const unsigned char aotx_source_profile_digest[32];
__device__ bool aotx_source_split(const unsigned char *text, uint32_t bytes, uint32_t *units,
    aotx_source_span *spans, uint32_t *count, bool trim);
__device__ __forceinline__ uint32_t aotx_source_escaped(const unsigned char *text, uint32_t bytes) {
    uint32_t count = bytes;
    for (uint32_t i = 0; i < bytes; ++i)
        count += text[i] == '"' || text[i] == '\\' || text[i] == '\t' || text[i] == '\n';
    return count;
}
#endif
