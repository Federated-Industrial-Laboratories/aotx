/* Purpose: Identify two source extents of the same embedding calculation.
 * Owns: Processor digests and exact compatibility checks.
 * Launch shape: One helper call per query or stored vector.
 * Lifetime: Recorded vectors retain their original processor identity. */
#ifndef AOTX_COGNITIVE_TEXT_SPACE_CUH
#define AOTX_COGNITIVE_TEXT_SPACE_CUH
#include "cognitive/codec.cuh"
static __device__ const unsigned char aotx_text_short_processor[32] = {0x7d, 0x12, 0xaf, 0x1d, 0x2c, 0xd1, 0xe5, 0x19, 0x4d, 0xef, 0x98, 0x3d, 0x1f, 0xd8, 0x07, 0x3d, 0x1c, 0x36, 0xc4, 0x44, 0xee, 0xa3, 0x9c, 0x2d, 0xcf, 0x9b, 0xbe, 0x39, 0x4e, 0x75, 0x89, 0x2d};
static __device__ const unsigned char aotx_text_full_processor[32] = {0x3f, 0x5f, 0x1f, 0xdc, 0x06, 0x71, 0x57, 0xf5, 0xb8, 0x1e, 0x88, 0x18, 0x88, 0x5b, 0xbb, 0xa3, 0x04, 0x84, 0xc2, 0x33, 0xbe, 0xfe, 0x9d, 0x41, 0xaa, 0x77, 0xc0, 0x37, 0xce, 0xba, 0x5a, 0x36};
__device__ inline bool aotx_text_space(const unsigned char *a, const unsigned char *b) {
    if (aotx_cog_equal(a, b, 32)) return true;
    bool left = aotx_cog_equal(a, aotx_text_short_processor, 32) || aotx_cog_equal(a, aotx_text_full_processor, 32);
    bool right = aotx_cog_equal(b, aotx_text_short_processor, 32) || aotx_cog_equal(b, aotx_text_full_processor, 32);
    return left && right;
}
__device__ inline const unsigned char *aotx_text_identity(uint32_t bytes) {
    return bytes <= 192 ? aotx_text_short_processor : aotx_text_full_processor;
}
#endif
