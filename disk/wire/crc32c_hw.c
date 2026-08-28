/* Purpose: Compute CRC-32C with the processor instruction, as the fast path.
 * Owns: Nothing.
 * Threading: One thread; the function holds no state.
 * Lifetime: The life of the process. */
#include "disk/wire/diskwire.h"

/* This file is built with the SSE4.2 instruction set, and no other file is. A machine
 * without that instruction set takes the table path, which aotx_crc32c selects. */
#ifdef __SSE4_2__
#include <nmmintrin.h>

uint32_t aotx_crc32c_hardware(const void *data, size_t bytes, uint32_t seed)
{
    const unsigned char *p = (const unsigned char *)data;
    uint64_t crc = ~seed;
    size_t i = 0;
    while (bytes - i >= 8) {
        uint64_t word;
        __builtin_memcpy(&word, p + i, 8);
        crc = _mm_crc32_u64(crc, word);
        i += 8;
    }
    while (i < bytes) {
        crc = _mm_crc32_u8((uint32_t)crc, p[i]);
        i++;
    }
    return ~(uint32_t)crc;
}
#endif
