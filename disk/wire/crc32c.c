/* Purpose: Compute CRC-32C over block bytes, with a table path and a hardware path.
 * Owns: The lookup table of the reference path, made once at the first call.
 * Threading: One thread; the table build is not safe against a second thread.
 * Lifetime: The life of the process. */
#include "disk/wire/diskwire.h"

/* The Castagnoli polynomial 0x1EDC6F41, in the reflected form that a table path uses. */
#define AOTX_CRC32C_POLY 0x82F63B78u

static uint32_t table[256];
static int table_ready;

static void build_table(void)
{
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int bit = 0; bit < 8; bit++) {
            c = (c & 1u) ? ((c >> 1) ^ AOTX_CRC32C_POLY) : (c >> 1);
        }
        table[i] = c;
    }
    table_ready = 1;
}

uint32_t aotx_crc32c_table(const void *data, size_t bytes, uint32_t seed)
{
    const unsigned char *p = (const unsigned char *)data;
    uint32_t crc = ~seed;
    if (!table_ready) {
        build_table();
    }
    for (size_t i = 0; i < bytes; i++) {
        crc = table[(crc ^ p[i]) & 0xffu] ^ (crc >> 8);
    }
    return ~crc;
}

#ifdef AOTX_CRC32C_HW
int aotx_crc32c_has_hardware(void)
{
    /* The hardware path holds a different instruction set. A machine without it must take
     * the table path, or the first call stops the process. */
    return __builtin_cpu_supports("sse4.2") ? 1 : 0;
}
#else
int aotx_crc32c_has_hardware(void)
{
    return 0;
}

uint32_t aotx_crc32c_hardware(const void *data, size_t bytes, uint32_t seed)
{
    return aotx_crc32c_table(data, bytes, seed);
}
#endif

uint32_t aotx_crc32c(const void *data, size_t bytes, uint32_t seed)
{
    if (aotx_crc32c_has_hardware()) {
        return aotx_crc32c_hardware(data, bytes, seed);
    }
    return aotx_crc32c_table(data, bytes, seed);
}
