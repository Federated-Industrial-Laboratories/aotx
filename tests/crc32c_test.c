/* Purpose: Check the checksum against the published value and check the two paths agree.
 * Owns: The buffers of the cases.
 * Threading: One thread.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include <stdlib.h>

#define AOTX_CHECK_VALUE 0xE3069283u

static void fill(unsigned char *buffer, size_t bytes, unsigned seed)
{
    size_t i;
    /* Every element carries different content, so a wrong index cannot hide. */
    for (i = 0; i < bytes; i++) {
        seed = seed * 1103515245u + 12345u;
        buffer[i] = (unsigned char)(seed >> 16);
    }
}

static void batch(int n)
{
    unsigned char buffer[4096];
    int i;
    for (i = 0; i < n; i++) {
        size_t bytes = (size_t)(1 + i * 37) % 3000u;
        uint32_t table_value;
        uint32_t chosen;
        fill(buffer, bytes, (unsigned)(i + 1));
        table_value = aotx_crc32c_table(buffer, bytes, 0);
        chosen = aotx_crc32c(buffer, bytes, 0);
        CHECK(table_value == chosen, "path values differ at element %d", i);
        if (aotx_crc32c_has_hardware()) {
            uint32_t hardware = aotx_crc32c_hardware(buffer, bytes, 0);
            CHECK(table_value == hardware, "hardware value differs at element %d", i);
        }
        if (bytes > 8) {
            uint32_t whole = aotx_crc32c_table(buffer, bytes, 0);
            uint32_t part = aotx_crc32c_table(buffer, 8, 0);
            part = aotx_crc32c_table(buffer + 8, bytes - 8, part);
            CHECK(whole == part, "the seed does not carry at element %d", i);
        }
    }
}

int main(void)
{
    const char *known = "123456789";
    printf("hardware path present: %d\n", aotx_crc32c_has_hardware());
    CHECK(aotx_crc32c_table(known, 9, 0) == AOTX_CHECK_VALUE,
          "the table path gives %08x", aotx_crc32c_table(known, 9, 0));
    CHECK(aotx_crc32c(known, 9, 0) == AOTX_CHECK_VALUE, "the chosen path gives another value");
    if (aotx_crc32c_has_hardware()) {
        CHECK(aotx_crc32c_hardware(known, 9, 0) == AOTX_CHECK_VALUE,
              "the hardware path gives another value");
    }
    CHECK(aotx_crc32c_table("", 0, 0) == 0, "an empty buffer must give zero");
    batch(1);
    batch(64);
    return aotx_report("crc32c_test", 100);
}
