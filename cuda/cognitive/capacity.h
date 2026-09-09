/* Purpose: Define configured GPU object and payload allocation bounds.
 * Owns: Resource sizes; stored byte layouts remain in format.h.
 * Launch shape: Shared by all batched device and disk consumers in one build.
 * Lifetime: One complete runtime build. */
#ifndef AOTX_COGNITIVE_CAPACITY_H
#define AOTX_COGNITIVE_CAPACITY_H
#ifndef AOTX_MEMORY_OBJECTS
#define AOTX_MEMORY_OBJECTS 8192u
#endif
#ifndef AOTX_MEMORY_BYTES
#define AOTX_MEMORY_BYTES 16777216u
#endif
#define AOTX_COG_OBJECTS (AOTX_MEMORY_OBJECTS + 0u)
#define AOTX_COG_PAYLOAD (AOTX_MEMORY_BYTES + 0u)
#define AOTX_COG_WORDS ((AOTX_COG_OBJECTS + 31u) / 32u)
/* A live load carries two complete images in one uint32 byte count. */
#if AOTX_MEMORY_OBJECTS < 1 || AOTX_MEMORY_BYTES < 1 || \
    AOTX_MEMORY_OBJECTS > 2147483639ull || AOTX_MEMORY_BYTES > 2147483639ull || \
    (128ull + 256ull * AOTX_MEMORY_OBJECTS + AOTX_MEMORY_BYTES) > 2147483639ull
#error "memory capacity exceeds the live transfer byte count"
#endif
#endif
