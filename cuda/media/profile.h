/* Purpose: Define independently configurable image storage and workspace capacities.
 * Owns: No state; a runtime profile records the capacities used by its sources.
 * Threading: Disk readers validate profile bytes before device allocation.
 * Lifetime: One runtime allocation. */
#ifndef AOTX_MEDIA_PROFILE_H
#define AOTX_MEDIA_PROFILE_H
#include <stdint.h>
#ifndef AOTX_MEDIA_OBJECTS
#define AOTX_MEDIA_OBJECTS 128u
#endif
#ifndef AOTX_MEDIA_BYTES
#define AOTX_MEDIA_BYTES 67108864ull
#endif
#ifndef AOTX_MEDIA_FEATURE_ROWS
#define AOTX_MEDIA_FEATURE_ROWS 131072u
#endif
#ifndef AOTX_MEDIA_WORKERS
#define AOTX_MEDIA_WORKERS 1u
#endif
#ifndef AOTX_MEDIA_PIXELS
#define AOTX_MEDIA_PIXELS 4194304u
#endif
#ifndef AOTX_MEDIA_DIMENSION
#define AOTX_MEDIA_DIMENSION 8192u
#endif
#ifndef AOTX_MEDIA_PATCHES
#define AOTX_MEDIA_PATCHES 8192u
#endif
#ifndef AOTX_MEDIA_HORIZONTAL
#define AOTX_MEDIA_HORIZONTAL 16777216ull
#endif
#define AOTX_MEDIA_PROFILE_BYTES 80u
/* The magic is AOTXIM01. The schema, object count and source bytes are at 8, 12 and 16.
 * Feature rows, workers, pixels, dimensions and patches are at 24, 28, 32, 36 and 40.
 * Horizontal float capacity is at 48. Bytes 44..47 and 56..79 are zero. */
typedef struct aotx_media_profile {
    uint64_t bytes, horizontal;
    uint32_t objects, feature_rows, workers, pixels, dimension, patches;
} aotx_media_profile;
#endif
