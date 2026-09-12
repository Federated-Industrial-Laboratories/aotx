/* Purpose: Define independent sound feature and workspace capacities.
 * Owns: No state; a portable profile records allocation requirements.
 * Threading: Disk readers validate fields before device allocation.
 * Lifetime: One runtime allocation. */
#ifndef AOTX_AUDIO_PROFILE_H
#define AOTX_AUDIO_PROFILE_H
#include <stdint.h>
#ifndef AOTX_AUDIO_FEATURE_ROWS
#define AOTX_AUDIO_FEATURE_ROWS 16384u
#endif
#ifndef AOTX_AUDIO_WORKERS
#define AOTX_AUDIO_WORKERS 1u
#endif
#ifndef AOTX_AUDIO_SOURCE_FRAMES
#define AOTX_AUDIO_SOURCE_FRAMES 1440000u
#endif
#define AOTX_AUDIO_PROFILE_BYTES 64u
/* The magic is AOTXAU01. Schema, feature rows, workers and source-frame capacity
 * are at 8, 12, 16 and 20. Bytes 24..63 are zero. */
typedef struct aotx_audio_profile {
    uint32_t feature_rows, workers, source_frames;
} aotx_audio_profile;
#endif
