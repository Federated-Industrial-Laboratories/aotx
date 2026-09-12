/* Purpose: Describe the trained sound encoder and validated weight offsets.
 * Owns: No storage; offsets name bytes in a validated weight extent.
 * Threading: Disk readers write descriptors before device use.
 * Lifetime: From component load to release. */
#ifndef AOTX_AUDIO_FORMAT_H
#define AOTX_AUDIO_FORMAT_H
#include <stdint.h>
#define AOTX_AUDIO_LAYERS 32u
#define AOTX_AUDIO_WIDTH 1280u
#define AOTX_AUDIO_HIDDEN 5120u
#define AOTX_AUDIO_OUTPUT 4096u
#define AOTX_AUDIO_SAMPLES 480000u
#define AOTX_AUDIO_FRAMES 3000u
#define AOTX_AUDIO_KEYS 1500u
#define AOTX_AUDIO_ROWS 750u
#define AOTX_AUDIO_PCM_HEAD 32u
#define AOTX_AUDIO_WAV 3u
#define AOTX_AUDIO_PCM 4u
#define AOTX_AUDIO_S16 1u
#define AOTX_AUDIO_F32 3u
/* Raw bytes start with AOTXPCM1. Encoding, rate and channels are at 8, 12 and 16.
 * Bytes 20..23 are zero. The sample-frame count is at 24. All fields are little endian. */
enum aotx_audio_base_tensor {
    AOTX_AUDIO_CONV1, AOTX_AUDIO_CONV1_BIAS, AOTX_AUDIO_CONV2, AOTX_AUDIO_CONV2_BIAS,
    AOTX_AUDIO_POSITION, AOTX_AUDIO_NORM, AOTX_AUDIO_NORM_BIAS,
    AOTX_AUDIO_PROJECT, AOTX_AUDIO_PROJECT_BIAS, AOTX_AUDIO_BASE_TENSORS
};
enum aotx_audio_layer_tensor {
    AOTX_AUDIO_LN1, AOTX_AUDIO_LN1_BIAS, AOTX_AUDIO_LN2, AOTX_AUDIO_LN2_BIAS,
    AOTX_AUDIO_Q, AOTX_AUDIO_Q_BIAS, AOTX_AUDIO_K, AOTX_AUDIO_V, AOTX_AUDIO_V_BIAS,
    AOTX_AUDIO_OUT, AOTX_AUDIO_OUT_BIAS, AOTX_AUDIO_UP, AOTX_AUDIO_UP_BIAS,
    AOTX_AUDIO_DOWN, AOTX_AUDIO_DOWN_BIAS, AOTX_AUDIO_LAYER_TENSORS
};
typedef struct aotx_audio_desc {
    uint64_t bytes;
    uint64_t base[AOTX_AUDIO_BASE_TENSORS];
    uint64_t layer[AOTX_AUDIO_LAYERS][AOTX_AUDIO_LAYER_TENSORS];
} aotx_audio_desc;
#endif
