/* Purpose: Resolve scoped image links and construct typed language input rows.
 * Owns: Per-slot source references and expanded token and position maps.
 * Launch shape: One thread per conversation slot; callers reset before a new prompt.
 * Lifetime: Prompt preparation through sequence installation. */
#ifndef AOTX_MEDIA_PROMPT_CUH
#define AOTX_MEDIA_PROMPT_CUH
#include "model/forward.cuh"
#include "profile/profile.cuh"
#define AOTX_MEDIA_REFS (AOTX_SAY_BYTES / 72u + 1u)
#define AOTX_MEDIA_VISION_START 248053u
#define AOTX_MEDIA_VISION_END 248054u
#define AOTX_MEDIA_IMAGE_PAD 248056u
#define AOTX_MEDIA_AUDIO_PAD 151646u
#define AOTX_MEDIA_AUDIO_START 151647u
#define AOTX_MEDIA_AUDIO_END 151648u
struct aotx_media_reference { unsigned object; unsigned long long generation; };
struct aotx_media_prompt_state {
    unsigned stage, count, extra, turn_extra, raw_length, raw_turn, raw_system;
    unsigned char digest[32];
    aotx_media_reference reference[AOTX_MEDIA_REFS];
};
extern __device__ aotx_media_prompt_state aotx_media_prompts[AOTX_SLOTS];
extern __device__ aotx_model_input aotx_media_input[AOTX_SLOTS][AOTX_SEQ_MAX_TOKENS];
__global__ void aotx_media_prepare(void);
__device__ unsigned aotx_media_expand(unsigned slot, unsigned count);
__device__ bool aotx_media_leased(unsigned object);
__device__ bool aotx_media_retry(unsigned slot);
#endif
