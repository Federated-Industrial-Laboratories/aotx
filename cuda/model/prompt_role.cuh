/* Purpose: Bind prompt framing to the language model selected for each slot.
 * Owns: The model role of each pending prompt and tool continuation.
 * Launch shape: One caller per slot; media checks take complete byte spans.
 * Lifetime: From prompt construction through the reply and its tool result. */
#ifndef AOTX_MODEL_PROMPT_ROLE_CUH
#define AOTX_MODEL_PROMPT_ROLE_CUH
#include "model/model.cuh"
#include "profile/profile.cuh"
extern __device__ unsigned aotx_prompt_roles[AOTX_SLOTS];
__device__ __forceinline__ unsigned aotx_prompt_role(unsigned slot)
{
    unsigned role = slot < AOTX_SLOTS ? aotx_prompt_roles[slot] : 0u;
    return role == AOTX_MODEL_LANGUAGE || role == AOTX_MODEL_LANGUAGE_Q4 ||
        role == AOTX_MODEL_LANGUAGE_AUDIO ? role : aotx_model_default_language();
}
/* Both media families require their own trained parent model. */
__device__ __forceinline__ unsigned aotx_prompt_select(const unsigned char *p, unsigned n)
{
    unsigned types = 0;
    for (unsigned i = 0; i + 7u <= n; ++i) {
        if (p[i] != '[' || p[i+6] != ':') continue;
        if (p[i+1]=='a' && p[i+2]=='u' && p[i+3]=='d' && p[i+4]=='i' && p[i+5]=='o') types |= 2u;
        if (p[i+1]=='i' && p[i+2]=='m' && p[i+3]=='a' && p[i+4]=='g' && p[i+5]=='e') types |= 1u;
    }
    return types == 3u ? AOTX_MODEL_ROLES : types == 2u ? AOTX_MODEL_LANGUAGE_AUDIO
        : aotx_model_default_language();
}
#endif
