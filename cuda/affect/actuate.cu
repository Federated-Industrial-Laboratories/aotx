/* Purpose: Apply the affect state to the sampler row of an open sequence.
 * Owns: No state; the affect state and its open law hold every input.
 * Launch shape: One device call for each sequence that opens.
 * Lifetime: One sequence snapshot. */
#include "affect/affect.cuh"

__device__ void aotx_affect_apply_how(unsigned int agent, aotx_model_how *how)
{
    if (agent >= AOTX_SLOTS || how == 0 || how->affect == 0u) {
        return;
    }
    const aotx_affect_agent_state *state = &aotx_affect_state[agent];
    const aotx_affect_law *law = &aotx_affect_laws[agent];
    float effective[AOTX_AFFECT_DATA_AXES];
    for (unsigned int j = 0u; j < AOTX_AFFECT_DATA_AXES; ++j) {
        int sum = (int)state->fast[j] + (int)state->slow[j];
        int bound = (int)rintf(law->cap[j] * 32768.0f);
        sum = (sum > bound) ? bound : (sum < -bound) ? -bound : sum;
        effective[j] = (float)sum / 32768.0f;
    }
    float heat = how->temperature * (1.0f + law->temperature_gain * effective[1]);
    how->temperature = fminf(fmaxf(heat, 0.0f), 2.0f);
    float voice = 1.0f + law->voice_gain * effective[0];
    how->voice_scale = fminf(fmaxf(voice, 0.0f), 2.0f);
}
