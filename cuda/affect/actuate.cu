/* Purpose: Apply the affect state to the sampler row of an open sequence.
 * Owns: No state; the affect state and its open law hold every input.
 * Launch shape: One device call for each sequence that opens.
 * Lifetime: One sequence snapshot. */
#include "affect/affect.cuh"
#include "cli/prompt.cuh"

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
    if (law->steer_gain != 0.0f && aotx_affect_composite_table.trusted != 0u
        && aotx_affect_steer != 0) {
        how->steer[AOTX_MODEL_CONDUCT_AFFECT] = AOTX_MODEL_CONDUCT_AFFECT;
        how->steer_strength[AOTX_MODEL_CONDUCT_AFFECT] = 1.0f;
    } else {
        how->steer[AOTX_MODEL_CONDUCT_AFFECT] = AOTX_MODEL_CONDUCT_NONE;
        how->steer_strength[AOTX_MODEL_CONDUCT_AFFECT] = 0.0f;
    }
}

__global__ void aotx_affect_build(void)
{
    unsigned int agent = blockIdx.x;
    if (agent >= AOTX_SLOTS) return;
    const aotx_say_slot *say = &aotx_say.slot[agent];
    unsigned int opens = say->wanted != 0u && aotx_say_count[agent] != 0u;
    if (opens == 0u) return;
    if (threadIdx.x == 0u) aotx_affect_snapshot(agent);
    __syncthreads();
    aotx_affect_agent_state *state = &aotx_affect_state[agent];
    const aotx_affect_law *law = &aotx_affect_laws[agent];
    const aotx_affect_composite_desc *table = &aotx_affect_composite_table;
    float effective[AOTX_AFFECT_DATA_AXES];
    float d[AOTX_AFFECT_DATA_AXES];
    for (unsigned int j = 0u; j < AOTX_AFFECT_DATA_AXES; ++j) {
        int sum = (int)state->fast[j] + (int)state->slow[j];
        int bound = (int)rintf(law->cap[j] * 32768.0f);
        sum = (sum > bound) ? bound : (sum < -bound) ? -bound : sum;
        effective[j] = (float)sum / 32768.0f;
        d[j] = law->steer_gain * effective[j];
    }
    float q = d[0] * (table->K[0][0] * d[0] + table->K[0][1] * d[1])
            + d[1] * (table->K[1][0] * d[0] + table->K[1][1] * d[1]);
    float scale = 1.0f;
    if (q > 2.0f * law->budget && q > 0.0f) {
        scale = sqrtf(fmaxf(0.0f, 2.0f * law->budget / q));
    }
    unsigned int applied = law->on != 0u && table->trusted != 0u && law->steer_gain != 0.0f
                         && (d[0] != 0.0f || d[1] != 0.0f);
    if (threadIdx.x == 0u) {
        state->scale = (unsigned short)rintf(scale * 65535.0f);
        state->axes = (unsigned short)AOTX_AFFECT_DATA_AXES;
        state->actuator_flags = (applied != 0u) ? AOTX_AFFECT_FLAG_COMPOSITE : 0u;
        if (applied != 0u && scale < 1.0f) state->actuator_flags |= AOTX_AFFECT_FLAG_BUDGET;
    }
    unsigned long long cells = (unsigned long long)table->layer_count * table->hidden;
    for (unsigned long long cell = threadIdx.x; cell < cells; cell += blockDim.x) {
        float value = 0.0f;
        if (applied != 0u) {
            for (unsigned int j = 0u; j < AOTX_AFFECT_DATA_AXES; ++j) {
                value += scale * d[j] * aotx_affect_composite[(unsigned long long)j * cells
                                                               + cell];
            }
        }
        aotx_affect_steer[(unsigned long long)agent * cells + cell] = value;
    }
}
