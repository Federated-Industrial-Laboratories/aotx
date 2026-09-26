/* Purpose: Admit exact accepted control doses at each device consumer.
 * Owns: No state; the conduct table holds the immutable evidence result.
 * Launch shape: One caller for each sequence in the input batch.
 * Lifetime: Each call checks the current selected controls. */
#include "model/conduct.cuh"
#include "model/control.cuh"

__device__ bool aotx_conduct_setting(const aotx_model_how *how) {
    if (!how) return true;
    unsigned active = 0, measured = 0;
    for (unsigned i = 0; i < AOTX_MODEL_STEERS; ++i) {
#ifdef AOTX_AFFECT
        if (i == AOTX_MODEL_CONDUCT_AFFECT) continue;
#endif
        if (how->steer[i] == AOTX_MODEL_CONDUCT_NONE || how->steer_strength[i] == 0.0f) continue;
        if (how->steer[i] >= aotx_conduct.vectors) return false;
        const aotx_steer_vector *v = aotx_conduct.vector + how->steer[i];
        const aotx_control_permit *p = &v->permit;
        if (!aotx_control_matches(&v->identity, aotx_model_default_language())) return false;
        ++active;
        if (p->status == AOTX_QUALIFICATION_MEASUREMENT) { ++measured; continue; }
        if (p->status != AOTX_QUALIFICATION_ACCEPTED || !p->count || p->count > AOTX_QUALIFICATION_DOSES) return false;
        bool matched = false;
        for (unsigned j = 0; j < p->count; ++j)
            if (p->dose[j] && how->steer_strength[i] == (float)p->dose[j] / 10000.0f) matched = true;
        if (!matched) return false;
    }
    if (!active || active == measured) return true;
    if (measured || active != 1 || how->affect || (how->voice != AOTX_MODEL_CONDUCT_NONE && how->voice_scale != 0.0f)) return false;
#ifdef AOTX_AFFECT
    if (how->steer[AOTX_MODEL_CONDUCT_AFFECT] != AOTX_MODEL_CONDUCT_NONE) return false;
#endif
    return true;
}
