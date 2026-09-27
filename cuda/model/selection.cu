/* Purpose: Resolve exact qualified request controls on the device.
 * Owns: No state; failed requests cannot change the caller's sampling row.
 * Launch shape: One caller for each request in the admission batch.
 * Lifetime: A request keeps the qualification digest across queueing and recovery. */
#include "model/selection.cuh"
#include "model/control.cuh"
static __device__ unsigned aotx_selection_word(const unsigned char *p) {
    unsigned v = 0;
    for (unsigned i = 0; i < 4; ++i) v |= (unsigned)p[i] << (8 * i);
    return v;
}
__device__ bool aotx_control_selection_shape(const unsigned char *p) {
    unsigned used = 0;
    for (unsigned i = 0; i < AOTX_CONTROL_SELECTION_BYTES; ++i) used |= p[i];
    if (!used) return true;
    int dose = (int)aotx_selection_word(p + 8);
    unsigned digest = 0;
    for (unsigned i = 16; i < 48; ++i) digest |= p[i];
    return aotx_selection_word(p) == 1 && aotx_selection_word(p + 4) == AOTX_CONTROL_VECTOR &&
        dose && dose >= -40000 && dose <= 40000 && !aotx_selection_word(p + 12) && digest;
}
__device__ unsigned aotx_control_select(const unsigned char *p, unsigned role, aotx_model_how *how) {
    if (!aotx_control_selection_shape(p)) return 400;
    if (!aotx_selection_word(p)) return 200;
    if (role != aotx_model_default_language()) return 503;
    unsigned found = AOTX_CONDUCT_VECTORS;
    for (unsigned i = 0; i < aotx_conduct.vectors; ++i) {
        const aotx_steer_vector &v = aotx_conduct.vector[i];
        bool same = v.permit.status == AOTX_QUALIFICATION_ACCEPTED && aotx_control_matches(&v.identity, role);
        for (unsigned j = 0; j < 32; ++j) same &= v.permit.digest[j] == p[16 + j];
        if (!same) continue;
        if (found != AOTX_CONDUCT_VECTORS) return 503;
        found = i;
    }
    if (found == AOTX_CONDUCT_VECTORS) return 503;
#ifdef AOTX_AFFECT
    if (how->steer[AOTX_MODEL_CONDUCT_AFFECT] != AOTX_MODEL_CONDUCT_NONE) return 503;
#endif
    float strength = (float)(int)aotx_selection_word(p + 8) / 10000.0f;
    if (!aotx_conduct_dose(&aotx_conduct.vector[found].permit, strength) || how->affect ||
        (how->voice != AOTX_MODEL_CONDUCT_NONE && how->voice_scale != 0.0f)) return 503;
    /* All checks precede the first change to the caller's sampling row. */
    for (unsigned i = 0; i < AOTX_MODEL_STEERS; ++i) {
        how->steer[i] = AOTX_MODEL_CONDUCT_NONE; how->steer_strength[i] = 0;
    }
    how->steer[0] = found;
    how->steer_strength[0] = strength;
    return 200;
}
