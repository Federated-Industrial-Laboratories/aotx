/* Purpose: Mark candidate registrations from isolated measurement programs.
 * Owns: No persistent storage or runtime option.
 * Launch shape: Host glue; the normal registration places the candidate batch.
 * Lifetime: Only the calling measurement program can use this admission. */
#include "model/conduct.cuh"
int aotx_conduct_register_measurement(const char *name, const unsigned *layers,
    unsigned count, unsigned hidden, const float *values, float potency, unsigned positions) {
    aotx_control_permit permit = {};
    permit.status = AOTX_QUALIFICATION_MEASUREMENT;
    return aotx_conduct_register_vector(name, layers, count, hidden, values, potency, positions, &permit);
}
