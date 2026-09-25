/* Purpose: Validate recorded appraisal rows without another model call.
 * Owns: Versioned output bounds, model references and call boundary admission.
 * Launch shape: One thread restores each independent source row.
 * Lifetime: One complete recorded decision before canonical byte comparison. */
#ifndef AOTX_APPRAISAL_REPLAY_CUH
#define AOTX_APPRAISAL_REPLAY_CUH
#include "appraisal/outcome_parse.cuh"
#include "model/load.cuh"

__device__ inline bool aotx_appraisal_saved_model(const unsigned char *digest) {
    for (uint32_t role = 0; role < AOTX_MODEL_ROLES; ++role)
        if (aotx_model_is_language(role) && aotx_model_load.resident[role].active &&
            aotx_cog_equal(digest, aotx_model_load.resident[role].body.digest, 32)) return true;
    for (uint32_t file = 0; file < aotx_model_load.files; ++file)
        if (aotx_model_is_language(aotx_model_load.file[file].role) &&
            aotx_cog_equal(digest, aotx_model_load.file[file].digest, 32)) return true;
    return false;
}
__device__ inline uint32_t aotx_appraisal_replay_row(uint32_t row,
    const unsigned char *p, uint32_t version, uint32_t status) {
    const unsigned char *queue = aotx_live_store.objects[aotx_appraisal.rows[row].queue];
    aotx_intake_row *r = aotx_intake.rows + row;
    r->bytes = aotx_cog_u32(p + 56); r->status = aotx_cog_u32(p + 60);
    r->phase = r->second_call = r->first_bytes = 0;
    bool model = !aotx_cog_zero(p + 24, 32);
    if (!aotx_cog_equal(p, queue + AOTX_CO_ID) || aotx_cog_u64(p + 16) != aotx_cog_u64(queue + AOTX_CO_VERSION) ||
        r->bytes > AOTX_INTAKE_REPLY || r->status != status ||
        (!status && (!r->bytes || !model)) || (r->bytes && !model) ||
        (model && !aotx_appraisal_saved_model(p + 24)) ||
        !aotx_cog_zero(p + 64 + min(r->bytes, AOTX_INTAKE_REPLY), AOTX_INTAKE_REPLY - min(r->bytes, AOTX_INTAKE_REPLY)))
        return AOTX_COG_REFERENCE;
    for (uint32_t j = 0; j < 32; ++j) {
        r->model[j] = p[24 + j]; aotx_appraisal.rows[row].first_model[j] = 0;
    }
    for (uint32_t j = 0; j < r->bytes; ++j) r->reply[j] = p[64 + j];
    if (version == 2) {
        r->first_bytes = aotx_cog_u32(p + 4160); r->second_call = aotx_cog_u32(p + 4164);
        r->phase = aotx_cog_u32(p + 4168);
        bool first_model = !aotx_cog_zero(p + 4192, 32);
        if (r->first_bytes > AOTX_INTAKE_REPLY || r->second_call > 1 || r->phase > 2 ||
            !aotx_cog_zero(p + 4172, 20) ||
            !aotx_cog_zero(p + 4224 + min(r->first_bytes, AOTX_INTAKE_REPLY), AOTX_INTAKE_REPLY - min(r->first_bytes, AOTX_INTAKE_REPLY)) ||
            (first_model && !aotx_cog_equal(p + 4192, p + 24, 32)) || (r->first_bytes && !first_model) ||
            (!r->phase && (r->first_bytes || r->bytes || r->second_call || first_model)) ||
            (r->phase == 1 && (r->bytes || r->second_call)) ||
            (r->phase == 2 && (!r->first_bytes || !first_model || (r->bytes && !r->second_call))) ||
            (!status && (r->phase != 2 || r->second_call != 1))) return AOTX_COG_FORMAT;
        for (uint32_t j = 0; j < r->first_bytes; ++j) r->first_reply[j] = p[4224 + j];
        for (uint32_t j = 0; j < 32; ++j) aotx_appraisal.rows[row].first_model[j] = p[4192 + j];
        if (r->phase == 2) {
            uint32_t checked = aotx_appraisal_outcome_parse(row, r->first_reply, r->first_bytes);
            if (checked) return checked;
        }
    }
    if (aotx_appraisal.recovery && (status != AOTX_COG_DENIED || model || r->bytes ||
        r->phase || r->first_bytes || r->second_call)) return AOTX_COG_REFERENCE;
    return 0;
}
#endif
