/* Purpose: Render bounded statement and classification prompts on the device.
 * Owns: Exact fixed-byte accounting and the captured model wrapper.
 * Launch shape: One prompt per leased source in the current batch.
 * Lifetime: Both internal calls and recorded capacity checks. */
#ifndef AOTX_COGNITIVE_INTAKE_PROMPT_CUH
#define AOTX_COGNITIVE_INTAKE_PROMPT_CUH
__device__ inline uint32_t aotx_intake_wrap(unsigned char *out, uint32_t at,
    const aotx_wrap *wrap, uint32_t span, bool prefix = false) {
    uint32_t length = prefix ? wrap->prefix_length : wrap->length[span];
    return aotx_recall_run(out, at, AOTX_SAY_BYTES, wrap->bytes + wrap->offset[span], length);
}
__device__ inline uint32_t aotx_intake_render(uint32_t row, uint32_t role,
    unsigned char *out, bool targets) {
    const aotx_wrap *wrap = aotx_wrap_active(role);
    const aotx_intake_row *r = aotx_intake.rows + row;
    const unsigned char *q = aotx_live.requests + 64 + row * AOTX_RECALL_QUERY;
    uint32_t cap = AOTX_SAY_BYTES, at = 0;
    at = aotx_intake_wrap(out, at, wrap, AOTX_WRAP_SYSTEM_HEAD, true);
    at = aotx_intake_wrap(out, at, wrap, AOTX_WRAP_SYSTEM_HEAD);
    at = aotx_recall_word(out, at, cap, r->phase == 1 ? aotx_intake_statement_instruction : aotx_intake_source_instruction);
    at = aotx_intake_wrap(out, at, wrap, AOTX_WRAP_SYSTEM_TAIL);
    at = aotx_intake_wrap(out, at, wrap, AOTX_WRAP_USER_HEAD);
    if (r->phase == 2) {
        at = aotx_recall_word(out, at, cap, "Prior assertions (index: quote):\n");
        if (targets) for (uint32_t j = 0; j < r->target_count; ++j)
            at = aotx_intake_target_text(r->target_index[j], j + 1, out, at, cap);
    }
    at = aotx_recall_word(out, at, cap, r->phase == 1 ? "Source actor: " : "Current statements:\n[source_actor=");
    at = aotx_cog_zero(q + AOTX_RECALL_ACTOR, 16) ? aotx_recall_word(out, at, cap, "unknown") :
        aotx_recall_hex(out, at, cap, q + AOTX_RECALL_ACTOR);
    at = aotx_recall_word(out, at, cap, r->phase == 1 ? "\n<source>\n" : "]\n");
    for (uint32_t j = 0; j < (r->phase == 1 ? r->source_count : r->first_count); ++j) {
        const aotx_intake_span *span = r->statements + j;
        at = aotx_recall_number(out, at, cap, j + 1);
        at = aotx_recall_word(out, at, cap, ": ");
        if (r->phase == 1) {
            at = aotx_recall_word(out, at, cap, "\"");
            for (uint32_t k = 0; k < span->length; ++k) {
                unsigned char c = q[4640 + span->start + k];
                if (c == '\t' || c == '\n' || c == '"' || c == '\\') {
                    at = aotx_recall_word(out, at, cap, "\\");
                    if (c == '\t') c = 't'; else if (c == '\n') c = 'n';
                }
                at = aotx_recall_run(out, at, cap, &c, 1);
            }
            at = aotx_recall_word(out, at, cap, "\"");
        } else at = aotx_recall_run(out, at, cap, q + 4640 + span->start, span->length);
        at = aotx_recall_word(out, at, cap, "\n");
    }
    at = aotx_recall_word(out, at, cap, r->phase == 1 ? aotx_intake_statement_reminder : aotx_intake_source_reminder);
    at = aotx_intake_wrap(out, at, wrap, AOTX_WRAP_USER_TAIL);
    at = aotx_intake_wrap(out, at, wrap, AOTX_WRAP_GENERATION_HEAD);
    at = aotx_intake_wrap(out, at, wrap, AOTX_WRAP_THINK_OPEN);
    return aotx_intake_wrap(out, at, wrap, AOTX_WRAP_THINK_CLOSE);
}
__device__ uint32_t aotx_intake_target_capacity(uint32_t row, uint32_t role) {
    if (role >= AOTX_MODEL_ROLES || !aotx_wrap_active(role)->usable) return UINT32_MAX;
    if (aotx_intake_span_capacity(row)) return UINT32_MAX;
    uint32_t bytes = aotx_intake_render(row, role, 0, false);
    return bytes > AOTX_SAY_BYTES ? UINT32_MAX : min(AOTX_INTAKE_TARGET_BUDGET, AOTX_SAY_BYTES - bytes);
}
static __device__ uint32_t aotx_intake_source_prompt(uint32_t row, uint32_t slot) {
    aotx_intake_row *r = aotx_intake.rows + row;
    const aotx_shared_receipt *shared = aotx_shared_request(slot);
    aotx_prompt_roles[slot] = r->phase == 1 ? (shared ? shared->role : aotx_model_default_language()) : r->target_role;
    for (uint32_t pass = 0; pass < (r->phase == 1 ? 2u : 1u); ++pass) {
        uint32_t role = aotx_prompt_role(slot);
        if (role >= AOTX_MODEL_ROLES || !aotx_model_is_language(role) || !aotx_model_load.resident[role].active ||
            !aotx_model_wrap[role].usable || aotx_say.slot[slot].live || aotx_say.slot[slot].wanted) return AOTX_COG_DENIED;
        if (r->phase == 2 && (!aotx_cog_equal(r->model, aotx_model_load.resident[role].body.digest, 32) ||
            !aotx_cog_equal((const unsigned char *)&r->wrapper, (const unsigned char *)aotx_wrap_active(role), sizeof(r->wrapper))))
            return AOTX_COG_LAYOUT;
        uint32_t capacity = aotx_intake_target_capacity(row, role);
        if (capacity == UINT32_MAX) return AOTX_COG_CAPACITY;
        uint32_t status = aotx_intake_targets_prepare(row, r->phase == 1 ? 0 : capacity);
        if (status) return status;
        uint32_t bytes = aotx_intake_render(row, role, aotx_say.prompt[slot], true);
        if (bytes > AOTX_SAY_BYTES) return AOTX_COG_CAPACITY;
        if (r->phase == 1) {
            uint32_t selected = shared ? shared->role : aotx_prompt_select(aotx_say.prompt[slot], bytes);
            if (selected != role) { aotx_prompt_roles[slot] = selected; continue; }
            r->target_role = role; r->wrapper = *aotx_wrap_active(role);
            for (uint32_t j = 0; j < 32; ++j) r->model[j] = aotx_model_load.resident[role].body.digest[j];
        }
        if (!aotx_intake_qualified(role)) return AOTX_COG_UNAVAILABLE;
        aotx_say.slot[slot].length = bytes; aotx_say.slot[slot].wanted = 1;
        aotx_media_prompts[slot].stage = 0;
        return AOTX_COG_OK;
    }
    return AOTX_COG_LAYOUT;
}
#endif
