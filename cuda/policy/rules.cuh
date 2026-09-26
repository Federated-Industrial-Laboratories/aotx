/* Purpose: Select a bounded memory work proposal for each valid observation.
 * Owns: Each row's proposal and portable counter state.
 * Launch shape: One thread per independent row; no shared state between rows.
 * Lifetime: One data-policy evaluation or an embedded native implementation. */
#ifndef AOTX_POLICY_RULES_CUH
#define AOTX_POLICY_RULES_CUH
#include "policy/abi.h"
static __device__ inline uint64_t aotx_policy_word(const unsigned char *p) {
    uint64_t v = 0;
    for (unsigned i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (8 * i);
    return v;
}
static __device__ inline void aotx_policy_word_set(unsigned char *p, uint64_t v) {
    for (unsigned i = 0; i < 8; ++i) p[i] = (unsigned char)(v >> (8 * i));
}
static __device__ inline void aotx_policy_rule_row(const aotx_policy_input *in,
    const unsigned char *before, aotx_policy_output *out, unsigned char *after, uint32_t stride) {
    if (!in->valid) return;
    *out = {};
    if (stride < 16) { out->status = 1; return; }
    for (uint32_t j = 0; j < stride; ++j) after[j] = before[j];
    uint64_t calls = aotx_policy_word(before), prior = aotx_policy_word(before + 8);
    aotx_policy_word_set(after, calls == UINT64_MAX ? calls : calls + 1);
    unsigned pressure = in->rule_pressure ? in->rule_pressure : in->pressure;
    bool ready = (in->enabled & 1u) && !in->paused && !in->foreground && pressure &&
        (in->objects * 100 >= in->object_capacity * pressure ||
         in->bytes * 100 >= in->byte_capacity * pressure);
    bool moved = in->source >= prior && in->source - prior >= in->minimum_move;
    bool cooled = !prior || (in->source >= prior && in->source - prior >= in->backoff);
    if (ready && moved && cooled) {
        out->action = AOTX_POLICY_MAINTAIN; out->reason = AOTX_POLICY_REASON_PRESSURE;
        aotx_policy_word_set(after + 8, in->source);
    } else if ((in->reserved0 == AOTX_POLICY_APPRAISAL_ABI || in->reserved0 == AOTX_POLICY_REVIEW_ABI) && in->reserved1[0] &&
        !in->paused && !in->foreground) {
        out->action = AOTX_POLICY_APPRAISE; out->reason = AOTX_POLICY_REASON_EVIDENCE;
    } else if (in->reserved0 == AOTX_POLICY_REVIEW_ABI && (in->enabled & 2u) && !in->paused && !in->foreground) {
        out->action = AOTX_POLICY_REVIEW; out->reason = AOTX_POLICY_REASON_EVIDENCE;
    }
}
#endif
