/* Purpose: Render source-labelled text and separate appraisal values.
 * Owns: Bounded output bytes; no ranking or state changes.
 * Launch shape: One rendering thread per query.
 * Lifetime: One exact selection through recorded recovery. */
#ifndef AOTX_COGNITIVE_RECALL_LABELS_CUH
#define AOTX_COGNITIVE_RECALL_LABELS_CUH
#include "cognitive/recall_contextual.cuh"

__device__ inline uint32_t aotx_recall_run(unsigned char *out, uint32_t at,
    uint32_t cap, const unsigned char *p, uint32_t n) {
    if (at > cap || n > cap - at) return cap + 1;
    if (out) for (uint32_t j = 0; j < n; ++j) out[at + j] = p[j];
    return at + n;
}
__device__ inline uint32_t aotx_recall_word(unsigned char *out, uint32_t at,
    uint32_t cap, const char *p) {
    uint32_t n = 0; while (p[n]) ++n;
    return aotx_recall_run(out, at, cap, (const unsigned char *)p, n);
}
__device__ inline uint32_t aotx_recall_number(unsigned char *out, uint32_t at,
    uint32_t cap, uint64_t value) {
    unsigned char digits[20]; uint32_t n = 0;
    do { digits[n++] = '0' + value % 10; value /= 10; } while (value);
    while (n) at = aotx_recall_run(out, at, cap, digits + --n, 1);
    return at;
}
__device__ inline uint32_t aotx_recall_hex(unsigned char *out, uint32_t at, uint32_t cap, const unsigned char *id) {
    const char *hex = "0123456789abcdef";
    for (uint32_t j = 0; j < 16; ++j) {
        unsigned char pair[2] = {(unsigned char)hex[id[j] >> 4], (unsigned char)hex[id[j] & 15]};
        at = aotx_recall_run(out, at, cap, pair, 2);
    }
    return at;
}
__device__ inline uint32_t aotx_recall_context_label(const unsigned char *q,
    unsigned char *out, uint32_t at, uint32_t cap) {
    if (!(aotx_context_flags(q) & AOTX_RECALL_TASKS)) return at;
    const unsigned char *c = q + AOTX_RECALL_EXTENSION;
    at = aotx_recall_word(out, at, cap, "[task cue=");
    at = aotx_recall_hex(out, at, cap, c + 16);
    at = aotx_recall_word(out, at, cap, " participants=");
    uint32_t count = aotx_cog_u32(c + 32);
    if (!count) at = aotx_recall_word(out, at, cap, "unknown");
    for (uint32_t j = 0; j < count; ++j) {
        if (j) at = aotx_recall_word(out, at, cap, ",");
        at = aotx_recall_hex(out, at, cap, c + 48 + j * 16);
    }
    return aotx_recall_word(out, at, cap, "]\n");
}
__device__ inline uint32_t aotx_recall_appraisal_text(const unsigned char *p,
    unsigned char *out, uint32_t at, uint32_t cap) {
    const char *labels[] = {"benefit=", " harm=", " arousal=", " consequence=", " confidence=", " units="};
    for (uint32_t j = 0; j < 6; ++j) {
        at = aotx_recall_word(out, at, cap, labels[j]);
        uint32_t value = aotx_cog_u32(p + 4 + j * 4);
        if (value == AOTX_COG_UNKNOWN) at = aotx_recall_word(out, at, cap, "unknown");
        else at = aotx_recall_number(out, at, cap, value);
    }
    return aotx_recall_word(out, at, cap, " assessment\n");
}
__device__ inline uint32_t aotx_recall_one(const aotx_cognitive_store *s,
    uint32_t index, uint32_t reason, unsigned char *out, uint32_t at, uint32_t cap) {
    const unsigned char *r = s->objects[index];
    const unsigned char *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    at = aotx_recall_word(out, at, cap, "[memory id=");
    const char *hex = "0123456789abcdef";
    for (uint32_t j = 0; j < 16; ++j) {
        unsigned char pair[2] = {(unsigned char)hex[r[AOTX_CO_ID + j] >> 4], (unsigned char)hex[r[AOTX_CO_ID + j] & 15]};
        at = aotx_recall_run(out, at, cap, pair, 2);
    }
    at = aotx_recall_word(out, at, cap, " version=");
    at = aotx_recall_number(out, at, cap, aotx_cog_u64(r + AOTX_CO_VERSION));
    at = aotx_recall_word(out, at, cap, " source=");
    at = aotx_recall_number(out, at, cap, aotx_cog_u32(r + AOTX_CO_SOURCE_KIND));
    at = aotx_recall_word(out, at, cap, " evidence=");
    at = aotx_recall_number(out, at, cap, aotx_cog_u32(r + AOTX_CO_EVIDENCE));
    at = aotx_recall_word(out, at, cap, " reason=");
    at = aotx_recall_number(out, at, cap, reason);
    bool contextual = aotx_recall_contextual(s, r);
    bool appraisal = aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_APPRAISAL;
    if (contextual || appraisal || reason == AOTX_RECALL_SIGNIFICANT) {
        at = aotx_recall_word(out, at, cap, " subject=");
        at = aotx_recall_hex(out, at, cap, r + AOTX_CO_SUBJECT);
    }
    if (contextual) {
        at = aotx_recall_word(out, at, cap, " task=");
        at = aotx_recall_hex(out, at, cap, p + 16);
    }
    if (appraisal) {
        at = aotx_recall_word(out, at, cap, " assesses=");
        at = aotx_recall_hex(out, at, cap, r + AOTX_CO_SOURCE);
        at = aotx_recall_word(out, at, cap, "@");
        at = aotx_recall_number(out, at, cap, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION));
    }
    at = aotx_recall_word(out, at, cap, "]\n");
    if (appraisal) return aotx_recall_appraisal_text(p, out, at, cap);
    at = aotx_recall_run(out, at, cap, p + (contextual ? 64 : 32), aotx_cog_u32(p + 12));
    return aotx_recall_word(out, at, cap, "\n");
}
#endif
