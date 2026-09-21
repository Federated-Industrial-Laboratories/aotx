/* Purpose: Render source-labelled text and separate appraisal values.
 * Owns: Bounded output bytes; no ranking or state changes.
 * Launch shape: One rendering thread per query.
 * Lifetime: One exact selection through recorded recovery. */
#ifndef AOTX_COGNITIVE_RECALL_LABELS_CUH
#define AOTX_COGNITIVE_RECALL_LABELS_CUH
#include "appraisal/recall.cuh"
#include "cognitive/recall_sources.cuh"

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
__device__ inline uint32_t aotx_recall_relation_text(const aotx_cognitive_store *s,
    const unsigned char *r, const unsigned char *p, unsigned char *out, uint32_t at, uint32_t cap) {
    at = aotx_recall_word(out, at, cap, "relationship exposure=1 units=1");
    const char *labels[] = {" regard_gain=", " regard_loss=", " task_trust_gain=", " task_trust_loss="};
    for (uint32_t j = 0; j < 4; ++j) {
        at = aotx_recall_word(out, at, cap, labels[j]);
        uint32_t value = aotx_cog_u32(p + 16 + j * 4);
        at = value == AOTX_COG_UNKNOWN ? aotx_recall_word(out, at, cap, "unknown") :
            aotx_recall_number(out, at, cap, value);
    }
    at = aotx_recall_word(out, at, cap, " task="); at = aotx_recall_hex(out, at, cap, p + 32);
    at = aotx_recall_word(out, at, cap, " reported assessment; no authority\n");
    uint32_t length = 0; const unsigned char *source = aotx_appraisal_source(s, r, &length);
    if (!source) return cap + 1;
    const char *quotes[] = {"evidence: ", "task quote: ", "reported commitment candidate: "};
    for (uint32_t j = 0; j < 3; ++j) {
        uint32_t start = aotx_cog_u32(p + 48 + j * 8), bytes = aotx_cog_u32(p + 52 + j * 8);
        if (start > length || bytes > length - start) return cap + 1;
        at = aotx_recall_word(out, at, cap, quotes[j]);
        at = bytes ? aotx_recall_run(out, at, cap, source + start, bytes) : aotx_recall_word(out, at, cap, "unknown");
        at = aotx_recall_word(out, at, cap, "\n");
    }
    return at;
}
__device__ inline uint32_t aotx_recall_one(const aotx_cognitive_store *s,
    uint32_t index, uint32_t reason, unsigned char *out, uint32_t at, uint32_t cap, bool sources = false) {
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
    if (sources) {
        uint32_t source = aotx_recall_source_index(s, index);
        const unsigned char *event = s->objects[source], *actor = aotx_recall_source_actor(s, source);
        at = aotx_recall_word(out, at, cap, " source_ref=");
        at = aotx_recall_hex(out, at, cap, event + AOTX_CO_ID);
        at = aotx_recall_word(out, at, cap, "@");
        at = aotx_recall_number(out, at, cap, aotx_cog_u64(event + AOTX_CO_VERSION));
        at = aotx_recall_word(out, at, cap, " source_actor=");
        at = actor ? aotx_recall_hex(out, at, cap, actor) : aotx_recall_word(out, at, cap, "unknown");
    }
    bool contextual = aotx_recall_contextual(s, r);
    bool interpreted = aotx_intake_payload(p, aotx_cog_u64(r + AOTX_CO_BYTES));
    if (interpreted) {
        const char *kinds[] = {"unknown", "participant mention", "task mention", "assertion", "correction"};
        uint32_t kind = aotx_cog_u32(p + 16);
        at = aotx_recall_word(out, at, cap, " inferred=");
        at = aotx_recall_word(out, at, cap, kinds[kind <= 4 ? kind : 0]);
        at = aotx_recall_word(out, at, cap, " from=");
        at = aotx_recall_hex(out, at, cap, r + AOTX_CO_SOURCE);
        at = aotx_recall_word(out, at, cap, "@");
        at = aotx_recall_number(out, at, cap, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION));
        at = aotx_recall_word(out, at, cap, " byte=");
        at = aotx_recall_number(out, at, cap, aotx_cog_u32(p + 20));
        at = aotx_recall_word(out, at, cap, " length=");
        at = aotx_recall_number(out, at, cap, aotx_cog_u32(p + 12));
        if (kind == AOTX_INTAKE_CORRECTION) {
            at = aotx_recall_word(out, at, cap, " replaces=");
            at = aotx_recall_hex(out, at, cap, r + AOTX_CO_SUPERSEDES);
            at = aotx_recall_word(out, at, cap, "@");
            at = aotx_recall_number(out, at, cap, aotx_cog_u64(r + AOTX_CO_SUPER_VERSION));
        }
    }
    bool appraisal = aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_APPRAISAL;
    uint32_t automatic = aotx_appraisal_recall_kind(s, r);
    if (contextual || appraisal || automatic || reason == AOTX_RECALL_SIGNIFICANT) {
        at = aotx_recall_word(out, at, cap, " subject=");
        at = aotx_recall_hex(out, at, cap, r + AOTX_CO_SUBJECT);
    }
    if (contextual) {
        at = aotx_recall_word(out, at, cap, " task=");
        at = aotx_recall_hex(out, at, cap, p + 16);
    }
    if (appraisal || automatic) {
        at = aotx_recall_word(out, at, cap, " assesses=");
        at = aotx_recall_hex(out, at, cap, r + AOTX_CO_SOURCE);
        at = aotx_recall_word(out, at, cap, "@");
        at = aotx_recall_number(out, at, cap, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION));
    }
    if (automatic && !aotx_cog_zero(r + AOTX_CO_SUPERSEDES, 16)) {
        at = aotx_recall_word(out, at, cap, " replaces=");
        at = aotx_recall_hex(out, at, cap, r + AOTX_CO_SUPERSEDES);
        at = aotx_recall_word(out, at, cap, "@");
        at = aotx_recall_number(out, at, cap, aotx_cog_u64(r + AOTX_CO_SUPER_VERSION));
    }
    at = aotx_recall_word(out, at, cap, "]\n");
    if (automatic == 2) return aotx_recall_relation_text(s, r, p, out, at, cap);
    if (automatic == 3) return aotx_recall_word(out, at, cap, "completed appraisal; no new exposure\n");
    if (appraisal) return aotx_recall_appraisal_text(p, out, at, cap);
    at = aotx_recall_run(out, at, cap, p + (interpreted ? AOTX_INTAKE_PAYLOAD : contextual ? 64 : 32), aotx_cog_u32(p + 12));
    return aotx_recall_word(out, at, cap, "\n");
}
#endif
