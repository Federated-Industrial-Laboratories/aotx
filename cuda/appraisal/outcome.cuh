/* Purpose: Constrain exact source quotes before numeric appraisal.
 * Owns: A bounded quote-first response and its independent complete reader.
 * Launch shape: One source row per thread; vocabulary threads test prefix copies.
 * Lifetime: The first of at most two internal model calls per source. */
#ifndef AOTX_APPRAISAL_OUTCOME_CUH
#define AOTX_APPRAISAL_OUTCOME_CUH
#include "appraisal/grammar.cuh"

__device__ __forceinline__ bool aotx_appraisal_outcome_byte(const aotx_intake_index_row *s,
    aotx_appraisal_prefix *p, uint32_t c) {
    if (p->stage == 3) {
        if (!p->quote.length && c == '"') return false;
        if (!aotx_intake_quote(s, &p->quote, c, 0, 0)) return false;
        if (p->quote.stage == 6) { ++p->evidence; p->stage = 7; }
        return true;
    }
    if (c == ' ' || c == '\n' || c == '\t' || c == '\r') return true;
    switch (p->stage) {
    case 0: if (c != '[') return false; p->stage = 1; return true;
    case 1:
        if (c == ']' && !p->evidence) { p->stage = 12; return true; }
        if (c != '"' || p->evidence >= 8) return false;
        p->quote = {}; p->stage = 3; return true;
    case 7:
        if (c == ']') { p->stage = 12; return true; }
        if (c != ',' || p->evidence >= 8) return false;
        p->stage = 1; return true;
    default: return false;
    }
}
#endif
