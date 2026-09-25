/* Purpose: Constrain appraisal JSON, numeric prefixes and exact source quotes.
 * Owns: Scalar output prefix state; source paths remain in the shared index.
 * Launch shape: Vocabulary threads test copies; one thread advances each source row.
 * Lifetime: One internal response before independent complete admission. */
#ifndef AOTX_APPRAISAL_GRAMMAR_CUH
#define AOTX_APPRAISAL_GRAMMAR_CUH
#include "appraisal/appraisal.cuh"
#include "cognitive/codec.cuh"
#include "cognitive/intake_grammar.cuh"

/* Bit zero records task admission at the immutable source-index cut. */
__device__ __forceinline__ const unsigned char *aotx_appraisal_task_index(uint32_t row,
    const aotx_intake_index_row *s, uint32_t *length) {
    *length = 0;
    uint32_t index = aotx_appraisal.rows[row].task_source;
    if (!(s->eligible & 1u) || index >= aotx_live_store.count) return 0;
    const unsigned char *r = aotx_live_store.objects[index];
    uint64_t at = aotx_cog_u64(r + AOTX_CO_OFFSET), bytes = aotx_cog_u64(r + AOTX_CO_BYTES);
    if (at > aotx_live_store.bytes || bytes > aotx_live_store.bytes - at || bytes < 32) return 0;
    const unsigned char *p = aotx_live_store.payload + at;
    uint32_t n = aotx_cog_u32(p + 12);
    if (!n || n > AOTX_RECALL_TEXT || bytes != 32ull + n) return 0;
    *length = n; return p + 32;
}
__device__ __forceinline__ bool aotx_appraisal_decimal_prefix(uint32_t value, uint32_t target) {
    if (!value) return !target;
    while (target > value) target /= 10;
    return target == value;
}
__device__ __forceinline__ const char *aotx_appraisal_key(const aotx_appraisal_prefix *p) {
    if (p->stage == 29) return "support";
    if (p->stage == 17) return "evidence";
    if (p->stage == 21) return "task";
    if (p->stage == 24) return "commitment";
    if (p->stage == 27) return "correction";
    if (p->stage != 20) return 0;
    switch (p->field) {
    case 0: return "benefit";
    case 1: return "harm";
    case 2: return "arousal";
    case 3: return "consequence";
    case 4: return "confidence";
    case 5: return "regard_gain";
    case 6: return "regard_loss";
    case 7: return "trust_gain";
    case 8: return "trust_loss";
    default: return 0;
    }
}
__device__ __forceinline__ bool aotx_appraisal_key_byte(aotx_appraisal_prefix *p, uint32_t c) {
    const char *key = aotx_appraisal_key(p);
    if (!key) return false;
    uint32_t length = 0;
    while (key[length]) ++length;
    bool space = c == ' ' || c == '\n' || c == '\t' || c == '\r';
    if ((!p->gap || p->gap == length + 2) && space) return true;
    if (p->gap <= length + 1) {
        uint32_t expected = !p->gap || p->gap == length + 1 ? '"' : (unsigned char)key[p->gap - 1];
        if (c != expected) return false;
        ++p->gap; return true;
    }
    if (p->gap != length + 2 || c != ':') return false;
    p->stage -= 16; p->gap = 0; return true;
}
__device__ __forceinline__ bool aotx_appraisal_number_valid(const aotx_intake_index_row *s,
    const aotx_appraisal_prefix *p, uint32_t value, bool task, bool prefix) {
    if (p->stage == 11) {
        if (!value) return true;
        if (!p->evidence) return false;
        for (uint32_t j = 1; j <= AOTX_RECALL_LIMIT; ++j)
            if ((s->eligible & (1u << j)) && (prefix ? aotx_appraisal_decimal_prefix(value, j) : value == j)) return true;
        return false;
    }
    if (p->field == 3) return p->support ? value <= 4u : value == 0;
    bool unknown = prefix ? aotx_appraisal_decimal_prefix(value, AOTX_COG_UNKNOWN) : value == AOTX_COG_UNKNOWN;
    if (!p->support || (p->field >= 7 && !task)) return unknown;
    return value <= AOTX_COG_SCALE || unknown;
}
__device__ __forceinline__ bool aotx_appraisal_quote_byte(const aotx_intake_index_row *s,
    aotx_appraisal_prefix *p, uint32_t c, const unsigned char *task, uint32_t task_bytes) {
    aotx_intake_prefix *q = &p->quote;
    uint32_t before = q->length;
    bool empty = c == '"' && !q->length && !q->escape;
    if (p->stage == 2 && empty == (p->evidence != 0)) return false;
    if ((p->stage == 6 || p->stage == 9) && !p->evidence && !empty) return false;
    if (p->stage == 6 && p->trust && q->length == task_bytes &&
        (c != '"' || q->escape || q->utf8_left)) return false;
    if (!empty && !aotx_intake_quote(s, q, c, 0, 0)) return false;
    if (p->stage == 6 && p->trust) {
        if (!task || q->length > task_bytes || empty) return false;
        uint32_t start = s->node[q->node].position + 1 - q->length;
        for (uint32_t j = before; j < q->length; ++j) if (s->source[start + j] != task[j]) return false;
        if (q->stage == 6 && q->length != task_bytes) return false;
    }
    if (!empty && q->stage != 6) return true;
    if (p->stage == 2) p->stage = 3;
    else if (p->stage == 6) {
        if ((!p->evidence && !empty) || (p->trust && empty)) return false;
        p->task = !empty; p->stage = 7;
    } else {
        if (!p->evidence && !empty) return false;
        p->stage = 10;
    }
    return true;
}
__device__ __forceinline__ bool aotx_appraisal_prefix_byte(const aotx_intake_index_row *s,
    aotx_appraisal_prefix *p, uint32_t c, const unsigned char *task, uint32_t task_bytes) {
    if (p->stage >= 16) return aotx_appraisal_key_byte(p, c);
    if (p->stage == 2 || p->stage == 6 || p->stage == 9) return aotx_appraisal_quote_byte(s, p, c, task, task_bytes);
    bool space = c == ' ' || c == '\n' || c == '\t' || c == '\r';
    if (space) { if ((p->stage == 4 || p->stage == 11) && p->digits) p->gap = 1; return true; }
    switch (p->stage) {
    case 0: if (c != '{') return false; p->stage = 29; p->gap = 0; return true;
    case 13:
        if (c != '0' && c != '1') return false;
        p->support = c - '0'; p->stage = 14; return true;
    case 14:
        if (c != ',') return false;
        p->stage = 20; p->gap = 0; return true;
    case 1: case 5: case 8:
        if (c != '"') return false;
        ++p->stage; p->quote = {}; return true;
    case 3:
        if (c != ',') return false;
        p->stage = 21; p->gap = 0; return true;
    case 4: case 11:
        if (c >= '0' && c <= '9') {
            uint32_t digit = c - '0';
            if (p->gap || (p->digits && !p->number) || p->number > (UINT32_MAX - digit) / 10) return false;
            uint32_t value = p->number * 10 + digit;
            if (!aotx_appraisal_number_valid(s, p, value, task && task_bytes, true)) return false;
            p->number = value; ++p->digits; return true;
        }
        if (!p->digits || !aotx_appraisal_number_valid(s, p, p->number, task && task_bytes, false)) return false;
        if (p->stage == 11) {
            if (c != '}') return false;
            p->stage = 12; return true;
        }
        if (c != ',') return false;
        if (p->number != (p->field == 3 ? 0 : AOTX_COG_UNKNOWN)) p->evidence = 1;
        if (p->field >= 7 && p->number != AOTX_COG_UNKNOWN) p->trust = 1;
        ++p->field; p->digits = p->number = p->gap = 0;
        if (p->field == AOTX_APPRAISAL_VALUES && p->support != p->evidence) return false;
        p->stage = p->field == AOTX_APPRAISAL_VALUES ? 17 : 20;
        return true;
    case 7: if (c != ',') return false; p->stage = 24; p->gap = 0; return true;
    case 10:
        if (c != ',') return false;
        p->stage = 27; p->digits = p->number = p->gap = 0; return true;
    default: return false;
    }
}
#endif
