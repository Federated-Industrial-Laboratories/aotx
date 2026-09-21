/* Purpose: Apply default Unicode sentence boundaries and the source whitespace profile.
 * Owns: Bounded code point work and complete original byte intervals.
 * Launch shape: One device thread per source in the intake batch.
 * Lifetime: One parse; the caller supplies all work and output storage. */
#include "cognitive/source_spans.cuh"

__device__ const unsigned char aotx_source_profile_digest[32] = AOTX_SOURCE_PROFILE_DIGEST;
static __device__ uint32_t aotx_source_class_of(uint32_t point) {
    uint32_t lo = 0, hi = aotx_source_property_count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (point < aotx_source_properties[mid][0]) hi = mid;
        else if (point > (aotx_source_properties[mid][1] >> 4)) lo = mid + 1;
        else return aotx_source_properties[mid][1] & 15;
    }
    return AOTX_SB_OTHER;
}
static __device__ bool aotx_source_para(uint32_t c) {
    return c == AOTX_SB_CR || c == AOTX_SB_LF || c == AOTX_SB_SEP;
}
static __device__ bool aotx_source_space(uint32_t c) {
    return c == AOTX_SB_SP || aotx_source_para(c);
}
static __device__ bool aotx_source_term(uint32_t c) {
    return c == AOTX_SB_ATERM || c == AOTX_SB_STERM;
}
static __device__ bool aotx_source_units(const unsigned char *text, uint32_t bytes,
    uint32_t *units, uint32_t *count) {
    uint32_t at = 0, n = 0;
    while (at < bytes) {
        uint32_t c = text[at++], point = c, left = 0, minimum = 0;
        if (c >= 0xc2 && c <= 0xdf) { point = c & 31; left = 1; minimum = 0x80; }
        else if (c >= 0xe0 && c <= 0xef) { point = c & 15; left = 2; minimum = 0x800; }
        else if (c >= 0xf0 && c <= 0xf4) { point = c & 7; left = 3; minimum = 0x10000; }
        else if (c >= 0x80) return false;
        if (left > bytes - at) return false;
        while (left--) {
            c = text[at++]; if ((c & 0xc0) != 0x80) return false;
            point = (point << 6) | (c & 63);
        }
        if (point < minimum || point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff)) return false;
        units[n++] = (at << 4) | aotx_source_class_of(point);
    }
    uint32_t next = AOTX_SB_OTHER;
    for (uint32_t i = n; i; --i) {
        uint32_t c = units[i - 1] & 15;
        if (c == AOTX_SB_OLETTER || c == AOTX_SB_UPPER || c == AOTX_SB_LOWER ||
            aotx_source_para(c) || aotx_source_term(c)) next = c;
        units[i - 1] |= next << 16;
    }
    *count = n; return true;
}
static __device__ bool aotx_source_emit(const uint32_t *units, uint32_t first, uint32_t last,
    aotx_source_span *spans, uint32_t *count, bool trim) {
    if (trim) {
        while (first < last && aotx_source_space(units[first] & 15)) ++first;
        while (last > first && aotx_source_space(units[last - 1] & 15)) --last;
    }
    if (first == last) return true;
    if (*count == AOTX_SOURCE_SPANS) return false;
    uint32_t start = first ? (units[first - 1] >> 4) & 4095 : 0;
    spans[(*count)++] = {start, ((units[last - 1] >> 4) & 4095) - start};
    return true;
}
__device__ bool aotx_source_split(const unsigned char *text, uint32_t bytes, uint32_t *units,
    aotx_source_span *spans, uint32_t *count, bool trim) {
    *count = 0;
    if (bytes > AOTX_SOURCE_BYTES) return false;
    uint32_t n = 0;
    if (!aotx_source_units(text, bytes, units, &n)) return false;
    uint32_t first = 0, last = AOTX_SB_OTHER, before = AOTX_SB_OTHER;
    uint32_t term = AOTX_SB_OTHER, spaces = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t c = units[i] & 15, previous = i ? units[i - 1] & 15 : AOTX_SB_OTHER;
        bool ignored = c == AOTX_SB_EXTEND || c == AOTX_SB_FORMAT, split = false;
        if (i) {
            if (previous == AOTX_SB_CR && c == AOTX_SB_LF) {}
            else if (aotx_source_para(previous)) split = true;
            else if (ignored) {}
            else if (last == AOTX_SB_ATERM && c == AOTX_SB_NUMERIC) {}
            else if ((before == AOTX_SB_UPPER || before == AOTX_SB_LOWER) &&
                     last == AOTX_SB_ATERM && c == AOTX_SB_UPPER) {}
            else if (term == AOTX_SB_ATERM && (units[i] >> 16) == AOTX_SB_LOWER) {}
            else if (aotx_source_term(term) && (c == AOTX_SB_CONTINUE || aotx_source_term(c))) {}
            else if (aotx_source_term(term) && !spaces && (c == AOTX_SB_CLOSE || aotx_source_space(c))) {}
            else if (aotx_source_term(term) && aotx_source_space(c)) {}
            else if (aotx_source_term(term)) split = true;
        }
        if (split) {
            if (!aotx_source_emit(units, first, i, spans, count, trim)) return false;
            first = i;
        }
        if (ignored && i && !aotx_source_para(previous)) continue;
        if (ignored) c = AOTX_SB_OTHER;
        if (aotx_source_term(c)) { term = c; spaces = 0; }
        else if (c == AOTX_SB_SP && aotx_source_term(term)) spaces = 1;
        else if (!(c == AOTX_SB_CLOSE && !spaces)) { term = AOTX_SB_OTHER; spaces = 0; }
        before = last; last = c;
    }
    return aotx_source_emit(units, first, n, spans, count, trim);
}
