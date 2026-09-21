/* Purpose: Check JSON prefixes and exact source quotations during token selection.
 * Owns: Bounded scalar prefix state; substring data stays in the source index.
 * Launch shape: Vocabulary threads test copies; one source thread advances accepted bytes.
 * Lifetime: One internal response; the complete parser validates admission separately. */
#ifndef AOTX_COGNITIVE_INTAKE_GRAMMAR_CUH
#define AOTX_COGNITIVE_INTAKE_GRAMMAR_CUH
#include "cognitive/intake_index.cuh"

__device__ __forceinline__ bool aotx_intake_byte(const aotx_intake_index_row *s, aotx_intake_prefix *p, uint32_t byte) {
    if (p->length == AOTX_RECALL_TEXT) return false;
    if (s->mode) {
        const aotx_intake_row *r = aotx_intake.rows + s->row;
        if (s->mode == 1 || p->items < r->first_count) {
            if (p->items >= (s->mode == 1 ? r->source_count : r->first_count)) return false;
            const aotx_intake_span *span = r->statements + p->items;
            if (p->length >= span->length || byte != s->source[span->start + p->length]) return false;
        } else {
            uint32_t next = aotx_intake_next(aotx_intake_filtered_rows + s->row, p->filtered, byte);
            if (next == UINT32_MAX) return false;
            p->filtered = next;
        }
    }
    if (p->utf8_left) {
        if ((byte & 192) != 128) return false;
        p->utf8_value = p->utf8_value * 64 + (byte & 63);
        if (!--p->utf8_left && (p->utf8_value < p->utf8_min || p->utf8_value > 0x10ffff ||
            (p->utf8_value >= 0x80 && p->utf8_value <= 0x9f) ||
            (p->utf8_value >= 0xd800 && p->utf8_value <= 0xdfff))) return false;
    } else if (byte < 128) {
        if ((!byte || byte < 32 || byte == 127) && byte != 9 && byte != 10) return false;
    } else if (byte >= 194 && byte <= 223) { p->utf8_left = 1; p->utf8_value = byte & 31; p->utf8_min = 128; }
    else if (byte >= 224 && byte <= 239) { p->utf8_left = 2; p->utf8_value = byte & 15; p->utf8_min = 2048; }
    else if (byte >= 240 && byte <= 244) { p->utf8_left = 3; p->utf8_value = byte & 7; p->utf8_min = 65536; }
    else return false;
    uint32_t next = aotx_intake_next(s, p->node, byte);
    if (next == UINT32_MAX) return false;
    if (s->mode == 2 && p->items >= aotx_intake.rows[s->row].first_count &&
        !aotx_intake_optional(s, next, p->length + 1, p->kind)) return false;
    p->node = next; ++p->length; return true;
}
__device__ __forceinline__ bool aotx_intake_point(const aotx_intake_index_row *s, aotx_intake_prefix *p, uint32_t point) {
    if (point < 128) return aotx_intake_byte(s, p, point);
    uint32_t n = point < 2048 ? 2 : point < 65536 ? 3 : 4;
    if (!aotx_intake_byte(s, p, (n == 2 ? 0xc0 : n == 3 ? 0xe0 : 0xf0) | (point >> (6 * (n - 1))))) return false;
    for (uint32_t j = n - 1; j; --j) if (!aotx_intake_byte(s, p, 0x80 | ((point >> (6 * (j - 1))) & 63))) return false;
    return true;
}
__device__ __forceinline__ bool aotx_intake_unit_matches(const aotx_intake_prefix *p, uint32_t point) {
    uint32_t unit = point;
    if (p->high) {
        if (point < 0x10000 || 0xd800 + ((point - 0x10000) >> 10) != p->high) return false;
        unit = 0xdc00 + ((point - 0x10000) & 1023);
    } else if (point >= 0x10000) unit = 0xd800 + ((point - 0x10000) >> 10);
    return (unit >> ((4 - p->digits) * 4)) == p->code;
}
template<unsigned left> __device__ __forceinline__ bool aotx_intake_unicode_edges(
    const aotx_intake_index_row *s, const aotx_intake_prefix *original, const aotx_intake_prefix *p) {
    for (uint32_t edge = s->node[p->node].edge; edge != UINT32_MAX; edge = s->edge[edge].next) {
        uint32_t byte = s->edge[edge].key & 255;
        aotx_intake_prefix next = *p;
        if (!aotx_intake_byte(s, &next, byte)) continue;
        if (!next.utf8_left) {
            uint32_t point = byte < 128 ? byte : next.utf8_value;
            if (aotx_intake_unit_matches(original, point)) return true;
        } else if constexpr (left > 1) {
            if (aotx_intake_unicode_edges<left - 1>(s, original, &next)) return true;
        }
    }
    return false;
}
__device__ __forceinline__ bool aotx_intake_unicode_possible(const aotx_intake_index_row *s,
    const aotx_intake_prefix *p) {
    if (p->utf8_left) return false;
    if (s->mode) {
        const aotx_intake_row *r = aotx_intake.rows + s->row;
        if (s->mode == 1 || p->items < r->first_count) {
            const aotx_intake_span *span = r->statements + p->items;
            if (p->length >= span->length) return false;
            uint32_t at = span->start + p->length, point = s->source[at++];
            uint32_t more = point < 128 ? 0 : point < 224 ? 1 : point < 240 ? 2 : 3;
            if (more) {
                point &= more == 1 ? 31 : more == 2 ? 15 : 7;
                for (uint32_t j = 0; j < more; ++j) point = point * 64 + (s->source[at++] & 63);
            }
            return aotx_intake_unit_matches(p, point);
        }
        return aotx_intake_unicode_edges<4>(s, p, p);
    }
    for (uint32_t i = 0; i < s->bytes;) {
        uint32_t point = s->source[i++], more = point < 128 ? 0 : point < 224 ? 1 : point < 240 ? 2 : 3;
        if (more) {
            point &= more == 1 ? 31 : more == 2 ? 15 : 7;
            for (uint32_t j = 0; j < more; ++j) point = point * 64 + (s->source[i++] & 63);
        }
        uint32_t unit = point;
        if (p->high) {
            if (point < 0x10000 || 0xd800 + ((point - 0x10000) >> 10) != p->high) continue;
            unit = 0xdc00 + ((point - 0x10000) & 1023);
        } else if (point >= 0x10000) unit = 0xd800 + ((point - 0x10000) >> 10);
        if ((unit >> ((4 - p->digits) * 4)) != p->code) continue;
        aotx_intake_prefix next = *p;
        if (aotx_intake_point(s, &next, point)) return true;
    }
    return false;
}
__device__ __forceinline__ bool aotx_intake_quote(const aotx_intake_index_row *s, aotx_intake_prefix *p,
    uint32_t c, const aotx_intake_item *seen, uint32_t count) {
    if (p->escape == 3) { p->escape = 4; return c == '\\'; }
    if (p->escape == 4) { p->escape = 2; p->digits = p->code = 0; return c == 'u'; }
    if (p->escape == 2) {
        uint32_t digit = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 :
            c >= 'A' && c <= 'F' ? c - 'A' + 10 : 16;
        if (digit == 16) return false;
        p->code = p->code * 16 + digit;
        ++p->digits;
        if (!aotx_intake_unicode_possible(s, p)) return false;
        if (p->digits != 4) return true;
        uint32_t point = p->code; p->escape = 0;
        if (p->high) {
            if (point < 0xdc00 || point > 0xdfff) return false;
            point = 0x10000 + ((p->high - 0xd800) << 10) + point - 0xdc00; p->high = 0;
        } else if (point >= 0xd800 && point <= 0xdbff) { p->high = point; p->escape = 3; return true; }
        else if (point >= 0xdc00 && point <= 0xdfff) return false;
        return aotx_intake_point(s, p, point);
    }
    if (p->escape == 1) {
        p->escape = 0;
        if (c == 'u') { p->escape = 2; p->digits = p->code = 0; return !s->mode || aotx_intake_unicode_possible(s, p); }
        if (c == 'n') c = '\n'; else if (c == 't') c = '\t';
        else if (c != '"' && c != '\\' && c != '/') return false;
        return aotx_intake_byte(s, p, c);
    }
    if (c == '"') {
        if (!p->length || p->utf8_left) return false;
        bool fixed = s->mode == 1 || (s->mode == 2 && p->items < aotx_intake.rows[s->row].first_count);
        if (fixed) {
            const aotx_intake_span *span = aotx_intake.rows[s->row].statements + p->items;
            if (p->length != span->length) return false;
            p->start = span->start;
        } else {
            if (s->node[p->node].ends != 1) return false;
            p->start = s->node[p->node].position + 1 - p->length;
            if (s->mode == 2 && aotx_intake_excluded(s->row, p->node, p->length, p->kind, false)) return false;
            for (uint32_t j = 0; j < count; ++j)
                if (seen[j].kind == p->kind && seen[j].start == p->start && seen[j].length == p->length) return false;
        }
        p->stage = 6; return true;
    }
    if (c == '\\') {
        if (p->utf8_left) return false;
        p->escape = 1; p->digits = p->code = 0;
        return !s->mode || aotx_intake_unicode_possible(s, p);
    }
    return c >= 32 && aotx_intake_byte(s, p, c);
}
__device__ __forceinline__ bool aotx_intake_number_prefix(const aotx_intake_index_row *s, const aotx_intake_prefix *p, uint32_t value) {
    uint32_t eligible = s->eligible & ~p->used;
    if (p->kind != AOTX_INTAKE_CORRECTION) return !value;
    if (!value || value > AOTX_RECALL_LIMIT) return false;
    for (uint32_t j = 1; j <= AOTX_RECALL_LIMIT; ++j)
        if ((eligible & (1u << j)) && (j == value || (value < 10 && j >= 10 && j / 10 == value))) return true;
    return false;
}
__device__ __forceinline__ bool aotx_intake_first_label(aotx_intake_prefix *p, uint32_t c) {
    if (p->stage == 12) {
        if (c != 's' && c != 'r') return false;
        p->kind = c == 's' ? AOTX_INTAKE_ASSERTION : AOTX_INTAKE_REJECT;
        p->digits = 1; p->stage = 13; return true;
    }
    const char *label = p->kind == AOTX_INTAKE_ASSERTION ? "statement" : "request";
    uint32_t bytes = p->kind == AOTX_INTAKE_ASSERTION ? 9 : 7;
    if (p->digits < bytes) return c == (uint32_t)label[p->digits++];
    if (c != '"') return false;
    p->stage = 14; return true;
}
__device__ __forceinline__ bool aotx_intake_first_prefix(const aotx_intake_index_row *s,
    aotx_intake_prefix *p, uint32_t c) {
    uint32_t count = aotx_intake.rows[s->row].source_count;
    switch (p->stage) {
    case 0: if (c != '[') return false; p->stage = 1; return true;
    case 1: if (c == ']') { if (count) return false; p->stage = 11; return true; }
        if (c != '[' || !count) return false; p->stage = 2; return true;
    case 2:
        if (c != '"' || p->items >= count) return false;
        p->node = p->length = p->utf8_left = p->escape = p->high = p->kind = p->number = 0;
        p->stage = 5; return true;
    case 6: if (c != ',') return false; p->stage = 7; return true;
    case 7: if (c != '"') return false; p->stage = 12; return true;
    case 14: if (c != ']') return false; p->stage = 9; ++p->items; return true;
    case 9: if (c == ']') { if (p->items != count) return false; p->stage = 11; }
        else if (c == ',' && p->items < count) p->stage = 10; else return false; return true;
    case 10: if (c != '[') return false; p->stage = 2; return true;
    default: return false;
    }
}
__device__ __forceinline__ bool aotx_intake_prefix_byte(const aotx_intake_index_row *s, aotx_intake_prefix *p,
    uint32_t c, const aotx_intake_item *seen, uint32_t count) {
    if (p->stage == 5) { p->spaces = 0; return aotx_intake_quote(s, p, c, seen, count); }
    if (s->mode == 1 && (p->stage == 12 || p->stage == 13)) {
        p->spaces = 0; return aotx_intake_first_label(p, c);
    }
    bool space = c == ' ' || c == '\n' || c == '\t' || c == '\r';
    if (space) { if (s->mode && ++p->spaces > 8) return false; if (p->stage == 8) p->gap = 1; return true; }
    p->spaces = 0;
    if (s->mode == 1) return aotx_intake_first_prefix(s, p, c);
    uint32_t required = s->mode == 2 ? aotx_intake.rows[s->row].first_count : 0;
    bool optional = s->mode == 2 && p->items >= required;
    switch (p->stage) {
    case 0: if (c != '[') return false; p->stage = 1; return true;
    case 1: if (c == ']') { if (required) return false; p->stage = 11; return true; }
        if (c != '[' || (optional && !aotx_intake_optional(s, 0, 0, 1) && !aotx_intake_optional(s, 0, 0, 2))) return false;
        p->stage = 2; return true;
    case 2:
        if (c < '1' || c > '4' || p->items == AOTX_INTAKE_ITEMS ||
            (c == '4' && !(s->eligible & ~p->used))) return false;
        if (s->mode == 2 && (p->items < required ? c < '3' : c > '2')) return false;
        if (optional && !aotx_intake_optional(s, 0, 0, c - '0')) return false;
        p->kind = c - '0'; p->stage = 3; return true;
    case 3: if (c != ',') return false; p->stage = 4; return true;
    case 4: if (c != '"') return false; p->stage = 5;
        p->node = p->filtered = p->length = p->utf8_left = p->escape = p->high = 0; return true;
    case 6: if (c != ',') return false; p->stage = 7; p->number = p->digits = p->gap = 0; return true;
    case 7:
        if (c < '0' || c > '9' || !aotx_intake_number_prefix(s, p, c - '0')) return false;
        p->number = c - '0'; p->digits = 1; p->stage = 8; return true;
    case 8:
        if (c >= '0' && c <= '9') {
            if (p->gap || !p->number || p->digits != 1 ||
                !aotx_intake_number_prefix(s, p, p->number * 10 + c - '0')) return false;
            p->number = p->number * 10 + c - '0'; ++p->digits; return true;
        }
        if (c != ']' || (p->kind == 4 && !(s->eligible & ~p->used & (1u << p->number)))) return false;
        if (p->number) p->used |= 1u << p->number;
        p->stage = 9; ++p->items; return true;
    case 9:
        if (c == ']') { if (p->items < required) return false; p->stage = 11; }
        else if (c == ',') {
            if (s->mode == 2 && (p->items == AOTX_INTAKE_ITEMS || (optional &&
                !aotx_intake_optional(s, 0, 0, 1) && !aotx_intake_optional(s, 0, 0, 2)))) return false;
            p->stage = 10;
        } else return false;
        return true;
    case 10: if (c != '[') return false; p->stage = 2; return true;
    default: return false;
    }
}
#endif
