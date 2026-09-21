/* Purpose: Remove emitted optional quotes from bounded completion data.
 * Owns: Sparse consumed and exhausted prefixes for each optional kind.
 * Launch shape: One advancing source thread; vocabulary threads only read.
 * Lifetime: One second-stage response, reset with its substring index. */
#include "cognitive/intake_index.cuh"

static __device__ bool aotx_intake_mark(uint32_t row, uint32_t node, uint32_t length, uint32_t kind, bool dead) {
    uint32_t at = aotx_intake_consumed_at(row, node, length);
    if (at == UINT32_MAX) return false;
    aotx_intake_consumed[row][at] |= ((node << 12) | length) + 1;
    aotx_intake_consumed[row][at] |= 1u << (28 + (kind - 1) * 2 + (dead ? 1 : 0));
    return true;
}
static __device__ bool aotx_intake_remaining(const aotx_intake_index_row *s,
    uint32_t node, uint32_t length, uint32_t kind) {
    uint32_t end = s->node[node].position + 1;
    if (length && s->node[node].ends == 1 && length <= aotx_intake_filtered_rows[s->row].order[node] &&
        (end == s->bytes || (s->source[end] & 192) != 128) &&
        !aotx_intake_excluded(s->row, node, length, kind, false)) return true;
    for (uint32_t edge = s->node[node].edge; edge != UINT32_MAX; edge = s->edge[edge].next) {
        uint32_t child = s->edge[edge].target, byte = s->edge[edge].key & 255;
        if (!length && !((byte >= 32 && byte < 127) || byte == 9 || byte == 10 || (byte >= 194 && byte <= 244))) continue;
        if (length + 1 <= aotx_intake_completion[s->row][child] &&
            !aotx_intake_excluded(s->row, child, length + 1, kind, true)) return true;
    }
    return false;
}
__device__ __noinline__ bool aotx_intake_consume(uint32_t row, const aotx_intake_item *item) {
    aotx_intake_index_row *s = aotx_intake_index_rows + row;
    uint32_t node = 0; s->sizes[0] = 0;
    for (uint32_t j = 0; j < item->length; ++j) {
        node = aotx_intake_next(s, node, s->source[item->start + j]);
        if (node == UINT32_MAX) return false;
        s->sizes[j + 1] = node;
    }
    if (!aotx_intake_mark(row, node, item->length, item->kind, false)) return false;
    for (uint32_t j = item->length + 1; j; --j) {
        uint32_t length = j - 1; node = s->sizes[length];
        if (aotx_intake_remaining(s, node, length, item->kind)) break;
        if (!aotx_intake_mark(row, node, length, item->kind, true)) return false;
    }
    return true;
}
