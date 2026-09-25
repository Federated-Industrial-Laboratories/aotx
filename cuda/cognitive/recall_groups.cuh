/* Purpose: Share repeated exact source labels within one bounded selection.
 * Owns: Temporary source groups; no stored references or payload changes.
 * Launch shape: One selection or rendering thread per query.
 * Lifetime: One prospective or recorded selection. */
#ifndef AOTX_COGNITIVE_RECALL_GROUPS_CUH
#define AOTX_COGNITIVE_RECALL_GROUPS_CUH
#include "cognitive/recall_labels.cuh"

__device__ inline uint32_t aotx_recall_group_labels(const aotx_cognitive_store *s,
    const unsigned char *q, const uint32_t *indices, uint32_t count, uint32_t *groups,
    unsigned char *out, uint32_t at, uint32_t cap) {
    for (uint32_t j = 0; j < count; ++j) groups[j] = UINT32_MAX;
    if (!aotx_context_groups(q)) return at;
    uint32_t sources[AOTX_RECALL_LIMIT], next = 0;
    for (uint32_t j = 0; j < count; ++j) sources[j] = aotx_recall_source_index(s, indices[j]);
    for (uint32_t j = 0; j < count; ++j) {
        if (groups[j] != UINT32_MAX) continue;
        uint32_t matches = 0;
        for (uint32_t k = j; k < count; ++k) if (sources[k] == sources[j]) ++matches;
        if (matches < 2) continue;
        for (uint32_t k = j; k < count; ++k) if (sources[k] == sources[j]) groups[k] = next;
        at = aotx_recall_word(out, at, cap, "[source_group=");
        at = aotx_recall_number(out, at, cap, next++);
        at = aotx_recall_source_label(s, indices[j], out, at, cap);
        at = aotx_recall_word(out, at, cap, "]\n");
    }
    return at;
}
#endif
