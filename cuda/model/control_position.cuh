/* Purpose: Restrict response controls to complete tokens after the user turn.
 * Owns: No persistent state.
 * Launch shape: One caller for each tokenized sequence in a batch.
 * Lifetime: The boundary stays fixed through prompt chunks and reply tokens. */
#ifndef AOTX_MODEL_CONTROL_POSITION_CUH
#define AOTX_MODEL_CONTROL_POSITION_CUH
#include "disk/runtime/control.h"

__device__ __forceinline__ int aotx_control_position(unsigned mode, unsigned first,
    const unsigned *base, unsigned seq, unsigned row) {
    if (mode == AOTX_CONTROL_ALL) return 1;
    return mode == AOTX_CONTROL_RESPONSE && first && base &&
        (unsigned long long)base[seq] + row + 1ull >= first;
}

/* A piece that crosses the prefix boundary stays unmodified. Zero refuses the control. */
__device__ __forceinline__ unsigned aotx_control_response(const aotx_wrap *wrap,
    const unsigned char *clean, unsigned start, unsigned length, const unsigned *piece_start,
    const unsigned *piece_length, const unsigned *chunk, unsigned pieces, unsigned tokens) {
    unsigned suffix = 0;
    for (unsigned s = AOTX_WRAP_GENERATION_HEAD; s < AOTX_WRAP_SPANS; ++s) suffix += wrap->length[s];
    if (!wrap->usable || !suffix || suffix > length || !tokens || start > ~0u - length) return 0;
    unsigned first = start + length - suffix, at = first;
    for (unsigned s = AOTX_WRAP_GENERATION_HEAD; s < AOTX_WRAP_SPANS; ++s) {
        if (wrap->offset[s] > AOTX_WRAP_BYTES || wrap->length[s] > AOTX_WRAP_BYTES - wrap->offset[s]) return 0;
        for (unsigned j = 0; j < wrap->length[s]; ++j)
            if (clean[at++] != wrap->bytes[wrap->offset[s] + j]) return 0;
    }
    unsigned total = 0, boundary = 0, end = start;
    for (unsigned p = 0; p < pieces; ++p) {
        if (piece_start[p] < end || piece_start[p] > start + length || !piece_length[p] ||
            piece_length[p] > start + length - piece_start[p] || !chunk[p] || chunk[p] > tokens - total) return 0;
        if (!boundary && piece_start[p] >= first) boundary = total + 1;
        total += chunk[p]; end = piece_start[p] + piece_length[p];
    }
    return total == tokens && end == start + length ? boundary : 0;
}
#endif
