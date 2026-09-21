/* Purpose: Admit decoded tokens against the source grammar within the sampler.
 * Owns: A scalar copy of the current prefix; persistent state stays unchanged.
 * Launch shape: One vocabulary thread for each candidate token.
 * Lifetime: One token choice; no temporary device allocation. */
#ifndef AOTX_COGNITIVE_INTAKE_TOKEN_CUH
#define AOTX_COGNITIVE_INTAKE_TOKEN_CUH
#include "cognitive/intake_grammar.cuh"
#include "appraisal/token.cuh"
#include "model/decode_state.cuh"
#include "text/utf8.cuh"
#include "model/vocab.cuh"

__device__ __forceinline__ bool aotx_intake_allows(uint32_t slot, uint32_t token) {
    if (aotx_appraisal.active) return aotx_appraisal_allows(slot, token);
    uint32_t row = aotx_intake.row[slot] - 1;
    const aotx_intake_row *r = aotx_intake.rows + row;
    const aotx_intake_index_row *s = aotx_intake_index_rows + row;
    if (!s->ready || r->status || r->bytes > AOTX_INTAKE_REPLY) return false;
    if (aotx_wrap_end(aotx_seqs.slot[slot].role, token) || token == aotx_seqs.slot[slot].stop)
        return r->prefix.stage == 11;
    const aotx_text_vocab *v = aotx_model_vocab(aotx_seqs.slot[slot].role);
    if (token >= v->tokens || aotx_text_is_control(v, token)) return false;
    uint64_t from = v->token_at[token], span = v->token_at[token + 1] - from;
    if (!span || span > UINT32_MAX) return false;
    const unsigned char *text = v->token_bytes + from;
    aotx_intake_prefix prefix = r->prefix;
    uint32_t walk = 0, bytes = 0;
    bool progress = false, completed = false;
    while (walk < span) {
        uint32_t point = 0, took = aotx_text_decode_point(text, (uint32_t)span, walk, &point);
        if (!took) return false;
        walk += took;
        uint32_t byte = aotx_text_point_byte(point);
        uint32_t n = byte < 256 || point < 128 ? 1 : point < 2048 ? 2 : point < 65536 ? 3 : 4;
        if (n > AOTX_INTAKE_REPLY - r->bytes - bytes) return false;
        bytes += n;
        for (uint32_t j = 0; j < n; ++j) {
            uint32_t encoded = byte < 256 ? byte : point;
            if (n > 1) encoded = j ? 0x80 | ((point >> (6 * (n - j - 1))) & 63) :
                (n == 2 ? 0xc0 : n == 3 ? 0xe0 : 0xf0) | (point >> (6 * (n - 1)));
            /* A token must add quoted content or structure, beyond space between fields. */
            progress |= prefix.stage == 5 || (encoded != ' ' && encoded != '\t' && encoded != '\n' && encoded != '\r');
            if (completed && prefix.stage == 9 && encoded == ',') return false;
            uint32_t before = prefix.items;
            if (!aotx_intake_prefix_byte(s, &prefix, encoded, r->items, r->prefix.items)) return false;
            completed |= s->mode == 2 && before >= r->first_count && prefix.items != before;
        }
    }
    return s->mode || progress;
}
#endif
