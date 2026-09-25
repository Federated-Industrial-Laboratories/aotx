/* Purpose: Admit sampled vocabulary tokens against the appraisal output contract.
 * Owns: A scalar prefix copy; persistent memory and accepted state stay unchanged.
 * Launch shape: One vocabulary thread per candidate token in each leased sequence.
 * Lifetime: One constrained token choice with no device allocation. */
#ifndef AOTX_APPRAISAL_TOKEN_CUH
#define AOTX_APPRAISAL_TOKEN_CUH
#include "appraisal/grammar.cuh"
#include "appraisal/outcome.cuh"
#include "model/decode_state.cuh"
#include "text/utf8.cuh"
#include "model/vocab.cuh"

__device__ __forceinline__ bool aotx_appraisal_allows(uint32_t slot, uint32_t token) {
    uint32_t row = aotx_intake.row[slot] - 1;
    const aotx_intake_row *r = aotx_intake.rows + row;
    const aotx_appraisal_row *a = aotx_appraisal.rows + row;
    const aotx_intake_index_row *s = aotx_intake_index_rows + row;
    if (!s->ready || r->status || r->bytes > AOTX_INTAKE_REPLY) return false;
    if (aotx_wrap_end(aotx_seqs.slot[slot].role, token) || token == aotx_seqs.slot[slot].stop)
        return a->prefix.stage == 12;
    const aotx_text_vocab *v = aotx_model_vocab(aotx_seqs.slot[slot].role);
    if (token >= v->tokens || aotx_text_is_control(v, token)) return false;
    uint64_t from = v->token_at[token], span = v->token_at[token + 1] - from;
    if (!span || span > UINT32_MAX) return false;
    const unsigned char *text = v->token_bytes + from;
    aotx_appraisal_prefix prefix = a->prefix;
    uint32_t walk = 0, bytes = 0;
    uint32_t task_bytes = 0;
    const unsigned char *task = aotx_appraisal_task_index(row, s, &task_bytes);
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
            bool valid = r->phase == 1 ? aotx_appraisal_outcome_byte(s, &prefix, encoded) :
                aotx_appraisal_prefix_byte(s, &prefix, encoded, task, task_bytes);
            if (!valid) return false;
        }
    }
    return bytes != 0;
}
#endif
