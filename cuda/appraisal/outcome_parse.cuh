/* Purpose: Admit complete quote arrays against the immutable source event.
 * Owns: Exact unique quotes and independent JSON validation.
 * Launch shape: One source row per thread in the admitted batch.
 * Lifetime: First-stage admission and recorded-output validation. */
#ifndef AOTX_APPRAISAL_OUTCOME_PARSE_CUH
#define AOTX_APPRAISAL_OUTCOME_PARSE_CUH
#include "appraisal/parse.cuh"

__device__ inline uint32_t aotx_appraisal_outcome_parse(uint32_t row,
    const unsigned char *output, uint32_t length) {
    aotx_intake_row *text = aotx_intake.rows + row;
    uint32_t bytes = 0, count = 0, starts[8], sizes[8];
    const unsigned char *source = aotx_appraisal_source(aotx_appraisal.rows[row].source, &bytes);
    const unsigned char *q = aotx_live.requests + 64 + row * AOTX_RECALL_QUERY;
    if (!source || bytes != aotx_cog_u32(q + 148) || !aotx_cog_equal(source, q + 4640, bytes) ||
        aotx_cog_zero(aotx_live_store.objects[aotx_appraisal.rows[row].source] + AOTX_CO_SUBJECT, 16)) return AOTX_COG_SOURCE;
    if (!length || length > AOTX_INTAKE_REPLY) return AOTX_COG_FORMAT;
    aotx_intake_reader r = {output, 0, length};
    if (!aotx_intake_take(&r, '[')) return AOTX_COG_FORMAT;
    aotx_intake_space(&r);
    if (r.at < r.bytes && r.p[r.at] != ']') for (;;) {
        if (count == 8) return AOTX_COG_FORMAT;
        uint32_t start = 0, size = 0;
        uint32_t status = aotx_appraisal_quote_parse(&r, text->quote, source, bytes, &start, &size);
        if (status) return status;
        if (!size) return AOTX_COG_SOURCE;
        for (uint32_t j = 0; j < count; ++j)
            if (starts[j] == start && sizes[j] == size) return AOTX_COG_SOURCE;
        starts[count] = start; sizes[count++] = size;
        aotx_intake_space(&r);
        if (r.at == r.bytes || r.p[r.at] != ',') break;
        ++r.at;
    }
    if (!aotx_intake_take(&r, ']')) return AOTX_COG_FORMAT;
    aotx_intake_space(&r);
    return r.at == r.bytes ? AOTX_COG_OK : AOTX_COG_FORMAT;
}
#endif
