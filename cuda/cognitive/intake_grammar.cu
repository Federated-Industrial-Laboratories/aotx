/* Purpose: Apply source grammar checks to decoded vocabulary tokens.
 * Owns: Token-prefix admission and the accepted internal prefix state.
 * Launch shape: One vocabulary thread per candidate; one thread advances a chosen row.
 * Lifetime: One leased generation; ordinary sampling does not call these checks. */
#include "cognitive/intake_grammar.cuh"
#include "appraisal/appraisal.cuh"
#include "model/decode_state.cuh"
#include "text/text.cuh"

__device__ bool aotx_intake_advance(uint32_t row, const unsigned char *bytes, uint32_t length) {
    if (aotx_appraisal.active) return aotx_appraisal_advance(row, bytes, length);
    aotx_intake_row *r = aotx_intake.rows + row;
    const aotx_intake_index_row *s = aotx_intake_index_rows + row;
    for (uint32_t j = 0; j < length; ++j) {
        uint32_t before = r->prefix.items;
        if (!aotx_intake_prefix_byte(s, &r->prefix, bytes[j], r->items, before)) return false;
        if (r->prefix.items != before) {
            aotx_intake_item *item = r->items + before;
            *item = {}; item->kind = r->prefix.kind; item->start = r->prefix.start;
            item->length = r->prefix.length; item->target = r->prefix.number;
        }
    }
    return true;
}
