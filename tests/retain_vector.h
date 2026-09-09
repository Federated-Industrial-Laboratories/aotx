/* Purpose: Check typed vector source validation and explicit focus selection.
 * Owns: Deliberately invalid component headers in isolated test stores.
 * Launch shape: One and 64 distinct retained inputs; the last vector is changed.
 * Lifetime: One GPU test process. */
#ifndef AOTX_TEST_RETAIN_VECTOR_H
#define AOTX_TEST_RETAIN_VECTOR_H
#include "retain_pressure.h"

__global__ void aotx_retain_vector_change(unsigned n, unsigned offset) {
    auto r = aotx_live_store.objects[(n - 1) * 3 + 1];
    aotx_live_store.payload[aotx_cog_u64(r + AOTX_CO_OFFSET) + offset] ^= 1;
}
static void aotx_retain_vectors(unsigned n) {
    aotx_live_device d(n); aotx_retain_open(d, n);
    d.send(aotx_retain_query(n, 0, 1), 4); d.idle(n);
    auto request = aotx_retain_bytes(n, 0, 1, 1, false);
    for (unsigned i = 0; i < n; ++i) aotx_put(request.data() + 64 + i * 160 + 128, AOTX_COG_UNKNOWN, 4);
    d.send(request, 8); aotx_check(!d.state().status, "retention without focus is valid");
    for (const auto &b : d.bindings(n)) aotx_check(!b.focus_count, "focus admission remains explicit");
    auto query = aotx_retain_query(n, 3 * n, 2);
    for (unsigned offset : {8u, 88u, 104u, 112u}) {
        aotx_retain_vector_change<<<1,1>>>(n, offset);
        d.send(query, 4);
        aotx_check(d.state().status == AOTX_COG_LAYOUT, "invalid typed vector source refuses the query batch");
        for (const auto &b : d.bindings(n)) aotx_check(b.ordinal == 1 && !b.focus_count, "invalid vector preserves every current request");
        aotx_retain_vector_change<<<1,1>>>(n, offset);
    }
    d.send(query, 4); aotx_check(!d.state().status, "valid retained vectors remain available to semantic recall");
    for (unsigned i = 0; i < n; ++i) aotx_check(aotx_selected(d.bindings(n)[i].choice, 0) == 300064 + i,
        "semantic recall selects the matching private retained input");
}
#endif
