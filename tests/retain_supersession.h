/* Purpose: Check batched supersession and caller focus with distinct references.
 * Owns: Seeded working memory, exact replacement checks and last-row refusals.
 * Launch shape: One and 64 bindings through the live device kernels.
 * Lifetime: One bounded test process without model weights. */
#ifndef AOTX_TEST_RETAIN_SUPERSESSION_H
#define AOTX_TEST_RETAIN_SUPERSESSION_H
#include "retain_pressure.h"

__global__ void aotx_retain_seed_replacement(unsigned n) {
    unsigned i = threadIdx.x; if (i >= n) return;
    auto b = aotx_live_bindings + i; b->focus_count = 1;
    aotx_cog_put(b->focus[0], 500000 + i, 8);
    b->focus[0][15] = 0xa7; aotx_cog_put(b->focus[0] + 16, 1, 8);
}
static void aotx_retain_unchanged(aotx_live_device &d, unsigned n, const aotx_bytes &store,
    const std::vector<aotx_live_binding> &bindings, unsigned status) {
    aotx_check(d.state().status == status, "bad final row returns the expected refusal");
    aotx_check(aotx_retain_store() == store, "batch refusal preserves every store byte");
    auto after = d.bindings(n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!memcmp(&after[i], &bindings[i], sizeof(after[i])),
        "batch refusal preserves each binding, query, ordinal and working reference");
}
static void aotx_retain_supersession(unsigned n) {
    unsigned start_checks = aotx_checks, start_failures = aotx_failures;
    aotx_fixture seed;
    for (unsigned i = 0; i < n; ++i) seed.add(aotx_memory_row(i, AOTX_COG_WORKING, 500000 + i, i + 1, 2),
        aotx_memory_text("old source " + std::to_string(i)));
    aotx_live_device d(n); d.send(aotx_live_load_bytes(seed.wire(false, n)), 1);
    d.send(aotx_live_binding_bytes(n, n, 2), 3);
    d.send(aotx_retain_query(n, n, 1, false, 2), 4); d.idle(n);
    aotx_check(!d.state().status, "distinct source queries are accepted before replacement");
    aotx_retain_seed_replacement<<<1,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize());
    auto before = aotx_retain_store(); auto bindings = d.bindings(n);
    auto request = aotx_retain_bytes(n, n, 1, 0, false);
    for (unsigned i = 0; i < n; ++i) {
        auto r = request.data() + 64 + i * 160;
        aotx_id(r + 80, 500000 + i); aotx_put(r + 96, 1);
    }
    auto bad = request; bad[64 + (n - 1) * 160 + 112] ^= 1;
    d.send(bad, 8); aotx_retain_unchanged(d, n, before, bindings, 7);
    if (n > 1) {
        bad = request; auto r = bad.data() + 64 + (n - 1) * 160;
        aotx_id(r + 80, 500000); aotx_id(r + 112, 3000);
        d.send(bad, 8); aotx_retain_unchanged(d, n, before, bindings, 11);
    }
    auto result = aotx_retain_result(d.send(request, 8));
    aotx_check(!d.state().status, "every distinct supersession succeeds as one batch");
    auto after = aotx_retain_store(); auto store = (const aotx_cognitive_store *)after.data();
    auto prior = (const aotx_cognitive_store *)before.data(); bindings = d.bindings(n);
    aotx_check(store->count == n * 4 && store->sequence == n * 4, "seed and retained objects fit the exact store bound");
    aotx_check(result.size() > 64 + n * 384, "replacement records all rows and a canonical tail");
    for (unsigned i = 0; i < n; ++i) {
        auto old = prior->objects[i], r = store->objects[n + i * 3 + 2];
        aotx_check(!memcmp(store->objects[i], old, 256) &&
            !memcmp(store->payload + aotx_get(old + AOTX_CO_OFFSET), prior->payload + aotx_get(old + AOTX_CO_OFFSET),
                aotx_get(old + AOTX_CO_BYTES)), "replacement preserves every old object and payload");
        aotx_check(aotx_get(r + AOTX_CO_ID) == 300064 + i && aotx_get(r + AOTX_CO_SUPERSEDES) == 500000 + i &&
            aotx_get(r + AOTX_CO_SUPER_VERSION) == 1, "each new memory names its own superseded reference");
        aotx_check(aotx_get(r + AOTX_CO_SOURCE) == 100064 + i && aotx_get(r + AOTX_CO_EMBEDDING) == 400064 + i,
            "each replacement retains its own source and component");
        aotx_check(bindings[i].focus_count == 1 && aotx_get(bindings[i].focus[0]) == 300064 + i &&
            aotx_get(bindings[i].focus[0] + 16) == 1, "focused replacement works with admission off");
        if (result.size() >= 64 + n * 384) {
            auto out = result.data() + 64 + i * 384;
            aotx_check(!memcmp(out, request.data() + 64 + i * 160, 160) && aotx_get(out + 160, 4) == 1 &&
                !memcmp(out + 192, bindings[i].focus, 192), "journal rows contain the exact replacement and complete focus");
        }
    }
    auto query = aotx_retain_query(n, n * 4, 2, true, 2);
    for (unsigned i = 0; i < n; ++i) {
        auto q = query.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
        aotx_put(q + 132, n == 1 ? 1 : 2, 4);
        aotx_pin(q, 1, 0, 300064 + (n > 1 && i % 2 ? (i + 1) % n : i));
        if (n > 1 && !(i % 2)) aotx_pin(q, 1, 1, 300064 + (i + 1) % n);
    }
    bad = query; aotx_pin(bad.data() + 128 + (n - 1) * AOTX_LIVE_QUERY_ROW, 1, 0, 500000 + n - 1);
    d.send(bad, 4); aotx_retain_unchanged(d, n, after, bindings, 10);
    d.send(query, 4); aotx_check(!d.state().status, "caller focus merges after a refused batch without advancing its ordinal");
    bindings = d.bindings(n);
    for (unsigned i = 0; i < n; ++i) {
        const auto &b = bindings[i]; unsigned count = n == 1 ? 1 : 2;
        aotx_check(b.ordinal == 2 && aotx_get(b.query + 144, 4) == count && b.choice.count == count,
            "exact references deduplicate and missing device references append");
        for (unsigned j = 0; j < count; ++j) {
            unsigned index = (i + ((i % 2) ^ j)) % n;
            aotx_check(aotx_get(b.query + 4448 + j * 24) == 300064 + index &&
                aotx_get(b.query + 4464 + j * 24) == 1 && aotx_selected(b.choice, j) == 300064 + index,
                "each merged query and selection retain caller order before appended focus");
        }
    }
    if (n > 1) {
        d.idle(n); bindings = d.bindings(n); query = aotx_retain_query(n, n * 4, 3, true, 2);
        auto q = query.data() + 128 + (n - 1) * AOTX_LIVE_QUERY_ROW;
        for (unsigned j = 0; j < 8; ++j) aotx_pin(q, 1, j, 300064 + j);
        d.send(query, 4); aotx_retain_unchanged(d, n, after, bindings, 2);
    }
    printf("supersession n=%u: %u checks, %u failures\n", n, aotx_checks - start_checks, aotx_failures - start_failures);
}
#endif
