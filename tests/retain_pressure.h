/* Purpose: Check working-set replacement, supersession and store pressure.
 * Owns: Independently seeded pressure states and expected retained references.
 * Launch shape: One and 64 distinct bindings with bounded shared seed memory.
 * Lifetime: One device test process. */
#ifndef AOTX_TEST_RETAIN_PRESSURE_H
#define AOTX_TEST_RETAIN_PRESSURE_H
#include "retain_fixture.h"
#include "cognitive/codec.cuh"

__global__ void aotx_retain_seed_focus(unsigned n) {
    unsigned i = threadIdx.x; if (i >= n) return;
    auto b = aotx_live_bindings + i; b->focus_count = 8;
    for (unsigned j = 0; j < 8; ++j) {
        aotx_cog_put(b->focus[j], 500000 + (i + j) % 8, 8);
        b->focus[j][15] = 0xa7; aotx_cog_put(b->focus[j] + 16, 1, 8);
    }
}
static void aotx_retain_pressure(unsigned n) {
    for (unsigned mode = 0; mode < 4; ++mode) {
        aotx_fixture seed;
        for (unsigned j = 0; j < 8; ++j) {
            auto r = aotx_memory_row(0, AOTX_COG_WORKING, 500000 + j, j + 1, 2);
            aotx_put(r.data() + AOTX_CO_RETENTION, mode < 3 ? mode : 0, 4);
            if (mode == 3) aotx_put(r.data() + AOTX_CO_FLAGS, AOTX_COG_PROTECTED, 4);
            seed.add(r, aotx_memory_text("seed " + std::to_string(j)));
        }
        aotx_live_device d(n); d.send(aotx_live_load_bytes(seed.wire(false, 8)), 1);
        d.send(aotx_live_binding_bytes(n, 8, 2), 3);
        d.send(aotx_retain_query(n, 8, 1, false, 2), 4); d.idle(n);
        aotx_retain_seed_focus<<<1,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize());
        auto before = aotx_retain_store(); auto bindings = d.bindings(n);
        d.send(aotx_retain_bytes(n, 8, 1), 8);
        aotx_check(d.state().status == (mode ? 2u : 0u), "only ordinary unprotected focus can leave at capacity");
        auto rows = d.bindings(n);
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(rows[i].focus_count == 8, "working set retains a fixed reference bound");
            if (mode) aotx_check(!memcmp(rows[i].focus, bindings[i].focus, sizeof(rows[i].focus)), "refused pressure keeps all focus references");
            else {
                for (unsigned j = 0; j < 7; ++j)
                    aotx_check(aotx_get(rows[i].focus[j]) == 500000 + (i + j + 1) % 8, "oldest ordinary reference leaves focus");
                aotx_check(aotx_get(rows[i].focus[7]) == 300064 + i, "new memory enters focus last");
            }
        }
        auto after = aotx_retain_store(); auto store = (const aotx_cognitive_store *)after.data();
        aotx_check(mode ? after == before : store->count == 8 + n * 3, "focus replacement never deletes stored objects");
    }
    for (unsigned payload = 0; payload < 2; ++payload) {
        aotx_fixture seed;
        unsigned count = payload ? 1 : AOTX_COG_OBJECTS + 1 - n * 3;
        for (unsigned i = 0; i < count; ++i) seed.add(aotx_memory_row(i, AOTX_COG_COMPONENT, 600000 + i, i + 1, 2),
            aotx_bytes(payload ? AOTX_COG_PAYLOAD - 64 : 1, 0x41));
        aotx_live_device d(n); d.send(aotx_live_load_bytes(seed.wire(false, count)), 1);
        aotx_check(!d.state().status && d.state().ready, "pressure store is admitted");
        d.send(aotx_live_binding_bytes(n, count), 3); d.send(aotx_retain_query(n, count, 1), 4); d.idle(n);
        aotx_check(!d.state().status, "pressure fixture admits source query"); auto before = aotx_retain_store();
        d.send(aotx_retain_bytes(n, count, 1), 8);
        aotx_check(d.state().status == 2 && aotx_retain_store() == before, "object or payload pressure refuses every row");
    }
}

static void aotx_retain_turnover(void) {
    aotx_live_device d(1); aotx_retain_open(d, 1);
    for (unsigned turn = 1; turn <= 10; ++turn) {
        unsigned cut = (turn - 1) * 3;
        d.send(aotx_retain_query(1, cut, turn, true), 4); d.idle(1);
        d.send(aotx_retain_bytes(1, cut, turn), 8);
        aotx_check(!d.state().status, "successive turns retain exact input");
        auto b = d.bindings(1)[0]; unsigned count = std::min(turn, 8u);
        aotx_check(b.focus_count == count, "turnover keeps eight working references");
        for (unsigned i = 0; i < count; ++i) aotx_check(aotx_get(b.focus[i]) == 300000 + (turn - count + i + 1) * 64,
            "turnover keeps ordered current focus");
    }
    d.send(aotx_retain_query(1, 30, 11, true), 4); d.idle(1);
    auto r = aotx_retain_bytes(1, 30, 11, 0, false); aotx_id(r.data() + 64 + 80, 300640); aotx_put(r.data() + 64 + 96, 1);
    auto before = aotx_retain_store(); auto bad = r; bad[64 + 112] ^= 1;
    d.send(bad, 8); aotx_check(d.state().status == 7 && aotx_retain_store() == before, "supersession requires the same subject");
    d.send(r, 8); aotx_check(!d.state().status, "explicit correction retains original source and replaces focused memory");
    auto b = d.bindings(1)[0];
    aotx_check(b.focus_count == 8 && aotx_get(b.focus[7]) == 300704, "superseded focus receives the new exact reference");
    auto all = aotx_retain_store(); auto s = (const aotx_cognitive_store *)all.data();
    aotx_check(s->count == 33 && aotx_get(s->objects[32] + AOTX_CO_SUPERSEDES) == 300640, "correction preserves immutable history");
    auto next = aotx_retain_query(1, 33, 12, true);
    aotx_pin(next.data() + 128, 1, 0, 300704);
    d.send(next, 4); aotx_check(!d.state().status, "caller focus and device focus deduplicate exact references");
    b = d.bindings(1)[0]; aotx_check(aotx_get(b.query + 144, 4) == 8 && aotx_selected(b.choice, 0) == 300704,
        "explicit focus stays first after merging"); d.idle(1);
    next = aotx_retain_query(1, 33, 13, true); aotx_pin(next.data() + 128, 1, 0, 300064);
    d.send(next, 4); aotx_check(d.state().status == 2 && d.bindings(1)[0].ordinal == 12, "merged focus overflow refuses the query");
    next = aotx_retain_query(1, 33, 13); aotx_pin(next.data() + 128, 0, 0, 300640);
    d.send(next, 4); aotx_check(d.state().status == 10, "superseded memory cannot return through a required pin");
}
#endif
