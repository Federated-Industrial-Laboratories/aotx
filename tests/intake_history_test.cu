/* Purpose: Verify correction targets at their recorded sequence positions.
 * Owns: Independent history images and exact batch publication checks.
 * Launch shape: Distinct N=1 and N=64 owners through restore and update.
 * Lifetime: One test process, with both state schemas and reordered images. */
#include "cognitive_fixture.h"
#include "cognitive/intake.h"
#include <algorithm>

static std::string aotx_quote(unsigned i) {
    return "Person" + std::to_string(i) + " will not cook.";
}
static aotx_row aotx_history_row(unsigned i, unsigned kind, unsigned id, unsigned sequence) {
    auto r = aotx_object(i, kind, id, sequence);
    memset(r.data() + AOTX_CO_SUBJECT, 0, 16); return r;
}
static aotx_bytes aotx_history_payload(unsigned i, unsigned kind) {
    auto text = aotx_quote(i); aotx_bytes p(AOTX_INTAKE_PAYLOAD + text.size(), 0);
    memcpy(p.data(), "AOTXMEM3", 8); aotx_put(p.data() + 8, 3, 4);
    aotx_put(p.data() + 12, text.size(), 4); aotx_put(p.data() + 16, kind, 4);
    memset(p.data() + 24, 0x51, 32); memset(p.data() + 56, 0x71, 32);
    memcpy(p.data() + AOTX_INTAKE_PAYLOAD, text.data(), text.size()); return p;
}
static void aotx_history_add(aotx_fixture &f, unsigned n, unsigned id, unsigned target = 0,
    unsigned version = 1, unsigned created = 0, unsigned target_version = 1) {
    unsigned first = f.rows.size() + 1;
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_history_row(i, AOTX_COG_ASSERTION, id + i, first + i);
        aotx_put(r.data() + AOTX_CO_VERSION, version);
        if (created) aotx_put(r.data() + AOTX_CO_CREATED, created + i);
        aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
        aotx_id(r.data() + AOTX_CO_SOURCE, 100 + i); aotx_put(r.data() + AOTX_CO_SOURCE_VERSION, 1);
        if (target) {
            aotx_id(r.data() + AOTX_CO_SUPERSEDES, target + i);
            aotx_put(r.data() + AOTX_CO_SUPER_VERSION, target_version);
        }
        f.add(r, aotx_history_payload(i, target ? AOTX_INTAKE_CORRECTION : AOTX_INTAKE_ASSERTION));
    }
}
static aotx_fixture aotx_history_source(unsigned n) {
    aotx_fixture f;
    for (unsigned i = 0; i < n; ++i) {
        auto text = aotx_quote(i); aotx_bytes p(32 + text.size(), 0);
        memcpy(p.data(), "AOTXMEM1", 8); aotx_put(p.data() + 8, 1, 4);
        aotx_put(p.data() + 12, text.size(), 4); memcpy(p.data() + 32, text.data(), text.size());
        f.add(aotx_history_row(i, AOTX_COG_EVENT, 100 + i, i + 1), p);
    }
    aotx_history_add(f, n, 1000); return f;
}
static void aotx_history_root(aotx_fixture &f) {
    auto original = f.rows;
    for (auto &r : f.rows) {
        for (unsigned field : {AOTX_CO_SOURCE, AOTX_CO_SUPERSEDES, AOTX_CO_EMBEDDING}) {
            unsigned version = field == AOTX_CO_SOURCE ? AOTX_CO_SOURCE_VERSION :
                field == AOTX_CO_SUPERSEDES ? AOTX_CO_SUPER_VERSION : AOTX_CO_EMBED_VERSION;
            if (!aotx_get(r.data() + version)) continue;
            for (const auto &old : original)
                if (!memcmp(r.data() + field, old.data() + AOTX_CO_ID, 16) &&
                    aotx_get(r.data() + version) == aotx_get(old.data() + AOTX_CO_VERSION)) {
                    aotx_put(r.data() + version, aotx_get(old.data() + AOTX_CO_UPDATED)); break;
                }
        }
        aotx_put(r.data() + AOTX_CO_VERSION, aotx_get(r.data() + AOTX_CO_UPDATED));
    }
}
static aotx_bytes aotx_history_wire(const aotx_fixture &f, bool tail, unsigned sequence,
    unsigned root, unsigned tick) {
    auto out = f.wire(tail, sequence, tick);
    if (root) {
        aotx_put(out.data() + 8, 2, 4); aotx_put(out.data() + 96, root);
        aotx_put(out.data() + 104, root); aotx_put(out.data() + 124, 80, 4);
    }
    return out;
}
static void aotx_history(unsigned n, unsigned mode, bool stale, bool snapshot, bool rooted) {
    auto f = aotx_history_source(n);
    unsigned cut = f.rows.size();
    aotx_history_add(f, n, 2000, 1000);
    if (mode == 1) aotx_history_add(f, n, 2000, 1000, 2, 2 * n + 1);
    if (mode == 3) aotx_history_add(f, n, 1000, 0, 2, n + 1);
    if (mode != 2) cut = f.rows.size();
    aotx_history_add(f, n, 3000, mode == 3 ? 1000 : 2000, 1, 0, mode % 2 ? 2 : 1);
    if (stale) {
        auto &last = f.rows.back();
        aotx_id(last.data() + AOTX_CO_SUPERSEDES, (mode == 1 ? 2000 : 1000) + n - 1);
        aotx_put(last.data() + AOTX_CO_SUPER_VERSION, 1);
    }
    if (rooted) aotx_history_root(f);
    aotx_fixture seed, tail;
    for (unsigned i = 0; i < f.rows.size(); ++i)
        (i < cut ? seed : tail).add(f.rows[i], f.payloads[i]);
    unsigned root = rooted ? cut : 0;
    aotx_device d; auto seed_bytes = aotx_history_wire(seed, false, cut, root, 5);
    aotx_check(!d.load(seed_bytes).status, "valid historical correction seed is admitted");
    aotx_check(d.checkpoint() == seed_bytes, "seed preserves every source and correction byte");
    if (snapshot) {
        std::reverse(f.rows.begin(), f.rows.end()); std::reverse(f.payloads.begin(), f.payloads.end());
    }
    auto expected = aotx_history_wire(f, false, f.rows.size(), root, 6);
    auto input = snapshot ? expected : aotx_history_wire(tail, true, cut + 1, root, 6);
    if (stale) d.rejects(input, !snapshot, AOTX_COG_STALE, "last stale correction refuses the complete batch");
    else {
        auto result = d.load(input, !snapshot);
        aotx_check(!result.status && result.applied == (snapshot ? f.rows.size() : tail.rows.size()),
            "current targets and later history remain valid at each recorded position");
        aotx_check(d.checkpoint() == expected, "accepted history preserves the exact complete image");
    }
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) for (unsigned mode = 0; mode < 4; ++mode)
        for (bool stale : {false, true}) for (bool snapshot : {false, true})
            for (bool rooted : {false, true}) aotx_history(n, mode, stale, snapshot, rooted);
    printf("correction history: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
