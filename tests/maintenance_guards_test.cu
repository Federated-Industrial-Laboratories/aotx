/* Purpose: Verify protected roots, exact version guards and automatic maintenance.
 * Owns: Independent source, appraisal, correction and full-capacity refusal fixtures.
 * Launch shape: Distinct N=1/N=64 batches through the maintained live consumer.
 * Lifetime: One bounded process; no production guard is bypassed. */
#include "maintenance_fixture.h"
#include "cognitive/lookup.cuh"

static aotx_bytes aotx_maint_root_image(const aotx_fixture &f, uint64_t sequence) {
    auto image = f.wire(false, sequence);
    aotx_put(image.data() + 8, 2, 4); aotx_put(image.data() + 96, sequence);
    aotx_put(image.data() + 104, sequence); aotx_put(image.data() + 124, 80, 4);
    return image;
}
static aotx_bytes aotx_maint_root_payload(unsigned i, unsigned kind) {
    if (kind != AOTX_COG_MEDIA) return aotx_bytes(32, (unsigned char)(i + 1));
    aotx_bytes p(AOTX_COG_MEDIA_HEADER + 6, 0);
    aotx_put(p.data(), 1, 4); aotx_put(p.data() + 4, AOTX_COG_IMAGE_MEDIA, 4);
    aotx_put(p.data() + 8, AOTX_COG_SOURCE_BYTES, 4); aotx_put(p.data() + 12, AOTX_COG_U8, 4);
    aotx_put(p.data() + 16, 1); aotx_put(p.data() + 24, 2); aotx_put(p.data() + 32, 3);
    aotx_put(p.data() + 48, 6); aotx_put(p.data() + 64, AOTX_COG_SPATIAL, 4);
    aotx_put(p.data() + 68, 3, 4); p[72] = (unsigned char)(i + 1);
    for (unsigned j = AOTX_COG_MEDIA_HEADER; j < p.size(); ++j) p[j] = (unsigned char)(i + j);
    return p;
}
static void aotx_maint_root_validation(unsigned n) {
    aotx_device d;
    for (unsigned kind : {AOTX_COG_EVENT, AOTX_COG_MEDIA, AOTX_COG_COMPONENT}) {
        aotx_fixture original;
        for (unsigned i = 0; i < n; ++i)
            original.add(aotx_object(i, kind, 100 + i, i + 1), aotx_maint_root_payload(i, kind));
        auto first = original.wire(false, n);
        aotx_check(!d.load(first).status, "immutable schema one creation batches admit");
        aotx_fixture revision = original;
        for (unsigned i = 0; i < n; ++i) {
            aotx_put(revision.rows[i].data() + AOTX_CO_VERSION, 2);
            aotx_put(revision.rows[i].data() + AOTX_CO_UPDATED, n + i + 1);
            revision.payloads[i].back() ^= 0x55;
        }
        d.rejects(revision.wire(true, n + 1), true, AOTX_COG_VERSION,
            "immutable payload changes refuse with their predecessor present");
        d.rejects(aotx_maint_root_image(revision, 2 * n), false, AOTX_COG_VERSION,
            "a root cannot admit immutable payload changes without their predecessor");
        auto legacy = aotx_maint_root_image(original, n);
        aotx_check(!d.load(legacy).status && d.checkpoint() == legacy,
            "legacy immutable creations remain exact across a root checkpoint");
        auto tombstones = revision;
        for (unsigned i = 0; i < n; ++i) {
            aotx_put(tombstones.rows[i].data() + AOTX_CO_FLAGS, AOTX_COG_TOMBSTONE, 4);
            tombstones.payloads[i].clear();
        }
        auto deleted = aotx_maint_root_image(tombstones, 2 * n);
        aotx_check(!d.load(deleted).status && d.checkpoint() == deleted,
            "valid immutable tombstones restore after their predecessor is retired");
        auto resolved = d.resolve(n, 100, 2);
        for (unsigned i = 0; i < n; ++i)
            aotx_check(resolved[i].status == AOTX_COG_DENIED, "each rooted deletion remains inaccessible");
        auto fresh = original;
        for (unsigned i = 0; i < n; ++i) {
            uint64_t sequence = 4 * n + i + 1;
            aotx_put(fresh.rows[i].data() + AOTX_CO_CREATED, sequence);
            aotx_put(fresh.rows[i].data() + AOTX_CO_UPDATED, sequence);
            aotx_put(fresh.rows[i].data() + AOTX_CO_VERSION, sequence);
        }
        auto current = aotx_maint_root_image(fresh, 5 * n);
        aotx_check(!d.load(current).status && d.checkpoint() == current,
            "sequence-version immutable creations remain exact across a root checkpoint");
        for (auto &r : fresh.rows) aotx_put(r.data() + AOTX_CO_VERSION, 3);
        d.rejects(aotx_maint_root_image(fresh, 5 * n), false, AOTX_COG_VERSION,
            "rooted creations cannot use an unrelated version");
    }
    auto mutable_rows = aotx_initial(n);
    for (auto &r : mutable_rows.rows) {
        aotx_put(r.data() + AOTX_CO_UPDATED, aotx_get(r.data() + AOTX_CO_CREATED) + n);
        aotx_put(r.data() + AOTX_CO_VERSION, 2);
    }
    auto valid = aotx_maint_root_image(mutable_rows, 2 * n);
    aotx_check(!d.load(valid).status && d.checkpoint() == valid,
        "mutable revisions restore without retired predecessors");
    for (auto &r : mutable_rows.rows) aotx_put(r.data() + AOTX_CO_VERSION, 1);
    d.rejects(aotx_maint_root_image(mutable_rows, 2 * n), false, AOTX_COG_VERSION,
        "a root cannot turn version one into a revised object");
}

static void aotx_maint_retry_coverage(unsigned n) {
    aotx_device d; auto root = aotx_initial(n);
    auto policy = std::make_unique<aotx_cognitive_store>(); policy->root_sequence = n;
    policy->keep_recent = n; policy->pressure_percent = 80;
    auto first = root.wire(false, n); aotx_maint_schema(first, *policy);
    aotx_check(!d.load(first).status, "a complete retained retry window restores");
    aotx_fixture fresh;
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_object(i, AOTX_COG_ASSERTION, 1000 + i, n + i + 1);
        aotx_put(r.data() + AOTX_CO_VERSION, n + i + 1);
        fresh.add(r, aotx_memory_text("new retry record " + std::to_string(i)));
    }
    auto tail = fresh.wire(true, n + 1, 6); aotx_maint_schema(tail, *policy);
    aotx_check(!d.load(tail, true).status && !d.load(tail, true).status,
        "the complete new batch applies and retries exactly");
    auto complete = d.checkpoint();
    aotx_check(!d.load(complete).status && d.checkpoint() == complete,
        "a checkpoint retains all admitted records after its root");
    auto missing = root.wire(false, 2 * n, 6); aotx_maint_schema(missing, *policy);
    d.rejects(missing, false, AOTX_COG_SEQUENCE, "missing post-root updates refuse before publication");
    aotx_check(!d.load(tail, true).status && d.checkpoint() == complete,
        "an exact retry still succeeds after the malformed import refuses");
    auto all = root; all.append(fresh);
    for (unsigned index : {0u, n, 2 * n - 1}) {
        auto gap = all; gap.rows.erase(gap.rows.begin() + index); gap.payloads.erase(gap.payloads.begin() + index);
        missing = gap.wire(false, 2 * n, 6); aotx_maint_schema(missing, *policy);
        d.rejects(missing, false, AOTX_COG_SEQUENCE, "a missing retry record refuses at each window position");
    }
    auto duplicate = all;
    duplicate.rows.back() = duplicate.rows.front(); duplicate.payloads.back() = duplicate.payloads.front();
    missing = duplicate.wire(false, 2 * n, 6); aotx_maint_schema(missing, *policy);
    d.rejects(missing, false, AOTX_COG_SEQUENCE, "duplicate updates cannot replace missing window coverage");
    policy->root_sequence = 2 * n; policy->retry_floor = n;
    auto retired = fresh.wire(false, 2 * n, 6); aotx_maint_schema(retired, *policy);
    aotx_check(!d.load(retired).status && d.checkpoint() == retired,
        "genuine retirement below the retry floor restores exactly");
    aotx_check(!d.load(tail, true).status, "retained retries accept their original pre-maintenance header");
    policy->retry_floor = 2 * n;
    aotx_fixture empty;
    auto retired_all = empty.wire(false, 2 * n, 6); aotx_maint_schema(retired_all, *policy);
    aotx_check(!d.load(retired_all).status && d.checkpoint() == retired_all,
        "an empty fully retired root remains a valid checkpoint");
    d.rejects(tail, true, AOTX_COG_STALE, "a retired retry is stale after an empty root restores");
    aotx_live_device live(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
    live.send(aotx_live_load_bytes(root.wire(false, 2 * n)), AOTX_LIVE_LOAD);
    auto before = aotx_maint_store();
    aotx_check(!live.state().status, "legacy sparse checkpoints retain their import contract");
    live.send(aotx_maint_policy(*before, 2 * n, 0), AOTX_LIVE_MAINTAIN);
    auto after = aotx_maint_store();
    aotx_check(live.state().status == AOTX_COG_SEQUENCE && !memcmp(before.get(), after.get(), sizeof(*before)),
        "maintenance cannot promise missing legacy retry records or change live state");
    live.send(aotx_maint_policy(*before, 0, 0), AOTX_LIVE_MAINTAIN);
    after = aotx_maint_store();
    aotx_check(!live.state().status && after->count == n && after->retry_floor == 2 * n,
        "a sparse legacy store can select an available empty retry window");
}

__global__ void aotx_maint_guard_lookup(unsigned n, unsigned *status) {
    unsigned i = threadIdx.x; if (i >= n) return;
    const uint64_t ids[] = {1ull + i, 101ull + i, 301ull + i};
    for (unsigned j = 0; j < 3; ++j) {
        aotx_cognitive_query q = {}; q.version = 1;
        aotx_cog_put(q.id, ids[j], 8); q.id[15] = 0xa7;
        aotx_cog_put(q.principal, 1000 + i, 8); q.principal[15] = 0xa7;
        aotx_cog_put(q.room, 2000 + i, 8); q.room[15] = 0xa7;
        status[i * 3 + j] = aotx_cog_resolve_one(&aotx_live_store, &q).status;
    }
}
static void aotx_maint_roots(unsigned n) {
    aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
    aotx_fixture f = aotx_initial(n), appraisal = aotx_appraisals(n);
    for (auto &r : appraisal.rows) aotx_put(r.data() + AOTX_CO_FLAGS, AOTX_COG_PROTECTED, 4);
    f.append(appraisal);
    for (unsigned v = 2; v <= 3; ++v) for (unsigned i = 0; i < n; ++i) {
        auto r = f.rows[i]; aotx_put(r.data() + AOTX_CO_VERSION, v);
        aotx_put(r.data() + AOTX_CO_UPDATED, f.rows.size() + 1);
        if (v == 3) aotx_put(r.data() + AOTX_CO_FLAGS, AOTX_COG_TOMBSTONE, 4);
        f.add(r, v == 3 ? aotx_bytes{} : aotx_memory_text("obsolete " + std::to_string(i)));
    }
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_object(i, AOTX_COG_ASSERTION, 301 + i, f.rows.size() + 1);
        aotx_put(r.data() + AOTX_CO_FLAGS, AOTX_COG_PROTECTED, 4);
        f.add(r, aotx_memory_text("prior statement " + std::to_string(i)));
    }
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_object(i, AOTX_COG_ASSERTION, 501 + i, f.rows.size() + 1);
        aotx_id(r.data() + AOTX_CO_SUPERSEDES, 301 + i); aotx_put(r.data() + AOTX_CO_SUPER_VERSION, 1);
        f.add(r, aotx_memory_text("correction " + std::to_string(i)));
    }
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_object(i, AOTX_COG_INTENTION, 701 + i, f.rows.size() + 1);
        aotx_put(r.data() + AOTX_CO_RETENTION, 2, 4);
        f.add(r, aotx_memory_text("pending task " + std::to_string(i)));
    }
    d.send(aotx_live_load_bytes(f.wire(false, f.rows.size())), AOTX_LIVE_LOAD);
    aotx_check(!d.state().status, "distinct protected and correction roots admit before maintenance");
    auto before = aotx_maint_store(); d.send(aotx_maint_policy(*before, 0, 1), AOTX_LIVE_MAINTAIN);
    auto after = aotx_maint_store();
    aotx_check(!d.state().status && after->count == 6 * n, "only unreferenced predecessor versions are removed");
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(aotx_maint_find(*after, 1 + i, 1) >= 0 && aotx_maint_find(*after, 1 + i, 3) >= 0 &&
            aotx_maint_find(*after, 1 + i, 2) < 0, "historical source and current deletion guard survive without the unused predecessor");
        int index = aotx_maint_find(*after, 101 + i, 1);
        aotx_check(index >= 0, "protected appraisal remains present");
        if (index >= 0) {
            const unsigned char *p = after->payload + aotx_get(after->objects[index] + AOTX_CO_OFFSET);
            aotx_check(aotx_get(p + 4, 4) == 700000 + i && aotx_get(p + 8, 4) == 800000 - i,
                "each distinct benefit and harm value remains separate and exact");
        }
        aotx_check(aotx_maint_find(*after, 501 + i, 1) >= 0 && aotx_maint_find(*after, 701 + i, 1) >= 0,
            "supersession guards and pending intentions survive age-based eligibility");
    }
    unsigned *out; AOTX_CUDA(cudaMalloc(&out, n * 3 * sizeof(unsigned)));
    aotx_maint_guard_lookup<<<1,64>>>(n, out); std::vector<unsigned> status(n * 3);
    AOTX_CUDA(cudaMemcpy(status.data(), out, status.size() * sizeof(unsigned), cudaMemcpyDeviceToHost)); cudaFree(out);
    for (unsigned i = 0; i < n; ++i)
        aotx_check(status[i * 3] == AOTX_COG_DENIED && status[i * 3 + 1] == AOTX_COG_DENIED &&
            status[i * 3 + 2] == AOTX_COG_STALE, "retained historical bytes cannot bypass current deletion or supersession");
    auto policy = aotx_maint_policy(*after, 0, 1); policy[40] ^= 1;
    d.send(policy, AOTX_LIVE_MAINTAIN); auto unchanged = aotx_maint_store();
    aotx_check(d.state().status == AOTX_COG_STALE && !memcmp(after.get(), unchanged.get(), sizeof(*after)),
        "a stale expected root refuses before mutation");
    aotx_fixture update;
    for (unsigned i = 0; i < n; ++i) {
        auto r = f.rows[4 * n + i];
        aotx_put(r.data() + AOTX_CO_UPDATED, after->sequence + i + 1);
        update.add(r, aotx_memory_text("invalid reused version " + std::to_string(i)));
    }
    auto tail = update.wire(true, after->sequence + 1, after->tick + 1); aotx_maint_schema(tail, *after);
    d.send(tail, AOTX_LIVE_UPDATE); unchanged = aotx_maint_store();
    aotx_check(d.state().status == AOTX_COG_VERSION && !memcmp(after.get(), unchanged.get(), sizeof(*after)),
        "new updates cannot reuse an old exact version");
}
static void aotx_maint_automatic(unsigned n) {
    aotx_live_records journal; std::unique_ptr<aotx_cognitive_store> expected; uint64_t hash = 0;
    {
        aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
        aotx_fixture f;
        for (unsigned i = 0; i < AOTX_COG_OBJECTS / 100 + n; ++i) {
            auto r = aotx_memory_row(i % n, AOTX_COG_EVENT, 9000000 + i, i + 1);
            aotx_put(r.data() + AOTX_CO_FLAGS, AOTX_COG_PROTECTED, 4);
            f.add(r, aotx_memory_text("protected " + std::to_string(i)));
        }
        aotx_maint_append(journal, d.send(aotx_live_load_bytes(f.wire(false, f.rows.size())), AOTX_LIVE_LOAD));
        auto s = aotx_maint_store();
        aotx_maint_append(journal, d.send(aotx_maint_policy(*s, 0, 1, 1, 1), AOTX_LIVE_MAINTAIN));
        s = aotx_maint_store(); aotx_fixture extra;
        for (unsigned i = 0; i < n; ++i) {
            uint64_t seq = s->sequence + i + 1;
            auto r = aotx_memory_row(i, AOTX_COG_EVENT, 9100000 + i, seq);
            aotx_put(r.data() + AOTX_CO_VERSION, seq); extra.add(r, aotx_memory_text("ordinary " + std::to_string(i)));
        }
        auto tail = extra.wire(true, s->sequence + 1, s->tick + 1); aotx_maint_schema(tail, *s);
        auto parts = aotx_live_parts(tail, AOTX_LIVE_UPDATE, 700);
        uint64_t accepted = d.state().accepted;
        aotx_maint_append(journal, d.process({parts[0]}, false, false));
        aotx_check(d.state().accepted == accepted && d.state().received, "automatic maintenance cannot cross a partial input transfer");
        parts.erase(parts.begin()); aotx_maint_append(journal, d.process(parts));
        accepted = d.state().accepted;
        auto automatic = d.process({}); aotx_maint_append(journal, automatic);
        expected = aotx_maint_store();
        aotx_check(!d.state().status && d.state().accepted == accepted + 1 &&
            expected->root_sequence == expected->sequence && expected->count == f.rows.size() + 1,
            "configured capacity pressure runs the same batched collector automatically");
        unsigned requests = 0;
        for (const auto &r : automatic) {
            const unsigned char *p = r.data() + AOTX_HEADER_BYTES;
            if (aotx_get(p + 4, 4) == AOTX_LIVE_MAINTAIN && aotx_get(p + 32 + 12, 4) == 1) ++requests;
        }
        aotx_check(requests == 1, "automatic maintenance records its exact policy request");
        accepted = d.state().accepted; d.process({});
        aotx_check(d.state().accepted == accepted, "unchanged pressure does not cause an unbounded repeated collector");
        hash = d.seam().apply.state_hash;
    }
    aotx_live_device restored(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
    restored.process(journal, true); auto actual = aotx_maint_store();
    aotx_check(!restored.state().fatal && !memcmp(actual.get(), expected.get(), sizeof(*actual)) &&
        restored.seam().apply.state_hash == hash, "automatic maintenance replay preserves complete state and hash");
}
static void aotx_maint_full(unsigned n) {
    aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
    aotx_fixture f;
    for (unsigned i = 0; i < AOTX_COG_OBJECTS; ++i) {
        auto r = aotx_memory_row(i % n, AOTX_COG_EVENT, 9200000 + i, i + 1);
        aotx_put(r.data() + AOTX_CO_FLAGS, AOTX_COG_PROTECTED, 4); f.add(r, aotx_bytes(16, (unsigned char)(i % 251 + 1)));
    }
    d.send(aotx_live_load_bytes(f.wire(false, f.rows.size())), AOTX_LIVE_LOAD);
    auto before = aotx_maint_store(); d.send(aotx_maint_policy(*before, 0, 1), AOTX_LIVE_MAINTAIN);
    before = aotx_maint_store();
    aotx_check(!d.state().status && before->count == AOTX_COG_OBJECTS, "a wholly protected store retains the full allocation");
    aotx_fixture extra;
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_memory_row(i, AOTX_COG_EVENT, 9300000 + i, before->sequence + i + 1);
        aotx_put(r.data() + AOTX_CO_VERSION, before->sequence + i + 1); extra.add(r, aotx_bytes(16, (unsigned char)(i + 1)));
    }
    auto tail = extra.wire(true, before->sequence + 1, before->tick + 1); aotx_maint_schema(tail, *before);
    d.send(tail, AOTX_LIVE_UPDATE); auto after = aotx_maint_store();
    aotx_check(d.state().status == AOTX_COG_CAPACITY && !memcmp(before.get(), after.get(), sizeof(*before)),
        "full protected storage refuses the complete new batch without loss or offload");
}
int main(int argc, char **argv) {
    bool state_only = argc == 2 && !strcmp(argv[1], "--state-only");
    if (argc > 1 && !state_only) return 2;
    for (unsigned n : {1u, AOTX_SLOTS < AOTX_RECALL_BATCH ? AOTX_SLOTS : AOTX_RECALL_BATCH}) {
        aotx_maint_root_validation(n); aotx_maint_retry_coverage(n);
        aotx_maint_roots(n); aotx_maint_automatic(n);
        if (!state_only) aotx_maint_full(n);
    }
    printf("maintenance guards: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
