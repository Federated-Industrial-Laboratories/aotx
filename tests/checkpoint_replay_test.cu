/* Purpose: Verify that journal recovery preserves direct memory pressure decisions.
 * Owns: Distinct source batches, exact store comparisons and missing-decision controls.
 * Launch shape: N=1 and the complete profile batch through the real live kernels.
 * Lifetime: One bounded process; all journals and snapshots remain test-owned. */
#include "checkpoint_fixture.h"

__global__ void aotx_checkpoint_test_end(void) { if (!threadIdx.x) aotx_live_restore_end(); }
static aotx_bytes aotx_checkpoint_store(void) {
    aotx_bytes bytes(sizeof(aotx_cognitive_store));
    AOTX_CUDA(cudaMemcpyFromSymbol(bytes.data(), aotx_live_store, bytes.size()));
    return bytes;
}
static void aotx_checkpoint_append(aotx_live_records &all, const aotx_live_records &part) {
    all.insert(all.end(), part.begin(), part.end());
}
static unsigned aotx_checkpoint_op(const aotx_live_record &r) {
    return aotx_get(r.data() + AOTX_HEADER_BYTES + 4, 4);
}
static aotx_bytes aotx_checkpoint_seed(unsigned n) {
    aotx_checkpoint_device d(n); auto corpus = aotx_memory_corpus(n);
    d.live.send(aotx_live_load_bytes(corpus.wire(false, corpus.rows.size())), AOTX_LIVE_LOAD);
    d.live.send(aotx_live_binding_bytes(n, corpus.rows.size()), AOTX_LIVE_BIND);
    d.publish(1); return d.image(1);
}
static void aotx_checkpoint_admission(unsigned n, unsigned op, const aotx_bytes &seed) {
    aotx_live_records journal;
    aotx_bytes expected_store;
    std::vector<aotx_live_binding> expected_bindings;
    aotx_live_view expected = {};
    uint64_t expected_hash = 0;
    auto corpus = aotx_memory_corpus(n); uint64_t cut = corpus.rows.size();
    {
        aotx_checkpoint_device d(n);
        if (op == AOTX_LIVE_UPDATE || op == AOTX_LIVE_BIND) {
            aotx_checkpoint_append(journal, d.live.send(aotx_live_load_bytes(corpus.wire(false, cut)), AOTX_LIVE_LOAD));
            d.publish(1);
            if (op == AOTX_LIVE_UPDATE) {
                if (AOTX_MEMORY_SNAPSHOTS == 1) {
                    d.ring()->ack_boot = d.ring()->boot; d.ring()->durable_revision = 1; d.ring()->consumed = 1;
                }
                aotx_checkpoint_append(journal, d.live.send(aotx_live_binding_bytes(n, cut), AOTX_LIVE_BIND));
                d.publish(2);
            }
        }
        aotx_bytes input;
        if (op == AOTX_LIVE_LOAD) input = aotx_live_load_bytes(corpus.wire(false, cut));
        else if (op == AOTX_LIVE_BIND) input = aotx_live_binding_bytes(n, cut);
        else if (op == AOTX_CP_RESUME) input = seed;
        else {
            aotx_fixture update;
            for (unsigned i = 0; i < n; ++i)
                update.add(aotx_memory_row(i, AOTX_COG_SOURCE, 999000 + i, cut + i + 1),
                    aotx_memory_text("refused source " + std::to_string(i)));
            input = update.wire(true, cut + 1, 6);
        }
        if (op != AOTX_LIVE_UPDATE || AOTX_MEMORY_SNAPSHOTS != 2) d.ring()->error = AOTX_COG_CAPACITY;
        auto before = aotx_checkpoint_store(); auto accepted = d.live.state().accepted;
        aotx_checkpoint_append(journal, d.live.send(input, op));
        expected = d.live.state(); expected_hash = d.live.seam().apply.state_hash;
        expected_store = aotx_checkpoint_store(); expected_bindings = d.live.bindings(n);
        aotx_check(expected.status && expected.accepted == accepted && expected_store == before,
            "disk pressure refuses a valid direct operation without store mutation");
        unsigned decisions = 0, flagged = 0;
        for (const auto &r : journal) {
            if (aotx_checkpoint_op(r) == AOTX_LIVE_ADMISSION) ++decisions;
            else if (((const aotx_record_header *)r.data())->flags & AOTX_FLAG_ADMISSION) ++flagged;
        }
        aotx_check(decisions == (op == AOTX_LIVE_UPDATE ? 3u : op == AOTX_LIVE_BIND ? 2u : 1u) && flagged,
            "each direct operation has a correlated admission and marked input");
    }
    {
        aotx_live_device restored(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
        restored.process(journal, true);
        auto actual = restored.state();
        aotx_check(actual.status == expected.status && actual.ready == expected.ready &&
            actual.accepted == expected.accepted && actual.refused == expected.refused && !actual.fatal,
            "recovery preserves direct refusal and operation counters");
        aotx_check(aotx_checkpoint_store() == expected_store, "recovery restores every original store byte");
        auto bindings = restored.bindings(n);
        for (unsigned i = 0; i < n; ++i)
            aotx_check(!memcmp(&bindings[i], &expected_bindings[i], sizeof(bindings[i])),
                "recovery preserves each distinct binding under pressure");
        aotx_check(restored.seam().apply.state_hash == expected_hash, "admission records share the exact journal hash");
    }
    for (unsigned mode = 0; mode < 2; ++mode) {
        auto bad = journal;
        if (!mode) bad.pop_back();
        else bad.back()[AOTX_HEADER_BYTES + AOTX_LIVE_PART + 32] ^= 1;
        aotx_live_device restored(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
        restored.process(bad, true, false);
        aotx_checkpoint_test_end<<<1,1>>>(); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_check(restored.state().fatal && aotx_checkpoint_store() == expected_store,
            "a missing or mismatched admission refuses recovery without applying the operation");
    }
}
static void aotx_checkpoint_legacy(unsigned n) {
    auto corpus = aotx_memory_corpus(n); aotx_live_records journal;
    aotx_bytes expected;
    {
        aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
        journal = d.send(aotx_live_load_bytes(corpus.wire(false, corpus.rows.size())), AOTX_LIVE_LOAD);
        expected = aotx_checkpoint_store();
    }
    aotx_live_records legacy;
    for (auto r : journal) if (aotx_checkpoint_op(r) != AOTX_LIVE_ADMISSION) {
        ((aotx_record_header *)r.data())->flags &= ~AOTX_FLAG_ADMISSION; legacy.push_back(r);
    }
    aotx_live_device restored(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
    restored.process(legacy, true);
    aotx_check(!restored.state().status && restored.state().accepted == 1 && aotx_checkpoint_store() == expected,
        "unmarked older direct input restores through its original path");
}
static void aotx_checkpoint_resume_replay(unsigned n, const aotx_bytes &seed) {
    aotx_live_records journal;
    aotx_bytes expected;
    std::vector<aotx_live_binding> bindings;
    uint64_t accepted = 0, hash = 0;
    {
        aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
        journal = d.send(seed, AOTX_CP_RESUME);
        aotx_check(!d.state().status, "a complete file resume records an allowed admission");
        expected = aotx_checkpoint_store(); bindings = d.bindings(n);
        accepted = d.state().accepted; hash = d.seam().apply.state_hash;
    }
    aotx_live_device restored(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
    restored.process(journal, true);
    aotx_check(!restored.state().status && !restored.state().fatal && restored.state().accepted == accepted &&
        aotx_checkpoint_store() == expected && restored.seam().apply.state_hash == hash,
        "an allowed file resume restores its exact store and journal outcome");
    auto actual = restored.bindings(n);
    for (unsigned i = 0; i < n; ++i)
        aotx_check(!memcmp(&actual[i], &bindings[i], sizeof(actual[i])), "allowed resume replay restores distinct bindings");
}
int main() {
    const unsigned batch = AOTX_SLOTS < AOTX_RECALL_BATCH ? AOTX_SLOTS : AOTX_RECALL_BATCH;
    for (unsigned n : {1u, batch}) {
        auto seed = aotx_checkpoint_seed(n);
        for (unsigned op : {AOTX_LIVE_LOAD, AOTX_LIVE_UPDATE, AOTX_LIVE_BIND, AOTX_CP_RESUME})
            aotx_checkpoint_admission(n, op, seed);
        aotx_checkpoint_legacy(n);
        aotx_checkpoint_resume_replay(n, seed);
    }
    printf("checkpoint replay: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
