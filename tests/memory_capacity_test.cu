/* Purpose: Verify configured memory bounds, late dependencies and exact recall recovery.
 * Owns: Independent full-store fixtures and per-query expected private results.
 * Launch shape: State batches and distinct recall batches at N=1 and N=64.
 * Lifetime: One test process without model weights. */
#include "recall_fixture.h"
#include "cognitive/live.cuh"
#include <chrono>

static void aotx_capacity_bounds(unsigned n) {
    aotx_fixture f;
    for (unsigned i = 0; i < AOTX_COG_OBJECTS; ++i)
        f.add(aotx_memory_row(i, AOTX_COG_COMPONENT, 4000000 + i, i + 1), { (unsigned char)i });
    aotx_device d;
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "all configured object slots are usable");
    auto expected = f.wire(false, f.rows.size());
    aotx_check(d.checkpoint() == expected, "full table checkpoint keeps every byte");
    aotx_fixture tail;
    for (unsigned i = 0; i < n; ++i)
        tail.add(aotx_memory_row(i, AOTX_COG_COMPONENT, 6000000 + i, AOTX_COG_OBJECTS + i + 1), { (unsigned char)i });
    d.rejects(tail.wire(true, AOTX_COG_OBJECTS + 1, 6), true, AOTX_COG_CAPACITY,
        "full table refuses a whole new batch");
    f = {};
    for (unsigned i = 0; i < n; ++i) {
        size_t bytes = AOTX_COG_PAYLOAD / n + (i < AOTX_COG_PAYLOAD % n);
        f.add(aotx_memory_row(i, AOTX_COG_COMPONENT, 7000000 + i, i + 1), aotx_bytes(bytes, (unsigned char)(i + 1)));
    }
    aotx_check(!d.load(f.wire(false, n)).status, "all configured payload bytes are usable");
    aotx_check(d.checkpoint() == f.wire(false, n), "full payload checkpoint keeps distinct bytes");
    tail = {};
    for (unsigned i = 0; i < n; ++i)
        tail.add(aotx_memory_row(i, AOTX_COG_COMPONENT, 8000000 + i, n + i + 1), { (unsigned char)i });
    d.rejects(tail.wire(true, n + 1, 6), true, AOTX_COG_CAPACITY, "full payload refuses a whole new batch");
}

static void aotx_capacity_recall(unsigned n) {
    aotx_fixture f;
    unsigned padding = AOTX_COG_OBJECTS - n * 4;
    for (unsigned i = 0; i < padding; ++i)
        f.add(aotx_memory_row(i + 100, AOTX_COG_COMPONENT, 4000000 + i, i + 1), { (unsigned char)i });
    for (unsigned i = 0; i < n; ++i)
        f.add(aotx_memory_row(i, AOTX_COG_COMPONENT, 900000 + i, f.rows.size() + 1), aotx_memory_vector(2 + i, 3, 1));
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_memory_row(i, AOTX_COG_ASSERTION, 10000 + i * 3, f.rows.size() + 1);
        aotx_id(r.data() + AOTX_CO_EMBEDDING, 900000 + i); aotx_put(r.data() + AOTX_CO_EMBED_VERSION, 1);
        aotx_id(r.data() + AOTX_CO_SOURCE, 900000 + i); aotx_put(r.data() + AOTX_CO_SOURCE_VERSION, 1);
        aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
        f.add(r, aotx_memory_text("private capacity memory " + std::to_string(i)));
    }
    unsigned cut = f.rows.size(); auto image = f.wire(false, cut);
    aotx_recall_device d;
    aotx_check(!d.load(image).status, "late private dependencies are admitted");
    auto q = aotx_memory_queries(n, cut);
    auto begin = std::chrono::steady_clock::now(); auto selected = d.search(q, n);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    aotx_status_rows(selected, 0, "late private memory recall");
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(selected[i].count == 1 && aotx_selected(selected[i], 0) == 10000 + i * 3,
            "each query selects its exact private memory");
        aotx_check(selected[i].index[0] == padding + n + i && selected[i].searches == 1,
            "selection reaches the configured table end through a real search");
        aotx_check(aotx_context(selected[i]).find("private capacity memory " + std::to_string(i)) != std::string::npos,
            "selected late memory reaches the context");
    }
    aotx_check(!d.record(q, n, true).status, "recorded batch fills the remaining object slots");
    auto checkpoint = d.checkpoint();
    aotx_check(aotx_get(checkpoint.data() + 20, 4) == AOTX_COG_OBJECTS, "recording uses the final configured slot");
    {
        aotx_recall_device restored;
        aotx_check(!restored.load(checkpoint).status, "full recorded checkpoint restores");
        auto saved = restored.saved_queries(n); auto replay = restored.search(saved, n, true);
        aotx_status_rows(replay, 0, "recorded late memory replays");
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(!replay[i].searches && replay[i].count == selected[i].count &&
                !memcmp(replay[i].selection, selected[i].selection, sizeof(replay[i].selection)) &&
                aotx_context(replay[i]) == aotx_context(selected[i]), "recovery preserves exact selections without search");
        }
    }
    aotx_check(!d.load(image).status, "original dependency state restores");
    auto row = f.rows[padding + n - 1];
    aotx_put(row.data() + AOTX_CO_VERSION, 2); aotx_put(row.data() + AOTX_CO_UPDATED, cut + 1);
    aotx_put(row.data() + AOTX_CO_FLAGS, AOTX_COG_TOMBSTONE, 4);
    aotx_fixture denial; denial.add(row, {});
    aotx_check(!d.load(denial.wire(true, cut + 1, 6), true).status, "last dependency receives a current tombstone");
    q = aotx_memory_queries(n, cut + 1);
    for (unsigned i = 0; i < n; ++i) aotx_pin(aotx_query_at(q, i), 0, 0, 10000 + i * 3);
    auto denied = d.search(q, n);
    for (unsigned i = 0; i < n; ++i)
        aotx_check(denied[i].status == (i + 1 == n ? AOTX_COG_DENIED : AOTX_COG_OK),
            "a late transitive tombstone denies only its dependent query");
    printf("capacity n=%u objects=%u payload_bytes=%u first_dependency=%u search_seconds=%.6f\n",
        n, AOTX_COG_OBJECTS, AOTX_COG_PAYLOAD, padding, seconds);
}

int main(int argc, char **argv) {
    if (AOTX_COG_OBJECTS < 256 || AOTX_COG_PAYLOAD < 65536) return 4;
    bool dependencies = argc == 2 && !strcmp(argv[1], "--dependencies");
    printf("allocation store=%zu live_state=%zu bindings=%zu search_scratch=%zu live_total=%zu dependency_bytes_per_thread=%u\n",
        sizeof(aotx_cognitive_store), sizeof(aotx_live_state), sizeof(aotx_live_binding) * AOTX_SLOTS,
        sizeof(aotx_recall_scratch) * AOTX_RECALL_BATCH,
        3 * sizeof(aotx_cognitive_store) + sizeof(aotx_live_state) + sizeof(aotx_live_binding) * AOTX_SLOTS +
            sizeof(aotx_recall_scratch) * AOTX_RECALL_BATCH, 2 * AOTX_COG_WORDS * 4);
    for (unsigned n : {1u, 64u}) {
        if (!dependencies) aotx_capacity_bounds(n);
        aotx_capacity_recall(n);
    }
    printf("memory capacity: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
