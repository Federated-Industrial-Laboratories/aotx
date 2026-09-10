/* Purpose: Verify retained roots, scalable reuse, exact retries and file recovery.
 * Owns: Distinct scoped sources and complete-store expected results at both batch sizes.
 * Launch shape: Real live admission, maintenance, checkpoint and import kernels.
 * Lifetime: One bounded process with test-owned CCIR files and journals. */
#include "maintenance_fixture.h"

static void aotx_maint_round(unsigned n) {
    char directory[] = "/tmp/aotx-maintenance-XXXXXX";
    aotx_check(mkdtemp(directory) != nullptr, "maintenance directory opens");
    std::string path = std::string(directory) + "/memory.aotxccir";
    aotx_live_records journal;
    aotx_bytes image;
    std::unique_ptr<aotx_cognitive_store> expected;
    std::vector<aotx_live_binding> bindings;
    uint64_t hash = 0, total_objects = 0, total_bytes = 0;
    const unsigned fill = AOTX_COG_OBJECTS / 2;
    const unsigned payload = AOTX_COG_PAYLOAD / AOTX_COG_OBJECTS;
    {
        aotx_checkpoint_device d(n);
        aotx_checkpoint_disk disk;
        aotx_check(!aotx_checkpoint_disk_open(&disk, d.transport.checkpoint_fd, path.c_str()), "maintenance drain opens");
        aotx_fixture seed;
        for (unsigned i = 0; i < fill; ++i) {
            auto r = aotx_memory_row(i % n, AOTX_COG_EVENT, 3000000 + i, i + 1);
            seed.add(r, aotx_bytes(payload, (unsigned char)(i % 251 + 1)));
        }
        auto corpus = aotx_memory_corpus(n);
        for (auto &r : corpus.rows) {
            aotx_put(r.data() + AOTX_CO_CREATED, aotx_get(r.data() + AOTX_CO_CREATED) + fill);
            aotx_put(r.data() + AOTX_CO_UPDATED, aotx_get(r.data() + AOTX_CO_UPDATED) + fill);
        }
        seed.append(corpus); total_objects = seed.rows.size(); total_bytes = (uint64_t)fill * payload;
        aotx_maint_append(journal, d.live.send(aotx_live_load_bytes(seed.wire(false, seed.rows.size())), AOTX_LIVE_LOAD));
        aotx_maint_append(journal, d.live.send(aotx_live_binding_bytes(n, seed.rows.size()), AOTX_LIVE_BIND));
        aotx_maint_append(journal, d.live.send(aotx_live_query_bytes(n, seed.rows.size(), 1), AOTX_LIVE_QUERY));
        d.live.prompt(n); d.live.idle(n); aotx_maint_flush(d, disk);
        uint64_t large = aotx_maint_size(path);
        auto before = aotx_maint_store(); auto old_bindings = d.live.bindings(n);
        aotx_maint_append(journal, d.live.send(aotx_maint_policy(*before, n, 1), AOTX_LIVE_MAINTAIN));
        auto after = aotx_maint_store(); bindings = d.live.bindings(n);
        aotx_check(!d.live.state().status && after->count == 2 * n && after->bytes < before->bytes,
            "complete dependency roots survive while unneeded slots and payload bytes are reclaimed");
        aotx_check(after->root_sequence == before->sequence && after->retry_floor == before->sequence - n,
            "root and exact retry floor name the accepted sequence");
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(bindings[i].choice.index[0] != old_bindings[i].choice.index[0], "selection indices actually move during compaction");
            old_bindings[i].choice.index[0] = bindings[i].choice.index[0];
            aotx_check(!memcmp(&bindings[i], &old_bindings[i], sizeof(bindings[i])), "each binding preserves exact logical query and context bytes");
        }
        image = aotx_maint_flush(d, disk);
        aotx_check(aotx_maint_size(path) < large / 2, "runtime mirror replacement removes retired file extents");
        aotx_bytes old = seed.wire(true, 1);
        auto stable = aotx_maint_store();
        aotx_maint_append(journal, d.live.send(old, AOTX_LIVE_UPDATE));
        after = aotx_maint_store();
        aotx_check(d.live.state().status == AOTX_COG_STALE && !memcmp(stable.get(), after.get(), sizeof(*after)),
            "a covered retry below the floor refuses with no state change");
        for (unsigned round = 0; round < 3; ++round) {
            before = aotx_maint_store(); aotx_fixture tail;
            for (unsigned i = 0; i < fill; ++i) {
                uint64_t seq = before->sequence + i + 1;
                auto r = aotx_memory_row(i % n, AOTX_COG_EVENT, 3000000 + round * fill + i, seq);
                aotx_put(r.data() + AOTX_CO_VERSION, seq);
                tail.add(r, aotx_bytes(payload, (unsigned char)((i + round) % 251 + 1)));
            }
            auto input = tail.wire(true, before->sequence + 1, before->tick + 1); aotx_maint_schema(input, *before);
            aotx_maint_append(journal, d.live.send(input, AOTX_LIVE_UPDATE));
            aotx_check(!d.live.state().status, "new sequence versions reuse retired IDs without reusing old exact references");
            total_objects += fill; total_bytes += (uint64_t)fill * payload;
            aotx_maint_flush(d, disk); before = aotx_maint_store();
            aotx_maint_append(journal, d.live.send(aotx_maint_policy(*before, n, 1), AOTX_LIVE_MAINTAIN));
            aotx_check(!d.live.state().status, "repeated complete maintenance remains usable");
            image = aotx_maint_flush(d, disk);
            after = aotx_maint_store();
            aotx_check(after->count == 3 * n, "recent exact retries and old bound dependencies survive together");
            aotx_fixture retry;
            for (unsigned i = fill - n; i < fill; ++i) retry.add(tail.rows[i], tail.payloads[i]);
            auto bytes = retry.wire(true, before->sequence - n + 1, before->tick); aotx_maint_schema(bytes, *before);
            aotx_maint_append(journal, d.live.send(bytes, AOTX_LIVE_UPDATE));
            aotx_check(!d.live.state().status, "the exact retained retry succeeds with its original root header");
            auto conflict = bytes; conflict.back() ^= 1;
            aotx_maint_append(journal, d.live.send(conflict, AOTX_LIVE_UPDATE));
            aotx_check(d.live.state().status == AOTX_COG_VERSION, "a conflicting retained retry refuses");
            before = aotx_maint_store(); aotx_maint_flush(d, disk);
            aotx_maint_append(journal, d.live.send(aotx_maint_policy(*before, 0, 1), AOTX_LIVE_MAINTAIN));
            aotx_maint_flush(d, disk);
        }
        aotx_check(total_objects > AOTX_COG_OBJECTS && total_bytes > AOTX_COG_PAYLOAD,
            "cumulative accepted objects and payload bytes exceed the configured resident capacities");
        before = aotx_maint_store();
        aotx_maint_append(journal, d.live.send(aotx_live_query_bytes(n, before->sequence, 2), AOTX_LIVE_QUERY));
        aotx_check(!d.live.state().status, "fresh query uses retained memory after repeated reclamation");
        auto prompts = d.live.prompt(n);
        for (unsigned i = 0; i < n; ++i) aotx_check(prompts[i].find("fact " + std::to_string(i)) != std::string::npos,
            "each distinct scoped memory reaches the model prompt after maintenance");
        d.live.idle(n); image = aotx_maint_flush(d, disk);
        expected = aotx_maint_store(); bindings = d.live.bindings(n); hash = d.live.seam().apply.state_hash;
        printf("maintenance n=%u accepted_objects=%llu accepted_payload=%llu live_objects=%u live_payload=%u file=%llu\n",
            n, (unsigned long long)total_objects, (unsigned long long)total_bytes, expected->count, expected->bytes,
            (unsigned long long)aotx_maint_size(path));
        aotx_checkpoint_disk_close(&disk);
    }
    {
        aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
        d.send(image, AOTX_CP_RESUME); auto actual = aotx_maint_store();
        aotx_check(!d.state().status && !memcmp(expected.get(), actual.get(), sizeof(*actual)), "file resume restores the exact compacted store and policy");
        auto b = d.bindings(n);
        for (unsigned i = 0; i < n; ++i) aotx_check(!memcmp(&b[i], &bindings[i], sizeof(b[i])), "file resume preserves each remapped binding exactly");
    }
    {
        aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
        aotx_maint_batch = n; d.process(journal, true, true, aotx_maint_replay_idle);
        auto actual = aotx_maint_store();
        aotx_check(!d.state().fatal && !memcmp(expected.get(), actual.get(), sizeof(*actual)) && d.seam().apply.state_hash == hash,
            "journal replay preserves the entire maintained store and authoritative hash");
    }
    unlink(path.c_str()); rmdir(directory);
}
int main() {
    const unsigned batch = AOTX_SLOTS < AOTX_RECALL_BATCH ? AOTX_SLOTS : AOTX_RECALL_BATCH;
    for (unsigned n : {1u, batch}) aotx_maint_round(n);
    printf("maintenance: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
