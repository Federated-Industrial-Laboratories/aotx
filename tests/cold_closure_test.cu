/* Purpose: Check complete dependency retrieval and interrupted result recovery.
 * Owns: Distinct private source pairs and exact recorded-result comparisons.
 * Launch shape: Real GPU and disk paths at N=1 and N=64.
 * Lifetime: One bounded process with isolated temporary files. */
#include "cold_fixture.h"

static void cold_closure(unsigned n, unsigned scope) {
    char directory[] = "/tmp/aotx-cold-closure-XXXXXX";
    aotx_check(mkdtemp(directory) != nullptr, "closure directory opens");
    std::string path = std::string(directory) + "/memory.aotxccir";
    aotx_fixture corpus;
    std::vector<unsigned> sources, leaves;
    for (unsigned i = 0; i < n; ++i) {
        corpus.add(aotx_memory_row(i, AOTX_COG_EVENT, 700000 + i, i + 1, scope), aotx_memory_text("source " + std::to_string(i)));
        sources.push_back(i); leaves.push_back(n + i);
    }
    for (unsigned i = 0; i < n; ++i) {
        auto row = aotx_memory_row(i, AOTX_COG_ASSERTION, 800000 + i, n + i + 1, scope);
        aotx_id(row.data() + AOTX_CO_SOURCE, 700000 + i); aotx_put(row.data() + AOTX_CO_SOURCE_VERSION, 1);
        corpus.add(row, aotx_memory_text("assertion " + std::to_string(i)));
    }
    aotx_live_records journal, partial, result;
    std::unique_ptr<aotx_cognitive_store> expected, off;
    {
        aotx_checkpoint_device d(n); aotx_checkpoint_disk disk;
        aotx_check(!aotx_checkpoint_disk_open(&disk, d.transport.checkpoint_fd, path.c_str()), "closure mirror opens");
        cold_disk = &disk;
        aotx_maint_append(journal, d.live.send(aotx_live_load_bytes(corpus.wire(false, 2 * n)), AOTX_LIVE_LOAD));
        aotx_check(!d.live.state().status, "private source pairs load");
        aotx_maint_flush(d, disk); auto original = aotx_maint_store();
        aotx_maint_append(journal, cold_send(d.live, cold_control(*original, AOTX_COLD_ENABLE)));
        aotx_maint_flush(d, disk); expected = aotx_maint_store();
        aotx_maint_append(journal, cold_send(d.live, cold_control(*expected, AOTX_COLD_OFFLOAD, sources)));
        aotx_check(d.live.state().status == AOTX_COG_REFERENCE, "resident dependent blocks source offload");
        cold_unchanged(*expected, "incomplete offload closure changes no object");
        aotx_maint_append(journal, cold_send(d.live, cold_control(*expected, AOTX_COLD_OFFLOAD, leaves)));
        aotx_check(!d.live.state().status, "leaf payload batch offloads");
        aotx_maint_flush(d, disk); auto state = aotx_maint_store();
        aotx_maint_append(journal, cold_send(d.live, cold_control(*state, AOTX_COLD_OFFLOAD, sources)));
        aotx_check(!d.live.state().status, "sources offload when every dependent is cold");
        aotx_maint_flush(d, disk); off = aotx_maint_store();
        if (scope == AOTX_COG_ROOM) {
            auto wrong = cold_control(*off, AOTX_COLD_FETCH, leaves);
            wrong[AOTX_COLD_HEADER + (n - 1) * AOTX_COLD_ROW + 40] ^= 1;
            aotx_maint_append(journal, cold_send(d.live, wrong));
            aotx_check(d.live.state().status == AOTX_COG_DENIED, "one wrong room refuses the complete retrieval batch");
            cold_unchanged(*off, "room denial preserves all cold payloads");
        }
        aotx_ccir_input inputs[AOTX_CCIR_SECTIONS] = {}; uint32_t count = 0;
        for (uint32_t i = 0; i < disk.view.count; ++i) if (disk.view.sections[i].type != AOTX_CCIR_COLD) {
            inputs[count].section = disk.view.sections[i]; inputs[count++].source = AOTX_CCIR_REUSE;
        }
        uint64_t generation = disk.view.generation;
        aotx_check(aotx_ccir_writer_append(&disk.view, inputs, count, &disk.view.meta, nullptr) != 0 &&
            disk.view.generation == generation, "cold checkpoint cannot publish without its required extent section");
        result = cold_send(d.live, cold_control(*off, AOTX_COLD_FETCH, leaves));
        aotx_check(!d.live.state().status, "one requested batch retrieves all transitive cold sources");
        cold_unchanged(*expected, "dependency retrieval restores every private row and payload exactly");
        for (const auto &r : result) {
            partial.push_back(r);
            if (((const aotx_record_header *)r.data())->type == AOTX_LIVE_RECORD &&
                aotx_get(r.data() + 68, 4) == AOTX_COLD_RESULT) break;
        }
        aotx_check(partial.size() < result.size(), "interruption cuts a nonempty result before completion");
        aotx_checkpoint_disk_close(&disk); cold_disk = nullptr;
    }
    aotx_live_records interrupted;
    {
        aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
        d.process(journal, true); d.process(partial, true);
        cold_unchanged(*off, "partial replay cannot install part of a dependency closure");
        unsigned *out; AOTX_CUDA(cudaMalloc(&out, 4));
        aotx_live_test_end<<<1,1>>>(out); AOTX_CUDA(cudaDeviceSynchronize()); cudaFree(out);
        interrupted = d.process({});
        aotx_check(!d.state().fatal && d.state().status == AOTX_COG_UNAVAILABLE,
            "interrupted recovery records an unavailable result");
        cold_unchanged(*off, "interrupted retrieval keeps the original cold store");
    }
    {
        aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
        d.process(journal, true); d.process(partial, true); d.process(interrupted, true);
        aotx_check(!d.state().fatal && d.state().phase == AOTX_LIVE_IDLE, "interruption marker survives repeated recovery");
        cold_unchanged(*off, "repeated interrupted recovery is exact");
    }
    {
        aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
        d.process(journal, true); d.process(result, true);
        aotx_check(!d.state().fatal, "complete recorded dependency result replays without a file");
        cold_unchanged(*expected, "complete result restores exact resident bytes without external reads");
    }
    unlink(path.c_str()); rmdir(directory);
}
int main() {
    for (unsigned n : {1u, 64u}) for (unsigned scope : {0u, 1u}) cold_closure(n, scope);
    printf("cold closure: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
