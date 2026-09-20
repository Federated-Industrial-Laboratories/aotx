/* Purpose: Verify cold-read failures, cancellation, scope and current corrections.
 * Owns: Distinct scoped text, injected file faults and byte-exact state checks.
 * Launch shape: Real live and disk consumers at N=1 and N=64.
 * Lifetime: One bounded process; each case restores or removes its test files. */
#include "cold_fixture.h"
#include <fcntl.h>
#include "cli/cli.cuh"
extern "C" {
#include "disk/ccir/internal.h"
}

__global__ void aotx_cold_test_cancel(void) { if (!threadIdx.x) aotx_cold_cancel(); }
__global__ void aotx_cold_test_status(void) {
    if (!threadIdx.x) aotx_cli_line((const unsigned char *)"memory", 6, aotx_time_tick);
}
static aotx_cold_transport *cold_ring(aotx_checkpoint_device &d) {
    return (aotx_cold_transport *)(d.transport.checkpoint_map + AOTX_CP_COLD_OFFSET);
}
static void cold_guards(unsigned n) {
    char directory[] = "/tmp/aotx-cold-guards-XXXXXX";
    aotx_check(mkdtemp(directory) != nullptr, "guard directory opens");
    std::string path = std::string(directory) + "/memory.aotxccir";
    aotx_checkpoint_device d(n); aotx_checkpoint_disk disk;
    aotx_check(!aotx_checkpoint_disk_open(&disk, d.transport.checkpoint_fd, path.c_str()), "guard mirror opens");
    auto corpus = aotx_memory_corpus(n);
    d.live.send(aotx_live_load_bytes(corpus.wire(false, corpus.rows.size())), AOTX_LIVE_LOAD);
    d.live.send(aotx_live_binding_bytes(n, corpus.rows.size()), AOTX_LIVE_BIND);
    aotx_maint_flush(d, disk); cold_disk = &disk;
    auto resident = aotx_maint_store(); cold_send(d.live, cold_control(*resident, AOTX_COLD_ENABLE));
    aotx_maint_flush(d, disk); resident = aotx_maint_store();
    std::vector<unsigned> rows;
    for (unsigned i = 0; i < n; ++i) rows.push_back(n + i);
    cold_send(d.live, cold_control(*resident, AOTX_COLD_OFFLOAD, rows));
    aotx_check(!d.live.state().status, "unselected text can become cold");
    aotx_maint_flush(d, disk); auto off = aotx_maint_store();
    d.live.send(aotx_live_query_bytes(n, off->sequence, 1), AOTX_LIVE_QUERY);
    aotx_check(d.live.state().status == AOTX_COG_UNAVAILABLE, "recall reports unavailable instead of omitting cold evidence");
    cold_unchanged(*off, "unavailable recall does not change memory");

    auto wrong = cold_control(*off, AOTX_COLD_FETCH, rows);
    wrong[AOTX_COLD_HEADER + (n - 1) * AOTX_COLD_ROW + 24] ^= 1;
    uint64_t request = cold_ring(d)->request;
    cold_send(d.live, wrong);
    aotx_check(d.live.state().status == AOTX_COG_DENIED && cold_ring(d)->request == request,
        "wrong scope refuses the complete batch before any file read");
    cold_unchanged(*off, "scope denial cannot install private payloads");

    cold_disk = nullptr;
    cold_send(d.live, cold_control(*off, AOTX_COLD_FETCH, rows), false);
    aotx_check(d.live.state().phase == AOTX_COLD_WAIT, "read waits asynchronously with live state unchanged");
    cold_unchanged(*off, "a pending read publishes no partial payload");
    uint64_t output = d.live.seam().dev.tail;
    aotx_cold_test_status<<<1,1>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    aotx_check(d.live.seam().dev.tail > output && d.live.state().phase == AOTX_COLD_WAIT,
        "the real console command completes while cold IO makes no progress");
    aotx_cold_test_cancel<<<1,1>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    d.live.process({}, false, true, cold_hook);
    aotx_check(d.live.state().status == AOTX_COG_UNAVAILABLE && d.live.state().phase == AOTX_LIVE_IDLE,
        "cancellation releases the memory operation without waiting for the disk");
    cold_unchanged(*off, "cancelled read preserves all retained state");
    cold_disk = &disk;
    for (unsigned pass = 0; pass < 2000 && cold_ring(d)->response != cold_ring(d)->request; ++pass) {
        aotx_cold_disk_pass(&disk); std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    aotx_check(cold_ring(d)->response == cold_ring(d)->request, "late response completes its original transport slot");
    d.live.process({}, false, true, cold_hook);
    cold_unchanged(*off, "a late response cannot publish after cancellation");

    cold_disk = nullptr;
    cold_send(d.live, cold_control(*off, AOTX_COLD_FETCH, rows), false);
    ++cold_ring(d)->generation;
    cold_disk = &disk; d.live.process({}, false, true, cold_hook);
    aotx_check(d.live.state().status == AOTX_COG_UNAVAILABLE, "a different file generation cannot satisfy a read");
    cold_unchanged(*off, "generation mismatch preserves the complete store");

    aotx_cold_catalog catalog;
    aotx_check(!aotx_cold_catalog_open(&disk.view, &catalog) && catalog.count == n, "fault injection opens the selected cold catalog");
    unsigned char byte = 0;
    uint64_t at = catalog.payload;
    aotx_check(pread(disk.view.fd, &byte, 1, at) == 1, "cold source byte reads");
    unsigned char changed = byte ^ 1;
    aotx_check(pwrite(disk.view.fd, &changed, 1, at) == 1, "cold payload fault is installed");
    cold_send(d.live, cold_control(*off, AOTX_COLD_FETCH, rows));
    aotx_check(d.live.state().status == AOTX_COG_UNAVAILABLE, "payload digest failure refuses the entire result");
    cold_unchanged(*off, "damaged file bytes cannot enter live memory");
    aotx_check(pwrite(disk.view.fd, &byte, 1, at) == 1, "cold source byte is restored");
    const aotx_ccir_section *section = nullptr;
    for (unsigned i = 0; i < disk.view.count; ++i)
        if (disk.view.sections[i].type == AOTX_CCIR_COLD) section = disk.view.sections + i;
    aotx_check(section != nullptr, "committed cold section exists");
    unsigned char digest[32];
    aotx_check(!aotx_ccir_hash_fd(disk.view.fd, section->offset, section->bytes, digest) &&
        !memcmp(digest, section->digest, 32), "original section has its committed digest");
    at = catalog.payload + 32;
    aotx_check(pread(disk.view.fd, &byte, 1, at) == 1, "original text byte reads");
    changed = 'X';
    aotx_check(pwrite(disk.view.fd, &changed, 1, at) == 1, "changed text keeps a valid payload schema");
    aotx_check(!aotx_ccir_hash_fd(disk.view.fd, catalog.payload,
        aotx_get(catalog.rows + AOTX_CO_BYTES), digest), "changed payload checksum is computed");
    uint64_t checksum = section->offset + AOTX_COLD_EXTENT_HEADER + AOTX_COG_OBJECT;
    aotx_check(pwrite(disk.view.fd, digest, 32, checksum) == 32, "catalog takes the changed payload checksum");
    aotx_check(!aotx_ccir_hash_fd(disk.view.fd, section->offset, section->bytes, digest) &&
        memcmp(digest, section->digest, 32), "changed catalog has no committed section digest");
    cold_send(d.live, cold_control(*off, AOTX_COLD_FETCH, rows));
    aotx_check(d.live.state().status == AOTX_COG_UNAVAILABLE, "an uncommitted catalog cannot authenticate changed text");
    cold_unchanged(*off, "catalog integrity failure preserves the complete cold store");
    aotx_check(pwrite(disk.view.fd, &byte, 1, at) == 1 &&
        pwrite(disk.view.fd, catalog.rows + AOTX_COG_OBJECT, 32, checksum) == 32,
        "original text and catalog checksum are restored");
    aotx_cold_catalog_close(&catalog);

    aotx_fixture corrections;
    for (unsigned i = 0; i < n; ++i) {
        aotx_row row; memcpy(row.data(), off->objects[n + i], row.size());
        aotx_put(row.data() + AOTX_CO_FLAGS, 0, 4);
        aotx_put(row.data() + AOTX_CO_VERSION, 2);
        aotx_put(row.data() + AOTX_CO_UPDATED, off->sequence + i + 1);
        corrections.add(row, aotx_memory_text("corrected fact " + std::to_string(i)));
    }
    d.live.send(corrections.wire(true, off->sequence + 1, off->tick + 1), AOTX_LIVE_UPDATE);
    aotx_check(!d.live.state().status, "a new current revision can replace a cold historical assertion");
    aotx_maint_flush(d, disk); auto corrected = aotx_maint_store();
    cold_send(d.live, cold_control(*corrected, AOTX_COLD_FETCH, rows));
    aotx_check(d.live.state().status == AOTX_COG_STALE, "historical cold bytes cannot replace a current correction");
    cold_unchanged(*corrected, "stale fetch leaves the correction intact");
    auto query = aotx_live_query_bytes(n, corrected->sequence, 1);
    for (unsigned i = 0; i < n; ++i) aotx_pin(query.data() + 64 + i * AOTX_LIVE_QUERY_ROW + 64, 0, 0, 10000 + i * 3, 2);
    d.live.send(query, AOTX_LIVE_QUERY);
    aotx_check(!d.live.state().status, "current corrected evidence remains usable with old versions cold");
    auto prompts = d.live.prompt(n); d.live.idle(n);
    for (unsigned i = 0; i < n; ++i)
        aotx_check(prompts[i].find("corrected fact " + std::to_string(i)) != std::string::npos,
            "each conversation receives its own current correction");
    aotx_maint_flush(d, disk); corrected = aotx_maint_store();
    std::vector<unsigned> current;
    for (unsigned i = 0; i < n; ++i) current.push_back(2 * n + i);
    cold_send(d.live, cold_control(*corrected, AOTX_COLD_OFFLOAD, current));
    aotx_check(d.live.state().status == AOTX_COG_REFERENCE, "active conversation selections remain resident");
    cold_unchanged(*corrected, "active reference protection is atomic");
    aotx_checkpoint_disk_close(&disk); cold_disk = nullptr;
    unlink(path.c_str()); rmdir(directory);
}
int main() {
    for (unsigned n : {1u, 64u}) cold_guards(n);
    printf("cold guards: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
