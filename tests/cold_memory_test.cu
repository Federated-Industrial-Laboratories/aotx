/* Purpose: Verify explicit offload, bounded retrieval and complete file recovery.
 * Owns: Distinct private payloads, real disk transport and exact expected state.
 * Launch shape: The live graph paths with N=1 and N=64 request batches.
 * Lifetime: One bounded process with isolated temporary files. */
#include "cold_fixture.h"
extern "C" void aotx_checkpoint_fault(int write_after, int sync_after);

static aotx_bytes cold_flush(aotx_checkpoint_device &d, aotx_checkpoint_disk &disk, unsigned line) {
    unsigned before = aotx_failures;
    auto image = aotx_maint_flush(d, disk);
    if (aotx_failures != before) fprintf(stderr, "cold checkpoint line %u: revision %llu generation %llu error %llu\n",
        line, (unsigned long long)d.live.state().accepted, (unsigned long long)disk.view.generation,
        (unsigned long long)d.ring()->error);
    return image;
}
#define aotx_maint_flush(d, disk) cold_flush(d, disk, __LINE__)

static aotx_bytes cold_fault_flush(aotx_checkpoint_device &d, aotx_checkpoint_disk &disk) {
    uint64_t serial = d.ring()->head + 1, before = disk.view.generation;
    d.publish(serial); auto image = d.image(serial);
    for (unsigned fault = 0; fault < 2; ++fault) {
        aotx_checkpoint_fault(fault ? -1 : 0, fault ? 0 : -1);
        disk.next_retry = 0;
        aotx_check(!aotx_checkpoint_disk_pass(&disk) && d.ring()->consumed < serial && d.ring()->error,
            "failed cold extent write or sync releases no snapshot");
        aotx_checkpoint_fault(-1, -1);
        aotx_ccir_view reader;
        int status = aotx_ccir_open(disk.path, nullptr, &reader);
        aotx_check(!status && reader.generation == before,
            "interrupted cold extent publication retains the complete prior generation");
        if (!status) aotx_ccir_close(&reader);
    }
    disk.next_retry = 0;
    aotx_check(aotx_checkpoint_disk_pass(&disk) == 1, "cold extent retry publishes one complete generation");
    d.step();
    aotx_check(!d.state().error && d.state().durable == d.live.state().accepted,
        "only the complete cold generation becomes durable");
    return image;
}

static void cold_round(unsigned n) {
    char directory[] = "/tmp/aotx-cold-XXXXXX";
    aotx_check(mkdtemp(directory) != nullptr, "cold test directory opens");
    std::string path = std::string(directory) + "/memory.aotxccir";
    aotx_fixture corpus;
    std::vector<unsigned> rows;
    for (unsigned i = 0; i < n; ++i) {
        corpus.add(aotx_memory_row(i, AOTX_COG_MEDIA, 500000 + i, i + 1),
            cold_media(i, AOTX_COG_PAYLOAD / (2 * n) - AOTX_COG_MEDIA_HEADER));
        rows.push_back(i);
    }
    aotx_live_records journal;
    aotx_bytes saved;
    std::unique_ptr<aotx_cognitive_store> original;
    {
        aotx_checkpoint_device d(n);
        aotx_checkpoint_disk disk;
        aotx_check(!aotx_checkpoint_disk_open(&disk, d.transport.checkpoint_fd, path.c_str()), "cold mirror opens");
        cold_disk = &disk;
        aotx_maint_append(journal, d.live.send(aotx_live_load_bytes(corpus.wire(false, n)), AOTX_LIVE_LOAD));
        original = aotx_maint_store(); aotx_maint_flush(d, disk);
        aotx_maint_append(journal, cold_send(d.live, cold_control(*original, AOTX_COLD_OFFLOAD, rows)));
        aotx_check(d.live.state().status == AOTX_COG_DENIED, "GPU-only mode does not offload implicitly");
        cold_unchanged(*original, "disabled offload leaves all memory unchanged");
        aotx_maint_append(journal, cold_send(d.live, cold_control(*original, AOTX_COLD_ENABLE)));
        aotx_check(!d.live.state().status, "explicit tiered mode is accepted");
        aotx_maint_flush(d, disk);
        auto resident = aotx_maint_store();
        auto bad = cold_control(*resident, AOTX_COLD_OFFLOAD, rows);
        bad[AOTX_COLD_HEADER + (n - 1) * AOTX_COLD_ROW + 24] ^= 1;
        aotx_maint_append(journal, cold_send(d.live, bad));
        aotx_check(d.live.state().status == AOTX_COG_DENIED, "one wrong principal refuses the entire batch");
        cold_unchanged(*resident, "scope refusal has no partial offload");
        aotx_maint_append(journal, cold_send(d.live, cold_control(*resident, AOTX_COLD_OFFLOAD, rows)));
        auto off = aotx_maint_store();
        aotx_check(!d.live.state().status && off->tiered && !off->bytes && off->count == n,
            "explicit offload releases resident payload capacity and retains every object");
        for (unsigned i = 0; i < n; ++i)
            aotx_check((aotx_get(off->objects[i] + AOTX_CO_FLAGS, 4) & AOTX_COG_COLD) &&
                !aotx_get(off->objects[i] + AOTX_CO_OFFSET), "each cold object has no resident address");
        saved = cold_fault_flush(d, disk);
        aotx_cold_catalog catalog;
        aotx_check(!aotx_cold_catalog_open(&disk.view, &catalog) && catalog.count == n,
            "the selected CCIR generation contains every cold payload");
        aotx_cold_catalog_close(&catalog);
        aotx_maint_append(journal, cold_send(d.live, cold_control(*off, AOTX_COLD_FETCH, rows)));
        auto fetched = aotx_maint_store();
        aotx_check(!d.live.state().status && fetched->tiered && fetched->bytes == resident->bytes,
            "asynchronous fetch restores the complete requested batch");
        aotx_check(!memcmp(fetched->payload, resident->payload, resident->bytes) &&
            !memcmp(fetched->objects, resident->objects, sizeof(resident->objects)),
            "all distinct payloads and object metadata survive the round trip exactly");
        aotx_maint_flush(d, disk);
        aotx_maint_append(journal, cold_send(d.live, cold_control(*fetched, AOTX_COLD_OFFLOAD, rows)));
        saved = aotx_maint_flush(d, disk); off = aotx_maint_store();
        aotx_maint_append(journal, cold_send(d.live, cold_control(*off, AOTX_COLD_GPU)));
        auto gpu = aotx_maint_store();
        aotx_check(!d.live.state().status && !gpu->tiered && gpu->bytes == original->bytes,
            "GPU-only mode fetches all cold data before releasing tiered mode");
        aotx_maint_flush(d, disk);
        aotx_maint_append(journal, cold_send(d.live, cold_control(*gpu, AOTX_COLD_ENABLE)));
        aotx_maint_flush(d, disk); resident = aotx_maint_store();
        aotx_maint_append(journal, cold_send(d.live, cold_control(*resident, AOTX_COLD_OFFLOAD, rows)));
        saved = aotx_maint_flush(d, disk);
        aotx_checkpoint_disk_close(&disk); cold_disk = nullptr;
    }
    {
        aotx_checkpoint_device d(n);
        unsigned char *image = nullptr; uint32_t length = 0;
        aotx_check(!aotx_checkpoint_file_read(path.c_str(), &image, &length), "file-only cold checkpoint reads");
        aotx_check(length == saved.size() && !memcmp(image, saved.data(), length), "file-only bytes match the saved cold cut");
        d.live.send(aotx_bytes(image, image + length), AOTX_CP_RESUME); free(image);
        aotx_check(!d.live.state().status, "cold metadata restores without reading payloads into GPU memory");
        aotx_checkpoint_disk disk;
        aotx_check(!aotx_checkpoint_disk_open(&disk, d.transport.checkpoint_fd, path.c_str()), "recovered cold mirror opens");
        cold_disk = &disk; aotx_maint_flush(d, disk);
        auto off = aotx_maint_store();
        aotx_fixture extra;
        for (unsigned i = 0; i < n; ++i)
            extra.add(aotx_memory_row(i, AOTX_COG_MEDIA, 600000 + i, n + i + 1),
                cold_media(i + 64, (AOTX_COG_PAYLOAD / n) * 3 / 4 - AOTX_COG_MEDIA_HEADER));
        d.live.send(extra.wire(true, n + 1, off->tick + 1), AOTX_LIVE_UPDATE);
        aotx_check(!d.live.state().status, "new independent memory uses capacity freed by offload");
        aotx_maint_flush(d, disk); auto large = aotx_maint_store();
        cold_send(d.live, cold_control(*large, AOTX_COLD_GPU));
        aotx_check(d.live.state().status == AOTX_COG_CAPACITY, "GPU-only transition checks the complete retained size");
        cold_unchanged(*large, "failed fit preserves tiered mode and all resident and cold memory");
        auto maintenance = aotx_maint_policy(*large, n, n);
        d.live.send(maintenance, AOTX_LIVE_MAINTAIN);
        aotx_check(!d.live.state().status, "configured maintenance can retire old cold objects");
        aotx_maint_flush(d, disk); auto reduced = aotx_maint_store();
        aotx_check(reduced->count == n && reduced->bytes == large->bytes, "retention removes only the old cold batch");
        cold_send(d.live, cold_control(*reduced, AOTX_COLD_GPU));
        aotx_check(!d.live.state().status && !aotx_maint_store()->tiered, "GPU-only mode succeeds after an explicit retention change");
        aotx_maint_flush(d, disk); aotx_checkpoint_disk_close(&disk); cold_disk = nullptr;
    }
    {
        aotx_live_device replay(n);
        AOTX_LIVE_CLEAR(aotx_checkpoint);
        replay.process(journal, true);
        auto restored = aotx_maint_store();
        aotx_check(!replay.state().fatal && !restored->bytes && restored->tiered && restored->count == n,
            "exact replay restores residency without a disk read transport");
    }
    unlink(path.c_str()); rmdir(directory);
}
int main() {
    for (unsigned n : {1u, 64u}) cold_round(n);
    printf("cold memory: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
