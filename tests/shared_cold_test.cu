/* Purpose: Check typed corrections, shared cold reads and publication after file recovery.
 * Owns: Distinct source graphs, copied files and exact expected quote bytes.
 * Launch shape: Real device and disk paths at N=1 and N=64.
 * Lifetime: One bounded process with isolated files and no model weights. */
#include "shared_cold_fixture.h"
#include <fstream>

static void shared_cold(unsigned n) {
    char directory[] = "/tmp/aotx-shared-cold-XXXXXX";
    aotx_check(mkdtemp(directory) != nullptr, "shared cold directory opens");
    std::string original = std::string(directory) + "/original.aotxccir";
    std::string copied = std::string(directory) + "/copied.aotxccir";
    aotx_bytes saved; std::vector<unsigned> rows[4];
    for (unsigned i = 0; i < n; ++i) for (unsigned k = 0; k < 4; ++k) rows[k].push_back(i * 4 + k);
    {
        aotx_checkpoint_device d(n); aotx_checkpoint_disk disk;
        aotx_check(!aotx_checkpoint_disk_open(&disk, d.transport.checkpoint_fd, original.c_str()), "source mirror opens");
        cold_disk = &disk; auto corpus = cold_shared_corpus(n);
        d.live.send(aotx_live_load_bytes(corpus.wire(false, n * 4)), AOTX_LIVE_LOAD);
        aotx_check(!d.live.state().status, "typed source graphs load through validated admission");
        aotx_maint_flush(d, disk); cold_shared_readers reader(n);
        reader.read(false); reader.publish(200);
        auto s = aotx_maint_store(); cold_send(d.live, cold_control(*s, AOTX_COLD_ENABLE));
        aotx_maint_flush(d, disk);
        for (unsigned k : {3u, 1u, 0u, 2u}) {
            s = aotx_maint_store(); cold_send(d.live, cold_control(*s, AOTX_COLD_OFFLOAD, rows[k]));
            unsigned expected = k == 2 ? 0 : k == 0 ? AOTX_COG_REFERENCE : AOTX_COG_DENIED;
            aotx_check(d.live.state().status == expected, "only permitted leaves offload while working sources remain resident");
            if (d.live.state().status != expected) {
                fprintf(stderr, "offload N=%u kind=%u status=%u expected=%u\n", n, k, d.live.state().status, expected);
                exit(1);
            }
            if (expected) {
                cold_unchanged(*s, "protected source graphs stay unchanged after refused offload");
                continue;
            }
            saved = aotx_maint_flush(d, disk);
        }
        auto off = aotx_maint_store(); reader.read(true); reader.publish(200);
        cold_unchanged(*off, "cold read and publication refusals change no store bytes");
        aotx_checkpoint_disk_close(&disk); cold_disk = nullptr;
    }
    { std::ifstream in(original, std::ios::binary); std::ofstream out(copied, std::ios::binary); out << in.rdbuf(); }
    aotx_check(!unlink(original.c_str()), "the original file is removed before recovery");
    {
        aotx_checkpoint_device d(n); unsigned char *image = nullptr; uint32_t length = 0;
        aotx_check(!aotx_checkpoint_file_read(copied.c_str(), &image, &length), "copied cold checkpoint reads");
        aotx_check(length == saved.size() && !memcmp(image, saved.data(), length), "copied checkpoint bytes are exact");
        d.live.send(aotx_bytes(image, image + length), AOTX_CP_RESUME); free(image);
        aotx_check(!d.live.state().status, "typed cold metadata restores without source files");
        aotx_checkpoint_disk disk;
        aotx_check(!aotx_checkpoint_disk_open(&disk, d.transport.checkpoint_fd, copied.c_str()), "copied mirror opens");
        cold_disk = &disk; aotx_maint_flush(d, disk); cold_shared_readers reader(n);
        reader.read(true); reader.publish(200);
        for (unsigned k : {2u}) {
            auto s = aotx_maint_store(); cold_send(d.live, cold_control(*s, AOTX_COLD_FETCH, rows[k]));
            aotx_check(!d.live.state().status, "requested leaves retrieve their exact source closure");
            aotx_maint_flush(d, disk);
        }
        reader.read(false); reader.publish(200);
        auto s = aotx_maint_store(); cold_send(d.live, cold_control(*s, AOTX_COLD_OFFLOAD, rows[2]));
        aotx_maint_flush(d, disk); s = aotx_maint_store();
        aotx_fixture correction;
        for (unsigned i = 0; i < n; ++i) {
            aotx_row row; memcpy(row.data(), s->objects[rows[2][i]], row.size());
            aotx_put(row.data() + AOTX_CO_FLAGS, 0, 4); aotx_put(row.data() + AOTX_CO_VERSION, 2);
            aotx_put(row.data() + AOTX_CO_UPDATED, s->sequence + i + 1);
            correction.add(row, cold_interpretation(i, true));
        }
        d.live.send(correction.wire(true, s->sequence + 1, s->tick + 1), AOTX_LIVE_UPDATE);
        aotx_check(!d.live.state().status, "a current typed quote replaces its cold historical version");
        aotx_maint_flush(d, disk); s = aotx_maint_store(); reader.read(false, true);
        cold_send(d.live, cold_control(*s, AOTX_COLD_FETCH, rows[2]));
        aotx_check(d.live.state().status == AOTX_COG_STALE, "a stale cold quote cannot overwrite the current correction");
        cold_unchanged(*s, "stale retrieval preserves the complete corrected store");
        reader.publish(200, true); auto published = aotx_maint_store();
        aotx_check(published->count == n * 8, "publication appends exactly one complete graph per owner");
        for (unsigned i = 0; i < n; ++i) {
            auto row = published->objects[n * 5 + i * 3];
            auto text = aotx_memory_text(cold_quote(i, false) + " " + cold_quote(i, true));
            aotx_check(aotx_get(row + AOTX_CO_OWNER) == 7000 + i &&
                !memcmp(published->payload + aotx_get(row + AOTX_CO_OFFSET), text.data(), text.size()),
                "publication retains each exact source in its explicit destination");
        }
        aotx_maint_flush(d, disk); aotx_checkpoint_disk_close(&disk); cold_disk = nullptr;
    }
    unlink(copied.c_str()); rmdir(directory);
}
int main(void) {
    shared_cold(1); shared_cold(64);
    printf("shared cold: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
