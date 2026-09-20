/* Purpose: Drive explicit residency batches with real GPU and disk consumers.
 * Owns: Test request images, temporary payloads and the disk progress hook.
 * Launch shape: Distinct batches at one and all profile slots.
 * Lifetime: One isolated cold-memory test process. */
#ifndef AOTX_COLD_FIXTURE_H
#define AOTX_COLD_FIXTURE_H
#include "maintenance_fixture.h"
#include "disk/cognitive/cold_io.h"
#include <thread>
#include <chrono>

static aotx_checkpoint_disk *cold_disk;
static void cold_hook(bool replay) {
    if (!replay && cold_disk) { aotx_cold_disk_pass(cold_disk); std::this_thread::yield(); }
}
static aotx_bytes cold_control(const aotx_cognitive_store &s, unsigned mode,
    const std::vector<unsigned> &rows = {}) {
    aotx_bytes p(AOTX_COLD_HEADER + rows.size() * AOTX_COLD_ROW);
    memcpy(p.data(), "AOTXTIR1", 8); aotx_put(p.data() + 8, 1, 4);
    aotx_put(p.data() + 12, mode, 4); aotx_put(p.data() + 16, rows.size(), 4);
    aotx_put(p.data() + 20, AOTX_COLD_ROW, 4); memcpy(p.data() + 24, s.lineage, 16);
    aotx_put(p.data() + 40, s.sequence);
    for (unsigned i = 0; i < rows.size(); ++i) {
        const auto *r = s.objects[rows[i]]; auto *q = p.data() + AOTX_COLD_HEADER + i * AOTX_COLD_ROW;
        memcpy(q, r + AOTX_CO_ID, 16); memcpy(q + 16, r + AOTX_CO_VERSION, 8);
        memcpy(q + 24, r + AOTX_CO_OWNER, 16); memcpy(q + 40, r + AOTX_CO_ROOM, 16);
    }
    return p;
}
static aotx_live_records cold_send(aotx_live_device &d, const aotx_bytes &p, bool finish = true) {
    return d.process(aotx_live_parts(p, AOTX_COLD_CONTROL, d.next_id++), false, finish, cold_hook);
}
static aotx_bytes cold_media(unsigned i, unsigned bytes) {
    bytes &= ~1u;
    aotx_bytes p(AOTX_COG_MEDIA_HEADER + bytes, (unsigned char)(i % 251 + 1));
    memset(p.data(), 0, AOTX_COG_MEDIA_HEADER);
    aotx_put(p.data(), 1, 4); aotx_put(p.data() + 4, AOTX_COG_AUDIO_MEDIA, 4);
    aotx_put(p.data() + 8, AOTX_COG_SOURCE_BYTES, 4); aotx_put(p.data() + 12, AOTX_COG_I16, 4);
    aotx_put(p.data() + 16, bytes / 2); aotx_put(p.data() + 24, 1);
    aotx_put(p.data() + 48, bytes); aotx_put(p.data() + 64, AOTX_COG_TEMPORAL, 4);
    aotx_put(p.data() + 68, 2, 4); memset(p.data() + 72, i % 251 + 1, 32);
    aotx_put(p.data() + 168, 16000, 4); return p;
}
static void cold_unchanged(const aotx_cognitive_store &before, const char *message) {
    auto after = aotx_maint_store(); aotx_check(!memcmp(&before, after.get(), sizeof(before)), message);
}
#endif
