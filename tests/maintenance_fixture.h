/* Purpose: Build lifecycle policies, versioned tails and durable GPU test state.
 * Owns: Test-only host buffers and exact expected byte comparisons.
 * Launch shape: The normal live and checkpoint nodes at N=1 and N=64.
 * Lifetime: One maintained memory test. */
#ifndef AOTX_MAINTENANCE_FIXTURE_H
#define AOTX_MAINTENANCE_FIXTURE_H
#include "checkpoint_fixture.h"
#include <memory>
#include <sys/stat.h>

static std::unique_ptr<aotx_cognitive_store> aotx_maint_store(void) {
    auto s = std::make_unique<aotx_cognitive_store>();
    AOTX_CUDA(cudaMemcpyFromSymbol(s.get(), aotx_live_store, sizeof(*s))); return s;
}
static aotx_bytes aotx_maint_policy(const aotx_cognitive_store &s, unsigned recent,
    unsigned age, unsigned automatic = 0, unsigned pressure = 90) {
    aotx_bytes p(64); memcpy(p.data(), "AOTXMNT1", 8); aotx_put(p.data() + 8, 1, 4);
    aotx_put(p.data() + 16, recent, 4); aotx_put(p.data() + 20, age, 4);
    aotx_put(p.data() + 24, automatic, 4); aotx_put(p.data() + 28, pressure, 4);
    aotx_put(p.data() + 32, s.sequence); aotx_put(p.data() + 40, s.root_sequence);
    memcpy(p.data() + 48, s.lineage, 16); return p;
}
static void aotx_maint_schema(aotx_bytes &p, const aotx_cognitive_store &s) {
    if (!s.pressure_percent) return;
    aotx_put(p.data() + 8, 2, 4); aotx_put(p.data() + 96, s.root_sequence);
    aotx_put(p.data() + 104, s.retry_floor); aotx_put(p.data() + 112, s.keep_recent, 4);
    aotx_put(p.data() + 116, s.max_age, 4); aotx_put(p.data() + 120, s.maintenance, 4);
    aotx_put(p.data() + 124, s.pressure_percent, 4);
}
static int aotx_maint_find(const aotx_cognitive_store &s, uint64_t id, uint64_t version) {
    for (uint32_t i = 0; i < s.count; ++i)
        if (aotx_get(s.objects[i] + AOTX_CO_ID) == id && aotx_get(s.objects[i] + AOTX_CO_VERSION) == version) return (int)i;
    return -1;
}
static void aotx_maint_append(aotx_live_records &out, const aotx_live_records &in) {
    out.insert(out.end(), in.begin(), in.end());
}
static uint64_t aotx_maint_size(const std::string &path) {
    struct stat s = {}; aotx_check(!stat(path.c_str(), &s), "mirror file has a readable size"); return (uint64_t)s.st_size;
}
static aotx_bytes aotx_maint_flush(aotx_checkpoint_device &d, aotx_checkpoint_disk &disk) {
    uint64_t serial = d.ring()->head + 1;
    d.publish(serial); auto image = d.image(serial);
    aotx_check(aotx_checkpoint_disk_pass(&disk) == 1, "one complete checkpoint becomes durable");
    d.step(); aotx_check(d.state().durable == d.live.state().accepted, "GPU sees the exact durable revision");
    return image;
}
static unsigned aotx_maint_batch;
static void aotx_maint_replay_idle(bool replay) {
    if (replay) aotx_live_test_idle<<<1,64>>>(aotx_maint_batch);
}
#endif
