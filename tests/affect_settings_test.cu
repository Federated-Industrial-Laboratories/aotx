/* Purpose: Check runtime affect setting grants, races, journal replay and sequence snapshots.
 * Owns: Distinct principals and mapped service requests.
 * Launch shape: N=1 and N=64 ordered device admission batches.
 * Lifetime: One isolated runtime and its recorded settings. */
#include "live_fixture.h"
#include "service/internal.cuh"
#include "settings/settings.cuh"
#ifdef AOTX_AFFECT
#include "affect/affect.cuh"
__global__ void aotx_affect_test_setup(unsigned n, unsigned actions, bool held, bool pending) {
    aotx_service.grant_count = n; aotx_service.cursor = 0; aotx_sched.held = held;
    for (unsigned i = 0; i < n; ++i) {
        auto &g = aotx_service.grants[i]; g = {};
        aotx_service_put(g.principal, i + 1, 8); g.revision = i + 11; g.actions = actions;
    }
    aotx_setting_table.pending_count = pending;
    if (pending) {
        auto &p = aotx_setting_table.pending[0]; p = {};
        const char *key = "affect.on"; p.key_len = 9; p.value = 1; p.scale = 1;
        for (unsigned i = 0; i < 9; ++i) p.key[i] = key[i];
    }
}
__global__ void aotx_affect_test_snapshot(unsigned n, bool capture, float *out) {
    unsigned i = threadIdx.x; if (i >= n) return;
    if (capture) aotx_affect_snapshot(i);
    out[i] = aotx_affect_laws[i].decay_fast;
}
__global__ void aotx_affect_test_revision(void) { aotx_setting_table.affect_revision = ~0ull; }
static aotx_settings_state aotx_affect_test_state(void) {
    aotx_settings_state s; AOTX_CUDA(cudaMemcpyFromSymbol(&s, aotx_setting_table, sizeof(s))); return s;
}
struct aotx_affect_test_fixture {
    aotx_live_device live;
    aotx_service_state s = {};
    aotx_service_mailbox *mailbox = nullptr;
    unsigned n; float *snapshot;
    explicit aotx_affect_test_fixture(unsigned count) : live(count), n(count) {
        AOTX_LIVE_CLEAR(aotx_checkpoint);
        AOTX_CUDA(cudaHostAlloc(&mailbox, AOTX_SERVICE_CHANNELS * sizeof(*mailbox), cudaHostAllocMapped));
        memset(mailbox, 0, AOTX_SERVICE_CHANNELS * sizeof(*mailbox));
        AOTX_CUDA(cudaHostGetDevicePointer(&s.mailbox, mailbox, 0));
        AOTX_CUDA(cudaMalloc(&s.frames, (size_t)AOTX_SERVICE_CHANNELS * AOTX_SERVICE_FRAME));
        AOTX_CUDA(cudaMalloc(&s.ready, AOTX_SERVICE_CHANNELS * sizeof(unsigned)));
        AOTX_CUDA(cudaMemset(s.ready, 0, AOTX_SERVICE_CHANNELS * sizeof(unsigned)));
        AOTX_CUDA(cudaMalloc(&s.grants, AOTX_SERVICE_PRINCIPALS * sizeof(aotx_service_grant)));
        AOTX_CUDA(cudaMalloc(&snapshot, n * sizeof(float)));
        s.enabled = 1; s.epoch = 71; AOTX_CUDA(cudaMemcpyToSymbol(aotx_service, &s, sizeof(s)));
    }
    ~aotx_affect_test_fixture() {
        AOTX_LIVE_CLEAR(aotx_service); cudaFree(s.frames); cudaFree(s.ready); cudaFree(s.grants); cudaFreeHost(mailbox); cudaFree(snapshot);
    }
    void law(bool capture, float expected) {
        aotx_affect_test_snapshot<<<1,64>>>(n, capture, snapshot); AOTX_CUDA(cudaDeviceSynchronize());
        std::vector<float> values(n); AOTX_CUDA(cudaMemcpy(values.data(), snapshot, n*sizeof(float), cudaMemcpyDeviceToHost));
        for (float value : values) aotx_check(value == expected, "a sequence changes its captured law only at the next open");
    }
    void send(const char *key, uint64_t revision, unsigned status, bool race = false,
              unsigned actions = 256, uint64_t epoch = 71, bool held = false, bool pending = false, int value = 2000) {
        for (unsigned i = 0; i < n; ++i) {
            auto &m = mailbox[i + 1]; memset(&m, 0, sizeof(m)); auto p = m.bytes;
            memcpy(p, AOTX_SERVICE_MAGIC, 8); aotx_service_put(p + 8, AOTX_SERVICE_AFFECT_SETTINGS, 4);
            aotx_service_put(p + 16, i + 1, 8); aotx_service_put(p + 32, i + 11, 8);
            if (key) {
                aotx_service_put(p + 40, epoch, 8); aotx_service_put(p + 88, 96, 4);
                aotx_service_put(p + 128, 1, 4); aotx_service_put(p + 132, strlen(key), 4);
                aotx_service_put(p + 136, revision + (race ? 0 : i), 8);
                aotx_service_put(p + 144, (uint64_t)value, 8); aotx_service_put(p + 152, 10000, 4);
                memcpy(p + 160, key, strlen(key));
            }
            m.length = AOTX_SERVICE_HEAD + (key ? 96 : 0); m.state = 1;
        }
        aotx_affect_test_setup<<<1,1>>>(n, actions, held, pending);
        aotx_service_copy<<<AOTX_SERVICE_CHANNELS,64>>>(); aotx_service_admit<<<1,1>>>(); AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) {
            auto &m = mailbox[i + 1]; unsigned expected = race && status == 200 && i ? 409 : status;
            if (m.state != 2 || aotx_service_get(m.bytes + 8, 4) != expected)
                fprintf(stderr, "row=%u key=%s revision=%llu actions=%u expected=%u actual=%llu\n", i, key ? key : "read",
                    (unsigned long long)revision, actions, expected, (unsigned long long)aotx_service_get(m.bytes + 8, 4));
            aotx_check(m.state == 2 && aotx_service_get(m.bytes + 8, 4) == expected, "each operator request has the expected status");
            if (expected == 200) {
                aotx_check(m.length == 128 + 32 + 13 * 64 && aotx_service_get(m.bytes + 128, 4) == 1 &&
                    aotx_service_get(m.bytes + 132, 4) == 13, "the response contains all bounded setting rows");
                aotx_check(aotx_service_get(m.bytes + 144, 4) == !!(actions & 256), "the readback reports the actual write permission");
            }
        }
    }
};
static void aotx_affect_settings_case(unsigned n) {
    aotx_live_records journal; unsigned long long revision;
    {
        aotx_affect_test_fixture f(n); f.law(true, 0.5f);
        f.send(nullptr, 0, 200); f.send("affect.decay_fast", 0, 200); revision = n;
        f.law(false, 0.5f); f.law(true, 0.2f);
        auto state = aotx_affect_test_state();
        aotx_check(state.affect_revision == revision && state.row[AOTX_SET_AFFECT_DECAY_FAST].value == 2000,
            "ordered changes in one tick retain every revision");
        auto tail = f.live.seam().dev.tail;
        f.send("affect.decay_fast", 0, 409, true); f.send("affect.decay_fast", revision, 410, false, 256, 70);
        f.send("affect.decay_fast", revision, 429, true, 256, 71, true);
        f.send("affect.decay_fast", revision, 429, true, 256, 71, false, true);
        f.send("affect.decay_fast", revision, 400, false, 256, 71, false, false, 9901);
        f.send("sample.temperature", revision, 400);
        for (unsigned actions : {1u, 8u, 64u, 128u}) {
            f.send("affect.decay_fast", revision, 403, false, actions);
            f.send(nullptr, 0, actions == 8 ? 200 : 403, false, actions);
        }
        aotx_check(f.live.seam().dev.tail == tail && aotx_affect_test_state().affect_revision == revision,
            "refused changes have no setting or journal effect");
        f.send("affect.decay_fast", revision, 200, true); ++revision;
        aotx_check(aotx_affect_test_state().affect_revision == revision, "one shared expected revision permits exactly one write");
        auto seam = f.live.seam(); journal.resize(seam.dev.tail);
        AOTX_CUDA(cudaMemcpy(journal.data(), f.live.out, journal.size() * AOTX_SLOT_BYTES, cudaMemcpyDeviceToHost));
        for (const auto &r : journal) aotx_check(((const aotx_record_header *)r.data())->type == AOTX_REC_SETTING,
            "accepted writes use the existing replayable setting record");
        aotx_affect_test_revision<<<1,1>>>(); AOTX_CUDA(cudaDeviceSynchronize());
        f.send("affect.decay_fast", ~0ull, 409, true);
        aotx_check(f.live.seam().dev.tail == seam.dev.tail, "revision overflow refuses before record publication");
    }
    {
        aotx_affect_test_fixture f(n); f.live.process(journal, true);
        auto state = aotx_affect_test_state();
        aotx_check(state.affect_revision == revision && state.row[AOTX_SET_AFFECT_DECAY_FAST].value == 2000,
            "journal replay restores the settings and exact revision");
    }
}
#endif
int main(void) {
#ifdef AOTX_AFFECT
    for (unsigned n : {1u, 64u}) aotx_affect_settings_case(n);
    printf("affect settings: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
#else
    return 77;
#endif
}
