/* Purpose: Check native policy authority, revision races and recorded controls.
 * Owns: Distinct principal grants, mapped mailboxes and exact journal comparisons.
 * Launch shape: Real device admission for N=1 and N=64 request batches.
 * Lifetime: Each fixture releases its transport and restores a complete control history. */
#include "live_fixture.h"
#include "policy/control.cuh"
#include "service/internal.cuh"
#include "appraisal/appraisal.cuh"
#include <memory>

__global__ void aotx_review_service_setup(unsigned n, unsigned actions) {
    if (threadIdx.x) return;
    aotx_policy.enabled = 1; aotx_policy.config.abi = 3;
    aotx_service.grant_count = n; aotx_service.cursor = 0;
    for (unsigned i = 0; i < n; ++i) {
        auto &g = aotx_service.grants[i]; g = {};
        aotx_service_put(g.principal, i + 1, 8); g.revision = i + 11; g.actions = actions;
    }
}
__global__ void aotx_review_service_cursor(unsigned held) { aotx_service.cursor = 0; aotx_sched.held = held; }
static std::unique_ptr<aotx_review_state> aotx_review_service_state(void) {
    auto s = std::make_unique<aotx_review_state>();
    AOTX_CUDA(cudaMemcpyFromSymbol(s.get(), aotx_review, sizeof(*s))); return s;
}
struct aotx_review_service_fixture {
    aotx_live_device live;
    aotx_service_state s = {};
    aotx_service_mailbox *mailbox = nullptr;
    unsigned n;
    explicit aotx_review_service_fixture(unsigned count) : live(count), n(count) {
        AOTX_LIVE_CLEAR(aotx_policy); AOTX_LIVE_CLEAR(aotx_appraisal); AOTX_LIVE_CLEAR(aotx_checkpoint);
        AOTX_CUDA(cudaHostAlloc(&mailbox, AOTX_SERVICE_CHANNELS * sizeof(*mailbox), cudaHostAllocMapped));
        memset(mailbox, 0, AOTX_SERVICE_CHANNELS * sizeof(*mailbox));
        AOTX_CUDA(cudaHostGetDevicePointer(&s.mailbox, mailbox, 0));
        AOTX_CUDA(cudaMalloc(&s.frames, (size_t)AOTX_SERVICE_CHANNELS * AOTX_SERVICE_FRAME));
        AOTX_CUDA(cudaMalloc(&s.ready, AOTX_SERVICE_CHANNELS * sizeof(unsigned)));
        AOTX_CUDA(cudaMemset(s.ready, 0, AOTX_SERVICE_CHANNELS * sizeof(unsigned)));
        AOTX_CUDA(cudaMalloc(&s.grants, AOTX_SERVICE_PRINCIPALS * sizeof(aotx_service_grant)));
        s.enabled = 1; s.epoch = 71; AOTX_CUDA(cudaMemcpyToSymbol(aotx_service, &s, sizeof(s)));
        aotx_review_service_setup<<<1,1>>>(n, 128); AOTX_CUDA(cudaDeviceSynchronize());
    }
    ~aotx_review_service_fixture() {
        AOTX_LIVE_CLEAR(aotx_service); cudaFree(s.frames); cudaFree(s.ready); cudaFree(s.grants); cudaFreeHost(mailbox);
    }
    void send(unsigned action, uint64_t revision, unsigned status, bool race = false, uint64_t epoch = 71, bool held = false) {
        for (unsigned i = 0; i < n; ++i) {
            auto &m = mailbox[i + 1]; memset(&m, 0, sizeof(m)); auto p = m.bytes;
            memcpy(p, AOTX_SERVICE_MAGIC, 8); aotx_service_put(p + 8, AOTX_SERVICE_POLICY, 4);
            aotx_service_put(p + 16, i + 1, 8); aotx_service_put(p + 32, i + 11, 8);
            if (action) {
                aotx_service_put(p + 40, epoch, 8); aotx_service_put(p + 88, 16, 4);
                aotx_service_put(p + 128, 1, 4); aotx_service_put(p + 132, action, 4);
                aotx_service_put(p + 136, revision + (race ? 0 : i), 8);
            }
            m.length = AOTX_SERVICE_HEAD + (action ? 16 : 0); m.state = 1;
        }
        aotx_review_service_cursor<<<1,1>>>(held);
        aotx_service_copy<<<AOTX_SERVICE_CHANNELS,64>>>(); aotx_service_admit<<<1,1>>>();
        AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) {
            auto &m = mailbox[i + 1]; unsigned wanted = race && i ? 409 : status;
            aotx_check(m.state == 2 && aotx_service_get(m.bytes + 8, 4) == wanted,
                "each scoped policy request has the exact expected admission status");
            if (wanted == 200) {
                aotx_check(m.length == 288 && aotx_service_get(m.bytes + 128, 4) == 1 &&
                    aotx_service_get(m.bytes + 40, 8) == 71, "each response has the bounded aggregate schema and current epoch");
                for (unsigned j = 252; j < 288; ++j) aotx_check(!m.bytes[j], "aggregate status has no undeclared source bytes");
            }
        }
    }
};
static void aotx_review_service_case(unsigned n) {
    aotx_live_records journal; uint64_t revision = 0;
    {
        aotx_review_service_fixture f(n);
        f.send(0, 0, 200); f.send(AOTX_POLICY_REVIEW_ON, 0, 200);
        revision = n;
        auto s = aotx_review_service_state();
        aotx_check(s->enabled && s->control_revision == revision, "distinct ordered operator requests apply once each");
        uint64_t tail = f.live.seam().dev.tail;
        f.send(AOTX_POLICY_REVIEW_OFF, 0, 409);
        f.send(AOTX_POLICY_REVIEW_OFF, revision, 410, false, 70);
        f.send(AOTX_POLICY_PAUSE, revision, 429, false, 71, true);
        f.send(0, 0, 200, false, 71, true);
        aotx_check(f.live.seam().dev.tail == tail && aotx_review_service_state()->control_revision == revision,
            "stale revisions, old epochs and scheduler holds have no journal or control effect");
        f.send(AOTX_POLICY_PAUSE, revision, 200, true); ++revision;
        aotx_check(aotx_review_service_state()->control_revision == revision,
            "competing requests with one expected revision have exactly one effect");
        for (unsigned actions : {1u, 8u, 64u}) {
            aotx_review_service_setup<<<1,1>>>(n, actions); AOTX_CUDA(cudaDeviceSynchronize());
            f.send(AOTX_POLICY_RESUME, revision, 403);
            f.send(0, 0, actions == 8 ? 200 : 403);
        }
        aotx_review_service_setup<<<1,1>>>(n, 128); AOTX_CUDA(cudaDeviceSynchronize());
        f.send(AOTX_POLICY_RESUME, revision, 200); revision += n;
        f.send(AOTX_POLICY_STOP, revision, 200); revision += n;
        auto seam = f.live.seam(); journal.resize(seam.dev.tail);
        AOTX_CUDA(cudaMemcpy(journal.data(), f.live.out, journal.size() * AOTX_SLOT_BYTES, cudaMemcpyDeviceToHost));
        for (const auto &r : journal) aotx_check(((const aotx_record_header *)r.data())->type == AOTX_REC_POLICY_CONTROL,
            "accepted native controls use the typed authoritative journal record");
    }
    {
        aotx_review_service_fixture f(n);
        f.live.process(journal, true);
        auto review = aotx_review_service_state(); aotx_policy_state policy;
        AOTX_CUDA(cudaMemcpyFromSymbol(&policy, aotx_policy, sizeof(policy)));
        aotx_check(!policy.fatal && review->control_revision == revision && review->enabled && policy.stopped && policy.paused,
            "journal replay restores all exact native control effects and the revision");
    }
}
int main(void) {
    for (unsigned n : {1u, 64u}) aotx_review_service_case(n);
    printf("reflection service: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
