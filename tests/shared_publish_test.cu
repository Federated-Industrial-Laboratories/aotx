/* Purpose: Check exact native memory publication, permission failures and replay copies.
 * Owns: Distinct source graphs, small shared tables and complete store comparisons.
 * Launch shape: An ordered N=1 or N=64 publication batch on the real GPU store.
 * Lifetime: Each case frees its shared tables and test buffers. */
#include "checkpoint_fixture.h"
#include "shared/host.h"
#include "shared/internal.cuh"
#include "cognitive/validate.cuh"
#include "disk/runtime/runtime.h"
#include <memory>

static aotx_fixture aotx_shared_publish_corpus(unsigned n, bool pressure) {
    aotx_fixture f;
    for (unsigned i = 0; i < n; ++i) for (unsigned k = 0; k < 3; ++k) {
        uint64_t first = i * 3 + 1;
        auto r = aotx_memory_row(i, k == 0 ? AOTX_COG_EVENT : k == 1 ? AOTX_COG_COMPONENT : AOTX_COG_WORKING,
            600000 + i * 3 + k, first + k);
        aotx_put(r.data() + AOTX_CO_VERSION, pressure ? first + k : 1);
        aotx_put(r.data() + AOTX_CO_EVIDENCE, i % 3, 4);
        aotx_put(r.data() + AOTX_CO_IMPORTANCE, 17000 + i * 11, 4);
        if (k) {
            aotx_id(r.data() + AOTX_CO_SOURCE, 600000 + i * 3);
            aotx_put(r.data() + AOTX_CO_SOURCE_VERSION, pressure ? first : 1);
        }
        if (k == 1) aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
        if (k == 2) {
            aotx_id(r.data() + AOTX_CO_EMBEDDING, 600001 + i * 3);
            aotx_put(r.data() + AOTX_CO_EMBED_VERSION, pressure ? first + 1 : 1);
        }
        auto p = aotx_memory_text("published source " + std::to_string(i));
        if (k == 1) {
            p = aotx_memory_vector(i + 1, i + 3, i + 7);
            memcpy(p.data(), "AOTXVEC2", 8); aotx_put(p.data() + 8, 2, 4);
            memset(p.data() + 88, 0, 40); aotx_id(p.data() + 88, 600000 + i * 3);
            aotx_put(p.data() + 104, pressure ? first : 1);
        }
        f.add(r, p);
    }
    return f;
}
__global__ void aotx_shared_publish_test_check(unsigned n, unsigned mode, unsigned *status) {
    if (threadIdx.x || blockIdx.x) return;
    for (unsigned i = 0; i < n; ++i) {
        aotx_shared_receipt *r = aotx_shared.receipts + i;
        aotx_service_grant grant = {};
        aotx_service_bytes(grant.principal, r->actor, 16); grant.actions = 127;
        unsigned char *working = aotx_live_store.objects[i * 3 + 2];
        unsigned char *vector = aotx_live_store.payload + aotx_cog_u64(aotx_live_store.objects[i * 3 + 1] + AOTX_CO_OFFSET);
        unsigned saved_bytes = aotx_live_store.bytes;
        uint64_t version = aotx_cog_u64(r->command + 128);
        if (mode == 1) grant.actions &= ~AOTX_SHARED_READ_ACTION;
        if (mode == 2) aotx_cog_put(r->command + 128, version + 1, 8);
        if (mode == 3) aotx_cog_put(working + AOTX_CO_KIND, AOTX_COG_ASSERTION, 2);
        if (mode == 4) vector[88] ^= 1;
        if (mode == 5) aotx_shared.spaces[i].owner[0] ^= 1;
        if (mode == 6) aotx_shared.spaces[n + i].owner[0] ^= 1;
        if (mode == 7) aotx_live_store.bytes = AOTX_COG_PAYLOAD;
        if (mode == 8) aotx_live.phase = AOTX_LIVE_WAIT;
        if (mode == 9) aotx_shared.slot[0] = 1;
        status[i] = aotx_shared_publish_check(r, &grant);
        aotx_cog_put(r->command + 128, version, 8);
        aotx_cog_put(working + AOTX_CO_KIND, AOTX_COG_WORKING, 2);
        if (mode == 4) vector[88] ^= 1;
        if (mode == 5) aotx_shared.spaces[i].owner[0] ^= 1;
        if (mode == 6) aotx_shared.spaces[n + i].owner[0] ^= 1;
        aotx_live_store.bytes = saved_bytes; aotx_live.phase = AOTX_LIVE_IDLE;
        aotx_shared.slot[0] = 0;
    }
}
__global__ void aotx_shared_publish_test_apply(unsigned n, unsigned replay, unsigned *status) {
    if (threadIdx.x || blockIdx.x) return;
    for (unsigned i = 0; i < n; ++i) {
        aotx_shared.transfer_serial = i + 1;
        status[i] = aotx_shared_publish_apply(i, replay != 0) ? 200 : 409;
        aotx_shared.serial = i + 1;
        if (status[i] == 200) for (unsigned k = 0; k < 3; ++k)
            if (aotx_cog_validate(&aotx_live_store, n * 3 + i * 3 + k)) status[i] = 500;
    }
}
__global__ void aotx_shared_publish_test_reject(unsigned n, unsigned *status) {
    if (threadIdx.x || blockIdx.x) return;
    for (unsigned i = 0; i < n; ++i) {
        unsigned char *p = aotx_shared.receipts[i].command + 128;
        uint64_t version = aotx_cog_u64(p); aotx_cog_put(p, version + 1, 8);
        aotx_shared.transfer_serial = i + 1;
        status[i] = aotx_shared_publish_apply(i, false) ? 200 : 409;
        aotx_cog_put(p, version, 8);
    }
}
static std::unique_ptr<aotx_cognitive_store> aotx_shared_publish_store(void) {
    auto s = std::make_unique<aotx_cognitive_store>();
    AOTX_CUDA(cudaMemcpyFromSymbol(s.get(), aotx_live_store, sizeof(*s)));
    return s;
}
static void aotx_shared_publish_case(unsigned n, bool pressure) {
    aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint); AOTX_LIVE_CLEAR(aotx_runtime_enabled);
    auto corpus = aotx_shared_publish_corpus(n, pressure);
    auto image = corpus.wire(false, n * 3);
    if (pressure) { aotx_put(image.data() + 8, 2, 4); aotx_put(image.data() + 124, 90, 4); }
    d.send(aotx_live_load_bytes(image), AOTX_LIVE_LOAD);
    aotx_check(d.state().ready && !d.state().status, "native source graph loads through validated memory admission");
    auto before = aotx_shared_publish_store();
    aotx_runtime_shared_profile profile;
    aotx_runtime_shared_current(&profile);
    profile.participants = n; profile.spaces = n * 2; profile.members = n;
    profile.conversations = 1; profile.receipts = n;
    aotx_check(!aotx_shared_open(&profile), "small explicit shared capacities allocate before state use");
    aotx_shared_state shared;
    AOTX_CUDA(cudaMemcpyFromSymbol(&shared, aotx_shared, sizeof(shared)));
    std::vector<aotx_shared_space> spaces(n * 2);
    std::vector<aotx_shared_participant> people(n);
    std::vector<aotx_shared_receipt> receipts(n);
    for (unsigned i = 0; i < n; ++i) {
        aotx_id(people[i].id, 90000 + i); people[i].active = 1;
        aotx_id(spaces[i].id, 1000 + i); spaces[i].active = 1;
        aotx_id(spaces[n + i].id, 7000 + i); spaces[n + i].active = 1; spaces[n + i].scope = i % 3;
        memcpy(spaces[i].owner, people[i].id, 16); memcpy(spaces[n + i].owner, people[i].id, 16);
        auto &r = receipts[i]; memcpy(r.actor, people[i].id, 16);
        r.operation = AOTX_SHARED_PUBLISH; r.participant = i; r.space = n + i;
        r.phase = AOTX_SHARED_QUEUED; r.admission_source = 1000 + i;
        aotx_id(r.command + 56, 600002 + i * 3); aotx_id(r.command + 72, 7000 + i);
        aotx_put(r.command + 128, pressure ? i * 3 + 3 : 1);
    }
    AOTX_CUDA(cudaMemcpy(shared.participants, people.data(), n * sizeof(people[0]), cudaMemcpyHostToDevice));
    AOTX_CUDA(cudaMemcpy(shared.spaces, spaces.data(), spaces.size() * sizeof(spaces[0]), cudaMemcpyHostToDevice));
    AOTX_CUDA(cudaMemcpy(shared.receipts, receipts.data(), n * sizeof(receipts[0]), cudaMemcpyHostToDevice));
    unsigned *status; AOTX_CUDA(cudaMallocManaged(&status, n * sizeof(*status)));
    const unsigned expected[] = {200, 404, 409, 409, 409, 404, 404, 429, 429, 429};
    for (unsigned mode = 0; mode < 10; ++mode) {
        aotx_shared_publish_test_check<<<1,1>>>(n, mode, status); AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) aotx_check(status[i] == expected[mode], "publication checks scope, exact graph and capacity");
        auto unchanged = aotx_shared_publish_store();
        aotx_check(!memcmp(before.get(), unchanged.get(), sizeof(*before)), "publication checks change no live store byte");
    }
    aotx_shared_publish_test_reject<<<1,1>>>(n, status); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) aotx_check(status[i] == 409, "a changed exact version refuses publication at commit");
    auto unchanged = aotx_shared_publish_store();
    aotx_check(!memcmp(before.get(), unchanged.get(), sizeof(*before)), "failed publication applies no partial object or payload batch");
    aotx_shared_publish_test_apply<<<1,1>>>(n, 0, status); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) aotx_check(status[i] == 200, "each published batch passes the complete object validator");
    auto after = aotx_shared_publish_store();
    std::vector<aotx_shared_receipt> complete(n);
    AOTX_CUDA(cudaMemcpy(complete.data(), shared.receipts, n * sizeof(complete[0]), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < n; ++i)
        aotx_check(complete[i].phase == AOTX_SHARED_DONE && complete[i].status == 200 &&
            complete[i].terminal_source == complete[i].admission_source && !complete[i].saved_terminal,
            "publication finishes at its admission source and waits for the actual saved acknowledgement");
    aotx_check(after->count == n * 6 && after->sequence == n * 6 && after->tick == before->tick + n,
        "publication appends exactly three objects and one store tick per operation");
    aotx_check(!memcmp(before->objects, after->objects, n * 3 * AOTX_COG_OBJECT) &&
        !memcmp(before->payload, after->payload, before->bytes), "publication preserves all original source bytes");
    for (unsigned i = 0; i < n; ++i) for (unsigned k = 0; k < 3; ++k) {
        const auto *r = after->objects[n * 3 + i * 3 + k], *old = before->objects[i * 3 + k];
        const auto *p = after->payload + aotx_get(r + AOTX_CO_OFFSET);
        const auto *q = before->payload + aotx_get(old + AOTX_CO_OFFSET);
        uint64_t version = pressure ? n * 3 + i * 3 + k + 1 : 1;
        aotx_check(!memcmp(r + AOTX_CO_ID, "AOTXSHP1", 8) && aotx_get(r + AOTX_CO_ID + 8) == (i + 1) * 4 + k + 1 &&
            aotx_get(r + AOTX_CO_VERSION) == version, "published IDs and versions follow the recorded operation serial");
        aotx_check(!memcmp(r + AOTX_CO_OWNER, spaces[n + i].id, 16) && aotx_get(r + AOTX_CO_SCOPE, 4) == i % 3,
            "each destination has its exact independent scope owner");
        aotx_check(aotx_get(r + AOTX_CO_SOURCE_KIND, 4) == aotx_get(old + AOTX_CO_SOURCE_KIND, 4) &&
            !memcmp(r + AOTX_CO_SUBJECT, old + AOTX_CO_SUBJECT, 16) &&
            !memcmp(r + AOTX_CO_EVIDENCE, old + AOTX_CO_EVIDENCE, 12), "publication preserves source and evidence metadata");
        aotx_check(k == 1 ? !memcmp(p, q, 88) && !memcmp(p + 112, q + 112, aotx_get(old + AOTX_CO_BYTES) - 112) :
            !memcmp(p, q, aotx_get(old + AOTX_CO_BYTES)), "text, model and vector bytes remain exact");
    }
    AOTX_CUDA(cudaMemcpyToSymbol(aotx_live_store, before.get(), sizeof(*before)));
    AOTX_CUDA(cudaMemcpy(shared.receipts, receipts.data(), n * sizeof(receipts[0]), cudaMemcpyHostToDevice));
    AOTX_CUDA(cudaMemset(shared.participants, 0, n * sizeof(people[0])));
    aotx_shared_publish_test_apply<<<1,1>>>(n, 1, status); AOTX_CUDA(cudaDeviceSynchronize());
    auto replay = aotx_shared_publish_store();
    for (unsigned i = 0; i < n; ++i) aotx_check(status[i] == 200, "recorded publication does not consult current grants");
    aotx_check(!memcmp(after.get(), replay.get(), sizeof(*after)), "publication replay restores the exact complete store bytes");
    cudaFree(status); aotx_shared_close();
}
int main(void) {
    for (unsigned n : {1u, 64u}) for (bool pressure : {false, true}) aotx_shared_publish_case(n, pressure);
    printf("shared publication: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures || aotx_checks < 1000 ? 1 : 0;
}
