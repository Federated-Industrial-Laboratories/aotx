/* Purpose: Build exact typed sources and shared readers for cold-memory checks.
 * Owns: Distinct source graphs, participants and bounded service reply buffers.
 * Launch shape: Ordered device read batches at N=1 and N=64.
 * Lifetime: One isolated test case without model weights. */
#ifndef AOTX_SHARED_COLD_FIXTURE_H
#define AOTX_SHARED_COLD_FIXTURE_H
#include "cold_fixture.h"
#include "cognitive/intake.h"
#include "shared/host.h"
#include "shared/internal.cuh"
#include "disk/runtime/runtime.h"

static std::string cold_quote(unsigned i, bool current) {
    return std::string(current ? "New fact " : "Old fact ") + std::to_string(i) + ".";
}
static aotx_bytes cold_interpretation(unsigned i, bool current) {
    auto quote = cold_quote(i, current);
    aotx_bytes p(AOTX_INTAKE_PAYLOAD + quote.size()); memcpy(p.data(), "AOTXMEM3", 8);
    aotx_put(p.data() + 8, 3, 4); aotx_put(p.data() + 12, quote.size(), 4);
    aotx_put(p.data() + 16, 3, 4);
    aotx_put(p.data() + 20, current ? cold_quote(i, false).size() + 1 : 0, 4);
    memset(p.data() + 24, 0x51, 32); memset(p.data() + 56, 0x71, 32);
    memcpy(p.data() + AOTX_INTAKE_PAYLOAD, quote.data(), quote.size()); return p;
}
static aotx_fixture cold_shared_corpus(unsigned n) {
    aotx_fixture f;
    for (unsigned i = 0; i < n; ++i) for (unsigned k = 0; k < 4; ++k) {
        uint64_t id = 600000 + i * 4;
        const unsigned kinds[] = {AOTX_COG_EVENT, AOTX_COG_COMPONENT, AOTX_COG_ASSERTION, AOTX_COG_WORKING};
        auto r = aotx_memory_row(i, kinds[k], id + k, f.rows.size() + 1);
        auto text = cold_quote(i, false) + " " + cold_quote(i, true);
        auto p = aotx_memory_text(text);
        if (k) { aotx_id(r.data() + AOTX_CO_SOURCE, id); aotx_put(r.data() + AOTX_CO_SOURCE_VERSION, 1); }
        if (k == 1) {
            p = aotx_memory_vector(i + 1, 2, 3); memcpy(p.data(), "AOTXVEC2", 8);
            aotx_put(p.data() + 8, 2, 4); memset(p.data() + 88, 0, 40);
            aotx_id(p.data() + 88, id); aotx_put(p.data() + 104, 1);
        }
        if (k == 2) {
            p = cold_interpretation(i, false); memset(r.data() + AOTX_CO_SUBJECT, 0, 16);
            aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
        }
        if (k >= 2) {
            aotx_id(r.data() + AOTX_CO_EMBEDDING, id + 1); aotx_put(r.data() + AOTX_CO_EMBED_VERSION, 1);
        }
        f.add(r, p);
    }
    return f;
}
__global__ void aotx_shared_cold_read_test(unsigned n, unsigned mode) {
    if (threadIdx.x || blockIdx.x) return;
    for (unsigned i = 0; i < n; ++i) {
        aotx_service_grant grant = {}; grant.actions = 127;
        aotx_service_bytes(grant.principal, aotx_shared.participants[i].id, 16);
        if (mode == 2) grant.principal[15] ^= 0xff;
        unsigned char read[96] = {};
        if (mode) aotx_service_bytes(read + 48, aotx_live_store.objects[i * 4 + 2] + AOTX_CO_ID, 16);
        aotx_service_put(read + 80, 64, 4);
        aotx_shared_memory_read(i, &grant, read, i);
    }
}
__global__ void aotx_shared_cold_publish_test(unsigned n, bool apply, unsigned *out) {
    if (threadIdx.x || blockIdx.x) return;
    for (unsigned i = 0; i < n; ++i) {
        aotx_service_grant grant = {}; grant.actions = 127;
        aotx_service_bytes(grant.principal, aotx_shared.participants[i].id, 16);
        out[i] = aotx_shared_publish_check(aotx_shared.receipts + i, &grant);
        if (apply && out[i] == 200) {
            aotx_shared.transfer_serial = ++aotx_shared.serial;
            out[i] = aotx_shared_publish_apply(i, false) ? 200 : 500;
        }
    }
}
struct cold_shared_readers {
    unsigned n;
    aotx_service_state service = {};
    unsigned *status;
    explicit cold_shared_readers(unsigned count) : n(count) {
        aotx_runtime_shared_profile profile; aotx_runtime_shared_current(&profile);
        profile.participants = n; profile.spaces = 2 * n; profile.members = n;
        profile.conversations = 1; profile.receipts = n;
        aotx_check(!aotx_shared_open(&profile), "shared test tables open");
        aotx_shared_state shared; AOTX_CUDA(cudaMemcpyFromSymbol(&shared, aotx_shared, sizeof(shared)));
        std::vector<aotx_shared_participant> people(n);
        std::vector<aotx_shared_space> spaces(2 * n);
        std::vector<aotx_shared_receipt> receipts(n);
        for (unsigned i = 0; i < n; ++i) {
            aotx_id(people[i].id, 90000 + i); people[i].active = 1;
            for (unsigned j : {i, n + i}) {
                aotx_id(spaces[j].id, (j < n ? 1000 : 7000) + i); spaces[j].active = 1;
                memcpy(spaces[j].owner, people[i].id, 16);
            }
            auto &r = receipts[i]; memcpy(r.actor, people[i].id, 16);
            r.operation = AOTX_SHARED_PUBLISH; r.participant = i; r.space = n + i;
            r.phase = AOTX_SHARED_QUEUED; r.admission_source = i + 1;
            aotx_id(r.command + 56, 600003 + i * 4); aotx_id(r.command + 72, 7000 + i);
            aotx_put(r.command + 128, 1);
        }
        AOTX_CUDA(cudaMemcpy(shared.participants, people.data(), n * sizeof(people[0]), cudaMemcpyHostToDevice));
        AOTX_CUDA(cudaMemcpy(shared.spaces, spaces.data(), spaces.size() * sizeof(spaces[0]), cudaMemcpyHostToDevice));
        AOTX_CUDA(cudaMemcpy(shared.receipts, receipts.data(), n * sizeof(receipts[0]), cudaMemcpyHostToDevice));
        AOTX_CUDA(cudaMallocManaged(&service.mailbox, n * sizeof(*service.mailbox)));
        AOTX_CUDA(cudaMallocManaged(&service.frames, n * AOTX_SERVICE_FRAME));
        AOTX_CUDA(cudaMallocManaged(&service.ready, n * sizeof(*service.ready)));
        AOTX_CUDA(cudaMemset(service.mailbox, 0, n * sizeof(*service.mailbox)));
        AOTX_CUDA(cudaMemset(service.frames, 0, n * AOTX_SERVICE_FRAME));
        AOTX_CUDA(cudaMemset(service.ready, 0, n * sizeof(*service.ready)));
        AOTX_CUDA(cudaMemcpyToSymbol(aotx_service, &service, sizeof(service)));
        AOTX_CUDA(cudaMallocManaged(&status, n * sizeof(*status)));
    }
    void read(bool cold, bool corrected = false) {
        for (unsigned mode = 0; mode < 3; ++mode) {
            aotx_shared_cold_read_test<<<1,1>>>(n, mode); AOTX_CUDA(cudaDeviceSynchronize());
            for (unsigned i = 0; i < n; ++i) {
                auto p = service.mailbox[i].bytes;
                unsigned expected = mode == 2 ? 404 : mode == 1 && cold ? 503 : 200;
                aotx_check(aotx_get(p + 8, 4) == expected, "shared memory reads enforce scope and residency");
                if (expected != 200) {
                    aotx_check(service.mailbox[i].length == AOTX_SERVICE_HEAD, "refused reads contain no private payload");
                } else if (!mode) {
                    auto out = p + AOTX_SERVICE_HEAD;
                    aotx_check(aotx_get(out + 192, 4) == 4 && !aotx_get(out + 200),
                        "the list retains exactly the four current source objects");
                } else {
                    auto expected_payload = cold_interpretation(i, corrected);
                    auto out = p + AOTX_SERVICE_HEAD + AOTX_SHARED_REPLY_HEAD;
                    aotx_check(aotx_get(out + 16) == (corrected ? 2 : 1) &&
                        !memcmp(out + 128, expected_payload.data(), expected_payload.size()),
                        "detail returns the exact current typed quote and version");
                }
            }
        }
    }
    void publish(unsigned expected, bool apply = false) {
        aotx_shared_cold_publish_test<<<1,1>>>(n, apply, status); AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) aotx_check(status[i] == expected, "publication has the required residency result");
    }
    ~cold_shared_readers() {
        cudaFree(status); cudaFree(service.mailbox); cudaFree(service.frames); cudaFree(service.ready);
        AOTX_LIVE_CLEAR(aotx_service); aotx_shared_close();
    }
};
#endif
