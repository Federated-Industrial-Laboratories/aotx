/* Purpose: Allocate bounded mapped shared-state fixtures with distinct principals.
 * Owns: Isolated device tables, a mapped mailbox and a retained record ring.
 * Launch shape: Production mailbox admission and record emission at N=1 and N=64.
 * Lifetime: One fixture; all allocations are released on exit. */
#ifndef AOTX_SHARED_STATE_FIXTURE_H
#define AOTX_SHARED_STATE_FIXTURE_H
#include "shared/internal.cuh"
#include "model/load.cuh"
#include "model/wrap.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
static unsigned checks, failures;
static void check(bool value, const char *label)
{ ++checks; if (!value) { ++failures; fprintf(stderr, "FAIL %s\n", label); } }
static void cu(cudaError_t status)
{ if (status != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(status)); exit(1); } }
static void put(std::vector<unsigned char> &p, unsigned at, unsigned long long value, unsigned bytes = 4)
{ aotx_service_put(p.data() + at, value, bytes); }
static std::vector<unsigned char> command(unsigned actor, unsigned operation, unsigned long long sequence,
                                          unsigned target = 0, unsigned space = 0, const std::string &text = "")
{
    std::vector<unsigned char> p(AOTX_SERVICE_HEAD + AOTX_SHARED_COMMAND_HEAD + text.size());
    memcpy(p.data(), AOTX_SERVICE_MAGIC, 8); put(p, 8, 10); put(p, 16, actor); put(p, 32, 1, 8);
    put(p, 88, p.size() - AOTX_SERVICE_HEAD);
    unsigned h = AOTX_SERVICE_HEAD;
    memcpy(p.data() + h, AOTX_SHARED_MAGIC, 8); put(p, h + 8, operation); put(p, h + 16, sequence, 8);
    put(p, h + 24, actor); put(p, h + 32, sequence, 8); put(p, h + 40, 77);
    put(p, h + 56, target); put(p, h + 72, space);
    if (operation == AOTX_SHARED_INPUT) {
        put(p, h + 104, AOTX_MODEL_LANGUAGE); put(p, h + 108, 16); put(p, h + 124, 0x3f800000);
        put(p, h + 136, text.size()); memcpy(p.data() + h + AOTX_SHARED_COMMAND_HEAD, text.data(), text.size());
    }
    return p;
}
static std::vector<unsigned char> read_frame(unsigned actor, unsigned kind, unsigned target = 0, unsigned parent = 0)
{
    std::vector<unsigned char> p(AOTX_SERVICE_HEAD + AOTX_SHARED_READ_HEAD);
    memcpy(p.data(), AOTX_SERVICE_MAGIC, 8); put(p, 8, 11); put(p, 16, actor); put(p, 32, 1, 8);
    put(p, 88, AOTX_SHARED_READ_HEAD); unsigned h = AOTX_SERVICE_HEAD;
    memcpy(p.data()+h, AOTX_SHARED_MAGIC, 8); put(p, h+8, kind); put(p, h+16, 77);
    put(p, h+32, target); put(p, h+48, parent); put(p, h+80, 64); return p;
}
__global__ void aotx_shared_test_reset(unsigned count)
{
    aotx_sched.held = 0; aotx_sched.start_ns = 1; aotx_seam.replaying = 0;
    aotx_live.ready = 1; aotx_live.fatal = 0; aotx_live.phase = AOTX_LIVE_IDLE; aotx_live.received = 0;
    aotx_live_store = {}; aotx_service_put(aotx_live_store.lineage, 77, 4);
    aotx_model_load.resident[AOTX_MODEL_LANGUAGE].active = 1;
    aotx_model_load.resident[AOTX_MODEL_LANGUAGE].body.digest[0] = 99;
    aotx_model_wrap[AOTX_MODEL_LANGUAGE].usable = 1;
    for (unsigned i = 0; i < count; ++i) {
        aotx_service_grant &g = aotx_service.grants[i]; g = {};
        aotx_service_put(g.principal, i+1, 4); g.revision = 1; g.actions = 127;
        g.models = 1u << AOTX_MODEL_LANGUAGE; g.pages = 64; g.tokens = 32; g.requests = 2;
    }
}
__global__ void aotx_shared_test_ack(void)
{
    unsigned char incarnation[16] = {8}, digest[32] = {9};
    aotx_shared_ack(aotx_seam.dev.tail, 1, incarnation, 17, digest);
}
__global__ void aotx_shared_test_result(unsigned index, unsigned mode, unsigned *out)
{
    aotx_shared_receipt &r = aotx_shared.receipts[index];
    if (mode == 0) {
        r.phase = AOTX_SHARED_RUNNING;
        unsigned char bytes[8] = {'r', (unsigned char)r.actor[0], 0xc3, 0xa9, ':', 'o', 'k', '\n'};
        *out = aotx_shared_output(index, bytes, 8);
    } else *out = aotx_shared_complete(index, 200, r.actor[0]+8, 3, 1);
}
__global__ void aotx_shared_test_replay(const unsigned char *records, unsigned count, unsigned *out, unsigned bad = 0)
{
    aotx_seam.replaying = 1;
    for (unsigned i = 0; i < count && !aotx_shared.fatal; ++i) {
        const aotx_record_header *h = (const aotx_record_header *)(records + (size_t)i*AOTX_SLOT_BYTES);
        if (h->type == AOTX_SHARED_RECORD) {
            unsigned char body[192];
            aotx_service_bytes(body, (const unsigned char *)h+AOTX_HEADER_BYTES, h->body_len);
            unsigned offset = aotx_shared_u32(body+28), target = AOTX_SHARED_ADMIT_HEAD+144;
            if (bad && aotx_shared_u64(body+8) == 1 && offset <= target && target < offset+h->body_len-32)
                body[32+target-offset] = 1;
            aotx_shared_part(body, h->body_len, h->seq);
        }
    }
    *out = aotx_shared_restore_end(); aotx_seam.replaying = 0;
}
struct fixture {
    aotx_shared_state shared = {}; aotx_service_state service = {}; aotx_seam_state seam = {};
    aotx_service_mailbox *mailbox = nullptr; unsigned *result = nullptr;
    fixture(unsigned n) {
        shared.participant_capacity = n+8; shared.space_capacity = n*2+8; shared.member_capacity = n+8;
        shared.conversation_capacity = n*2+8; shared.receipt_capacity = n*8+32; shared.enabled = 1;
#define ALLOCATE(field, count) cu(cudaMalloc(&shared.field, (count)*sizeof(*shared.field))); cu(cudaMemset(shared.field, 0, (count)*sizeof(*shared.field)))
        ALLOCATE(participants, shared.participant_capacity); ALLOCATE(spaces, shared.space_capacity);
        ALLOCATE(members, shared.member_capacity); ALLOCATE(conversations, shared.conversation_capacity);
        ALLOCATE(receipts, shared.receipt_capacity);
#undef ALLOCATE
        cu(cudaHostAlloc(&mailbox, AOTX_SERVICE_CHANNELS*sizeof(*mailbox), cudaHostAllocMapped));
        memset(mailbox, 0, AOTX_SERVICE_CHANNELS*sizeof(*mailbox)); cu(cudaHostGetDevicePointer(&service.mailbox, mailbox, 0));
        cu(cudaMalloc(&service.frames, (size_t)AOTX_SERVICE_CHANNELS*AOTX_SERVICE_FRAME));
        cu(cudaMalloc(&service.ready, AOTX_SERVICE_CHANNELS*sizeof(unsigned))); cu(cudaMemset(service.ready, 0, AOTX_SERVICE_CHANNELS*sizeof(unsigned)));
        cu(cudaMalloc(&service.grants, (n+1)*sizeof(*service.grants))); service.grant_count = n+1; service.enabled = 1; service.epoch = 17;
        cu(cudaMalloc(&seam.dev.base, 32768ull*AOTX_SLOT_BYTES)); cu(cudaMemset(seam.dev.base, 0, 32768ull*AOTX_SLOT_BYTES));
        seam.dev.slot_count = 32768; seam.dev.mask = 32767; seam.apply.state_hash = AOTX_FNV_BASIS;
        cu(cudaMalloc(&result, sizeof(unsigned)));
        cu(cudaMemcpyToSymbol(aotx_shared, &shared, sizeof(shared))); cu(cudaMemcpyToSymbol(aotx_service, &service, sizeof(service)));
        cu(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof(seam))); aotx_shared_test_reset<<<1,1>>>(n+1); cu(cudaDeviceSynchronize());
    }
    ~fixture() {
        cudaFree(shared.participants); cudaFree(shared.spaces); cudaFree(shared.members); cudaFree(shared.conversations); cudaFree(shared.receipts);
        cudaFree(service.frames); cudaFree(service.ready); cudaFree(service.grants); cudaFreeHost(mailbox); cudaFree(seam.dev.base); cudaFree(result);
    }
    void send(unsigned channel, const std::vector<unsigned char> &p) {
        memcpy(mailbox[channel].bytes, p.data(), p.size()); mailbox[channel].length = p.size();
        __atomic_store_n(&mailbox[channel].state, 1ull, __ATOMIC_RELEASE);
    }
    void tick() {
        aotx_service_copy<<<AOTX_SERVICE_CHANNELS,256>>>(); aotx_service_admit<<<1,1>>>(); aotx_shared_emit<<<1,1>>>(); cu(cudaDeviceSynchronize());
    }
    unsigned status(unsigned channel) { check(mailbox[channel].state == 2, "complete mapped response"); return (unsigned)aotx_service_get(mailbox[channel].bytes+8, 4); }
    void batch(const std::vector<std::vector<unsigned char>> &commands, unsigned expected) {
        std::vector<bool> done(commands.size()); unsigned completed = 0;
        for (unsigned turn = 0; turn < commands.size()*8+8 && completed < commands.size(); ++turn) {
            for (unsigned i = 0; i < commands.size(); ++i) if (!done[i]) send(i+1, commands[i]);
            tick();
            for (unsigned i = 0; i < commands.size(); ++i) if (!done[i]) {
                unsigned code = status(i+1);
                if (expected == 202 && code == 429) continue;
                check(code == expected, "bounded command batch status"); done[i] = true; ++completed;
            }
        }
        check(completed == commands.size(), "every requested row reaches a response");
        for (unsigned i = 0; i < 8; ++i) { aotx_shared_emit<<<1,1>>>(); cu(cudaDeviceSynchronize()); }
    }
    std::vector<aotx_shared_receipt> receipts() {
        std::vector<aotx_shared_receipt> rows(shared.receipt_capacity);
        cu(cudaMemcpy(rows.data(), shared.receipts, rows.size()*sizeof(rows[0]), cudaMemcpyDeviceToHost)); return rows;
    }
    unsigned value() { unsigned v; cu(cudaMemcpy(&v, result, sizeof(v), cudaMemcpyDeviceToHost)); return v; }
};
#endif
