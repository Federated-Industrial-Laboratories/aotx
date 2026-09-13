/* Purpose: Check runtime publication sources and complete-work capture boundaries.
 * Owns: Distinct record batches and a real GPU checkpoint transport at N=1 and N=64.
 * Launch shape: Parallel record publication and the normal checkpoint graph nodes.
 * Lifetime: Each case releases all device and mapped buffers. */
#include "checkpoint_fixture.h"
#include "model/load.cuh"

__global__ void aotx_runtime_test_records(unsigned char *records, unsigned n, unsigned mode) {
    unsigned i = threadIdx.x;
    if (i >= n) return;
    auto *r = (aotx_record_header *)(records + i * AOTX_SLOT_BYTES);
    const char *reads[] = {"memory", "mem", "stats", "settings", "models", "agents",
                          "modules", "roles", "skills", "tools", "bus", "help"};
    const char *text = mode == 1 ? "set sample.seed 19" : reads[i % 12];
    const char *catalog[] = {"module conductor", "modules role", "modules skill", "modules tool",
                            "roles", "skills", "tools", "module-change"};
    if (mode >= 4) text = catalog[mode - 4];
    unsigned char *body = aotx_seam_body(r); unsigned bytes = 0;
    body[bytes++] = ' '; body[bytes++] = '\t';
    while (*text) body[bytes++] = (unsigned char)*text++;
    if (mode >= 4) {
        body[bytes++] = ' '; body[bytes++] = (unsigned char)('0' + i / 10);
        body[bytes++] = (unsigned char)('0' + i % 10);
    }
    body[bytes++] = ' ';
    unsigned type = mode == 3 ? AOTX_REC_SETTING : AOTX_REC_INPUT_LINE;
    unsigned flags = mode == 2 ? AOTX_FLAG_FRAGMENT : 0;
    aotx_seam_publish(r, 1000 + 100 * mode + i, AOTX_WRITER_CONSOLE, AOTX_CLASS_A, type, flags, bytes);
}
__global__ void aotx_runtime_test_work(unsigned slot, unsigned mode, unsigned on) {
    if (mode == 0) aotx_say.slot[slot].wanted = on;
    if (mode == 1) aotx_seqs.slot[slot].state = on ? AOTX_SEQ_STATE_PREFILL : AOTX_SEQ_STATE_FREE;
    if (mode == 2) { aotx_task_used[slot] = on; aotx_agents.task[slot].state = AOTX_TASK_PENDING; }
    if (mode == 3) aotx_model_load.pending_count = on;
    if (mode == 4) aotx_catalog.arriving[slot % AOTX_CATALOG_ARRIVING_MAX].import = on;
    aotx_live_bindings[slot].active = !on;
}
static uint64_t source() {
    unsigned long long value;
    AOTX_CUDA(cudaMemcpyFromSymbol(&value, aotx_runtime_dirty, sizeof(value)));
    return value;
}
static void records(unsigned char *data, unsigned n, unsigned mode) {
    aotx_runtime_test_records<<<1,64>>>(data, n, mode); AOTX_CUDA(cudaDeviceSynchronize());
}
static void aotx_runtime_test_round(unsigned n) {
    aotx_checkpoint_device d(n);
    AOTX_LIVE_CLEAR(aotx_runtime_dirty); AOTX_LIVE_CLEAR(aotx_runtime_enabled);
    AOTX_LIVE_CLEAR(aotx_model_load); AOTX_LIVE_CLEAR(aotx_task_used);
    unsigned char *data; AOTX_CUDA(cudaMalloc(&data, n * AOTX_SLOT_BYTES));
    records(data, n, 1);
    aotx_check(source() == 0, "base mode records do not request runtime publication");
    auto corpus = aotx_memory_corpus(n); uint64_t cut = corpus.rows.size();
    d.live.send(aotx_live_load_bytes(corpus.wire(false, cut)), AOTX_LIVE_LOAD);
    d.live.send(aotx_live_binding_bytes(n, cut), AOTX_LIVE_BIND);
    aotx_check(!d.live.state().status, "distinct prepared memory and bindings load");
    unsigned enabled = 1;
    AOTX_CUDA(cudaMemcpyToSymbol(aotx_runtime_enabled, &enabled, sizeof(enabled)));
    d.ring()->reserved[0] = 1;
    records(data, n, 0);
    aotx_check(source() == 0, "complete status reads do not request another runtime checkpoint");
    for (unsigned mode = 4; mode < 11; ++mode) {
        records(data, n, mode);
        aotx_check(source() == 0, "catalog detail and filtered reads do not request publication");
    }
    records(data, n, 11);
    aotx_check(source() == 2100 + n - 1, "a command prefix alone does not suppress publication");
    AOTX_LIVE_CLEAR(aotx_runtime_dirty);
    records(data, n, 1);
    uint64_t first = 1100 + n - 1;
    aotx_check(source() == first, "parallel state input records retain their greatest source sequence");
    d.publish(1); auto before = d.image(1);
    const unsigned char *slot = (const unsigned char *)(d.ring() + 1);
    aotx_check(aotx_get(slot + 24) == first, "the complete image carries its runtime source sequence");
    d.acknowledge(1);
    records(data, n, 0); d.step();
    aotx_check(d.state().runtime_durable == first && !d.state().copying && d.ring()->head == 1,
        "runtime acknowledgement is visible and status reads leave the mirror unchanged");
    records(data, n, 2);
    aotx_check(source() == 1200 + n - 1, "fragmented input remains a runtime publication source");
    records(data, n, 3); uint64_t next = 1300 + n - 1;
    aotx_check(source() == next, "a setting record remains a publication source regardless of its text");
    for (unsigned mode = 0; mode < 5; ++mode) {
        aotx_runtime_test_work<<<1,1>>>(n - 1, mode, 1); AOTX_CUDA(cudaDeviceSynchronize());
        d.step();
        aotx_check(!d.state().copying && d.ring()->head == 1,
            "pending work in an unbound slot prevents complete runtime capture");
        aotx_runtime_test_work<<<1,1>>>(n - 1, mode, 0); AOTX_CUDA(cudaDeviceSynchronize());
    }
    d.publish(2); auto after = d.image(2);
    slot = (const unsigned char *)(d.ring() + 1) + AOTX_CP_SLOT_BYTES;
    aotx_check(aotx_get(slot + 24) == next && aotx_get(after.data() + 64) == aotx_get(before.data() + 64),
        "runtime changes publish while the memory operation revision stays unchanged");
    aotx_check(before == after, "runtime publication preserves the complete unchanged memory image");
    AOTX_LIVE_CLEAR(aotx_runtime_enabled); AOTX_LIVE_CLEAR(aotx_runtime_dirty);
    cudaFree(data);
}
static void aotx_runtime_test_restore(unsigned n) {
    for (unsigned mode = 0; mode < 4; ++mode) {
        aotx_live_device d(n);
        AOTX_LIVE_CLEAR(aotx_runtime_enabled); AOTX_LIVE_CLEAR(aotx_runtime_dirty);
        auto corpus = aotx_memory_corpus(n); uint64_t cut = corpus.rows.size();
        d.send(aotx_live_load_bytes(corpus.wire(false, cut)), AOTX_LIVE_LOAD);
        auto before = d.seam(); before.in.consumed = 0;
        aotx_check(before.apply.applied_count > n && before.apply.state_hash != 14695981039346656037ull,
            "restore checks follow a nonempty distinct state batch");
        aotx_live_records records(n);
        for (unsigned i = 0; i < n; ++i) {
            records[i].fill(0);
            auto *h = (aotx_record_header *)records[i].data();
            h->magic = AOTX_WIRE_MAGIC; h->layout = AOTX_WIRE_LAYOUT; h->header_bytes = AOTX_HEADER_BYTES;
            h->boot_id = before.boot_id; h->seq = i + 1; h->writer = AOTX_WRITER_RESTORE;
            h->cls = AOTX_CLASS_B; h->type = AOTX_REC_RESTORE; h->body_len = sizeof(aotx_restore_body);
            auto *body = (aotx_restore_body *)(records[i].data() + AOTX_HEADER_BYTES);
            body->restored_boot_id = 7000 + i; body->last_tick = 42 + i;
            body->state_hash = before.apply.state_hash;
            body->replayed_count = before.apply.applied_count;
            if (mode == 1 || mode == 3) body->state_hash ^= i + 1;
            if (mode == 2 || mode == 3) body->replayed_count += i + 1;
        }
        unsigned enabled = mode != 3;
        AOTX_CUDA(cudaMemcpyToSymbol(aotx_runtime_enabled, &enabled, sizeof(enabled)));
        AOTX_CUDA(cudaMemcpyToSymbol(aotx_seam, &before, sizeof(before)));
        AOTX_CUDA(cudaMemcpy(d.in, records.data(), n * AOTX_SLOT_BYTES, cudaMemcpyHostToDevice));
        aotx_live_test_start<<<1,1>>>(n);
        aotx_seam_apply_inbound<<<AOTX_APPLY_BLOCKS,AOTX_APPLY_THREADS>>>();
        AOTX_CUDA(cudaDeviceSynchronize());
        auto after = d.seam();
        aotx_check(after.in.consumed == n, "the complete restore report batch reaches the device");
        aotx_check(after.apply.rejected == before.apply.rejected + ((mode == 1 || mode == 2) ? n : 0),
            "runtime restore refuses wrong hashes or record counts and preserves base behavior");
        aotx_check(after.apply.state_hash == before.apply.state_hash &&
            after.apply.applied_count == before.apply.applied_count,
            "restore reports do not alter the authoritative state hash or count");
        AOTX_LIVE_CLEAR(aotx_runtime_enabled); AOTX_LIVE_CLEAR(aotx_runtime_dirty);
    }
}
int main(void) {
    aotx_runtime_test_round(1); aotx_runtime_test_round(64);
    aotx_runtime_test_restore(1); aotx_runtime_test_restore(64);
    printf("runtime state: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures || aotx_checks < 40 ? 1 : 0;
}
