/* Purpose: Verify coherent live checkpoints, durable pressure and exact file recovery.
 * Owns: Distinct N=1/N=64 conversations, disk fault controls and expected source bytes.
 * Launch shape: Real live admission, checkpoint and import kernels; actual CCIR file operations.
 * Lifetime: One bounded process; no language weights are required. */
#include "checkpoint_fixture.h"
extern "C" void aotx_checkpoint_fault(int write_after, int sync_after);

static void aotx_checkpoint_recover(const aotx_bytes &image, unsigned n,
    const std::vector<aotx_live_binding> &expected, uint64_t sequence, unsigned ordinal) {
    aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
    d.send(image, AOTX_CP_RESUME);
    aotx_check(d.state().ready && !d.state().status, "complete checkpoint imports into fresh slots");
    aotx_check(d.state().searches == 0, "checkpoint import performs no vector search");
    auto actual = d.bindings(n);
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(!memcmp(&actual[i], &expected[i], sizeof(actual[i])), "binding bytes and distinct focus survive restore");
        aotx_check(actual[i].ordinal == ordinal, "restored ordinal has no duplicate input");
    }
    auto q = aotx_live_query_bytes(n, sequence, ordinal + 1);
    d.send(q, AOTX_LIVE_QUERY);
    aotx_check(!d.state().status, "fresh input follows checkpoint recovery");
    auto prompt = d.prompt(n);
    for (unsigned i = 0; i < n; ++i)
        aotx_check(prompt[i].find("fact " + std::to_string(i) + " item 0") != std::string::npos,
            "restored memory reaches the real scoped prompt consumer");
}
static void aotx_checkpoint_refuse(const aotx_bytes &image, unsigned n) {
    for (unsigned mode = 0; mode < 7; ++mode) {
        aotx_bytes bad = image;
        unsigned char *last = bad.data() + AOTX_CP_HEADER + (n - 1) * AOTX_CP_ROW;
        if (mode == 0) last[24] ^= 1;
        if (mode == 1) aotx_put(last + 72, AOTX_RECALL_PINS + 1, 4);
        if (mode == 2) last[88] = 1;
        if (mode == 3) aotx_put(bad.data() + 48, aotx_get(bad.data() + 48) + 1);
        if (mode == 4) last[128 + AOTX_RECALL_QUERY + 184 + AOTX_RECALL_SELECTION] ^= 1;
        if (mode == 5) aotx_put(last, AOTX_SLOTS, 4);
        if (mode == 6) bad.pop_back();
        aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
        d.send(bad, AOTX_CP_RESUME);
        aotx_check(!d.state().ready && d.state().status, "malformed checkpoint refuses the complete import");
        for (const auto &b : d.bindings(n)) aotx_check(!b.active && !b.ordinal, "failed import changes no binding");
        uint32_t count = 1;
        AOTX_CUDA(cudaMemcpyFromSymbol(&count, aotx_live_store, sizeof(count), offsetof(aotx_cognitive_store, count)));
        aotx_check(count == 0, "failed binding validation publishes no object store");
    }
}
static void aotx_checkpoint_disk_faults(aotx_checkpoint_disk *disk, const aotx_bytes &saved) {
    auto observation = saved;
    observation[72] ^= 1;
    observation[AOTX_CP_HEADER + 128 + AOTX_RECALL_QUERY + 12] ^= 1;
    uint64_t generation = disk->view.generation;
    aotx_check(!aotx_checkpoint_file_write(disk, observation.data(), observation.size()) && disk->view.generation == generation,
        "replay search counts and capture time do not create another memory generation");
    observation[AOTX_CP_HEADER + 24] ^= 1;
    aotx_check(aotx_checkpoint_file_write(disk, observation.data(), observation.size()) == AOTX_CCIR_CHANGED,
        "a changed principal remains a conflicting snapshot");
    aotx_ccir_input inputs[AOTX_CCIR_SECTIONS] = {};
    for (uint32_t j = 0; j < disk->view.count; ++j) {
        inputs[j].section = disk->view.sections[j]; inputs[j].source = AOTX_CCIR_REUSE;
    }
    uint32_t count = disk->view.count;
    unsigned char optional[] = {1, 7, 3, 9};
    auto &in = inputs[count];
    in.section.type = AOTX_CCIR_LIVE; in.section.schema = 99; in.section.id[0] = 90;
    in.section.bytes = sizeof(optional); in.section.alignment = 8;
    in.source = AOTX_CCIR_MEMORY; in.data = optional;
    aotx_check(!aotx_ccir_writer_append(&disk->view, inputs, count + 1, &disk->view.meta, nullptr),
        "unknown optional live section remains opaque beside the required section");
    aotx_bytes image = saved;
    for (int fail = 0; fail < 3; ++fail) {
        aotx_put(image.data() + 64, aotx_get(image.data() + 64) + 1);
        uint64_t before = disk->view.generation;
        aotx_checkpoint_fault(-1, fail);
        int status = aotx_checkpoint_file_write(disk, image.data(), image.size());
        aotx_check(status == AOTX_CCIR_IO, "failed payload, commit or root sync reports no durable success");
        aotx_checkpoint_fault(-1, -1);
        aotx_ccir_view reader;
        aotx_check(!aotx_ccir_open(disk->path, nullptr, &reader), "flush failure leaves a complete readable generation");
        aotx_check(reader.generation == before + (fail == 2), "only a published root selects the new generation");
        aotx_ccir_close(&reader);
        aotx_check(!aotx_checkpoint_file_write(disk, image.data(), image.size()) && disk->view.generation == before + 1,
            "retry publishes or synchronizes the same update exactly once");
        aotx_check(disk->view.count == count + 1 && disk->view.sections[count].schema == 99,
            "mirror append preserves the unknown optional section");
    }
    aotx_checkpoint_fault(-1, 0);
    aotx_check(aotx_checkpoint_file_write(disk, image.data(), image.size()) == AOTX_CCIR_IO,
        "repeated generation still requires successful synchronization");
    aotx_checkpoint_fault(-1, -1);
    std::string moved = std::string(disk->path) + ".moved";
    aotx_check(!rename(disk->path, moved.c_str()), "test moves the leased inode");
    aotx_check(aotx_checkpoint_file_write(disk, image.data(), image.size()) == AOTX_CCIR_CHANGED,
        "an unlinked mirror pathname cannot receive a durable acknowledgement");
    aotx_check(!rename(moved.c_str(), disk->path), "test restores the leased pathname");
}
static void aotx_checkpoint_round(unsigned n) {
    char path[] = "/tmp/aotx-checkpoint-XXXXXX";
    aotx_check(mkdtemp(path) != nullptr, "checkpoint test directory opens");
    std::string file = std::string(path) + "/state.aotxccir";
    aotx_bytes saved;
    std::vector<aotx_live_binding> expected;
    uint64_t sequence = 0;
    {
        aotx_checkpoint_device d(n);
        auto corpus = aotx_memory_corpus(n);
        sequence = corpus.rows.size();
        d.live.send(aotx_live_load_bytes(corpus.wire(false, sequence)), AOTX_LIVE_LOAD);
        auto bind = aotx_live_binding_bytes(n, sequence);
        for (unsigned i = 0; i < n; ++i) aotx_put(bind.data() + 64 + i * 64 + 60, 1, 4);
        d.live.send(bind, AOTX_LIVE_BIND);
        d.live.send(aotx_live_query_bytes(n, sequence, 1), AOTX_LIVE_QUERY); sequence += 3 * n;
        d.step(); aotx_check(d.ring()->head == 0 && !d.state().copying, "queued input is not a completed checkpoint");
        d.live.prompt(n); d.step();
        aotx_check(d.ring()->head == 0, "an active prompt does not advance durability");
        d.live.idle(n); d.step();
        uint64_t first_revision = d.live.state().accepted;
        d.publish(1);
        auto first = d.image(1);
        aotx_check(aotx_get(first.data() + 64) == first_revision, "checkpoint names the captured operation revision");
        for (unsigned round = 2; round <= AOTX_MEMORY_SNAPSHOTS; ++round) {
            d.live.send(aotx_live_query_bytes(n, sequence, round), AOTX_LIVE_QUERY); sequence += 3 * n;
            aotx_check(!d.live.state().status, "reserved checkpoint capacity admits the next batch");
            d.live.prompt(n); d.live.idle(n); d.publish(round);
        }
        expected = d.live.bindings(n);
        d.live.send(aotx_live_query_bytes(n, sequence, AOTX_MEMORY_SNAPSHOTS + 1), AOTX_LIVE_QUERY);
        aotx_check(d.live.state().status == AOTX_COG_CAPACITY, "full persistence ring refuses memory production");
        for (const auto &b : d.live.bindings(n))
            aotx_check(b.ordinal == AOTX_MEMORY_SNAPSHOTS, "pressure refusal keeps every ordinal unchanged");
        aotx_check(d.image(1) == first, "unacknowledged image remains immutable across later input");
        d.ring()->ack_boot = d.ring()->boot ^ 1; d.ring()->consumed = 1; d.step();
        aotx_check(d.state().error == AOTX_COG_SEQUENCE, "another boot cannot release snapshot ownership");
        d.ring()->consumed = 0; d.ring()->ack_boot = 0;
        aotx_checkpoint_disk disk;
        aotx_check(!aotx_checkpoint_disk_open(&disk, d.transport.checkpoint_fd, file.c_str()), "disk consumer attaches");
        aotx_checkpoint_fault(0, -1);
        aotx_check(aotx_checkpoint_disk_pass(&disk) == 0 && d.ring()->consumed == 0 && d.ring()->error,
            "disk-full failure releases no unacknowledged snapshot");
        aotx_checkpoint_fault(-1, -1); disk.next_retry = 0;
        aotx_check(aotx_checkpoint_disk_pass(&disk) == (int)AOTX_MEMORY_SNAPSHOTS,
            "recovered disk publishes the complete pending batch");
        d.step();
        aotx_check(!d.state().error && d.state().durable == d.live.state().accepted,
            "GPU distinguishes and then observes the durable operation boundary");
        aotx_ccir_view other;
        aotx_check(aotx_ccir_writer_open(file.c_str(), nullptr, &other) == AOTX_CCIR_BUSY,
            "the continuous lease excludes a second writer");
        unsigned char *read = nullptr; uint32_t bytes = 0;
        aotx_check(!aotx_checkpoint_file_read(file.c_str(), &read, &bytes), "reader inspects a complete generation during service");
        if (read) { saved.assign(read, read + bytes); free(read); }
        aotx_check(saved == d.image(AOTX_MEMORY_SNAPSHOTS), "file sections reconstruct the exact published snapshot");
        uint64_t generation = disk.view.generation;
        aotx_check(!aotx_checkpoint_file_write(&disk, saved.data(), saved.size()) && disk.view.generation == generation,
            "a repeated acknowledged snapshot creates no second generation");
        aotx_checkpoint_disk_close(&disk);
        d.ring()->consumed = AOTX_MEMORY_SNAPSHOTS - 1;
        aotx_check(!aotx_checkpoint_disk_open(&disk, d.transport.checkpoint_fd, file.c_str()), "drain restarts with its retained ring");
        aotx_check(aotx_checkpoint_disk_pass(&disk) == 1 && disk.view.generation == generation,
            "restart reconciles an already committed but unacknowledged update once");
        aotx_checkpoint_disk_faults(&disk, saved);
        aotx_checkpoint_disk_close(&disk);
    }
    aotx_checkpoint_recover(saved, n, expected, sequence, AOTX_MEMORY_SNAPSHOTS);
    aotx_checkpoint_refuse(saved, n);
    unlink(file.c_str()); rmdir(path);
}
static void aotx_checkpoint_stale(unsigned n) {
    if (AOTX_MEMORY_SNAPSHOTS < 2) return;
    aotx_bytes saved;
    std::vector<aotx_live_binding> expected;
    {
        aotx_checkpoint_device d(n);
        auto corpus = aotx_memory_corpus(n); uint64_t cut = corpus.rows.size();
        d.live.send(aotx_live_load_bytes(corpus.wire(false, cut)), AOTX_LIVE_LOAD);
        d.live.send(aotx_live_binding_bytes(n, cut), AOTX_LIVE_BIND);
        d.live.send(aotx_live_query_bytes(n, cut, 1), AOTX_LIVE_QUERY);
        d.live.prompt(n); d.live.idle(n); d.step();
        if (n == 64) aotx_check(d.state().copying && !d.ring()->head, "large binding batch needs several bounded copies");
        aotx_fixture update;
        for (unsigned i = 0; i < n; ++i)
            update.add(aotx_memory_row(i, AOTX_COG_SOURCE, 999000 + i, cut + i + 1), aotx_memory_text("later source"));
        d.live.send(update.wire(true, cut + 1, 6), AOTX_LIVE_UPDATE);
        aotx_check(!d.live.state().status, "later mutation proceeds while the captured image is immutable");
        d.publish(1);
        aotx_check(aotx_get(d.image(1).data() + 48) == cut, "multi-tick publication keeps the earlier object boundary");
        d.publish(2); saved = d.image(2); expected = d.live.bindings(n);
        aotx_check(aotx_get(saved.data() + 48) == cut + n, "next snapshot includes the complete later mutation");
    }
    aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
    d.send(saved, AOTX_CP_RESUME);
    aotx_check(d.state().ready && !d.state().status, "stale context imports without a replacement search");
    auto restored = d.bindings(n);
    for (unsigned i = 0; i < n; ++i)
        aotx_check(!memcmp(&expected[i], &restored[i], sizeof(expected[i])), "stale binding bytes remain exact");
    unsigned *status; AOTX_CUDA(cudaMallocManaged(&status, n * sizeof(*status)));
    aotx_live_test_continuation<<<1,64>>>(n, status); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) aotx_check(status[i] == 0, "restored stale context cannot admit a continuation");
    cudaFree(status);
}
static void aotx_checkpoint_full(unsigned n) {
    aotx_checkpoint_device d(n);
    aotx_fixture corpus;
    for (unsigned i = 0; i < n; ++i) {
        size_t size = AOTX_COG_PAYLOAD / n + (i < AOTX_COG_PAYLOAD % n);
        corpus.add(aotx_memory_row(i, AOTX_COG_COMPONENT, 7000000 + i, i + 1), aotx_bytes(size, (unsigned char)(i + 1)));
    }
    auto object = corpus.wire(false, n);
    d.live.send(aotx_live_load_bytes(object), AOTX_LIVE_LOAD);
    d.live.send(aotx_live_binding_bytes(n, n), AOTX_LIVE_BIND);
    aotx_check(!d.live.state().status, "full payload store admits the complete binding batch");
    cudaEvent_t begin, end; AOTX_CUDA(cudaEventCreate(&begin)); AOTX_CUDA(cudaEventCreate(&end));
    float maximum = 0, total = 0; unsigned ticks = 0;
    while (d.ring()->head == 0 && ticks < AOTX_CP_BYTES / AOTX_CP_COPY + 4) {
        AOTX_CUDA(cudaEventRecord(begin)); aotx_checkpoint_capture(nullptr);
        AOTX_CUDA(cudaEventRecord(end)); AOTX_CUDA(cudaEventSynchronize(end));
        float ms = 0; AOTX_CUDA(cudaEventElapsedTime(&ms, begin, end));
        if (ms > maximum) maximum = ms;
        total += ms; ++ticks;
    }
    aotx_check(d.ring()->head == 1, "full configured payload publishes within its copy bound");
    auto image = d.image(1); size_t base = AOTX_CP_HEADER + n * AOTX_CP_ROW;
    aotx_check(image.size() == base + object.size() && !memcmp(image.data() + base, object.data(), object.size()),
        "every configured payload byte survives coherent publication");
    printf("checkpoint capacity n=%u objects=%u payload=%u image=%zu ticks=%u total_ms=%.6f max_tick_ms=%.6f\n",
        n, AOTX_COG_OBJECTS, AOTX_COG_PAYLOAD, image.size(), ticks, total, maximum);
    cudaEventDestroy(begin); cudaEventDestroy(end);
}
int main(int argc, char **argv) {
    bool capacity = argc == 1;
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--state-only"))) {
        fprintf(stderr, "usage: aotx_checkpoint_test [--state-only]\n"); return 2;
    }
    printf("checkpoint allocation image=%u staged_bindings=%zu mapped_transport=%llu\n",
        AOTX_CP_BYTES, sizeof(aotx_live_binding) * AOTX_SLOTS, (unsigned long long)((AOTX_CP_RING_BYTES + 4095) & ~4095ull));
    const unsigned batch = AOTX_SLOTS < AOTX_RECALL_BATCH ? AOTX_SLOTS : AOTX_RECALL_BATCH;
    for (unsigned n : {1u, batch}) {
        aotx_checkpoint_round(n); aotx_checkpoint_stale(n);
        if (capacity) aotx_checkpoint_full(n);
    }
    if (!capacity) printf("checkpoint: full payload capacity checks omitted\n");
    printf("checkpoint: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
