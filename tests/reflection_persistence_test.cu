/* Purpose: Check saved task review cues and their versioned withdrawal.
 * Owns: Distinct GPU evidence and memory/file feature scans.
 * Launch shape: Actual review construction and restore at N=1 and N=64.
 * Lifetime: One test process with temporary checkpoint files. */
#include "appraisal_recall_fixture.h"
#include "reflection/build.cuh"
extern "C" {
#include "disk/runtime/appraisal.h"
}

__global__ void aotx_review_persistence_build(const aotx_cognitive_store *s,
    const unsigned char *q, const uint32_t *indices, uint32_t count,
    unsigned char *tail, aotx_cognitive_result *result) {
    aotx_review_build_block(s, q + 64, indices, count, tail, result);
}
static void aotx_review_scan(const aotx_bytes &memory) {
    uint32_t required = 0;
    aotx_check(!aotx_runtime_appraisal_scan(memory.data(), -1, 0, memory.size(), nullptr, &required),
        "valid review state passes the memory save check");
    aotx_check((required & AOTX_RUNTIME_REVIEW) != 0, "saved review state requires a compatible reader");
    FILE *file = tmpfile();
    aotx_check(file != nullptr, "the checkpoint file opens");
    if (!file) return;
    unsigned char prefix[16] = {0};
    aotx_check(fwrite(prefix, 1, sizeof(prefix), file) == sizeof(prefix), "the file prefix is written");
    aotx_check(fwrite(memory.data(), 1, memory.size(), file) == memory.size(), "the checkpoint is written");
    aotx_check(!fflush(file), "checkpoint writes reach the file");
    required = 0;
    aotx_check(!aotx_runtime_appraisal_scan(nullptr, fileno(file), sizeof(prefix), memory.size(), nullptr, &required),
        "valid review state passes the file packing check");
    aotx_check((required & AOTX_RUNTIME_REVIEW) != 0, "file inspection retains the review feature");
    fclose(file);
}
static void aotx_review_persistence(unsigned count) {
    auto source = aotx_ar_corpus(count);
    auto queries = aotx_ar_queries(count, source.rows.size());
    std::vector<uint32_t> indices(count);
    for (unsigned i = 0; i < count; ++i)
        for (unsigned j = 0; j < source.rows.size(); ++j)
            if (aotx_get(source.rows[j].data() + AOTX_CO_ID) == aotx_ar_id(i, 6)) indices[i] = j;
    aotx_device d;
    aotx_check(!d.load(source.wire(false, source.rows.size())).status, "distinct supported sources are admitted");
    uint32_t *device_indices;
    unsigned char *device_queries;
    AOTX_CUDA(cudaMalloc(&device_indices, count * sizeof(uint32_t)));
    AOTX_CUDA(cudaMalloc(&device_queries, queries.size()));
    AOTX_CUDA(cudaMemcpy(device_indices, indices.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    AOTX_CUDA(cudaMemcpy(device_queries, queries.data(), queries.size(), cudaMemcpyHostToDevice));
    aotx_review_persistence_build<<<1,64>>>(d.live, device_queries, device_indices, count, d.image, d.result);
    auto built = d.finish();
    aotx_check(!built.status && built.applied == 2 * count, "each source produces a complete review group");
    if (built.status) { cudaFree(device_indices); cudaFree(device_queries); return; }
    aotx_cognitive_apply<<<1,64>>>(d.live, d.stage, d.image, built.bytes, d.result);
    auto applied = d.finish();
    aotx_check(!applied.status && applied.applied == 2 * count, "the review batch publishes");
    if (applied.status) { cudaFree(device_indices); cudaFree(device_queries); return; }
    auto before = d.checkpoint();
    aotx_review_scan(before);
    uint64_t cut = aotx_get(before.data() + 32);
    aotx_fixture withdrawal;
    for (unsigned i = 0; i < count; ++i) {
        aotx_row row;
        memcpy(row.data(), before.data() + AOTX_COG_HEADER +
            (source.rows.size() + 2 * i + 1) * AOTX_COG_OBJECT, AOTX_COG_OBJECT);
        aotx_check(aotx_get(row.data() + AOTX_CO_KIND, 2) == AOTX_COG_REVIEW, "each withdrawal targets its own cue");
        aotx_put(row.data() + AOTX_CO_FLAGS, AOTX_COG_TOMBSTONE, 4);
        aotx_put(row.data() + AOTX_CO_VERSION, 2);
        aotx_put(row.data() + AOTX_CO_UPDATED, cut + i + 1);
        withdrawal.add(row, {});
    }
    auto withdrawn = d.load(withdrawal.wire(true, cut + 1, 7), true);
    aotx_check(!withdrawn.status && withdrawn.applied == count, "all cue withdrawals are admitted");
    auto after = d.checkpoint();
    aotx_review_scan(after);
    aotx_check(!d.load(after).status, "withdrawn review state restores on the device");
    aotx_check(d.checkpoint() == after, "restored review bytes are identical");
    unsigned tombstones = 0;
    for (unsigned i = 0; i < aotx_get(after.data() + 20, 4); ++i) {
        auto row = after.data() + AOTX_COG_HEADER + i * AOTX_COG_OBJECT;
        if (aotx_get(row + AOTX_CO_KIND, 2) != AOTX_COG_REVIEW ||
            !(aotx_get(row + AOTX_CO_FLAGS, 4) & AOTX_COG_TOMBSTONE)) continue;
        ++tombstones;
        auto damaged = after;
        auto bad = damaged.data() + AOTX_COG_HEADER + i * AOTX_COG_OBJECT;
        uint32_t required = 0;
        aotx_put(bad + AOTX_CO_BYTES, 1);
        aotx_check(aotx_runtime_appraisal_scan(damaged.data(), -1, 0, damaged.size(), nullptr, &required) != 0,
            "a tombstone with payload bytes is refused");
        aotx_put(bad + AOTX_CO_BYTES, 0);
        aotx_put(bad + AOTX_CO_FLAGS, 0, 4);
        aotx_check(aotx_runtime_appraisal_scan(damaged.data(), -1, 0, damaged.size(), nullptr, &required) != 0,
            "an empty live cue is refused");
    }
    aotx_check(tombstones == count, "every distinct withdrawn cue is checked");
    cudaFree(device_indices); cudaFree(device_queries);
}
int main(void) {
    for (unsigned count : {1u, 64u}) aotx_review_persistence(count);
    printf("reflection persistence: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
