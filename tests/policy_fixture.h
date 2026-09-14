/* Purpose: Prepare exact policy bundles and native graph calls for regression tests.
 * Owns: Temporary assets, device arrays and graph lifetimes.
 * Launch shape: Independent policy batches and the live maintenance nodes.
 * Lifetime: One test-owned policy selection. */
#ifndef AOTX_POLICY_FIXTURE_H
#define AOTX_POLICY_FIXTURE_H
#include "maintenance_fixture.h"
#include "policy/host.h"
#include "policy/state.cuh"
#include "disk/policy/file.h"
#include <fstream>

static aotx_bytes aotx_policy_read_bytes(const char *path) {
    std::ifstream in(path, std::ios::binary);
    aotx_check((bool)in, "native test image opens");
    return aotx_bytes(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}
static void aotx_policy_patch(aotx_live_records &parts, unsigned offset, uint64_t value, unsigned bytes) {
    if (parts.empty() || !bytes || bytes > 8) {
        aotx_check(false, "an event field has a bounded nonempty width"); return;
    }
    unsigned total = aotx_get(parts.front().data() + 68, 4), visited = 0, changed = 0, altered = 0;
    if (offset > total || bytes > total - offset) {
        aotx_check(false, "the changed event field is inside the recorded event"); return;
    }
    for (auto &record : parts) {
        auto h = (const aotx_record_header *)record.data(); unsigned char *p = record.data() + 64;
        unsigned first = aotx_get(p + 8, 4), count = aotx_get(p + 12, 4);
        if (h->body_len <= AOTX_POLICY_PART || h->body_len > AOTX_BODY_BYTES || count != h->body_len - AOTX_POLICY_PART ||
            aotx_get(p + 4, 4) != total || first != visited || first > total || count > total - first) {
            aotx_check(false, "each changed event field has valid ordered fragment bounds"); return;
        }
        for (unsigned j = 0; j < bytes; ++j)
            if (offset + j >= first && offset + j - first < count) {
                unsigned char want = (unsigned char)(value >> (j * 8));
                unsigned char *cell = p + AOTX_POLICY_PART + offset + j - first;
                altered += *cell != want; *cell = want; ++changed;
            }
        visited += count;
    }
    aotx_check(visited == total && changed == bytes && altered,
        "each fault changes only its complete declared event field");
}
struct aotx_policy_asset {
    std::string directory, path, trust;
    explicit aotx_policy_asset(unsigned mode, unsigned stride = 16, const char *entry = "aotx_creator_maintenance",
        const char *image = AOTX_POLICY_TEST_PTX, unsigned format = 1, unsigned registers = 255,
        unsigned architecture = AOTX_ARCH, unsigned abi = AOTX_POLICY_ABI) {
        char folder[] = "/tmp/aotx-policy-XXXXXX";
        aotx_check(mkdtemp(folder) != nullptr, "policy directory opens");
        directory = folder; path = directory + "/policy.bin";
        aotx_policy_source source = {};
        source.config = {mode, 1, mode == AOTX_POLICY_NATIVE ? stride : 16, 0, 64, 0, 0, 0, 40, 1, 8, 0, abi};
        source.provenance = "Build-qualified CUDA maintenance source";
        source.provenance_bytes = strlen((const char *)source.provenance);
        source.license = "Apache-2.0"; source.license_bytes = 10;
        aotx_bytes code;
        if (mode == AOTX_POLICY_NATIVE) {
            code = aotx_policy_read_bytes(image); source.image = code.data(); source.image_bytes = code.size();
            source.entry = entry; source.config.architecture = architecture; source.config.registers = registers;
            source.config.format = format;
        }
        aotx_check(!aotx_policy_file_write(path.c_str(), &source), "complete policy bundle is written");
        aotx_policy_file file = {};
        aotx_check(!aotx_policy_file_read(path.c_str(), nullptr, 0, &file), "bundle inspection is inert");
        for (unsigned char c : file.digest) { trust += "0123456789abcdef"[c >> 4]; trust += "0123456789abcdef"[c & 15]; }
        aotx_policy_file_close(&file);
    }
    ~aotx_policy_asset() { unlink(path.c_str()); rmdir(directory.c_str()); }
    void open() { aotx_check(!aotx_policy_open(path.c_str(), trust.c_str()), "selected policy passes native admission"); }
};
static std::unique_ptr<aotx_policy_state> aotx_policy_read_state(void) {
    auto out = std::make_unique<aotx_policy_state>();
    AOTX_CUDA(cudaMemcpyFromSymbol(out.get(), aotx_policy, sizeof(*out))); return out;
}
struct aotx_policy_graph {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    aotx_policy_graph() {
        AOTX_CUDA(cudaStreamCreate(&stream));
        AOTX_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        aotx_check(aotx_policy_capture(stream) == 3, "live graph contains preparation, conditional work and publication");
        AOTX_CUDA(cudaStreamEndCapture(stream, &graph));
        AOTX_CUDA(cudaGraphInstantiate(&exec, graph, 0));
        size_t nodes = 0; AOTX_CUDA(cudaGraphGetNodes(graph, nullptr, &nodes));
        aotx_check(nodes == 3, "live top-level policy graph has exactly three nodes");
    }
    void tick() { AOTX_CUDA(cudaGraphLaunch(exec, stream)); AOTX_CUDA(cudaStreamSynchronize(stream)); }
    ~aotx_policy_graph() { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream); }
};
static aotx_policy_graph *aotx_policy_active_graph;
static void aotx_policy_test_hook(bool) { aotx_policy_active_graph->tick(); }
__global__ void aotx_policy_test_control(unsigned value) {
    if (!threadIdx.x) {
        aotx_policy.paused = value == 1; aotx_sched.held = value == 2;
        aotx_agents.agent[0].state = value == 3 ? AOTX_AGENT_STATE_RUN : AOTX_AGENT_STATE_IDLE;
    }
}
__global__ void aotx_policy_test_part(const unsigned char *p, unsigned n, unsigned flags, unsigned *ok) {
    if (!threadIdx.x) *ok = aotx_policy_part(p, n, flags);
}
#endif
