/* Purpose: Capture complete logit rows at fixed token positions.
 * Owns: Input batches, output rows, model and cache buffers.
 * Launch shape: Forward graphs at one or all sequence slots.
 * Lifetime: One model capture. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "kvcache/kvcache.cuh"
#include "mem/mem.cuh"
#include "model/forward.cuh"
#include "model/roles.h"

/* The file uses little-endian 32-bit words. Each group has a sequence count, a
 * step count, prefill lengths, and sequence-major token lists. Step zero consumes
 * each prefill in bounded chunks. Later steps consume one supplied token per sequence.
 * A multi-sequence group has one prefill length for all sequences. */
struct aotx_logits_group {
    uint32_t seqs, steps;
    std::vector<uint32_t> lengths;
    std::vector<std::vector<int>> ids;
};

static bool aotx_logits_word(FILE *in, uint32_t &value)
{
    unsigned char bytes[4];
    if (fread(bytes, 1u, sizeof bytes, in) != sizeof bytes) return false;
    value = (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8u
        | (uint32_t)bytes[2] << 16u | (uint32_t)bytes[3] << 24u;
    return true;
}

static bool aotx_logits_input(const char *path, uint32_t &vocab, uint32_t &rows,
                              std::vector<aotx_logits_group> &groups)
{
    FILE *in = fopen(path, "rb");
    if (in == NULL) return false;
    char magic[8];
    uint32_t count = 0u;
    bool ok = fread(magic, 1u, 8u, in) == 8u && memcmp(magic, "AOTXAC01", 8u) == 0
        && aotx_logits_word(in, vocab) && vocab > 1u && vocab <= 1048576u
        && aotx_logits_word(in, count) && count > 0u && count <= 1024u;
    rows = 0u;
    for (uint32_t g = 0u; g < count && ok; ++g) {
        aotx_logits_group group = {};
        ok = aotx_logits_word(in, group.seqs) && group.seqs > 0u
            && group.seqs <= AOTX_SLOTS && aotx_logits_word(in, group.steps)
            && group.steps > 0u && group.steps <= 64u;
        if (!ok) break;
        group.lengths.resize(group.seqs);
        group.ids.resize(group.seqs);
        for (uint32_t s = 0u; s < group.seqs && ok; ++s) {
            ok = aotx_logits_word(in, group.lengths[s]) && group.lengths[s] > 0u
                && group.lengths[s] <= AOTX_MODEL_MAX_TOKENS
                && group.lengths[s] + group.steps - 1u <= AOTX_SEQ_MAX_TOKENS;
        }
        for (uint32_t s = 1u; s < group.seqs && ok; ++s)
            ok = group.lengths[s] == group.lengths[0];
        for (uint32_t s = 0u; s < group.seqs && ok; ++s) {
            group.ids[s].resize(group.lengths[s] + group.steps - 1u);
            for (int &id : group.ids[s]) {
                uint32_t value = 0u;
                if (!aotx_logits_word(in, value) || value >= vocab) { ok = false; break; }
                id = (int)value;
            }
        }
        rows += group.seqs * group.steps;
        if (ok) groups.push_back(group);
    }
    ok = ok && rows > 0u && fgetc(in) == EOF && !ferror(in);
    return fclose(in) == 0 && ok;
}

__global__ void aotx_logits_release(unsigned int count)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot < count) aotx_kv_release(slot);
}

static bool aotx_logits_capture(unsigned int role, const aotx_logits_group &group,
                                unsigned int vocab, aotx_kv_map &pages, FILE *out,
                                uint32_t &written)
{
    int *ids = NULL;
    unsigned int *offset = NULL, *agent = NULL;
    float *logits = NULL;
    aotx_check_runtime(cudaMalloc((void **)&ids, AOTX_MODEL_MAX_TOKENS * sizeof(int)), "ids");
    aotx_check_runtime(cudaMalloc((void **)&offset, (AOTX_SLOTS + 1u) * sizeof(unsigned int)), "offset");
    aotx_check_runtime(cudaMalloc((void **)&agent, AOTX_SLOTS * sizeof(unsigned int)), "agent");
    aotx_check_runtime(cudaMalloc((void **)&logits,
        (size_t)AOTX_MODEL_MAX_TOKENS * vocab * sizeof(float)), "logits");
    std::vector<float> row(vocab);
    unsigned int agents[AOTX_SLOTS], offsets[AOTX_SLOTS + 1u];
    for (unsigned int s = 0u; s < group.seqs; ++s) agents[s] = s;
    aotx_check_runtime(cudaMemcpy(agent, agents, group.seqs * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice), "agents");
    aotx_model_forget();
    bool ok = true;
    for (unsigned int step = 0u; step < group.steps && ok; ++step) {
        unsigned int width = AOTX_MODEL_MAX_TOKENS / group.seqs;
        unsigned int parts = step == 0u ? (group.lengths[0] + width - 1u) / width : 1u;
        for (unsigned int part = 0u; part < parts && ok; ++part) {
            std::vector<int> tokens;
            offsets[0] = 0u;
            for (unsigned int s = 0u; s < group.seqs; ++s) {
                if (step == 0u) {
                    unsigned int start = part * width;
                    unsigned int end = start + width;
                    if (end > group.lengths[s]) end = group.lengths[s];
                    tokens.insert(tokens.end(), group.ids[s].begin() + start,
                                  group.ids[s].begin() + end);
                }
                else tokens.push_back(group.ids[s][group.lengths[s] + step - 1u]);
                offsets[s + 1u] = (unsigned int)tokens.size();
            }
            aotx_check_runtime(cudaMemcpy(ids, tokens.data(), tokens.size() * sizeof(int),
                                          cudaMemcpyHostToDevice), "tokens");
            aotx_check_runtime(cudaMemcpy(offset, offsets, (group.seqs + 1u) * sizeof(unsigned int),
                                          cudaMemcpyHostToDevice), "offsets");
            ok = aotx_model_pages(role, offset, group.seqs, agent) == 0;
            aotx_kv_serve(&pages, 0);
            if (ok) ok = aotx_model_prefill(role, ids, offset, group.seqs, agent, logits, NULL) == 0;
            if (ok) ok = aotx_model_faulted() == 0u;
            for (unsigned int s = 0u; s < group.seqs && ok && part + 1u == parts; ++s) {
                aotx_check_runtime(cudaMemcpy(row.data(), logits + (size_t)(offsets[s + 1u] - 1u) * vocab,
                                              vocab * sizeof(float), cudaMemcpyDeviceToHost), "row");
                for (float value : row) if (!isfinite(value)) { ok = false; break; }
                if (ok) ok = fwrite(row.data(), sizeof(float), vocab, out) == vocab;
                if (ok) ++written;
            }
        }
    }
    aotx_logits_release<<<1, AOTX_SLOTS>>>(group.seqs);
    aotx_check_runtime(cudaDeviceSynchronize(), "release");
    aotx_kv_serve(&pages, 0);
    cudaFree(logits); cudaFree(agent); cudaFree(offset); cudaFree(ids);
    return ok;
}

int main(int argc, char **argv)
{
    const uint32_t endian = 1u;
    if (argc != 5 || *(const unsigned char *)&endian != 1u || sizeof(float) != 4u) {
        fprintf(stderr, "usage: arch_logits STORE ROLE INPUT OUTPUT\n");
        return 1;
    }
    uint32_t vocab = 0u, rows = 0u, written = 0u;
    std::vector<aotx_logits_group> groups;
    unsigned int role = aotx_role_of(argv[2]);
    if (role >= AOTX_MODEL_ROLES || !aotx_model_is_language(role)
        || !aotx_logits_input(argv[3], vocab, rows, groups)) {
        fprintf(stderr, "arch_logits: invalid role or input\n");
        return 1;
    }
    FILE *out = fopen(argv[4], "wbx");
    if (out == NULL) { fprintf(stderr, "arch_logits: output did not open\n"); return 1; }
    aotx_check_runtime(cudaFree(0), "context");
    aotx_mem_map memory;
    aotx_kv_map pages;
    if (aotx_mem_reserve(&memory) != 0 || aotx_kv_open(&pages) != 0) return 1;
    bool loaded = aotx_boot_models(argv[1], argv[2], NULL) == 0;
    aotx_model_desc desc = {};
    if (loaded) aotx_check_runtime(cudaMemcpyFromSymbol(&desc, aotx_model, sizeof desc,
        (size_t)role * sizeof desc), "descriptor");
    bool opened = loaded && desc.vocab == vocab && aotx_model_open(role, AOTX_MODEL_MAX_TOKENS) == 0;
    bool ok = opened && fwrite("AOTXAR01", 1u, 8u, out) == 8u
        && fwrite(&vocab, sizeof vocab, 1u, out) == 1u
        && fwrite(&rows, sizeof rows, 1u, out) == 1u;
    for (size_t g = 0u; g < groups.size() && ok; ++g) {
        ok = aotx_logits_capture(role, groups[g], vocab, pages, out, written);
        printf("arch_logits: group %zu N=%u steps=%u rows=%u/%u %s\n", g,
               groups[g].seqs, groups[g].steps, written, rows, ok ? "ok" : "FAILED");
        fflush(stdout);
    }
    if (opened) aotx_model_shut(role);
    if (loaded) aotx_boot_models_release();
    aotx_kv_close(&pages);
    aotx_mem_release(&memory);
    ok = fclose(out) == 0 && ok && written == rows;
    printf("arch_logits: %u/%u complete rows, %u failures, 0 skips\n", written, rows, !ok);
    return ok ? 0 : 1;
}
