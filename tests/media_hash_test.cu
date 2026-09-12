/* Purpose: Check bounded device source digests against the disk digest and fixed vectors.
 * Owns: Distinct byte streams and digest states for each batch.
 * Launch shape: N=1 and N=64 streams with varied padding boundaries.
 * Lifetime: One test process. */
#include "media/hash.cuh"
extern "C" {
#include "disk/wire/diskwire.h"
}
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
static unsigned checks, failures;
static void check(bool good, const char *what) {
    ++checks; if (!good) { ++failures; fprintf(stderr, "FAIL %s\n", what); }
}
static void cu(cudaError_t rc) {
    if (rc != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(rc)); exit(1); }
}
static void run(const std::vector<std::vector<unsigned char>> &sources, unsigned quantum) {
    unsigned count = (unsigned)sources.size();
    std::vector<aotx_media_hash> jobs(count); aotx_media_hash *device;
    for (unsigned i = 0; i < count; ++i) {
        unsigned char *source; cu(cudaMalloc(&source, std::max<size_t>(1, sources[i].size())));
        if (!sources[i].empty()) cu(cudaMemcpy(source, sources[i].data(), sources[i].size(), cudaMemcpyHostToDevice));
        jobs[i].source = source; jobs[i].bytes = sources[i].size(); jobs[i].active = 1;
    }
    cu(cudaMalloc(&device, count*sizeof(*device)));
    cu(cudaMemcpy(device, jobs.data(), count*sizeof(*device), cudaMemcpyHostToDevice));
    bool done = false;
    for (unsigned step = 0; step < 2000 && !done; ++step) {
        aotx_media_hash_step<<<(count+63)/64,64>>>(device, count, quantum); cu(cudaDeviceSynchronize());
        cu(cudaMemcpy(jobs.data(), device, count*sizeof(*device), cudaMemcpyDeviceToHost));
        done = std::all_of(jobs.begin(), jobs.end(), [](const auto &j) { return j.done != 0; });
    }
    check(done, "all source digests finish within the work bound");
    for (unsigned i = 0; i < count; ++i) {
        aotx_sha256 state; unsigned char digest[32];
        aotx_sha256_init(&state); aotx_sha256_update(&state, sources[i].data(), sources[i].size());
        aotx_sha256_final(&state, digest);
        check(!jobs[i].status && !memcmp(digest, jobs[i].digest, 32), "device source digest matches the disk digest");
        if (sources[i].size() == 3 && !memcmp(sources[i].data(), "abc", 3)) {
            char text[65]; aotx_sha256_text(jobs[i].digest, text);
            check(!strcmp(text, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
                  "the fixed SHA-256 vector matches");
        }
        cudaFree((void *)jobs[i].source);
    }
    cudaFree(device);
}
int main(int argc, char **argv) {
    if (argc == 2) {
        std::ifstream f(argv[1], std::ios::binary);
        if (!f) return 1;
        std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)), {});
        run({bytes}, 128);
    } else {
        run({{'a','b','c'}}, 1);
        std::vector<std::vector<unsigned char>> sources(64);
        const unsigned edges[] = {0,1,55,56,63,64,65,119,120,127,128,129};
        for (unsigned i = 0; i < 64; ++i) {
            sources[i].resize(i < 12 ? edges[i] : 73*i);
            for (unsigned j = 0; j < sources[i].size(); ++j) sources[i][j] = (unsigned char)(i*31+j*7);
        }
        run(sources, 3); std::reverse(sources.begin(), sources.end()); run(sources, 17);
    }
    printf("checks=%u failures=%u\n", checks, failures); return failures ? 1 : 0;
}
