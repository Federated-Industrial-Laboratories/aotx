/* Purpose: Compare batched native image features with independent reference tensors.
 * Owns: Model bytes, reference cases, device work spans and one captured graph.
 * Launch shape: N=1 or N=64 distinct image jobs; the case list supplies dimensions.
 * Lifetime: One test process. */
#include "vision/vision.cuh"
#include "disk/modelfile/vision.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static unsigned checks, failures;
static void check(bool good, const char *what) {
    ++checks; if (!good) { ++failures; fprintf(stderr, "FAIL %s\n", what); }
}
static void cu(cudaError_t status) {
    if (status != cudaSuccess) { fprintf(stderr, "CUDA %s\n", cudaGetErrorString(status)); exit(1); }
}
template<class T> static std::vector<T> read(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
    size_t bytes = (size_t)file.tellg();
    if (bytes % sizeof(T)) { fprintf(stderr, "invalid reference size\n"); exit(1); }
    std::vector<T> out(bytes / sizeof(T)); file.seekg(0);
    file.read((char *)out.data(), bytes);
    if (!file) exit(1);
    return out;
}
template<class T> static void save(const std::string &path, const std::vector<T> &data) {
    std::ofstream file(path, std::ios::binary); file.write((const char *)data.data(), data.size()*sizeof(T));
}
struct image_case {
    std::string path;
    unsigned width, height, seen = 0;
};
static void compare(const std::string &path, const float *device, size_t count, unsigned width) {
    auto expected = read<float>(path);
    check(expected.size() == count, "reference tensor dimensions match");
    if (expected.size() != count) return;
    std::vector<float> got(count); cu(cudaMemcpy(got.data(), device, count*sizeof(float), cudaMemcpyDeviceToHost));
    double square = 0, reference = 0, max_abs = 0, min_cos = 1;
    size_t outside = 0, nonfinite = 0;
    for (size_t row = 0; row < count; row += width) {
        double a = 0, b = 0, dot = 0;
        for (unsigned c = 0; c < width; ++c) {
            size_t i = row + c;
            if (!std::isfinite(got[i]) || !std::isfinite(expected[i])) ++nonfinite;
            double e = (double)got[i] - expected[i];
            square += e*e; reference += (double)expected[i]*expected[i];
            max_abs = std::max(max_abs, fabs(e));
            outside += fabs(e) > 0.02 + 0.01*fabs(expected[i]);
            a += (double)got[i]*got[i]; b += (double)expected[i]*expected[i]; dot += (double)got[i]*expected[i];
        }
        double cosine = (a > 0 && b > 0) ? dot / sqrt(a*b) : a == b ? 1 : 0;
        min_cos = std::min(min_cos, cosine);
    }
    double relative = reference > 0 ? sqrt(square/reference) : square == 0 ? 0 : INFINITY;
    bool good = !nonfinite && !outside && relative <= 0.005 && (width != 1024u || min_cos >= 0.9995);
    printf("%s values=%zu relative=%.9g maximum=%.9g cosine=%.9g outside=%zu finite_fail=%zu\n",
        path.c_str(), count, relative, max_abs, min_cos, outside, nonfinite);
    check(good, "native tensor meets the declared precision limits");
    if (!good) save(path + ".actual", got);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: vision_test MODEL CASES COUNT\n"); return 2; }
    unsigned count = (unsigned)strtoul(argv[3], nullptr, 10);
    if (count != 1 && count != 64) return 2;
    aotx_modelfile *file = nullptr; aotx_vision_desc descriptor{};
    if (aotx_modelfile_open(argv[1], &file) || aotx_vision_file(file, &descriptor)) {
        fprintf(stderr, "invalid vision model\n"); aotx_modelfile_close(file); return 1;
    }
    unsigned char *weights = nullptr; aotx_vision_desc *desc = nullptr;
    cu(cudaMalloc(&weights, descriptor.bytes)); cu(cudaMalloc(&desc, sizeof(descriptor)));
    cu(cudaMemcpy(desc, &descriptor, sizeof(descriptor), cudaMemcpyHostToDevice));
    std::vector<unsigned char> chunk(2u*1024u*1024u);
    for (uint64_t at = 0; at < descriptor.bytes; at += chunk.size()) {
        size_t n = (size_t)std::min<uint64_t>(chunk.size(), descriptor.bytes - at);
        if (aotx_modelfile_read(file, at, n, chunk.data())) return 1;
        cu(cudaMemcpy(weights + at, chunk.data(), n, cudaMemcpyHostToDevice));
    }
    aotx_modelfile_close(file);
    std::ifstream list(std::string(argv[2]) + "/cases.tsv");
    std::vector<image_case> cases(count); std::vector<aotx_vision_job> jobs(count);
    unsigned capacity = 0;
    for (unsigned i = 0; i < count; ++i) {
        auto &j = jobs[i]; auto &c = cases[i]; std::string name;
        if (!(list >> name >> j.width >> j.height >> c.width >> c.height >> j.max_pixels)) return 1;
        c.path = std::string(argv[2]) + "/" + name;
        auto rgb = read<unsigned char>(c.path + ".rgb");
        j.source_bytes = rgb.size(); check(j.source_bytes == (size_t)j.width*j.height*3u, "source dimensions match");
        j.patch_capacity = c.width*c.height/256u; j.feature_capacity = j.patch_capacity/4u;
        capacity = std::max(capacity, j.patch_capacity);
        j.horizontal_values = (size_t)j.height*c.width*3u; j.rgb_bytes = (size_t)c.width*c.height*3u;
        unsigned char *source = nullptr; cu(cudaMalloc(&source, rgb.size())); j.source = source;
        cu(cudaMemcpy(source, rgb.data(), rgb.size(), cudaMemcpyHostToDevice));
        cu(cudaMalloc(&j.horizontal, j.horizontal_values*sizeof(float)));
        cu(cudaMalloc(&j.rgb, j.rgb_bytes));
        cu(cudaMalloc(&j.residual, (size_t)j.patch_capacity*768u*sizeof(float)));
        cu(cudaMalloc(&j.product, (size_t)j.patch_capacity*3072u*sizeof(float)));
        cu(cudaMalloc(&j.input, (size_t)j.patch_capacity*3072u*sizeof(half)));
        cu(cudaMalloc(&j.input_low, (size_t)j.patch_capacity*3072u*sizeof(half)));
        cu(cudaMalloc(&j.qkv, (size_t)j.patch_capacity*2304u*sizeof(float)));
        cu(cudaMalloc(&j.features, (size_t)j.feature_capacity*1024u*sizeof(float)));
    }
    aotx_vision_job *device = nullptr; cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t exec;
    cu(cudaMalloc(&device, count*sizeof(*device))); cu(cudaStreamCreate(&stream));
    cu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    aotx_vision_capture(stream, device, count, weights, desc, capacity, 256);
    cu(cudaStreamEndCapture(stream, &graph)); cu(cudaGraphInstantiate(&exec, graph, 0));
    cu(cudaMemcpy(device, jobs.data(), count*sizeof(*device), cudaMemcpyHostToDevice));
    unsigned steps = 0; bool done = false;
    auto start = std::chrono::steady_clock::now();
    for (; steps < 4000 && !done; ++steps) {
        cu(cudaGraphLaunch(exec, stream)); cu(cudaStreamSynchronize(stream));
        cu(cudaMemcpy(jobs.data(), device, count*sizeof(*device), cudaMemcpyDeviceToHost));
        done = true;
        for (unsigned i = 0; i < count; ++i) {
            auto &j = jobs[i]; auto &c = cases[i];
            if (j.phase != AOTX_VISION_READY && j.phase != AOTX_VISION_REFUSED) done = false;
            if (j.phase == AOTX_VISION_PATCH && !(c.seen & 1u)) {
                check(j.resized_width == c.width && j.resized_height == c.height, "smart resize dimensions match");
                auto expected = read<unsigned char>(c.path + ".resize"); std::vector<unsigned char> got(j.rgb_bytes);
                cu(cudaMemcpy(got.data(), j.rgb, got.size(), cudaMemcpyDeviceToHost));
                check(got == expected, "resized RGB bytes match exactly");
                if (got != expected) save(c.path + ".resize.actual", got);
                c.seen |= 1u;
            }
            if (j.phase == AOTX_VISION_PREPARE && j.layer == 0 && !(c.seen & 2u)) {
                compare(c.path + ".patch.f32", j.residual, (size_t)j.patches*768u, 768); c.seen |= 2u;
            }
            if (j.phase == AOTX_VISION_PREPARE && j.layer == 1 && !(c.seen & 4u)) {
                compare(c.path + ".block0.f32", j.residual, (size_t)j.patches*768u, 768); c.seen |= 4u;
            }
            if (j.phase == AOTX_VISION_PREPARE && j.layer == 6 && !(c.seen & 8u)) {
                compare(c.path + ".block5.f32", j.residual, (size_t)j.patches*768u, 768); c.seen |= 8u;
            }
            if (j.phase == AOTX_VISION_MERGE && !(c.seen & 16u)) {
                compare(c.path + ".block11.f32", j.residual, (size_t)j.patches*768u, 768); c.seen |= 16u;
            }
            if (j.phase == AOTX_VISION_READY && !(c.seen & 32u)) {
                compare(c.path + ".features.f32", j.features, (size_t)j.rows*1024u, 1024); c.seen |= 32u;
            }
        }
    }
    check(done, "all jobs reach a terminal state");
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    for (unsigned i = 0; i < count; ++i) {
        auto &j = jobs[i];
        if (j.status) fprintf(stderr, "job=%u phase=%u status=%u\n", i, j.phase, j.status);
        check(j.phase == AOTX_VISION_READY && !j.status && cases[i].seen == 63u, "complete image feature path passes");
        cudaFree((void *)j.source); cudaFree(j.horizontal); cudaFree(j.rgb); cudaFree(j.residual);
        cudaFree(j.product); cudaFree(j.input); cudaFree(j.input_low); cudaFree(j.qkv); cudaFree(j.features);
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    cudaFree(device); cudaFree(weights); cudaFree(desc);
    printf("N=%u steps=%u seconds=%.6f checks=%u failures=%u\n", count, steps, seconds, checks, failures);
    return failures ? 1 : 0;
}
