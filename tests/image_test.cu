/* Purpose: Check batched device image decoding against independent coefficients and pixels.
 * Owns: Distinct generated files, device buffers and one captured decoder graph.
 * Launch shape: N=1 and N=64 image jobs with varied formats and dimensions.
 * Lifetime: One test process; all image bytes remain local. */
#include "media/image.cuh"
#include "image_reference.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>

static unsigned aotx_image_checks, aotx_image_failures;
static void aotx_image_check(bool good, const char *text) {
    ++aotx_image_checks;
    if (!good) { ++aotx_image_failures; fprintf(stderr, "FAIL %s\n", text); }
}
static void aotx_image_cuda(cudaError_t rc) {
    if (rc != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(rc)); exit(1); }
}
struct aotx_image_batch {
    std::vector<aotx_image_job> rows;
    aotx_image_job *device = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    unsigned steps = 0;
    explicit aotx_image_batch(const std::vector<aotx_image_reference> &refs) : rows(refs.size()) {
        for (unsigned i = 0; i < rows.size(); ++i) {
            auto &j = rows[i]; const auto &r = refs[i];
            j.format = AOTX_IMAGE_JPEG; j.bytes = r.bytes.size();
            j.dimension_limit = 8192; j.pixel_limit = (uint64_t)r.width * r.height;
            j.rgb_bytes = j.pixel_limit * 3;
            j.coefficient_count = (uint64_t)((r.width + 15) / 16) * ((r.height + 15) / 16) * 12 * 64;
            j.plane_bytes = j.coefficient_count;
            unsigned char *source = nullptr;
            aotx_image_cuda(cudaMalloc(&source, j.bytes)); j.source = source;
            aotx_image_cuda(cudaMalloc(&j.coefficients, j.coefficient_count * sizeof(int32_t)));
            aotx_image_cuda(cudaMalloc(&j.planes, j.plane_bytes));
            aotx_image_cuda(cudaMalloc(&j.rgb, j.rgb_bytes));
            aotx_image_cuda(cudaMemcpy(source, r.bytes.data(), j.bytes, cudaMemcpyHostToDevice));
            aotx_image_cuda(cudaMemset(j.rgb, 0xA5, j.rgb_bytes));
        }
        aotx_image_cuda(cudaMalloc(&device, rows.size() * sizeof(*device)));
        aotx_image_cuda(cudaStreamCreate(&stream));
        aotx_image_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        aotx_image_capture(stream, device, (uint32_t)rows.size(), 37);
        aotx_image_cuda(cudaStreamEndCapture(stream, &graph));
        aotx_image_cuda(cudaGraphInstantiate(&exec, graph, 0));
    }
    void run() {
        aotx_image_cuda(cudaMemcpy(device, rows.data(), rows.size() * sizeof(*device), cudaMemcpyHostToDevice));
        bool done = false;
        for (steps = 0; steps < 20000 && !done; ++steps) {
            aotx_image_cuda(cudaGraphLaunch(exec, stream));
            aotx_image_cuda(cudaStreamSynchronize(stream));
            aotx_image_cuda(cudaMemcpy(rows.data(), device, rows.size() * sizeof(*device), cudaMemcpyDeviceToHost));
            done = std::all_of(rows.begin(), rows.end(), [](const auto &j) {
                return j.phase == AOTX_IMAGE_READY || j.phase == AOTX_IMAGE_REFUSED;
            });
        }
        aotx_image_check(done, "all image jobs reach a terminal state");
    }
    ~aotx_image_batch() {
        cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream); cudaFree(device);
        for (auto &j : rows) { cudaFree((void *)j.source); cudaFree(j.coefficients); cudaFree(j.planes); cudaFree(j.rgb); }
    }
};
static void aotx_image_compare(const std::vector<aotx_image_reference> &refs, const char *name) {
    aotx_image_batch batch(refs);
    auto start = std::chrono::steady_clock::now(); batch.run();
    unsigned max_rgb = 0, max_plane = 0; uint64_t coefficients = 0, pixels = 0, absolute = 0;
    for (unsigned i = 0; i < refs.size(); ++i) {
        const auto &r = refs[i]; const auto &j = batch.rows[i];
        if (j.phase != AOTX_IMAGE_READY) fprintf(stderr, "image %u phase=%u status=%u scan=%u unit=%u cursor=%llu\n",
            i, j.phase, j.status, j.scans, j.unit, (unsigned long long)j.cursor);
        aotx_image_check(j.phase == AOTX_IMAGE_READY && !j.status, "valid image completes");
        if (j.phase != AOTX_IMAGE_READY) continue;
        aotx_image_check(j.width == r.width && j.height == r.height && j.components == r.components,
                         "dimensions and component count match");
        std::vector<int32_t> coeff(j.blocks * 64);
        std::vector<unsigned char> planes(j.blocks * 64), rgb(r.rgb.size());
        aotx_image_cuda(cudaMemcpy(coeff.data(), j.coefficients, coeff.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        aotx_image_cuda(cudaMemcpy(planes.data(), j.planes, planes.size(), cudaMemcpyDeviceToHost));
        aotx_image_cuda(cudaMemcpy(rgb.data(), j.rgb, rgb.size(), cudaMemcpyDeviceToHost));
        uint64_t wrong = 0;
        for (unsigned c = 0; c < r.components; ++c) {
            const auto &channel = j.channel[c];
            for (unsigned y = 0; y < r.rows[c]; ++y)
                for (unsigned x = 0; x < r.cols[c]; ++x)
                    for (unsigned k = 0; k < 64; ++k) {
                        size_t got = ((size_t)channel.base + (size_t)y * channel.stride + x) * 64 + k;
                        size_t expected = ((size_t)y * r.cols[c] + x) * 64 + k;
                        wrong += coeff[got] != r.coefficients[c][expected]; ++coefficients;
                    }
            for (unsigned y = 0; y < r.ch[c]; ++y)
                for (unsigned x = 0; x < r.cw[c]; ++x) {
                    size_t got = (size_t)channel.base * 64 + (size_t)y * channel.stride * 8 + x;
                    unsigned error = (unsigned)abs((int)planes[got] - r.planes[c][(size_t)y * r.cw[c] + x]);
                    max_plane = std::max(max_plane, error);
                }
        }
        if (wrong) fprintf(stderr, "image %u wrong coefficients=%llu\n", i, (unsigned long long)wrong);
        aotx_image_check(!wrong, "every true quantized coefficient matches exactly");
        for (size_t p = 0; p < rgb.size(); ++p) {
            unsigned error = (unsigned)abs((int)rgb[p] - r.rgb[p]);
            max_rgb = std::max(max_rgb, error); absolute += error; ++pixels;
        }
    }
    aotx_image_check(coefficients > 0 && pixels > 0, "reference comparison applies coefficients and pixels");
    aotx_image_check(max_plane <= 1, "IDCT component error is at most one count");
    aotx_image_check(max_rgb <= 3, "RGB error is at most three counts");
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printf("%s N=%zu steps=%u coefficients=%llu bytes=%llu plane_max=%u rgb_max=%u rgb_mean=%.8f seconds=%.6f\n",
        name, refs.size(), batch.steps, (unsigned long long)coefficients, (unsigned long long)pixels,
        max_plane, max_rgb, pixels ? (double)absolute / pixels : 0, elapsed);
}
static size_t aotx_image_marker(const std::vector<unsigned char> &bytes, unsigned marker) {
    for (size_t i = 0; i + 1 < bytes.size(); ++i)
        if (bytes[i] == 255 && bytes[i + 1] == marker) return i;
    return bytes.size();
}
static void aotx_image_guards(unsigned n) {
    auto original = aotx_image_reference_read(aotx_image_encode(7, 39, 35, 2, true, 2));
    std::vector<aotx_image_reference> refs(n, original);
    for (unsigned i = 0; i < n; ++i) {
        auto &b = refs[i].bytes;
        unsigned which = i % 10;
        if (which == 0) b[0] = 0;
        if (which == 1) b.resize(b.size() - 2);
        if (which == 2) b[aotx_image_marker(b, 194) + 4] = 12;
        if (which == 3) b[aotx_image_marker(b, 196) + 5] = 2;
        if (which == 4) b[aotx_image_marker(b, 219) + 5] = 0;
        if (which == 5) b[aotx_image_marker(b, 218) + 6] = 255;
        if (which == 6) b[aotx_image_marker(b, 208) + 1] = 209;
        if (which == 7) b.insert(b.begin() + 2, {255,226,0,2});
    }
    aotx_image_batch batch(refs);
    for (unsigned i = 0; i < n; ++i) {
        if (i % 10 == 8) batch.rows[i].pixel_limit = 1;
        if (i % 10 == 9) batch.rows[i].cancel = 1;
    }
    batch.run();
    for (const auto &j : batch.rows) {
        aotx_image_check(j.phase == AOTX_IMAGE_REFUSED && j.status, "malformed or cancelled input is refused");
        std::vector<unsigned char> rgb(j.rgb_bytes);
        aotx_image_cuda(cudaMemcpy(rgb.data(), j.rgb, rgb.size(), cudaMemcpyDeviceToHost));
        aotx_image_check(std::all_of(rgb.begin(), rgb.end(), [](unsigned char v) { return v == 0xA5; }),
                         "refusal publishes no partial RGB pixels");
    }
}
static void aotx_image_raw(unsigned n) {
    std::vector<aotx_image_reference> refs(n);
    for (unsigned i = 0; i < n; ++i) {
        auto &r = refs[i]; r.width = 3 + i; r.height = 5 + i % 7;
        r.bytes.resize((size_t)r.width * r.height * 3);
        for (size_t k = 0; k < r.bytes.size(); ++k) r.bytes[k] = (unsigned char)(k * 17 + i);
    }
    aotx_image_batch batch(refs);
    for (unsigned i = 0; i < n; ++i) {
        batch.rows[i].format = AOTX_IMAGE_RGB8;
        batch.rows[i].width = refs[i].width; batch.rows[i].height = refs[i].height;
    }
    batch.run();
    for (unsigned i = 0; i < n; ++i) {
        std::vector<unsigned char> rgb(refs[i].bytes.size());
        aotx_image_cuda(cudaMemcpy(rgb.data(), batch.rows[i].rgb, rgb.size(), cudaMemcpyDeviceToHost));
        aotx_image_check(batch.rows[i].phase == AOTX_IMAGE_READY && rgb == refs[i].bytes,
                         "raw RGB preserves distinct bytes exactly");
    }
}
int main(int argc, char **argv) {
    if (argc == 2) {
        std::ifstream file(argv[1], std::ios::binary);
        std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (bytes.empty()) return 2;
        aotx_image_compare({aotx_image_reference_read(std::move(bytes))}, "file");
    } else {
        for (unsigned i = 0; i < 8; ++i)
            aotx_image_compare({aotx_image_reference_read(aotx_image_encode(i, 31 + i, 27 + i, i % 4, i / 4, i % 3))}, "single");
        std::vector<aotx_image_reference> refs;
        for (unsigned i = 0; i < 64; ++i)
            refs.push_back(aotx_image_reference_read(aotx_image_encode(i, 1 + i * 7 % 96, 1 + i * 11 % 80,
                i % 4, (i / 4) % 2, i % 5)));
        aotx_image_compare(refs, "batch");
        std::reverse(refs.begin(), refs.end()); aotx_image_compare(refs, "permuted");
        aotx_image_guards(1); aotx_image_guards(64); aotx_image_raw(1); aotx_image_raw(64);
    }
    printf("%u checks, %u failures\n", aotx_image_checks, aotx_image_failures);
    return aotx_image_failures ? 1 : 0;
}
