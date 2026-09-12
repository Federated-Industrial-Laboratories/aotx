/* Purpose: Check batched image admission, cancellation and finite work progress.
 * Owns: Device job records and sentinel spans; no model allocation is needed.
 * Launch shape: N=1 and N=64 jobs with distinct dimensions and limit failures.
 * Lifetime: One contract test process. */
#include "vision/vision.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

static unsigned checks, failures;
static void check(bool good, const char *what) {
    ++checks; if (!good) { ++failures; fprintf(stderr, "FAIL %s\n", what); }
}
static void cu(cudaError_t rc) {
    if (rc != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(rc)); exit(1); }
}
static aotx_vision_job valid(float *sentinel, unsigned i) {
    aotx_vision_job j{};
    j.width = 256u + 32u*i; j.height = 256u + 32u*(i%3u);
    j.source = (unsigned char *)sentinel; j.source_bytes = (size_t)j.width*j.height*3;
    j.max_pixels = AOTX_VISION_MAX_PIXELS;
    j.patch_capacity = j.width*j.height/256u + i;
    j.horizontal = sentinel; j.horizontal_values = j.source_bytes + 3u*i;
    j.rgb = (unsigned char *)sentinel; j.rgb_bytes = j.source_bytes + 6u*i;
    j.residual = sentinel; j.product = sentinel;
    j.input = (half *)sentinel; j.input_low = (half *)sentinel; j.qkv = sentinel;
    j.features = sentinel; j.feature_capacity = j.width*j.height/1024u + i;
    return j;
}
static void run(unsigned count) {
    float *sentinel; aotx_vision_job *device;
    cu(cudaMalloc(&sentinel, 256u*count)); cu(cudaMemset(sentinel, 0xa5, 256u*count));
    cu(cudaMalloc(&device, count*sizeof(*device)));
    std::vector<aotx_vision_job> jobs(count);
    for (unsigned i = 0; i < count; ++i) jobs[i] = valid(sentinel + 64u*i, i);
    cu(cudaMemcpy(device, jobs.data(), count*sizeof(*device), cudaMemcpyHostToDevice));
    aotx_vision_step<<<(count+63)/64,64>>>(device, count); cu(cudaDeviceSynchronize());
    cu(cudaMemcpy(jobs.data(), device, count*sizeof(*device), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < count; ++i) {
        const auto &j = jobs[i];
        check(j.phase == AOTX_VISION_RESIZE_H && !j.status, "valid jobs are admitted");
        check(j.resized_width == 256u + 32u*i && j.resized_height == 256u + 32u*(i%3u),
              "smart resize preserves the expected grid");
        check(j.patches == j.resized_width*j.resized_height/256u && j.rows == j.patches/4u,
              "feature count follows the full image grid");
    }
    for (unsigned kind = 0; kind < 13; ++kind) {
        for (unsigned i = 0; i < count; ++i) {
            jobs[i] = valid(sentinel + 64u*i, i); auto &j = jobs[i];
            if (count > 1 && (i + kind)%3u == 0) continue;
            switch (kind) {
            case 0: j.width = 0; break;
            case 1: --j.source_bytes; break;
            case 2: j.max_pixels = AOTX_VISION_MIN_PIXELS-1; break;
            case 3: j.max_pixels = AOTX_VISION_MAX_PIXELS+1; break;
            case 4: j.width = 201; j.height = 1; break;
            case 5: j.patch_capacity = 255; break;
            case 6: j.feature_capacity = 63; break;
            case 7: j.rgb_bytes = 100; break;
            case 8: j.horizontal_values = 100; break;
            case 9: j.input_low = nullptr; break;
            case 10: j.cancel = 1; break;
            case 11: j.source = nullptr; break;
            case 12: j.features = nullptr; break;
            }
        }
        cu(cudaMemcpy(device, jobs.data(), count*sizeof(*device), cudaMemcpyHostToDevice));
        aotx_vision_step<<<(count+63)/64,64>>>(device, count); cu(cudaDeviceSynchronize());
        cu(cudaMemcpy(jobs.data(), device, count*sizeof(*device), cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < count; ++i) {
            const auto &j = jobs[i];
            bool bad = count == 1 || (i + kind)%3u != 0;
            check(bad ? j.phase == AOTX_VISION_REFUSED && j.status != 0 :
                        j.phase == AOTX_VISION_RESIZE_H && j.status == 0,
                  "each mixed input has its own admission or refusal");
        }
    }
    for (unsigned phase = AOTX_VISION_RESIZE_H; phase <= AOTX_VISION_MERGE; ++phase) {
        for (unsigned i = 0; i < count; ++i) {
            jobs[i] = valid(sentinel + 64u*i, i); jobs[i].phase = phase; jobs[i].cancel = 1;
        }
        cu(cudaMemcpy(device, jobs.data(), count*sizeof(*device), cudaMemcpyHostToDevice));
        aotx_vision_step<<<(count+63)/64,64>>>(device, count);
        aotx_vision_finish<<<(count+63)/64,64>>>(device, count, 256); cu(cudaDeviceSynchronize());
        cu(cudaMemcpy(jobs.data(), device, count*sizeof(*device), cudaMemcpyDeviceToHost));
        for (auto &j : jobs) check(j.phase == AOTX_VISION_REFUSED && j.status == AOTX_VISION_CANCELLED,
                                  "cancellation stops every active encoder stage");
    }
    for (unsigned i = 0; i < count; ++i) {
        jobs[i] = valid(sentinel + 64u*i, i); jobs[i].phase = AOTX_VISION_ATTEND;
        jobs[i].patches = 256u + 4u*i; jobs[i].query = 128u + 4u*i;
    }
    cu(cudaMemcpy(device, jobs.data(), count*sizeof(*device), cudaMemcpyHostToDevice));
    aotx_vision_finish<<<(count+63)/64,64>>>(device, count, 256); cu(cudaDeviceSynchronize());
    cu(cudaMemcpy(jobs.data(), device, count*sizeof(*device), cudaMemcpyDeviceToHost));
    for (auto &j : jobs) check(j.phase == AOTX_VISION_BLOCK && j.query == 0,
                              "a partial final query slice completes once");
    std::vector<unsigned char> bytes(256u*count);
    cu(cudaMemcpy(bytes.data(), sentinel, bytes.size(), cudaMemcpyDeviceToHost));
    bool intact = true; for (auto byte : bytes) intact &= byte == 0xa5;
    check(intact, "admission and refusal leave all output spans unchanged");
    cudaFree(device); cudaFree(sentinel);
}
int main() {
    run(1); run(64); printf("checks=%u failures=%u\n", checks, failures); return failures ? 1 : 0;
}
