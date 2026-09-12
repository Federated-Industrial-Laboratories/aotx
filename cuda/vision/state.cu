/* Purpose: Admit image jobs and advance their finite graph work.
 * Owns: Each job holds its dimensions, progress and terminal status.
 * Launch shape: One thread per job in a batch.
 * Lifetime: One encoder request. */
#include "vision/vision.cuh"

__global__ void aotx_vision_step(aotx_vision_job *jobs, unsigned count)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    aotx_vision_job &j = jobs[i];
    if (j.phase == AOTX_VISION_READY || j.phase == AOTX_VISION_REFUSED) return;
    if (j.cancel) { j.status = AOTX_VISION_CANCELLED; j.phase = AOTX_VISION_REFUSED; return; }
    if (j.phase != AOTX_VISION_NEW) return;
    j.status = 0; j.layer = 0; j.query = 0; j.rows = 0; j.patches = 0;
    if (!j.width || !j.height || !j.source || !j.horizontal || !j.rgb || !j.residual ||
        !j.product || !j.input || !j.input_low || !j.qkv || !j.features ||
        (unsigned long long)j.width * j.height > j.source_bytes / 3u ||
        j.max_pixels < AOTX_VISION_MIN_PIXELS || j.max_pixels > AOTX_VISION_MAX_PIXELS ||
        (double)max(j.width, j.height) / min(j.width, j.height) > 200.0) {
        j.status = AOTX_VISION_INVALID; j.phase = AOTX_VISION_REFUSED; return;
    }
    double h = nearbyint((double)j.height / 32.0) * 32.0;
    double w = nearbyint((double)j.width / 32.0) * 32.0;
    double original = (double)j.height * j.width;
    if (h * w > j.max_pixels) {
        double beta = sqrt(original / j.max_pixels);
        h = fmax(32.0, floor(j.height / beta / 32.0) * 32.0);
        w = fmax(32.0, floor(j.width / beta / 32.0) * 32.0);
    } else if (h * w < AOTX_VISION_MIN_PIXELS) {
        double beta = sqrt((double)AOTX_VISION_MIN_PIXELS / original);
        h = ceil(j.height * beta / 32.0) * 32.0;
        w = ceil(j.width * beta / 32.0) * 32.0;
    }
    unsigned long long pixels = (unsigned long long)h * (unsigned long long)w;
    if (!pixels || pixels > j.max_pixels || pixels / 256u > j.patch_capacity ||
        pixels / 1024u > j.feature_capacity || pixels > j.rgb_bytes / 3u ||
        (unsigned long long)j.height * (unsigned long long)w > j.horizontal_values / 3u) {
        j.status = AOTX_VISION_LIMIT; j.phase = AOTX_VISION_REFUSED; return;
    }
    j.resized_width = (unsigned)w; j.resized_height = (unsigned)h;
    j.patches = (unsigned)(pixels / 256u); j.rows = j.patches / 4u;
    j.phase = AOTX_VISION_RESIZE_H;
}

__global__ void aotx_vision_finish(aotx_vision_job *jobs, unsigned count, unsigned quantum)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    aotx_vision_job &j = jobs[i];
    if (j.phase == AOTX_VISION_READY || j.phase == AOTX_VISION_REFUSED) return;
    if (j.status) { j.phase = AOTX_VISION_REFUSED; return; }
    if (j.phase == AOTX_VISION_ATTEND) {
        j.query += min(quantum, j.patches - j.query);
        if (j.query == j.patches) { j.query = 0; j.phase = AOTX_VISION_BLOCK; }
    } else if (j.phase == AOTX_VISION_BLOCK) {
        if (++j.layer == AOTX_VISION_LAYERS) j.phase = AOTX_VISION_MERGE;
        else j.phase = AOTX_VISION_PREPARE;
    } else if (j.phase >= AOTX_VISION_RESIZE_H && j.phase <= AOTX_VISION_MERGE) {
        ++j.phase;
    }
}
