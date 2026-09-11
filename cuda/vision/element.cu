/* Purpose: Apply positions, residuals and activations to image feature batches.
 * Owns: No allocation; writes only the admitted row spans.
 * Launch shape: One grid plane per job, with independent element threads.
 * Lifetime: The current encoder graph step. */
#include "vision/vision.cuh"

__global__ void aotx_vision_position(aotx_vision_job *jobs, unsigned count,
    const unsigned char *weights, const aotx_vision_desc *desc, unsigned temporal)
{
    if (blockIdx.y >= count) return;
    aotx_vision_job &j = jobs[blockIdx.y];
    if (j.phase != AOTX_VISION_PATCH) return;
    const float *position = (const float *)(weights + desc->base[AOTX_VISION_POSITION]);
    const float *bias = (const float *)(weights + desc->base[AOTX_VISION_PATCH_BIAS]);
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
         i < j.patches * 768u; i += gridDim.x * blockDim.x) {
        float value = j.product[i];
        if (temporal) {
            unsigned y, x, e = i % 768u;
            aotx_vision_xy(i / 768u, j.resized_width, y, x);
            float py = (float)y * 47.0f / (j.resized_height / 16u - 1u);
            float px = (float)x * 47.0f / (j.resized_width / 16u - 1u);
            unsigned y0 = (unsigned)py, x0 = (unsigned)px;
            unsigned y1 = min(y0 + 1u, 47u), x1 = min(x0 + 1u, 47u);
            float dy = py - y0, dx = px - x0;
            float a = position[(y0 * 48u + x0) * 768u + e];
            float b = position[(y0 * 48u + x1) * 768u + e];
            float c = position[(y1 * 48u + x0) * 768u + e];
            float d = position[(y1 * 48u + x1) * 768u + e];
            value += j.residual[i] + bias[e];
            value += ((1.0f - dx) * a + dx * b) * (1.0f - dy)
                   + ((1.0f - dx) * c + dx * d) * dy;
        }
        j.residual[i] = aotx_vision_finite(j, value);
    }
}

__global__ void aotx_vision_activation(aotx_vision_job *jobs, unsigned count,
    const unsigned char *weights, const aotx_vision_desc *desc, unsigned merger)
{
    if (blockIdx.y >= count) return;
    aotx_vision_job &j = jobs[blockIdx.y];
    if (j.phase != (merger ? AOTX_VISION_MERGE : AOTX_VISION_BLOCK)) return;
    const float *bias = (const float *)(weights + (merger
        ? desc->base[AOTX_VISION_MERGE0_BIAS] : desc->layer[j.layer][AOTX_VISION_UP_BIAS]));
    unsigned values = (merger ? j.rows : j.patches) * 3072u;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
         i < values; i += gridDim.x * blockDim.x) {
        float x = j.product[i] + bias[i % 3072u];
        float y = merger ? 0.5f * x * (1.0f + erff(x * 0.7071067811865475f))
            : 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        aotx_vision_input(j, i, y);
    }
}

__global__ void aotx_vision_residual(aotx_vision_job *jobs, unsigned count,
    const unsigned char *weights, const aotx_vision_desc *desc, unsigned operation)
{
    if (blockIdx.y >= count) return;
    aotx_vision_job &j = jobs[blockIdx.y];
    if (j.phase != (operation == 2u ? AOTX_VISION_MERGE : AOTX_VISION_BLOCK)) return;
    unsigned width = operation == 2u ? 1024u : 768u;
    unsigned values = width * (operation == 2u ? j.rows : j.patches);
    const float *bias = (const float *)(weights + (operation == 2u
        ? desc->base[AOTX_VISION_MERGE1_BIAS]
        : desc->layer[j.layer][operation ? AOTX_VISION_DOWN_BIAS : AOTX_VISION_OUT_BIAS]));
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
         i < values; i += gridDim.x * blockDim.x) {
        float value = j.product[i] + bias[i % width];
        if (operation != 2u) value += j.residual[i];
        value = aotx_vision_finite(j, value);
        if (operation == 2u) j.features[i] = value;
        else j.residual[i] = value;
    }
}
