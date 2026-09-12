/* Purpose: Apply affine LayerNorm and two-axis rotary positions to image rows.
 * Owns: The caller supplies half inputs and Q, K and V storage.
 * Launch shape: Four warps per block, one row or head per warp.
 * Lifetime: One encoder graph step. */
#include "vision/vision.cuh"

__global__ void aotx_vision_norm(aotx_vision_job *jobs, unsigned count,
    const unsigned char *weights, const aotx_vision_desc *desc, unsigned operation)
{
    if (blockIdx.y >= count) return;
    aotx_vision_job &j = jobs[blockIdx.y];
    unsigned phase = operation == 0u ? AOTX_VISION_PREPARE
        : operation == 1u ? AOTX_VISION_BLOCK : AOTX_VISION_MERGE;
    if (j.phase != phase) return;
    const float *scale = (const float *)(weights + (operation == 2u
        ? desc->base[AOTX_VISION_NORM]
        : desc->layer[j.layer][operation ? AOTX_VISION_LN2 : AOTX_VISION_LN1]));
    const float *bias = (const float *)(weights + (operation == 2u
        ? desc->base[AOTX_VISION_NORM_BIAS]
        : desc->layer[j.layer][operation ? AOTX_VISION_LN2_BIAS : AOTX_VISION_LN1_BIAS]));
    unsigned lane = threadIdx.x % 32u;
    for (unsigned p = blockIdx.x * 4u + threadIdx.x / 32u; p < j.patches; p += gridDim.x * 4u) {
        const float *row = j.residual + p * 768u;
        float sum = 0;
        for (unsigned c = lane; c < 768u; c += 32u) sum += row[c];
        float mean = aotx_vision_sum(sum) / 768.0f;
        float square = 0;
        for (unsigned c = lane; c < 768u; c += 32u) {
            float d = row[c] - mean; square += d * d;
        }
        float inverse = rsqrtf(aotx_vision_sum(square) / 768.0f + 1.0e-6f);
        for (unsigned c = lane; c < 768u; c += 32u) {
            float value = (row[c] - mean) * inverse * scale[c] + bias[c];
            aotx_vision_input(j, p * 768u + c, value);
        }
    }
}

__global__ void aotx_vision_rope(aotx_vision_job *jobs, unsigned count,
    const unsigned char *weights, const aotx_vision_desc *desc)
{
    if (blockIdx.y >= count) return;
    aotx_vision_job &j = jobs[blockIdx.y];
    if (j.phase != AOTX_VISION_PREPARE) return;
    const float *bias = (const float *)(weights + desc->layer[j.layer][AOTX_VISION_QKV_BIAS]);
    unsigned lane = threadIdx.x % 32u;
    for (unsigned h = blockIdx.x * 4u + threadIdx.x / 32u; h < j.patches * 36u;
         h += gridDim.x * 4u) {
        unsigned p = h / 36u, group = (h % 36u) / 12u, head = h % 12u;
        unsigned y, x;
        aotx_vision_xy(p, j.resized_width, y, x);
        unsigned c = group * 768u + head * 64u + lane, at = p * 2304u + c;
        float a = j.product[at] + bias[c], b = j.product[at + 32u] + bias[c + 32u];
        if (group < 2u) {
            float angle = (float)(lane < 16u ? y : x) * powf(10000.0f, -(float)(lane % 16u) / 16.0f);
            float cs = cosf(angle), sn = sinf(angle);
            float t = a * cs - b * sn; b = b * cs + a * sn; a = t;
        }
        j.qkv[at] = aotx_vision_finite(j, a); j.qkv[at + 32u] = aotx_vision_finite(j, b);
    }
}
