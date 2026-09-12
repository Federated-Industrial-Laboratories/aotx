/* Purpose: Resize RGB bytes and form spatially grouped patch rows on the device.
 * Owns: The caller supplies the horizontal, RGB and half row spans.
 * Launch shape: One grid plane per job, with independent sample threads.
 * Lifetime: The resize and patch steps of an admitted image. */
#include "vision/vision.cuh"

__device__ __forceinline__ static float cubic(float x)
{
    x = fabsf(x);
    if (x < 1.0f) return ((1.5f * x - 2.5f) * x) * x + 1.0f;
    if (x < 2.0f) return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * -0.5f;
    return 0.0f;
}

__global__ void aotx_vision_resize(aotx_vision_job *jobs, unsigned count, unsigned vertical)
{
    if (blockIdx.y >= count) return;
    aotx_vision_job &j = jobs[blockIdx.y];
    if (j.phase != (vertical ? AOTX_VISION_RESIZE_V : AOTX_VISION_RESIZE_H)) return;
    unsigned width = j.resized_width;
    unsigned height = vertical ? j.resized_height : j.height;
    unsigned in_size = vertical ? j.height : j.width;
    unsigned out_size = vertical ? j.resized_height : j.resized_width;
    float scale = (float)in_size / out_size;
    float stretch = fmaxf(scale, 1.0f), support = 2.0f * stretch;
    float inverse = scale >= 1.0f ? 1.0f / scale : 1.0f;
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < (unsigned long long)width * height * 3u; i += gridDim.x * blockDim.x) {
        unsigned c = i % 3u, x = (i / 3u) % width, y = i / (3u * width);
        float center = scale * ((vertical ? y : x) + 0.5f);
        int first = max((int)(center - support + 0.5f), 0);
        int last = min((int)(center + support + 0.5f), (int)in_size);
        float origin = first - center;
        float total = 0.0f;
        for (int t = first; t < last; ++t)
            total += cubic(((t - first) + origin + 0.5f) * inverse);
        float value = 0.0f;
        for (int t = first; t < last; ++t) {
            float w = cubic(((t - first) + origin + 0.5f) * inverse) / total;
            float v = vertical ? j.horizontal[((unsigned long long)t * width + x) * 3u + c]
                : (float)j.source[((unsigned long long)y * j.width + t) * 3u + c];
            value += w * v;
        }
        if (vertical) j.rgb[i] = (unsigned char)__float2uint_rn(fminf(255.0f, fmaxf(0.0f, value)));
        else j.horizontal[i] = value;
    }
}

__global__ void aotx_vision_patch(aotx_vision_job *jobs, unsigned count)
{
    if (blockIdx.y >= count) return;
    aotx_vision_job &j = jobs[blockIdx.y];
    if (j.phase != AOTX_VISION_PATCH) return;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
         i < j.patches * 768u; i += gridDim.x * blockDim.x) {
        unsigned p = i / 768u, e = i % 768u, y, x;
        aotx_vision_xy(p, j.resized_width, y, x);
        unsigned at = ((y * 16u + (e % 256u) / 16u) * j.resized_width
                     + x * 16u + e % 16u) * 3u + e / 256u;
        aotx_vision_input(j, i, ((float)j.rgb[at] / 255.0f - 0.5f) / 0.5f);
    }
}
