/* Purpose: Attend to every patch within each image without a square score allocation.
 * Owns: Two output values and online softmax state per lane.
 * Launch shape: Four query warps per block, heads in y and jobs in z.
 * Lifetime: One bounded query slice; completed rows remain in the job input span. */
#include "vision/vision.cuh"

__global__ void aotx_vision_attention(aotx_vision_job *jobs, unsigned count, unsigned quantum)
{
    if (blockIdx.z >= count) return;
    aotx_vision_job &j = jobs[blockIdx.z];
    if (j.phase != AOTX_VISION_ATTEND) return;
    unsigned local = blockIdx.x * 4u + threadIdx.x / 32u;
    unsigned p = j.query + local, lane = threadIdx.x % 32u, head = blockIdx.y;
    if (local >= quantum || p >= j.patches || head >= 12u) return;
    const float *q = j.qkv + p * 2304u + head * 64u;
    float q0 = q[lane], q1 = q[lane + 32u];
    float top = -INFINITY, mass = 0.0f, s0 = 0.0f, s1 = 0.0f;
    for (unsigned k = 0; k < j.patches; ++k) {
        const float *key = j.qkv + k * 2304u + 768u + head * 64u;
        float dot = q0 * key[lane] + q1 * key[lane + 32u];
        dot = aotx_vision_sum(dot) * 0.125f;
        float raised = fmaxf(top, dot), shift = expf(top - raised), weight = expf(dot - raised);
        const float *v = key + 768u;
        s0 = s0 * shift + weight * v[lane];
        s1 = s1 * shift + weight * v[lane + 32u];
        mass = mass * shift + weight; top = raised;
    }
    unsigned at = p * 768u + head * 64u + lane;
    aotx_vision_input(j, at, s0 / mass);
    aotx_vision_input(j, at + 32u, s1 / mass);
}
