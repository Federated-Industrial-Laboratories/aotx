/* Purpose: Compute causal convolution and gated delta recurrence.
 * Owns: Ordered updates to the matrix state and convolution history of each slot.
 * Launch shape: Sequences, heads, and value rows run in parallel; tokens run in order.
 * Lifetime: Scratch lasts one pass; matrix state and history last until position reset. */
#include "model/forward.cuh"
#include "model/delta.cuh"
#include "model/kinds_data.h"

__device__ __forceinline__ float aotx_delta_sum(float value)
{
    for (unsigned int step = 16u; step; step >>= 1u)
        value += __shfl_down_sync(0xffffffffu, value, step);
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float aotx_delta_sigmoid(float value)
{
    return 1.0f / (1.0f + expf(-value));
}

__global__ void aotx_model_delta_conv(unsigned int role, unsigned int layer)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    const aotx_delta_work *delta = &work->delta;
    unsigned int sequence = blockIdx.x;
    unsigned int channel = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int channels = 2u * desc->delta_key_heads * desc->delta_dim + desc->delta_inner;
    if (sequence >= run->seqs || channel >= channels) return;
    unsigned int first = run->offset[sequence];
    unsigned int end = run->offset[sequence + 1u];
    if (first == end) return;
    if (run->agent[sequence] >= AOTX_SLOTS) {
        if (channel == 0u) atomicAdd(&aotx_model_faults, 1u);
        return;
    }
    unsigned int width = desc->delta_conv;
    unsigned long long index =
        ((unsigned long long)run->agent[sequence] * delta->layers + delta->layer[layer]) *
        channels * (width - 1u) + (unsigned long long)channel * (width - 1u);
    const float *weight = (const float *)(work->weights + desc->layer[layer].offset[AOTX_DELTA_CONV]);
    float history[AOTX_DELTA_CONV_MAX - 1u];
    float kernel[AOTX_DELTA_CONV_MAX];
#pragma unroll
    for (unsigned int i = 0u; i < AOTX_DELTA_CONV_MAX; ++i) {
        kernel[i] = i < width ? weight[channel * width + i] : 0.0f;
    }
#pragma unroll
    for (unsigned int i = 0u; i < AOTX_DELTA_CONV_MAX - 1u; ++i) {
        history[i] = i + 1u < width && work->base[sequence] != 0u ?
                     delta->history[index + i] : 0.0f;
    }
    for (unsigned int token = first; token < end; ++token) {
        float input = delta->qkv[(unsigned long long)token * channels + channel];
        float sum = 0.0f;
#pragma unroll
        for (unsigned int i = 0u; i < AOTX_DELTA_CONV_MAX; ++i) {
            if (i < AOTX_DELTA_CONV_MAX - 1u && i + 1u < width) sum += history[i] * kernel[i];
            if (i + 1u == width) sum += input * kernel[i];
        }
        delta->conv[(unsigned long long)token * channels + channel] = sum * aotx_delta_sigmoid(sum);
#pragma unroll
        for (unsigned int i = 0u; i < AOTX_DELTA_CONV_MAX - 1u; ++i) {
            if (i + 1u < AOTX_DELTA_CONV_MAX - 1u && i + 2u < width) history[i] = history[i + 1u];
            if (i + 2u == width) history[i] = input;
        }
    }
#pragma unroll
    for (unsigned int i = 0u; i < AOTX_DELTA_CONV_MAX - 1u; ++i) {
        if (i + 1u < width) delta->history[index + i] = history[i];
    }
}

__global__ void aotx_model_delta_qk(unsigned int role, unsigned int layer)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    const aotx_delta_work *delta = &work->delta;
    unsigned int token = blockIdx.x;
    unsigned int head = blockIdx.y;
    unsigned int lane = threadIdx.x;
    if (token >= run->tokens) return;
    unsigned int dim = desc->delta_dim;
    unsigned int channels = 2u * desc->delta_key_heads * dim + desc->delta_inner;
    float *row = delta->conv + (unsigned long long)token * channels + head * dim;
    float values[AOTX_DELTA_DIM_MAX / 32u];
    float sum = 0.0f;
#pragma unroll
    for (unsigned int i = 0u; i < AOTX_DELTA_DIM_MAX / 32u; ++i) {
        unsigned int key = i * 32u + lane;
        values[i] = key < dim ? row[key] : 0.0f;
        sum += values[i] * values[i];
    }
    float scale = 1.0f / fmaxf(sqrtf(aotx_delta_sum(sum)), desc->rms_eps);
#pragma unroll
    for (unsigned int i = 0u; i < AOTX_DELTA_DIM_MAX / 32u; ++i) {
        unsigned int key = i * 32u + lane;
        if (key < dim) row[key] = values[i] * scale;
    }
    const float *a = (const float *)(work->weights + desc->layer[layer].offset[AOTX_DELTA_A]);
    const float *dt = (const float *)(work->weights + desc->layer[layer].offset[AOTX_DELTA_DT]);
    for (unsigned int value_head = head * 32u + lane; value_head < desc->delta_heads;
         value_head += 2u * desc->delta_key_heads * 32u) {
        unsigned long long index = (unsigned long long)token * desc->delta_heads + value_head;
        float biased = delta->alpha[index] + dt[value_head];
        float softplus = fmaxf(biased, 0.0f) + log1pf(expf(-fabsf(biased)));
        delta->alpha[index] = expf(a[value_head] * softplus);
        delta->beta[index] = aotx_delta_sigmoid(delta->beta[index]);
    }
}

__global__ void aotx_model_delta_scan(unsigned int role, unsigned int layer)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    const aotx_delta_work *delta = &work->delta;
    unsigned int sequence = blockIdx.x;
    unsigned int head = blockIdx.y;
    unsigned int value = blockIdx.z * AOTX_DELTA_VALUES + threadIdx.x / 32u;
    unsigned int lane = threadIdx.x % 32u;
    unsigned int dim = desc->delta_dim;
    if (sequence >= run->seqs || value >= dim) return;
    unsigned int first = run->offset[sequence];
    unsigned int end = run->offset[sequence + 1u];
    if (first == end) return;
    if (run->agent[sequence] >= AOTX_SLOTS) {
        if (head == 0u && value == 0u && lane == 0u) atomicAdd(&aotx_model_faults, 1u);
        return;
    }
    unsigned int key_width = desc->delta_key_heads * dim;
    unsigned int channels = 2u * key_width + desc->delta_inner;
    unsigned int key_head = head % desc->delta_key_heads;
    unsigned long long index =
        (((unsigned long long)run->agent[sequence] * delta->layers + delta->layer[layer]) *
         desc->delta_heads + head) * dim * dim + value * dim;
    float state[AOTX_DELTA_DIM_MAX / 32u];
#pragma unroll
    for (unsigned int i = 0u; i < AOTX_DELTA_DIM_MAX / 32u; ++i) {
        unsigned int key = i * 32u + lane;
        state[i] = key < dim && work->base[sequence] != 0u ? delta->state[index + key] : 0.0f;
    }
    float query_scale = rsqrtf((float)dim);
    for (unsigned int token = first; token < end; ++token) {
        const float *row = delta->conv + (unsigned long long)token * channels;
        float decay = 0.0f;
        float beta = 0.0f;
        if (lane == 0u) {
            unsigned long long gate_index = (unsigned long long)token * desc->delta_heads + head;
            decay = delta->alpha[gate_index];
            beta = delta->beta[gate_index];
        }
        decay = __shfl_sync(0xffffffffu, decay, 0);
        beta = __shfl_sync(0xffffffffu, beta, 0);
        float keys[AOTX_DELTA_DIM_MAX / 32u];
        float query[AOTX_DELTA_DIM_MAX / 32u];
        float prediction = 0.0f;
#pragma unroll
        for (unsigned int i = 0u; i < AOTX_DELTA_DIM_MAX / 32u; ++i) {
            unsigned int key = i * 32u + lane;
            keys[i] = key < dim ? row[key_width + key_head * dim + key] : 0.0f;
            query[i] = key < dim ? row[key_head * dim + key] * query_scale : 0.0f;
            state[i] *= decay;
            prediction += state[i] * keys[i];
        }
        prediction = aotx_delta_sum(prediction);
        float change = beta * (row[2u * key_width + head * dim + value] - prediction);
        float output = 0.0f;
#pragma unroll
        for (unsigned int i = 0u; i < AOTX_DELTA_DIM_MAX / 32u; ++i) {
            state[i] += keys[i] * change;
            output += state[i] * query[i];
        }
        output = aotx_delta_sum(output);
        if (lane == 0u)
            delta->out[(unsigned long long)token * desc->delta_inner + head * dim + value] = output;
    }
#pragma unroll
    for (unsigned int i = 0u; i < AOTX_DELTA_DIM_MAX / 32u; ++i) {
        unsigned int key = i * 32u + lane;
        if (key < dim) delta->state[index + key] = state[i];
    }
}

__global__ void aotx_model_delta_gate(unsigned int role, unsigned int layer)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    const aotx_delta_work *delta = &work->delta;
    unsigned int token = blockIdx.x;
    unsigned int head = blockIdx.y;
    unsigned int lane = threadIdx.x;
    if (token >= run->tokens) return;
    unsigned int dim = desc->delta_dim;
    unsigned long long start = (unsigned long long)token * desc->delta_inner + head * dim;
    const float *norm = (const float *)(work->weights + desc->layer[layer].offset[AOTX_DELTA_NORM]);
    float values[AOTX_DELTA_DIM_MAX / 32u];
    float sum = 0.0f;
#pragma unroll
    for (unsigned int i = 0u; i < AOTX_DELTA_DIM_MAX / 32u; ++i) {
        unsigned int value = i * 32u + lane;
        values[i] = value < dim ? delta->out[start + value] : 0.0f;
        sum += values[i] * values[i];
    }
    float scale = rsqrtf(aotx_delta_sum(sum) / (float)dim + desc->rms_eps);
#pragma unroll
    for (unsigned int i = 0u; i < AOTX_DELTA_DIM_MAX / 32u; ++i) {
        unsigned int value = i * 32u + lane;
        if (value < dim) {
            float gate = delta->z[start + value];
            float output = values[i] * scale * norm[value] * (gate * aotx_delta_sigmoid(gate));
            delta->act[start + value] = __float2half_rn(output);
        }
    }
}
