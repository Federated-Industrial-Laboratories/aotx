/* Purpose: Read the probe rows of one residual row and add the readouts to its agent.
 * Owns: Nothing; the tables and the sums live with the turn node.
 * Launch shape: The block of one residual row of the conduct kernel; the block reduces.
 * Lifetime: The whole run. */
#include "affect/affect.cuh"

#include "model/decode_state.cuh"

/* The block reduces the dot product of the row with each probe row of this layer. Every
 * warp adds with a shuffle, then one thread adds over the warps. The last prompt row gives
 * the prompt readout of the axis, and a reply row goes in the running sum. The first row
 * of the table alone counts the reply rows. Every row of the batch passes the layer of
 * that row one time, so the count is exact. The whole block makes the call, because the
 * block synchronizes here. */
__device__ void aotx_affect_readout(const aotx_model_run *run, unsigned int hidden,
                                    const float *resid, unsigned int row,
                                    unsigned int seq, unsigned int layer)
{
    __shared__ float part[AOTX_MODEL_ROW_THREADS / 32u];
    unsigned int agent = run->agent[seq];
    /* The shuffle takes whole warps, so a block that is not a multiple of 32 reads none. */
    if (agent >= AOTX_SLOTS || aotx_affect_rows.hidden != hidden
        || blockDim.x > AOTX_MODEL_ROW_THREADS || (blockDim.x & 31u) != 0u) {
        return;
    }
    unsigned int position = aotx_decode.first[agent] + (row - run->offset[seq]);
    unsigned int prompt = aotx_seqs.slot[agent].prompt;
    for (unsigned int p = 0u; p < aotx_affect_rows.count; ++p) {
        const aotx_affect_row *probe = &aotx_affect_rows.row[p];
        if (probe->layer != layer) {
            continue;
        }
        const float *direction = aotx_affect_probe + (unsigned long long)p * hidden;
        float sum = 0.0f;
        for (unsigned int x = threadIdx.x; x < hidden; x += blockDim.x) {
            sum += resid[x] * direction[x];
        }
        for (unsigned int lane = 16u; lane > 0u; lane >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, lane);
        }
        if ((threadIdx.x & 31u) == 0u) {
            part[threadIdx.x >> 5] = sum;
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            float total = 0.0f;
            for (unsigned int w = 0u; w < (blockDim.x + 31u) / 32u; ++w) {
                total += part[w];
            }
            float value = (total - probe->mean) / probe->scale;
            aotx_affect_sums *acc = &aotx_affect_acc[agent];
            if (position + 1u == prompt) {
                acc->prompt[probe->axis] = value;
            } else if (position >= prompt) {
                atomicAdd(&acc->reply_sum[probe->axis], value);
                if (p == 0u) {
                    atomicAdd(&acc->reply_rows, 1u);
                }
            }
        }
        __syncthreads();
    }
}
