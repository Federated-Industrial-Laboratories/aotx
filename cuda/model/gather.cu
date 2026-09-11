/* Purpose: Read the token rows, keep the cache positions, and gather the head rows.
 * Owns: Nothing; the buffer block and the position table hold the state.
 * Launch shape: One thread for each sequence, or one block for each row.
 * Lifetime: One pass of the forward graph. */
#include "model/blocks.cuh"
#include "model/forward.cuh"

__global__ void aotx_model_request(unsigned int role)
{
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= run->seqs) {
        return;
    }
    unsigned int agent = run->agent[s];
    if (agent >= AOTX_SLOTS) {
        return;
    }
    unsigned int end = aotx_model_seen[agent] + (run->offset[s + 1u] - run->offset[s]);
    unsigned int need = aotx_kvl_pages(&work->shape, end);
    unsigned int held = aotx_kv.count[agent];
    if (need > held) {
        aotx_kv_request(agent, need - held);
    }
}

__global__ void aotx_model_open_rows(unsigned int role)
{
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= run->seqs) {
        return;
    }
    unsigned int agent = run->agent[s];
    work->base[s] = (agent < AOTX_SLOTS) ? aotx_model_seen[agent] : 0u;
}

__global__ void aotx_model_shut_rows(unsigned int role)
{
    const aotx_model_run *run = &aotx_model_call[role];
    unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= run->seqs) {
        return;
    }
    unsigned int agent = run->agent[s];
    if (agent < AOTX_SLOTS) {
        aotx_model_seen[agent] += run->offset[s + 1u] - run->offset[s];
    }
}

__global__ void aotx_model_gather(unsigned int role)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    const void *table = aotx_block_tensor(work->weights, desc->token_embd);
    if (table == 0) {
        return;
    }

    /* One block takes a run of rows, so the grid holds the machine and not the batch. */
    for (unsigned int t = blockIdx.x; t < run->tokens; t += gridDim.x) {
        int id = run->ids[t];
        unsigned int row = (id < 0 || (unsigned int)id >= desc->vocab) ? 0u
                                                                      : (unsigned int)id;
        unsigned long long first = (unsigned long long)row * desc->hidden;
        const aotx_model_input *input = run->input ? run->input + t : 0;
        const float *feature = input ? input->feature : 0;
        bool invalid = feature && input->width != desc->hidden;
        if (invalid && !threadIdx.x) atomicAdd(&aotx_model_faults, 1u);
        for (unsigned int d = threadIdx.x; d < desc->hidden; d += blockDim.x) {
            float value = invalid ? 0.0f : feature ? feature[d]
                : aotx_block_at(table, desc->embd_type, first + d);
            work->resid[(unsigned long long)t * desc->hidden + d] = value;
            work->x[(unsigned long long)t * desc->hidden + d] = __float2half(value);
        }
    }
}

__global__ void aotx_model_select(unsigned int role)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int r = blockIdx.x;
    if (r >= run->rows) {
        return;
    }

    /* The head takes every row of the batch, or the last row of each sequence. The last row
     * of a sequence is the row before the first row of the sequence after it. */
    unsigned int src = (run->select == AOTX_MODEL_ROWS_ALL) ? r : (run->offset[r + 1u] - 1u);
    if (src >= run->tokens) {
        src = 0u;
    }
    if (threadIdx.x == 0u) {
        work->row[r] = src;
    }
    for (unsigned int d = threadIdx.x; d < desc->hidden; d += blockDim.x) {
        work->sel[(unsigned long long)r * desc->hidden + d] =
            work->xnorm[(unsigned long long)src * desc->hidden + d];
    }
}
