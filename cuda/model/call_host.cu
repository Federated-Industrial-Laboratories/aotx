/* Purpose: Give the callers of the module one pass for each kind of result.
 * Owns: Nothing; the buffers and the graph belong to model_host.cu.
 * Launch shape: Host glue only; the graph holds the kernels.
 * Lifetime: One call. */
#include <stdio.h>
#include <string.h>

#include "boot/check.h"
#include "model/forward.cuh"

/* Read the offset table of a batch and fill the parts of the call block that every pass
 * sets. The table comes back from the device, because the caller states the batch there.
 * The return is zero when every sequence holds at least one token. */
static int aotx_call_set(aotx_model_run *set, const int *ids, const unsigned int *offset,
                         unsigned int seqs, const unsigned int *agent)
{
    unsigned int rows[AOTX_MODEL_MAX_SEQS + 1u];
    memset(set, 0, sizeof *set);
    if (seqs == 0u || seqs > AOTX_MODEL_MAX_SEQS) {
        fprintf(stderr, "a pass of %u sequences is outside the bounds\n", seqs);
        return 1;
    }
    aotx_check_runtime(cudaMemcpy(rows, offset, (seqs + 1u) * sizeof *rows,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    for (unsigned int s = 0u; s < seqs; ++s) {
        if (rows[s + 1u] <= rows[s]) {
            fprintf(stderr, "the sequence %u of the pass holds no token\n", s);
            return 1;
        }
    }
    set->ids = ids;
    set->offset = offset;
    set->agent = agent;
    set->seqs = seqs;
    set->tokens = rows[seqs];
    set->top_p = 1.0f;
    return 0;
}

int aotx_model_prefill(unsigned int role, const int *ids, const unsigned int *offset,
                       unsigned int seqs, const unsigned int *agent, float *logits,
                       float *pooled)
{
    aotx_model_run set;
    if (aotx_call_set(&set, ids, offset, seqs, agent) != 0) {
        return 1;
    }
    set.logits = logits;
    set.pooled = pooled;

    /* A caller which asks for logits asks for the row of every token. A caller which asks
     * for a pooled row takes the last row of each sequence. */
    set.select = (logits != 0) ? AOTX_MODEL_ROWS_ALL : AOTX_MODEL_ROWS_LAST;
    set.rows = (set.select == AOTX_MODEL_ROWS_ALL) ? set.tokens : seqs;
    return aotx_model_launch(role, &set);
}

int aotx_model_rerank(const int *ids, const unsigned int *offset, unsigned int seqs,
                      const unsigned int *agent, float *score)
{
    aotx_model_run set;
    if (aotx_call_set(&set, ids, offset, seqs, agent) != 0) {
        return 1;
    }
    set.score = score;
    set.select = AOTX_MODEL_ROWS_LAST;
    set.rows = seqs;
    return aotx_model_launch(AOTX_MODEL_RERANKER, &set);
}

int aotx_model_sample(unsigned int role, const int *ids, const unsigned int *offset,
                      unsigned int seqs, const unsigned int *agent,
                      const aotx_model_how *how, int *token, unsigned int *draw,
                      unsigned long long *seed)
{
    aotx_model_run set;
    if (aotx_call_set(&set, ids, offset, seqs, agent) != 0) {
        return 1;
    }
    set.token = token;
    set.draw = draw;
    set.select = AOTX_MODEL_ROWS_LAST;
    set.rows = seqs;
    set.seed = how->seed;
    set.top_k = how->top_k;
    set.top_p = how->top_p;
    set.temperature = how->temperature;
    int state = aotx_model_launch(role, &set);
    if (state == 0 && seed != 0) {
        *seed = how->seed;
    }
    return state;
}
