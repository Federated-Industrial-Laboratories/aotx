/* Purpose: Check source group identity independently of text and actor equality.
 * Owns: Temporary device fixture mutations and exact restoration checks.
 * Launch shape: One independent source pair per thread at N=1 and N=64.
 * Lifetime: One test kernel; all source rows are restored before return. */
#ifndef AOTX_RECALL_GROUP_IDENTITY_H
#define AOTX_RECALL_GROUP_IDENTITY_H
#include "cognitive/recall_groups.cuh"

__global__ void aotx_group_identity(aotx_cognitive_store *s, const unsigned char *queries,
    const unsigned *indices, unsigned *answers, unsigned n) {
    unsigned i = threadIdx.x; if (i >= n) return;
    const unsigned *selected = indices + i * 4; const unsigned char *q = queries + 64 + i * AOTX_RECALL_QUERY;
    unsigned char *first = s->objects[selected[0]], *second = s->objects[selected[2]], *working = s->objects[selected[3]];
    unsigned char saved_event[256], saved_working[256], saved_actor[16];
    for (unsigned j = 0; j < 256; ++j) { saved_event[j] = second[j]; saved_working[j] = working[j]; }
    for (unsigned j = 0; j < 16; ++j) saved_actor[j] = first[AOTX_CO_SUBJECT + j];
    unsigned groups[4];
    /* Equal body locations and actors cannot join different event identities. */
    aotx_cog_put(second + AOTX_CO_OFFSET, aotx_cog_u64(first + AOTX_CO_OFFSET), 8);
    for (unsigned j = 0; j < 16; ++j) second[AOTX_CO_SUBJECT + j] = first[AOTX_CO_SUBJECT + j];
    __syncthreads();
    aotx_recall_group_labels(s, q, selected, 4, groups, 0, 0, 4096);
    __syncthreads();
    answers[i * 4] = groups[0] == 0 && groups[1] == 0 && groups[2] == 1 && groups[3] == 1;
    for (unsigned j = 0; j < 16; ++j) second[AOTX_CO_ID + j] = working[AOTX_CO_SOURCE + j] = first[AOTX_CO_ID + j];
    aotx_cog_put(second + AOTX_CO_VERSION, 2, 8); aotx_cog_put(working + AOTX_CO_SOURCE_VERSION, 2, 8);
    __syncthreads();
    aotx_recall_group_labels(s, q, selected, 4, groups, 0, 0, 4096);
    __syncthreads();
    answers[i * 4 + 1] = groups[0] == 0 && groups[1] == 0 && groups[2] == 1 && groups[3] == 1;
    aotx_cog_put(working + AOTX_CO_SOURCE_VERSION, 1, 8);
    __syncthreads();
    aotx_recall_group_labels(s, q, selected, 4, groups, 0, 0, 4096);
    __syncthreads();
    answers[i * 4 + 2] = groups[0] == 0 && groups[1] == 0 && groups[2] == UINT32_MAX && groups[3] == 0;
    for (unsigned j = 0; j < 16; ++j) first[AOTX_CO_SUBJECT + j] = 0;
    unsigned char text[512] = {};
    __syncthreads();
    unsigned bytes = aotx_recall_group_labels(s, q, selected, 4, groups, text, 0, sizeof(text));
    const char *ending = " source_actor=unknown]\n"; unsigned length = 0; while (ending[length]) ++length;
    bool unknown = bytes >= length && bytes < sizeof(text);
    for (unsigned j = 0; j < length && unknown; ++j) unknown &= text[bytes - length + j] == ending[j];
    __syncthreads();
    answers[i * 4 + 3] = unknown;
    for (unsigned j = 0; j < 256; ++j) { second[j] = saved_event[j]; working[j] = saved_working[j]; }
    for (unsigned j = 0; j < 16; ++j) first[AOTX_CO_SUBJECT + j] = saved_actor[j];
}
static void aotx_group_identities(aotx_recall_device &d, const aotx_fixture &f, const aotx_bytes &q, unsigned n) {
    std::vector<unsigned> indices, answers(n * 4);
    for (unsigned i = 0; i < n; ++i) for (unsigned part : {3u, 4u, 35u, 36u}) {
        unsigned j = 0;
        while (j < f.rows.size() && aotx_get(f.rows[j].data() + AOTX_CO_ID) != aotx_ar_id(2 * i, part)) ++j;
        aotx_check(j < f.rows.size(), "source identity fixture index exists"); indices.push_back(j);
    }
    unsigned *di, *da; AOTX_CUDA(cudaMalloc(&di, indices.size() * sizeof(unsigned)));
    AOTX_CUDA(cudaMalloc(&da, answers.size() * sizeof(unsigned)));
    AOTX_CUDA(cudaMemcpy(di, indices.data(), indices.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
    AOTX_CUDA(cudaMemcpy(d.requests, q.data(), q.size(), cudaMemcpyHostToDevice));
    aotx_group_identity<<<1,n>>>(d.live, d.requests, di, da, n);
    AOTX_CUDA(cudaMemcpy(answers.data(), da, answers.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    for (unsigned value : answers) aotx_check(value == 1, "group labels preserve exact source versions and unknown actors");
    aotx_check(d.checkpoint() == f.wire(false, f.rows.size()), "source identity controls restore the complete original state");
    cudaFree(di); cudaFree(da);
}
#endif
