/* Purpose: Verify explicit correction targets through live token admission and sampling.
 * Owns: Independent target tokens, finite logits and complete parser checks.
 * Launch shape: N=1 and N=64 after the real live source table and index are ready.
 * Lifetime: One generation prefix; no model weights or semantic quality claim. */
#ifndef AOTX_TEST_SOURCE_MASK_FIXTURE_H
#define AOTX_TEST_SOURCE_MASK_FIXTURE_H
#include "cognitive/intake_token.cuh"

__global__ void aotx_source_mask_prepare(const unsigned char *prefix, const unsigned *lengths,
    unsigned *out, float *head, int *tokens, unsigned *agents, unsigned n) {
    unsigned i = threadIdx.x;
    if (!i) {
        aotx_model_wrap[AOTX_MODEL_LANGUAGE].end_count = 0;
        aotx_model[AOTX_MODEL_LANGUAGE].vocab = 5; aotx_model_space[AOTX_MODEL_LANGUAGE].head = head;
        auto *call = aotx_model_call + AOTX_MODEL_LANGUAGE; *call = {};
        call->rows = n; call->agent = agents; call->token = tokens;
    }
    __syncthreads();
    if (i >= n) return;
    agents[i] = i; aotx_intake.row[i] = i + 1;
    aotx_seqs.slot[i].role = AOTX_MODEL_LANGUAGE; aotx_seqs.slot[i].stop = 99;
    auto *r = aotx_intake.rows + i; r->prefix = {}; r->status = r->count = r->bytes = 0;
    for (unsigned j = 0; j < lengths[i]; ++j) r->reply[j] = prefix[i * AOTX_INTAKE_REPLY + j];
    out[i * 10] = aotx_intake_advance(i, r->reply, lengths[i]); r->bytes = lengths[i];
    out[i * 10 + 1] = aotx_intake_index_rows[i].eligible;
    out[i * 10 + 2] = aotx_live.results[i].count; out[i * 10 + 3] = r->target_count;
    for (unsigned j = 0; j < 5; ++j) out[i * 10 + 4 + j] = aotx_intake_allows(i, j);
    auto before = r->prefix; r->prefix.used |= 1u << 2;
    out[i * 10 + 9] = !aotx_intake_allows(i, 1); r->prefix = before;
}
__global__ void aotx_source_mask_finish(unsigned target, unsigned *out, unsigned n) {
    unsigned i = threadIdx.x; if (i >= n) return;
    auto *r = aotx_intake.rows + i; auto before = r->prefix; unsigned length = r->bytes;
    const char *tail = target == 2 ? "2],[3,\"The earlier cooking plan changed.\",0]]" :
        "16],[3,\"The earlier cooking plan changed.\",0]]";
    unsigned count = 0; while (tail[count]) ++count;
    for (unsigned j = 0; j < count; ++j) r->reply[length + j] = tail[j];
    out[i] = aotx_intake_advance(i, r->reply + length, count); r->bytes += count;
    out[n + i] = !aotx_intake_parse(i);
    r->prefix = before; r->bytes = length; r->count = 0;
}
static void aotx_source_mask_check(const std::vector<std::string> &prefixes, unsigned expected) {
    unsigned n = prefixes.size();
    aotx_intake_index<<<n,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    unsigned char *raw, *prefix; unsigned long long *offset; unsigned *lengths, *out, *agents;
    float *head; int *tokens;
    AOTX_CUDA(cudaMallocManaged(&raw, 7)); memcpy(raw, "1216170", 7);
    AOTX_CUDA(cudaMallocManaged(&offset, 6 * sizeof(*offset)));
    unsigned long long positions[] = {0,1,2,4,6,7}; memcpy(offset, positions, sizeof(positions));
    aotx_text_vocab vocab = {}; vocab.tokens = 5; vocab.token_at = offset; vocab.token_bytes = raw;
    AOTX_CUDA(cudaMemcpyToSymbol(aotx_text_vocab_table, &vocab, sizeof(vocab)));
    AOTX_CUDA(cudaMallocManaged(&prefix, n * AOTX_INTAKE_REPLY));
    AOTX_CUDA(cudaMallocManaged(&lengths, n * sizeof(*lengths)));
    AOTX_CUDA(cudaMallocManaged(&out, n * 10 * sizeof(*out)));
    AOTX_CUDA(cudaMallocManaged(&agents, n * sizeof(*agents)));
    AOTX_CUDA(cudaMallocManaged(&tokens, n * sizeof(*tokens)));
    AOTX_CUDA(cudaMallocManaged(&head, n * 5 * sizeof(*head)));
    for (unsigned i = 0; i < n; ++i) {
        lengths[i] = prefixes[i].size(); memcpy(prefix + i * AOTX_INTAKE_REPLY, prefixes[i].data(), lengths[i]);
    }
    aotx_source_mask_prepare<<<1,64>>>(prefix, lengths, out, head, tokens, agents, n);
    AOTX_CUDA(cudaDeviceSynchronize());
    printf("target mask N=%u first_pre=%u first_targets=%u first_mask=%u target2=%u target16=%u\n",
        n, out[2], out[3], out[1], out[5], out[6]); fflush(stdout);
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(out[i * 10] && out[i * 10 + 3] == expected, "exact correction prefix uses the declared separate target table");
        aotx_check(out[i * 10 + 1] == (1u << (expected + 1)) - 2, "the live grammar includes every explicit table target");
        for (unsigned j = 0; j < 5; ++j) {
            unsigned target[] = {1,2,16,17,0};
            aotx_check(out[i * 10 + 4 + j] == (target[j] && target[j] <= expected),
                "decoded target tokens use table bounds instead of reply selection bounds");
        }
        aotx_check(out[i * 10 + 9], "an already used target remains masked");
    }
    for (unsigned target : {2u, 16u}) {
        if (target > expected) continue;
        unsigned wanted = target == 2 ? 1 : 2;
        for (unsigned i = 0; i < n; ++i) for (unsigned j = 0; j < 5; ++j)
            head[i * 5 + j] = j >= 3 ? 1000.0f : j == wanted ? 100.0f : 1.0f;
        aotx_model_pick<<<n,AOTX_MODEL_ROW_THREADS>>>(AOTX_MODEL_LANGUAGE); AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) aotx_check(tokens[i] == (int)wanted,
            "live sampling can choose each eligible target above invalid token maxima");
        aotx_source_mask_finish<<<1,64>>>(target, out, n); AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) aotx_check(out[i] && out[n + i],
            "the generated target advances to the same independently parsed correction");
    }
    cudaFree(raw); cudaFree(offset); cudaFree(prefix); cudaFree(lengths); cudaFree(out);
    cudaFree(agents); cudaFree(tokens); cudaFree(head);
}
#endif
