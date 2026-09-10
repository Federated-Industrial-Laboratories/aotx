/* Purpose: Verify source indexes, JSON prefixes and constrained token selection.
 * Owns: Independent substring/count oracles and malformed response expectations.
 * Launch shape: N=1 and N=64 distinct sources and vocabulary rows.
 * Lifetime: One test process without model weights. */
#include "intake_fixture.h"
#include "cognitive/intake_grammar.cuh"
#include "text/text.cuh"

__global__ void aotx_grammar_probe(const unsigned char *queries, const unsigned *lengths, unsigned *out, unsigned n) {
    unsigned i = blockIdx.x, j = threadIdx.x; if (i >= n || j >= 64) return;
    const auto *s = aotx_intake_index_rows + i; unsigned node = 0;
    for (unsigned k = 0; k < lengths[i * 64 + j] && node != UINT32_MAX; ++k)
        node = aotx_intake_next(s, node, queries[(i * 64 + j) * 2048 + k]);
    out[i * 64 + j] = node == UINT32_MAX ? 0 : s->node[node].ends;
}
__global__ void aotx_grammar_parse(const unsigned char *bytes, const unsigned *lengths, unsigned *out, unsigned n, unsigned split) {
    unsigned i = threadIdx.x; if (i >= n) return;
    auto *r = aotx_intake.rows + i; r->prefix = {}; r->bytes = lengths[i];
    for (unsigned j = 0; j < r->bytes; ++j) r->reply[j] = bytes[i * AOTX_INTAKE_REPLY + j];
    bool valid = true;
    for (unsigned j = 0; j < r->bytes && valid; j += split)
        valid = aotx_intake_advance(i, r->reply + j, min(split, r->bytes - j));
    out[i * 2] = valid && r->prefix.stage == 11;
    out[i * 2 + 1] = aotx_intake_parse(i) == 0;
}
__global__ void aotx_grammar_pick_setup(unsigned n, float *head, int *tokens, unsigned *agents, unsigned mode) {
    unsigned i = threadIdx.x;
    if (i < n) {
        agents[i] = i; aotx_intake.row[i] = mode == 1 ? 0 : i + 1;
        aotx_intake.rows[i].status = 0; aotx_intake.rows[i].prefix = {}; aotx_intake.rows[i].bytes = 0;
        if (mode == 2) aotx_intake.rows[i].prefix.stage = 11;
        aotx_seqs.slot[i].role = AOTX_MODEL_LANGUAGE; aotx_seqs.slot[i].stop = 99;
    }
    if (!i) {
        aotx_model[AOTX_MODEL_LANGUAGE].vocab = 3; aotx_model_space[AOTX_MODEL_LANGUAGE].head = head;
        auto *r = aotx_model_call + AOTX_MODEL_LANGUAGE; *r = {};
        r->rows = n; r->agent = agents; r->token = tokens;
    }
}
static std::string aotx_grammar_source(unsigned i) {
    return "Iris" + std::to_string(i) + " may not cook. Ren\xc3\xa9 says \"check\". Soup \xf0\x9f\x8d\xb2.\n";
}
static void aotx_grammar_open(aotx_intake_device &d, unsigned n, const std::vector<std::string> &sources) {
    aotx_fixture empty; d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
    auto p = aotx_intake_query(n, 0, 1);
    for (unsigned i = 0; i < n; ++i) {
        auto q = p.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
        memset(q + 4640, 0, 2048); memcpy(q + 4640, sources[i].data(), sources[i].size()); aotx_put(q + 148, sources[i].size(), 4);
    }
    d.process(aotx_live_parts(p, 4, d.next_id++), false, false);
    aotx_check(d.state().phase == AOTX_INTAKE_RUN, "source index fixture reaches the internal lease");
    aotx_intake_index<<<n,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
}
static void aotx_grammar_index(unsigned n, unsigned pattern) {
    aotx_intake_device d(n); std::vector<std::string> sources;
    unsigned seed = 37;
    for (unsigned i = 0; i < n; ++i) {
        std::string s;
        for (unsigned j = 0; j < 2048; ++j) { seed = seed * 1664525u + 1013904223u; s += pattern ? char('a' + ((seed >> 20) & 7)) : 'a'; }
        sources.push_back(s);
    }
    aotx_grammar_open(d, n, sources);
    unsigned char *queries; unsigned *lengths, *out;
    AOTX_CUDA(cudaMallocManaged(&queries, n * 64 * 2048)); AOTX_CUDA(cudaMallocManaged(&lengths, n * 64 * sizeof(unsigned)));
    AOTX_CUDA(cudaMallocManaged(&out, n * 64 * sizeof(unsigned)));
    std::vector<unsigned> expected(n * 64);
    for (unsigned i = 0; i < n; ++i) for (unsigned j = 0; j < 64; ++j) {
        unsigned length = j == 63 ? 2048 : 1 + (j * 31) % 512, start = j == 63 ? 0 : (i * 71 + j * 97) % (2049 - length);
        auto query = sources[i].substr(start, length); if (j % 5 == 0) query[length - 1] = 'Z';
        lengths[i * 64 + j] = length; memcpy(queries + (i * 64 + j) * 2048, query.data(), length);
        for (size_t at = 0; (at = sources[i].find(query, at)) != std::string::npos; ++at) ++expected[i * 64 + j];
    }
    aotx_grammar_probe<<<n,64>>>(queries, lengths, out, n); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n * 64; ++i) aotx_check(out[i] == expected[i], "sparse substring occurrence counts match the independent byte oracle");
    cudaFree(queries); cudaFree(lengths); cudaFree(out);
}
static void aotx_grammar_responses(unsigned n) {
    aotx_intake_device d(n); std::vector<std::string> sources;
    for (unsigned i = 0; i < n; ++i) sources.push_back(aotx_grammar_source(i));
    aotx_grammar_open(d, n, sources);
    unsigned char *bytes; unsigned *lengths, *out;
    AOTX_CUDA(cudaMallocManaged(&bytes, n * AOTX_INTAKE_REPLY)); AOTX_CUDA(cudaMallocManaged(&lengths, n * sizeof(unsigned)));
    AOTX_CUDA(cudaMallocManaged(&out, n * 2 * sizeof(unsigned)));
    for (unsigned mode = 0; mode < 16; ++mode) {
        for (unsigned i = 0; i < n; ++i) {
            const std::string name = "Iris" + std::to_string(i);
            std::vector<std::string> cases = {"[]", "[[3,\"" + name + " may not cook.\",0]]", "[[1,\"Ren\\u00e9\",0]]",
                "[[3,\"Soup \\ud83c\\udf72.\",0]]", "[[3,\"Ren\xc3\xa9 says \\\"check\\\".\",0]]",
                "[\"[3,\\\"" + name + "\\\",0]\"]", "[[3,\"absent\",0]]", "[[3,\"" + name + "\",0],]",
                "[[0,\"" + name + "\",0]]", "[[3,\"" + name + "\",\"0\"]]", "[[3,\"" + name + "\",00]]",
                "[[1,\"\\ud800\",0]]", "[[1,\"\\u0000\",0]]", "[[4,\"" + name + "\",1]]",
                "[[1,\"" + name + "\",0],[1,\"" + name + "\",0]]", "[] extra"};
            lengths[i] = cases[mode].size(); memcpy(bytes + i * AOTX_INTAKE_REPLY, cases[mode].data(), lengths[i]);
        }
        for (unsigned split : {1u, 2u, 7u, AOTX_INTAKE_REPLY}) {
            aotx_grammar_parse<<<1,64>>>(bytes, lengths, out, n, split); AOTX_CUDA(cudaDeviceSynchronize());
            for (unsigned i = 0; i < n; ++i) {
                aotx_check(out[i * 2] == (mode < 5), "incremental grammar accepts exactly the declared source response");
                aotx_check(out[i * 2 + 1] == (mode < 5), "independent whole-response parser agrees across every byte boundary");
            }
        }
    }
    cudaFree(bytes); cudaFree(lengths); cudaFree(out);
    float *head; int *tokens; unsigned *agents; unsigned char *raw; unsigned long long *offset;
    AOTX_CUDA(cudaMallocManaged(&head, n * 3 * sizeof(float))); AOTX_CUDA(cudaMallocManaged(&tokens, n * sizeof(int)));
    AOTX_CUDA(cudaMallocManaged(&agents, n * sizeof(unsigned))); AOTX_CUDA(cudaMallocManaged(&raw, 3));
    AOTX_CUDA(cudaMallocManaged(&offset, 4 * sizeof(unsigned long long)));
    memcpy(raw, "Z[]", 3); for (unsigned j = 0; j < 4; ++j) offset[j] = j;
    aotx_text_vocab vocab = {}; vocab.tokens = 3; vocab.token_at = offset; vocab.token_bytes = raw;
    AOTX_CUDA(cudaMemcpyToSymbol(aotx_text_vocab_table, &vocab, sizeof(vocab)));
    for (unsigned i = 0; i < n; ++i) { head[i * 3] = 100; head[i * 3 + 1] = 10; head[i * 3 + 2] = 1; }
    for (unsigned mode = 0; mode < 3; ++mode) {
        aotx_grammar_pick_setup<<<1,64>>>(n, head, tokens, agents, mode);
        aotx_model_pick<<<n,AOTX_MODEL_ROW_THREADS>>>(AOTX_MODEL_LANGUAGE); AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) aotx_check(tokens[i] == (mode == 0 ? 1 : mode == 1 ? 0 : 99),
            "grammar masks invalid maxima only for internal work and refuses an empty candidate set");
    }
    for (unsigned i = 0; i < n; ++i) {
        unsigned status = 0;
        AOTX_CUDA(cudaMemcpyFromSymbol(&status, aotx_intake, sizeof(status),
            offsetof(aotx_intake_state, rows) + i * sizeof(aotx_intake_row) + offsetof(aotx_intake_row, status)));
        aotx_check(status == AOTX_COG_CAPACITY, "empty candidate sets report capacity refusal for every source");
    }
    cudaFree(head); cudaFree(tokens); cudaFree(agents); cudaFree(raw); cudaFree(offset);
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) {
        aotx_grammar_index(n, 0); aotx_grammar_index(n, 1); aotx_grammar_responses(n);
    }
    printf("interpretation grammar: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
