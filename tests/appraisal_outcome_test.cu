/* Purpose: Verify source-first grammar and independent complete output admission.
 * Owns: Distinct source rows, malformed controls and vocabulary boundary checks.
 * Launch shape: N=1 and N=64 with four token splits per response.
 * Lifetime: One test process without model weights. */
#include "appraisal_model_fixture.h"
#include "appraisal/token.cuh"

__global__ void aotx_appraisal_outcome_mode(unsigned n, unsigned phase) {
    unsigned i = threadIdx.x;
    if (i < n) aotx_intake.rows[i].phase = phase;
}
static void aotx_appraisal_outcome_responses(unsigned n) {
    aotx_appraisal_model_device d(n);
    for (unsigned mode = 0; mode < 26; ++mode) {
        std::vector<std::string> responses;
        for (unsigned i = 0; i < n; ++i) {
            std::string quote = "I helped with task " + std::to_string(i) + ".";
            std::string item = "\"" + quote + "\"";
            std::string eight = "[";
            for (const char *part : {"I helped", "helped with", "with task", "Ren\\u00e9", "Soup", "I promise", "promise to", "to finish."}) {
                if (eight.size() > 1) eight += ",";
                eight += std::string("\"") + part + "\"";
            }
            eight += "]";
            std::vector<std::string> cases = {
                "[]", "[" + item + "]", "[\"Ren\\u00e9\"]", "[\"Soup \\ud83c\\udf72.\"]",
                " \n[" + item + "]\t", eight,
                "[\"absent\"]", "[\"repeat\"]", "[\"\"]", "[[" + item + ",1]]",
                "[[" + item + ",2]]", "[[" + item + ",3]]", "[[" + item + ",4]]",
                "[[" + item + ",5]]", "[1]", "[" + item + ",]",
                "[" + item + "]extra", "[" + item, "[null]", "[\"\\ud800\"]",
                "[" + item + "," + item + "]", eight.substr(0, eight.size() - 1) + "," + item + "]",
                "[" + item + ",1]", "{}", "[\"I promise to finish.\"]", "[\"René\"]"
            };
            responses.push_back(cases[mode]);
        }
        d.responses(responses);
        aotx_appraisal_outcome_mode<<<1,64>>>(n, 1);
        for (unsigned split : {1u, 2u, 7u, AOTX_INTAKE_REPLY}) {
            aotx_appraisal_model_parse<<<1,64>>>(d.reply, d.lengths, d.out, n, split);
            AOTX_CUDA(cudaDeviceSynchronize());
            bool valid = mode < 6 || mode >= 24;
            for (unsigned i = 0; i < n; ++i) {
                aotx_check(d.out[i * 16] == (valid || mode == 20), "quote grammar follows each fixed syntax and source expectation");
                aotx_check((d.out[i * 16 + 1] == AOTX_COG_OK) == valid, "complete quote admission rejects malformed or duplicate source evidence");
            }
        }
    }
    aotx_appraisal_outcome_mode<<<1,64>>>(n, 0); AOTX_CUDA(cudaDeviceSynchronize());
}
__global__ void aotx_appraisal_outcome_tokens(unsigned n, unsigned *out) {
    unsigned i = threadIdx.x; if (i >= n) return;
    aotx_seqs.slot[i].role = AOTX_MODEL_LANGUAGE; aotx_seqs.slot[i].stop = 99;
    aotx_intake.rows[i].phase = 1; aotx_intake.rows[i].bytes = 0; aotx_intake.rows[i].status = 0;
    aotx_appraisal.rows[i].prefix = {};
    out[i * 16] = aotx_appraisal_allows(i, 0);
    out[i * 16 + 1] = aotx_appraisal_allows(i, 1);
    out[i * 16 + 2] = aotx_appraisal_allows(i, 99);
    aotx_appraisal.rows[i].prefix.stage = 12;
    out[i * 16 + 3] = aotx_appraisal_allows(i, 99);
    out[i * 16 + 4] = aotx_appraisal_allows(i, 1);
}
static void aotx_appraisal_outcome_token_test(unsigned n) {
    aotx_appraisal_model_device d(n);
    unsigned char *raw; unsigned long long *offset;
    AOTX_CUDA(cudaMallocManaged(&raw, 2)); AOTX_CUDA(cudaMallocManaged(&offset, 3 * sizeof(unsigned long long)));
    memcpy(raw, "{[", 2); for (unsigned i = 0; i < 3; ++i) offset[i] = i;
    aotx_text_vocab v = {}; v.tokens = 2; v.token_bytes = raw; v.token_at = offset;
    AOTX_CUDA(cudaMemcpyToSymbol(aotx_text_vocab_table, &v, sizeof(v)));
    aotx_appraisal_outcome_tokens<<<1,64>>>(n, d.out); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) for (unsigned j = 0; j < 5; ++j)
        aotx_check(d.out[i * 16 + j] == (j == 1 || j == 3), "outcome token admission permits only the array start and complete stop");
    cudaFree(raw); cudaFree(offset);
}
int main() {
    for (unsigned n : {1u, 64u}) { aotx_appraisal_outcome_responses(n); aotx_appraisal_outcome_token_test(n); }
    printf("appraisal outcome: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
