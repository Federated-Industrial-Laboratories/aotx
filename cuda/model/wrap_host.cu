/* Purpose: Run the load wrap check without changing live decode state.
 * Owns: Temporary tokenizer, graph and cache allocations.
 * Launch shape: Host glue; byte checks, tokenization and the prefill run on the device.
 * Lifetime: One model load. */
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include "boot/check.h"
#include "model/graph_host.h"
#include "model/wrap_check.cuh"
#include "kvcache/kvcache.cuh"

static void aotx_wrap_tokenize(aotx_wrap_check_work *work, unsigned int role)
{
    if(role==AOTX_MODEL_LANGUAGE_AUDIO)aotx_text_vocab_select<<<1,128>>>(2u);
    aotx_wrap_check_build<<<1, 32>>>(role, work);
    aotx_text_batch raw = { &work->raw[0][0], work->start, work->length,
                            AOTX_WRAP_CHECK_ROWS };
    aotx_text_clean<<<1, 32>>>(raw, &work->clean[0][0], work->clean_start,
                               work->clean_length, AOTX_WRAP_CHECK_CLEAN);
    aotx_text_batch batch = { &work->clean[0][0], work->clean_start,
                              work->clean_length, AOTX_WRAP_CHECK_ROWS };
    aotx_text_pieces pieces = { work->piece_start, work->piece_length, work->piece_token,
        work->piece_count, work->work, &work->works, AOTX_WRAP_CHECK_BYTES };
    aotx_text_tokens tokens = { &work->id[0][0], work->count, work->chunk, work->scratch,
        work->merge, AOTX_WRAP_CHECK_WARPS, AOTX_WRAP_CHECK_BYTES };
    aotx_text_pretok<<<1, 32>>>(batch, pieces);
    aotx_text_merge<<<2, 64>>>(batch, pieces, tokens);
    aotx_text_gather<<<1, 32>>>(batch, pieces, tokens);
    aotx_wrap_check_tokens<<<1, 1>>>(role, work);
    if(role==AOTX_MODEL_LANGUAGE_AUDIO)aotx_text_vocab_select<<<1,128>>>(0u);
}

/* A live model replacement can hold a captured pass already. Save its host handles and
 * device bindings while the check uses a short pass and private pages in slot zero. */
static int aotx_wrap_prefill(unsigned int role, aotx_wrap_check_work *work,
                             unsigned int tokens)
{
    aotx_model_hold *hold = aotx_model_hold_of(role);
    aotx_model_hold saved = *hold;
    aotx_model_work space;
    aotx_model_run call;
    aotx_kv_table cache;
    unsigned int seen, faults;
    aotx_check_runtime(cudaMemcpyFromSymbol(&space, aotx_model_space, sizeof space,
                                            role * sizeof space), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&call, aotx_model_call, sizeof call,
                                            role * sizeof call), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&cache, aotx_kv, sizeof cache), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&seen, aotx_model_seen, sizeof seen), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&faults, aotx_model_faults, sizeof faults), "cudaMemcpyFromSymbol");
    int bad = aotx_model_open(role, tokens);
    void *pages = NULL;
    float *logits = NULL;
    if (bad == 0) {
        unsigned int count = aotx_kvl_pages(&hold->work.shape, tokens);
        if (count == 0u || count > AOTX_KV_PAGES_EACH) bad = 1;
        else {
            aotx_check_runtime(cudaMalloc(&pages, (size_t)count * AOTX_KV_PAGE_BYTES), "cudaMalloc");
            aotx_check_runtime(cudaMalloc(&logits, (size_t)hold->desc.vocab * sizeof(float)), "cudaMalloc");
            aotx_wrap_check_cache<<<(AOTX_KV_PAGES_EACH + 127u) / 128u, 128>>>(
                (unsigned long long)pages, count);
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            aotx_model_run run = {};
            run.ids = (const int *)&work->id[0][0]; run.offset = work->offset;
            run.agent = &work->agent; run.seqs = 1u; run.tokens = tokens;
            run.rows = 1u; run.select = AOTX_MODEL_ROWS_LAST; run.logits = logits;
            run.top_p = 1.0f;
            /* This private pass must not add attention mass to live page records. */
            bad = aotx_model_launch(role, &run);
            if (bad == 0) aotx_wrap_check_argmax<<<1, 256>>>(role, logits, work);
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }
    }
    aotx_model_shut(role);
    *hold = saved;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_space, &space, sizeof space,
                                          role * sizeof space), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &call, sizeof call,
                                          role * sizeof call), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_kv, &cache, sizeof cache), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_seen, &seen, sizeof seen), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_faults, &faults, sizeof faults), "cudaMemcpyToSymbol");
    cudaFree(logits);
    cudaFree(pages);
    return bad;
}

int aotx_model_wrap_check(unsigned int role, const aotx_wrap *expected, const char *file)
{
    if (role >= AOTX_MODEL_ROLES || expected == NULL) return 1;
    aotx_wrap observed;
    aotx_check_runtime(cudaMemcpyFromSymbol(&observed, aotx_model_wrap, sizeof observed,
                                            role * sizeof observed), "cudaMemcpyFromSymbol");
    unsigned int zero = 0u;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_wrap, &zero, sizeof zero,
        role * sizeof observed + offsetof(aotx_wrap, usable)), "cudaMemcpyToSymbol");
    if (!aotx_wrap_valid(expected) || !aotx_wrap_valid(&observed)
        || !aotx_wrap_matches(&observed)) {
        printf("wrap: %s spans=FAIL ends=not-run prefill=not-run usable=no\n", file);
        return 1;
    }
    aotx_wrap_check_work *work = NULL;
    aotx_check_runtime(cudaMalloc(&work, sizeof *work), "cudaMalloc");
    aotx_check_runtime(cudaMemset(work, 0, sizeof *work), "cudaMemset");
    aotx_check_runtime(cudaMemcpy(&work->expected, expected, sizeof *expected,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_wrap_tokenize(work, role);
    unsigned int results[4] = { 0u, 0u, 0u, ~0u }, tokens = 0u;
    aotx_check_runtime(cudaMemcpy(results, &work->order_ok, sizeof results,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(&tokens, work->count, sizeof tokens,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    int language = aotx_model_is_language(role);
    if (results[0] && results[1] && language) aotx_wrap_prefill(role, work, tokens);
    aotx_check_runtime(cudaMemcpy(results, &work->order_ok, sizeof results,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    int good = results[0] && results[1] && results[2];
    printf("wrap: %s spans=%s ends=%s prefill=%s argmax=%u usable=%s\n", file,
           results[0] ? "pass" : "FAIL", results[1] ? "pass" : "FAIL",
           !language ? "no-language-head" : !results[0] || !results[1] ? "not-run"
                                         : results[2] ? "pass" : "FAIL",
           results[3], good ? "yes" : "no");
    cudaFree(work);
    return good ? 0 : 1;
}
