/* Purpose: Place an explicit turn table for tests that do not load a model file.
 * Owns: The fixed role-tag fixture and its device copies.
 * Launch shape: Host test setup; no model computation.
 * Lifetime: One test process. */
#ifndef AOTX_TEST_WRAP_FIXTURE_H
#define AOTX_TEST_WRAP_FIXTURE_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "boot/check.h"
#include "model/wrap.cuh"
#include "model/call_format.cuh"

static void aotx_test_call_upload(unsigned int kind)
{
    aotx_call_format form = {};
    if (aotx_call_format_make(kind, &form) != 0) {
        fprintf(stderr, "call fixture: the selected form is invalid\n");
        exit(1);
    }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call_format, &form, sizeof form,
                        AOTX_MODEL_LANGUAGE * sizeof form), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call_format, &form, sizeof form,
                        AOTX_MODEL_LANGUAGE_Q4 * sizeof form), "cudaMemcpyToSymbol");
}

static aotx_wrap aotx_test_wrap_table(void)
{
    static const char *const span[AOTX_WRAP_SPANS] = {
        "<|im_start|>system\n", "<|im_end|>\n",
        "<|im_start|>user\n", "<|im_end|>\n",
        "<|im_start|>assistant\n", "<|im_end|>\n",
        "<|im_start|>assistant\n", "<think>\n\n", "</think>\n\n"
    };
    aotx_wrap wrap = {};
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < AOTX_WRAP_SPANS; ++i) {
        wrap.offset[i] = (uint16_t)at;
        wrap.length[i] = (uint8_t)strlen(span[i]);
        memcpy(wrap.bytes + at, span[i], wrap.length[i]);
        at += wrap.length[i];
    }
    wrap.end_ids[0] = 151645u;
    wrap.end_ids[1] = 151643u;
    wrap.think_open_id = 151667u;
    wrap.think_close_id = 151668u;
    wrap.end_count = 2u;
    wrap.usable = 1u;
    return wrap;
}

static void aotx_test_wrap_upload(const aotx_wrap *wrap)
{
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_wrap, wrap, sizeof *wrap,
                        AOTX_MODEL_LANGUAGE * sizeof *wrap), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_wrap, wrap, sizeof *wrap,
                        AOTX_MODEL_LANGUAGE_Q4 * sizeof *wrap), "cudaMemcpyToSymbol");
}

static void aotx_test_wrap_open(void)
{
    aotx_wrap wrap = aotx_test_wrap_table();
    aotx_test_wrap_upload(&wrap);
    aotx_test_call_upload(AOTX_CALL_HERMES);
}

#endif
