/* Purpose: Share the bounded workspace of the load wrap check.
 * Owns: Prompt bytes, token rows and check results during one load.
 * Launch shape: One thread for each tokenizer row; one block for the argmax.
 * Lifetime: One load check. */
#ifndef AOTX_MODEL_WRAP_CHECK_CUH
#define AOTX_MODEL_WRAP_CHECK_CUH
#include "model/wrap.cuh"
#include "text/text.cuh"

#define AOTX_WRAP_CHECK_ROWS (2u + AOTX_WRAP_SPANS + AOTX_WRAP_ENDS)
#define AOTX_WRAP_CHECK_BYTES 2048u
#define AOTX_WRAP_CHECK_CLEAN (3u * AOTX_WRAP_CHECK_BYTES)
#define AOTX_WRAP_CHECK_WARPS 4u

typedef struct aotx_wrap_check_work {
    aotx_wrap expected;
    unsigned char raw[AOTX_WRAP_CHECK_ROWS][AOTX_WRAP_CHECK_BYTES];
    unsigned char clean[AOTX_WRAP_CHECK_ROWS][AOTX_WRAP_CHECK_CLEAN];
    unsigned int start[AOTX_WRAP_CHECK_ROWS], length[AOTX_WRAP_CHECK_ROWS];
    unsigned int clean_start[AOTX_WRAP_CHECK_ROWS], clean_length[AOTX_WRAP_CHECK_ROWS];
    unsigned int piece_start[AOTX_WRAP_CHECK_ROWS * AOTX_WRAP_CHECK_BYTES];
    unsigned int piece_length[AOTX_WRAP_CHECK_ROWS * AOTX_WRAP_CHECK_BYTES];
    unsigned int piece_token[AOTX_WRAP_CHECK_ROWS * AOTX_WRAP_CHECK_BYTES];
    unsigned int piece_count[AOTX_WRAP_CHECK_ROWS];
    unsigned int work[AOTX_WRAP_CHECK_ROWS * AOTX_WRAP_CHECK_BYTES], works;
    unsigned int id[AOTX_WRAP_CHECK_ROWS][AOTX_WRAP_CHECK_BYTES];
    unsigned int count[AOTX_WRAP_CHECK_ROWS];
    unsigned int chunk[AOTX_WRAP_CHECK_ROWS * AOTX_WRAP_CHECK_BYTES];
    unsigned int scratch[AOTX_WRAP_CHECK_ROWS * AOTX_WRAP_CHECK_CLEAN];
    unsigned char merge[AOTX_WRAP_CHECK_WARPS * AOTX_TEXT_WARP_BYTES];
    unsigned int offset[2], agent;
    unsigned int order_ok, ends_ok, prefill_ok, argmax;
} aotx_wrap_check_work;

__global__ void aotx_wrap_check_build(unsigned int role, aotx_wrap_check_work *work);
__global__ void aotx_wrap_check_tokens(unsigned int role, aotx_wrap_check_work *work);
__global__ void aotx_wrap_check_argmax(unsigned int role, const float *logits,
                                      aotx_wrap_check_work *work);
__global__ void aotx_wrap_check_cache(unsigned long long pages, unsigned int count);
#endif
