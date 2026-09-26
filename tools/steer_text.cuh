/* Purpose: Share the bounded tokenizer shape with measurement kernels.
 * Owns: No allocation; the measurement program supplies every buffer.
 * Launch shape: One thread for each text in a tokenizer batch.
 * Lifetime: One measurement program. */
#ifndef AOTX_TOOLS_STEER_TEXT_CUH
#define AOTX_TOOLS_STEER_TEXT_CUH
#include "text/text.cuh"
#include "model/forward.cuh"
#define AOTX_STEER_TEXTS 64u
#define AOTX_STEER_STRIDE 1024u
#define AOTX_STEER_BYTES (256u * 1024u)
#define AOTX_STEER_CLEAN 8192u
#define AOTX_STEER_BLOCKS 64u
typedef struct aotx_steer_text {
    unsigned char *bytes, *clean;
    unsigned int *start, *length, *clean_start, *clean_length;
    aotx_text_pieces pieces;
    aotx_text_tokens tokens;
    void *piece[32];
    unsigned int count;
} aotx_steer_text;
__global__ void aotx_steer_positions(aotx_steer_text, unsigned, unsigned, aotx_model_how *, unsigned *);
#endif
