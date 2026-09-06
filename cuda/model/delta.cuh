/* Purpose: Declare the fixed recurrent state and delta kernels.
 * Owns: The recurrent matrix, convolution history, and delta scratch layout.
 * Launch shape: Sequences and heads run in parallel; each state scans tokens in order.
 * Lifetime: The model role owns state and scratch from open to shut. */
#ifndef AOTX_MODEL_DELTA_CUH
#define AOTX_MODEL_DELTA_CUH

#include <cuda_fp16.h>
#include "model/names.h"

#define AOTX_DELTA_THREADS 256u
#define AOTX_DELTA_VALUES 8u
#define AOTX_DELTA_DIM_MAX 128u
#define AOTX_DELTA_CONV_MAX 16u

typedef struct aotx_delta_work {
    float *state; /* [slot][compact layer][value head][value][key], F32. */
    float *history; /* [slot][compact layer][channel][oldest to newest], F32. */
    float *qkv;
    float *z;
    float *alpha; /* Projection, then decay after the query/key norm node. */
    float *beta; /* Projection, then sigmoid after the query/key norm node. */
    float *conv;
    float *out;
    half *act;
    unsigned int layers;
    unsigned char layer[AOTX_MODEL_MAX_LAYERS]; /* Physical to compact; 0xff has no recurrent state. */
    unsigned long long state_elements; /* Total across slots and compact layers. */
    unsigned long long history_elements; /* Total across slots and compact layers. */
} aotx_delta_work;

/* One thread scans one channel of a sequence, oldest convolution weight first. */
__global__ void aotx_model_delta_conv(unsigned int role, unsigned int layer);

/* One warp normalizes one query or key head; lanes also prepare each value head's gates. */
__global__ void aotx_model_delta_qk(unsigned int role, unsigned int layer);

/* One warp scans one value row; each lane owns up to four key elements, 32 positions apart. */
__global__ void aotx_model_delta_scan(unsigned int role, unsigned int layer);

/* One warp applies the per-head RMS norm, norm weight, and SiLU output gate. */
__global__ void aotx_model_delta_gate(unsigned int role, unsigned int layer);

#endif
