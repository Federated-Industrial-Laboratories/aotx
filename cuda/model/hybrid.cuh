/* Purpose: Declare fixed-state allocation and the hybrid layer capture operations.
 * Owns: Nothing; the model hold owns the storage.
 * Launch shape: Each kernel takes the token batch of one model role.
 * Lifetime: From model open to model close. */
#ifndef AOTX_MODEL_HYBRID_CUH
#define AOTX_MODEL_HYBRID_CUH

#include "model/forward.cuh"

struct aotx_model_hold;
int aotx_model_hybrid_open(struct aotx_model_hold *hold);
unsigned long long aotx_model_hybrid_bytes(const aotx_model_desc *desc,
                                           unsigned int tokens);
void aotx_model_hybrid_matrix(struct aotx_model_hold *hold, unsigned int layer,
                               unsigned int slot, unsigned int n, unsigned int k,
                               const half *x, float *y);
void aotx_model_hybrid_ffn(struct aotx_model_hold *hold, unsigned int role,
                            unsigned int layer);
__global__ void aotx_model_hybrid_product(unsigned int role, const void *weight,
                                          unsigned int type, unsigned int n,
                                          unsigned int k, const half *x, float *y);
__global__ void aotx_model_gated_qkv(unsigned int role, unsigned int layer);
__global__ void aotx_model_gated_attend(unsigned int role, unsigned int layer);

#endif
