/* Purpose: Score query and candidate pairs.
 * Owns: Nothing.
 * Launch shape: One block for each pair.
 * Lifetime: The tick. */
#ifndef RERANK_CUH
#define RERANK_CUH

#include "model/forward.cuh"

/* The class head gives two logits for each pair. The first row of the class tensor stands
 * for the answer yes and the second for the answer no. The model file states this pooling
 * with the value 4 and names the two classes in that order. */
#define AOTX_RERANK_CLASSES    2u
#define AOTX_RERANK_POOL_RANK  4u

__global__ void aotx_rerank_score(unsigned int role);

#endif
