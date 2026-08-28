/* Purpose: Compute embeddings and search them.
 * Owns: Nothing yet; the vector store in the arena comes with the agent module.
 * Launch shape: One block for each sequence; the threads hold the hidden width.
 * Lifetime: The whole run. */
#ifndef EMBED_CUH
#define EMBED_CUH

#include "model/forward.cuh"

/* The embedding head. The vector of a sequence is the last row of the sequence after the
 * last norm, made a unit vector. The model file states this pooling with the value 3. */
#define AOTX_EMBED_POOL_LAST   3u

__global__ void aotx_embed_pool(unsigned int role);

#endif
