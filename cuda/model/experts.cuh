/* Purpose: Declare the routed expert kernels and their launch dimensions.
 * Owns: Nothing; the forward buffers hold the selected experts and their sums.
 * Launch shape: Token rows for route and add; weight rows by token rows for matrix.
 * Lifetime: One forward graph. */
#ifndef AOTX_MODEL_EXPERTS_CUH
#define AOTX_MODEL_EXPERTS_CUH

#include "model/forward.cuh"

#define AOTX_EXPERT_ROUTE_THREADS 256u
#define AOTX_EXPERT_MATRIX_THREADS AOTX_GEMV_THREADS
#define AOTX_EXPERT_MATRIX_ROWS AOTX_GEMV_ROWS_CTA
#define AOTX_EXPERT_ADD_THREADS 256u

/* Route and add step through tokens by gridDim.x. Matrix steps by gridDim.y.
 * Matrix grid.x is ceil(n / AOTX_EXPERT_MATRIX_ROWS). All grids may exceed the batch.
 * Router weights are F32. Matrix k follows the block width rules of matrix.cuh. */
__global__ void aotx_model_expert_route(unsigned int role, unsigned int layer);
__global__ void aotx_model_expert_matrix(unsigned int role, unsigned int rank,
                                        const void *w, unsigned int type,
                                        unsigned int n, unsigned int k,
                                        const half *x, float *y);

/* Launch ranks in ascending order after each down product. The final rank writes proj. */
__global__ void aotx_model_expert_add(unsigned int role, unsigned int rank);

#endif
