/* Purpose: Hold the region table that device code reads to find a region and check a bound.
 * Owns: The region table.
 * Launch shape: No kernels; storage only.
 * Lifetime: The whole run. */
#include "mem/mem.cuh"

/* The table is empty until the host glue maps the regions and writes it. */
__device__ aotx_mem_table aotx_mem_region_table = { 0u, 0u, { { 0ull, 0ull, 0u, 0u } } };

/* The budget is zero until the host glue reads free device memory at start. */
__device__ aotx_mem_budget aotx_mem_budget_table = { 0ull, 0ull, 0ull, 0ull };
