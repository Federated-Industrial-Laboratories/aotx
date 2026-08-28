/* Purpose: Hold the table of the tensors that the weights region holds.
 * Owns: The tensor table.
 * Launch shape: One thread for each tensor of one model file.
 * Lifetime: From model load to the end of the run. */
#include "disk/modelfile/modelfile.h"
#include "mem/mem.cuh"

/* The layout of one tensor of the reader crosses to the device without change, so the
 * device reads the same structure the reader writes. The kernel makes the mix of each name
 * here and not on the host, because a mix of the bytes of a name is work. */

/* The table is empty until the host glue streams a model file into the region. */
__device__ aotx_mem_tensor_table aotx_mem_tensor_list;

__device__ unsigned long long aotx_mem_name(const char *name, unsigned int max)
{
    unsigned long long mix = 14695981039346656037ull;
    for (unsigned int i = 0u; i < max && name[i] != '\0'; ++i) {
        mix ^= (unsigned long long)(unsigned char)name[i];
        mix *= 1099511628211ull;
    }
    return mix;
}

__device__ const aotx_mem_tensor *aotx_mem_tensor_find(unsigned long long name,
                                                       unsigned int model)
{
    /* The count goes above the maximum when the table is full. The walk stops at the
     * entries the table holds and never reads past the array. */
    unsigned int held = aotx_mem_tensor_list.count;
    if (held > AOTX_MEM_TENSOR_MAX) {
        held = AOTX_MEM_TENSOR_MAX;
    }
    for (unsigned int i = 0u; i < held; ++i) {
        const aotx_mem_tensor *one = &aotx_mem_tensor_list.tensor[i];
        if (one->name == name && one->model == model) {
            return one;
        }
    }
    return 0;
}

__global__ void aotx_mem_tensor_add(const void *infos, const unsigned long long *place,
                                    unsigned int count, unsigned int model)
{
    const aotx_tensor_info *table = (const aotx_tensor_info *)infos;
    unsigned int step = blockDim.x * gridDim.x;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += step) {
        unsigned int at = atomicAdd(&aotx_mem_tensor_list.count, 1u);
        if (at >= AOTX_MEM_TENSOR_MAX) {
            /* The table is full. The tensor is refused and counted, and the host glue
             * reads the count and stops the load. */
            atomicAdd(&aotx_mem_tensor_list.refused, 1u);
            continue;
        }
        aotx_mem_tensor *one = &aotx_mem_tensor_list.tensor[at];
        one->name = aotx_mem_name(table[i].name, AOTX_TENSOR_NAME_BYTES);
        one->offset = place[i];
        one->bytes = table[i].bytes;
        for (unsigned int d = 0u; d < AOTX_MEM_TENSOR_DIMS; ++d) {
            one->dims[d] = (d < table[i].dim_count) ? table[i].dims[d] : 1ull;
        }
        one->type = table[i].type;
        one->model = model;
    }
}
