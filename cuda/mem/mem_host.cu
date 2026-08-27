/* Purpose: Reserve one virtual range and map the record ring and the scratch arena in it.
 * Owns: The virtual range, the physical allocations, and the region table content.
 * Launch shape: Host glue only; no kernels.
 * Lifetime: From the map at start to the release at exit. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>

#include "boot/check.h"
#include "mem/mem.cuh"

/* The virtual memory calls take sizes that are a whole number of granules. */
static size_t aotx_mem_round(size_t bytes, size_t granule)
{
    return ((bytes + granule - 1) / granule) * granule;
}

/* One region: physical memory, a map into the range, and read and write access for the
 * device that runs the system. */
static void aotx_mem_map_one(CUdeviceptr address, size_t bytes,
                             const CUmemAllocationProp *prop,
                             unsigned long long *handle)
{
    CUmemGenericAllocationHandle physical = 0;
    aotx_check_driver(cuMemCreate(&physical, bytes, prop, 0), "cuMemCreate");
    aotx_check_driver(cuMemMap(address, bytes, 0, physical, 0), "cuMemMap");
    CUmemAccessDesc access;
    memset(&access, 0, sizeof access);
    access.location = prop->location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    aotx_check_driver(cuMemSetAccess(address, bytes, &access, 1), "cuMemSetAccess");
    aotx_check_driver(cuMemsetD8(address, 0, bytes), "cuMemsetD8");
    *handle = (unsigned long long)physical;
}

int aotx_mem_reserve(aotx_mem_map *map)
{
    CUdevice device = 0;
    aotx_check_driver(cuCtxGetDevice(&device), "cuCtxGetDevice");

    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof prop);
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = (int)device;

    size_t granule = 0;
    aotx_check_driver(cuMemGetAllocationGranularity(&granule, &prop,
                                                    CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                      "cuMemGetAllocationGranularity");
    if (granule == 0) {
        return 1;
    }
    size_t ring = aotx_mem_round(AOTX_MEM_RING_BYTES, granule);
    size_t scratch = aotx_mem_round(AOTX_MEM_SCRATCH_BYTES, granule);
    size_t guard = aotx_mem_round(AOTX_MEM_GUARD_BYTES, granule);
    size_t total = ring + guard + scratch + guard;

    CUdeviceptr range = 0;
    aotx_check_driver(cuMemAddressReserve(&range, total, granule, 0, 0),
                      "cuMemAddressReserve");

    map->range = (unsigned long long)range;
    map->range_bytes = (unsigned long long)total;
    map->ring = (unsigned long long)range;
    map->ring_bytes = (unsigned long long)ring;
    map->scratch = (unsigned long long)(range + ring + guard);
    map->scratch_bytes = (unsigned long long)scratch;
    aotx_mem_map_one((CUdeviceptr)map->ring, ring, &prop, &map->handle[0]);
    aotx_mem_map_one((CUdeviceptr)map->scratch, scratch, &prop, &map->handle[1]);

    /* The table gives device code the base and the bound of each region. The gaps have no
     * entry, because no code may touch them. */
    aotx_mem_table table;
    memset(&table, 0, sizeof table);
    table.count = 2;
    table.region[0].base = map->ring;
    table.region[0].bytes = map->ring_bytes;
    table.region[0].kind = AOTX_MEM_KIND_RING;
    table.region[1].base = map->scratch;
    table.region[1].bytes = map->scratch_bytes;
    table.region[1].kind = AOTX_MEM_KIND_SCRATCH;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_mem_region_table, &table, sizeof table),
                       "cudaMemcpyToSymbol");
    return 0;
}

void aotx_mem_release(aotx_mem_map *map)
{
    if (map->range == 0) {
        return;
    }
    cuMemUnmap((CUdeviceptr)map->ring, (size_t)map->ring_bytes);
    cuMemUnmap((CUdeviceptr)map->scratch, (size_t)map->scratch_bytes);
    cuMemRelease((CUmemGenericAllocationHandle)map->handle[0]);
    cuMemRelease((CUmemGenericAllocationHandle)map->handle[1]);
    cuMemAddressFree((CUdeviceptr)map->range, (size_t)map->range_bytes);
    map->range = 0;
}
