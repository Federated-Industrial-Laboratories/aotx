/* Purpose: Reserve one virtual range and put the regions of the system in it.
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

/* The weights region gives its physical memory back at release, so the pieces which have
 * memory behind them are held here. One map holds this state, and the boot glue opens one
 * map. An entry of zero means the piece has no memory behind it. */
static CUmemGenericAllocationHandle aotx_mem_weights_piece[AOTX_MEM_WEIGHTS_PIECES];
static CUdeviceptr aotx_mem_weights_first = 0;
static unsigned long long aotx_mem_weights_bytes = 0ull;
static CUmemAllocationProp aotx_mem_weights_prop;

/* The budget table is read from the driver, never assumed. The display and the browser hold
 * memory of this device that the system does not control, and the read states that. */
static void aotx_mem_budget_write(unsigned long long reserved, int first)
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    aotx_mem_budget table;
    memset(&table, 0, sizeof table);
    if (first == 0) {
        aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_mem_budget_table, sizeof table),
                           "cudaMemcpyFromSymbol");
    }
    aotx_check_driver(cuMemGetInfo(&free_bytes, &total_bytes), "cuMemGetInfo");
    table.total = (unsigned long long)total_bytes;
    table.free_now = (unsigned long long)free_bytes;
    if (first != 0) {
        table.free_boot = (unsigned long long)free_bytes;
    }
    table.reserved += reserved;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_mem_budget_table, &table, sizeof table),
                       "cudaMemcpyToSymbol");
}

void aotx_mem_budget_read(void)
{
    aotx_mem_budget_write(0ull, 0);
}

void aotx_mem_budget_add(unsigned long long bytes)
{
    aotx_mem_budget_write(bytes, 0);
}

int aotx_mem_reserve(aotx_mem_map *map)
{
    CUdevice device = 0;
    aotx_check_driver(cuCtxGetDevice(&device), "cuCtxGetDevice");
    aotx_mem_budget_write(0ull, 1);

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
    size_t weights = aotx_mem_round(AOTX_MEM_WEIGHTS_BYTES, granule);
    size_t total = ring + guard + scratch + guard + weights + guard;

    CUdeviceptr range = 0;
    aotx_check_driver(cuMemAddressReserve(&range, total, granule, 0, 0),
                      "cuMemAddressReserve");

    map->range = (unsigned long long)range;
    map->range_bytes = (unsigned long long)total;
    map->ring = (unsigned long long)range;
    map->ring_bytes = (unsigned long long)ring;
    map->scratch = (unsigned long long)(range + ring + guard);
    map->scratch_bytes = (unsigned long long)scratch;
    map->weights = (unsigned long long)(range + ring + guard + scratch + guard);
    map->weights_bytes = (unsigned long long)weights;
    aotx_mem_map_one((CUdeviceptr)map->ring, ring, &prop, &map->handle[0]);
    aotx_mem_map_one((CUdeviceptr)map->scratch, scratch, &prop, &map->handle[1]);
    /* The weights region has no memory behind it at start. Each tensor asks for the pieces
     * it needs while it streams in. */
    memset(aotx_mem_weights_piece, 0, sizeof aotx_mem_weights_piece);
    aotx_mem_weights_first = (CUdeviceptr)map->weights;
    aotx_mem_weights_bytes = 0ull;
    aotx_mem_weights_prop = prop;

    /* The table gives device code the base and the bound of each region. The gaps have no
     * entry, because no code may touch them. */
    aotx_mem_table table;
    memset(&table, 0, sizeof table);
    table.count = 3;
    table.region[0].base = map->ring;
    table.region[0].bytes = map->ring_bytes;
    table.region[0].kind = AOTX_MEM_KIND_RING;
    table.region[1].base = map->scratch;
    table.region[1].bytes = map->scratch_bytes;
    table.region[1].kind = AOTX_MEM_KIND_SCRATCH;
    table.region[2].base = map->weights;
    table.region[2].bytes = map->weights_bytes;
    table.region[2].kind = AOTX_MEM_KIND_WEIGHTS;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_mem_region_table, &table, sizeof table),
                       "cudaMemcpyToSymbol");
    /* The budget counts the memory the system holds, and the weights region holds none of
     * it until a tensor arrives. The map of a piece adds that piece to the budget. */
    aotx_mem_budget_write(map->range_bytes - map->weights_bytes, 0);
    return 0;
}

unsigned long long aotx_mem_weights_base(void)
{
    return (unsigned long long)aotx_mem_weights_first;
}

unsigned long long aotx_mem_weights_held(void)
{
    return aotx_mem_weights_bytes;
}

int aotx_mem_weights_map(unsigned long long offset, unsigned long long bytes)
{
    if (aotx_mem_weights_first == 0 || bytes == 0ull
        || offset + bytes > AOTX_MEM_WEIGHTS_BYTES) {
        return 1;
    }
    unsigned long long first = offset / AOTX_MEM_WEIGHTS_GRAIN;
    unsigned long long last = (offset + bytes - 1ull) / AOTX_MEM_WEIGHTS_GRAIN;
    unsigned long long made = 0ull;
    CUmemAccessDesc access;
    memset(&access, 0, sizeof access);
    access.location = aotx_mem_weights_prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    for (unsigned long long piece = first; piece <= last; ++piece) {
        if (aotx_mem_weights_piece[piece] != 0) {
            continue;
        }
        CUdeviceptr at = aotx_mem_weights_first + piece * AOTX_MEM_WEIGHTS_GRAIN;
        CUmemGenericAllocationHandle physical = 0;
        aotx_check_driver(cuMemCreate(&physical, (size_t)AOTX_MEM_WEIGHTS_GRAIN,
                                      &aotx_mem_weights_prop, 0), "cuMemCreate");
        aotx_check_driver(cuMemMap(at, (size_t)AOTX_MEM_WEIGHTS_GRAIN, 0, physical, 0),
                          "cuMemMap");
        aotx_check_driver(cuMemSetAccess(at, (size_t)AOTX_MEM_WEIGHTS_GRAIN, &access, 1),
                          "cuMemSetAccess");
        aotx_mem_weights_piece[piece] = physical;
        made += AOTX_MEM_WEIGHTS_GRAIN;
    }
    if (made != 0ull) {
        aotx_mem_weights_bytes += made;
        aotx_mem_budget_add(made);
    }
    return 0;
}

int aotx_mem_weights_trim(unsigned long long bytes)
{
    unsigned long long first = (bytes + AOTX_MEM_WEIGHTS_GRAIN - 1ull)
                             / AOTX_MEM_WEIGHTS_GRAIN;
    unsigned long long released = 0ull;
    if (first > AOTX_MEM_WEIGHTS_PIECES) {
        return 1;
    }
    for (unsigned long long piece = first; piece < AOTX_MEM_WEIGHTS_PIECES; ++piece) {
        if (aotx_mem_weights_piece[piece] == 0) {
            continue;
        }
        CUdeviceptr at = aotx_mem_weights_first + piece * AOTX_MEM_WEIGHTS_GRAIN;
        aotx_check_driver(cuMemUnmap(at, (size_t)AOTX_MEM_WEIGHTS_GRAIN), "cuMemUnmap");
        aotx_check_driver(cuMemRelease(aotx_mem_weights_piece[piece]), "cuMemRelease");
        aotx_mem_weights_piece[piece] = 0;
        released += AOTX_MEM_WEIGHTS_GRAIN;
    }
    aotx_mem_weights_bytes -= released;
    if (released != 0ull) {
        aotx_mem_budget table;
        aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_mem_budget_table, sizeof table),
                           "cudaMemcpyFromSymbol");
        table.reserved -= released;
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_mem_budget_table, &table, sizeof table),
                           "cudaMemcpyToSymbol");
        aotx_mem_budget_read();
    }
    return 0;
}

void aotx_mem_release(aotx_mem_map *map)
{
    if (map->range == 0) {
        return;
    }
    for (unsigned long long piece = 0ull; piece < AOTX_MEM_WEIGHTS_PIECES; ++piece) {
        if (aotx_mem_weights_piece[piece] == 0) {
            continue;
        }
        CUdeviceptr at = (CUdeviceptr)map->weights + piece * AOTX_MEM_WEIGHTS_GRAIN;
        cuMemUnmap(at, (size_t)AOTX_MEM_WEIGHTS_GRAIN);
        cuMemRelease(aotx_mem_weights_piece[piece]);
        aotx_mem_weights_piece[piece] = 0;
    }
    aotx_mem_weights_first = 0;
    aotx_mem_weights_bytes = 0ull;
    cuMemUnmap((CUdeviceptr)map->ring, (size_t)map->ring_bytes);
    cuMemUnmap((CUdeviceptr)map->scratch, (size_t)map->scratch_bytes);
    cuMemRelease((CUmemGenericAllocationHandle)map->handle[0]);
    cuMemRelease((CUmemGenericAllocationHandle)map->handle[1]);
    cuMemAddressFree((CUdeviceptr)map->range, (size_t)map->range_bytes);
    map->range = 0;
}
