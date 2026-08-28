/* Purpose: Map and unmap the key and value pages that device code asks for.
 * Owns: The virtual range of the pages, the physical memory of each position.
 * Launch shape: Host glue; the stamp kernel writes the header of each page.
 * Lifetime: From the reservation at start to the release at exit. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <string.h>

#include "boot/check.h"
#include "kvcache/kvcache.cuh"
#include "mem/mem.cuh"

/* Physical memory of the device that runs the system, with read and write access for it. */
static void aotx_kv_property(CUmemAllocationProp *prop)
{
    CUdevice device = 0;
    aotx_check_driver(cuCtxGetDevice(&device), "cuCtxGetDevice");
    memset(prop, 0, sizeof *prop);
    prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop->location.id = (int)device;
}

/* Map one free position. The physical memory of a position is made once and kept, so a
 * position that was given back is mapped again with no new allocation. */
static int aotx_kv_take(aotx_kv_map *map, unsigned long long *address)
{
    if (map->free_count == 0u) {
        return 1;
    }
    unsigned int at = map->free_at[--map->free_count];
    CUdeviceptr page = (CUdeviceptr)(map->range
                                     + (unsigned long long)at * AOTX_KV_PAGE_BYTES);
    CUmemAllocationProp prop;
    aotx_kv_property(&prop);
    if (map->handle[at] == 0ull) {
        CUmemGenericAllocationHandle physical = 0;
        aotx_check_driver(cuMemCreate(&physical, (size_t)AOTX_KV_PAGE_BYTES, &prop, 0),
                          "cuMemCreate");
        map->handle[at] = (unsigned long long)physical;
        map->created += 1u;
    }
    aotx_check_driver(cuMemMap(page, (size_t)AOTX_KV_PAGE_BYTES, 0,
                               (CUmemGenericAllocationHandle)map->handle[at], 0),
                      "cuMemMap");
    CUmemAccessDesc access;
    memset(&access, 0, sizeof access);
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    aotx_check_driver(cuMemSetAccess(page, (size_t)AOTX_KV_PAGE_BYTES, &access, 1),
                      "cuMemSetAccess");
    map->held[at] = 1u;
    *address = (unsigned long long)page;
    return 0;
}

/* Unmap one position and put it back on the stack of free positions. */
static void aotx_kv_give(aotx_kv_map *map, unsigned long long address)
{
    if (address < map->range) {
        return;
    }
    unsigned long long at = (address - map->range) / AOTX_KV_PAGE_BYTES;
    if (at >= (unsigned long long)AOTX_KV_PAGES || map->held[at] == 0u) {
        return;
    }
    aotx_check_driver(cuMemUnmap((CUdeviceptr)address, (size_t)AOTX_KV_PAGE_BYTES),
                      "cuMemUnmap");
    map->held[at] = 0u;
    map->free_at[map->free_count] = (unsigned int)at;
    map->free_count += 1u;
}

int aotx_kv_open(aotx_kv_map *map)
{
    CUmemAllocationProp prop;
    size_t granule = 0;
    memset(map, 0, sizeof *map);
    aotx_kv_property(&prop);
    aotx_check_driver(cuMemGetAllocationGranularity(&granule, &prop,
                                                    CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                      "cuMemGetAllocationGranularity");
    if (granule == 0 || (AOTX_KV_PAGE_BYTES % (unsigned long long)granule) != 0ull) {
        return 1;
    }

    /* The range holds every page position and one unmapped gap at the end. A read past the
     * last page faults, and does not reach a neighbor. */
    size_t total = (size_t)(AOTX_KV_RANGE_BYTES + AOTX_MEM_GUARD_BYTES);
    CUdeviceptr range = 0;
    aotx_check_driver(cuMemAddressReserve(&range, total, (size_t)AOTX_KV_PAGE_BYTES, 0, 0),
                      "cuMemAddressReserve");
    map->range = (unsigned long long)range;
    map->range_bytes = (unsigned long long)total;
    for (unsigned int i = 0u; i < AOTX_KV_PAGES; ++i) {
        map->free_at[i] = AOTX_KV_PAGES - 1u - i;
    }
    map->free_count = AOTX_KV_PAGES;

    aotx_kv_table table;
    memset(&table, 0, sizeof table);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_kv, &table, sizeof table),
                       "cudaMemcpyToSymbol");
    aotx_mem_budget_add(map->range_bytes);
    return 0;
}

int aotx_kv_serve(aotx_kv_map *map, cudaStream_t stream)
{
    aotx_kv_table table;
    unsigned int marks[2] = { 0u, 0u };

    /* The marks come first, because a tick with no request must cost one small read. */
    aotx_check_runtime(cudaMemcpyFromSymbol(marks, aotx_kv, sizeof marks,
                                            offsetof(aotx_kv_table, made),
                                            cudaMemcpyDeviceToHost),
                       "cudaMemcpyFromSymbol");
    if (marks[0] == marks[1]) {
        return 0;
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_kv, sizeof table),
                       "cudaMemcpyFromSymbol");
    if (table.served == table.made) {
        return 0;
    }
    unsigned int done = 0u;
    for (unsigned int at = table.served; at != table.made; ++at) {
        aotx_kv_entry entry = table.queue[at & (AOTX_KV_QUEUE_MAX - 1u)];
        done += 1u;
        if (entry.agent >= AOTX_KV_AGENTS) {
            continue;
        }
        if (entry.pages == 0u) {
            for (unsigned int i = 0u; i < AOTX_KV_PAGES_EACH; ++i) {
                aotx_kv_give(map, table.page[entry.agent][i]);
                table.page[entry.agent][i] = 0ull;
            }
            table.count[entry.agent] = 0u;
            continue;
        }
        unsigned int want = table.count[entry.agent] + entry.pages;
        if (want > AOTX_KV_PAGES_EACH) {
            want = AOTX_KV_PAGES_EACH;
        }
        for (unsigned int i = table.count[entry.agent]; i < want; ++i) {
            unsigned long long address = 0ull;
            if (aotx_kv_take(map, &address) != 0) {
                /* Every position is mapped. The slot keeps what it holds, and the count of
                 * requests that no position could fill says so. */
                table.short_of += 1u;
                break;
            }
            table.page[entry.agent][i] = address;
            table.count[entry.agent] = i + 1u;
        }
    }

    /* The host glue owns the page table, the page count and the served mark. The made mark
     * and the refused count belong to device code, so they are not written over. */
    table.mapped_pages = AOTX_KV_PAGES - map->free_count;
    table.served = table.made;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_kv, &table, offsetof(aotx_kv_table, made),
                                          0, cudaMemcpyHostToDevice),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_kv, &table.served, sizeof table.served,
                                          offsetof(aotx_kv_table, served),
                                          cudaMemcpyHostToDevice),
                       "cudaMemcpyToSymbol");
    aotx_kv_stamp<<<AOTX_KV_AGENTS, AOTX_KV_PAGES_EACH, 0, stream>>>();
    aotx_check_runtime(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    return (int)done;
}

void aotx_kv_close(aotx_kv_map *map)
{
    if (map->range == 0ull) {
        return;
    }
    for (unsigned int i = 0u; i < AOTX_KV_PAGES; ++i) {
        unsigned long long page = map->range + (unsigned long long)i * AOTX_KV_PAGE_BYTES;
        if (map->held[i] != 0u) {
            cuMemUnmap((CUdeviceptr)page, (size_t)AOTX_KV_PAGE_BYTES);
            map->held[i] = 0u;
        }
        if (map->handle[i] != 0ull) {
            cuMemRelease((CUmemGenericAllocationHandle)map->handle[i]);
            map->handle[i] = 0ull;
        }
    }
    cuMemAddressFree((CUdeviceptr)map->range, (size_t)map->range_bytes);
    map->range = 0ull;
}
