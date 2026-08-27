/* Purpose: Map device memory: regions, guard gaps, slabs, handles.
 * Owns: The region table and the handle table.
 * Launch shape: One thread for each handle lookup.
 * Lifetime: The whole run. */
#ifndef AOTX_MEM_CUH
#define AOTX_MEM_CUH

/* The virtual range holds the regions in this order: the record ring, a guard gap, the
 * scratch arena, a guard gap. A guard gap has no physical memory behind it, so a write that
 * goes past the end of a region faults. */
#define AOTX_MEM_RING_BYTES     (16ull * 1024ull * 1024ull)
#define AOTX_MEM_SCRATCH_BYTES  (64ull * 1024ull * 1024ull)
#define AOTX_MEM_GUARD_BYTES    (2ull * 1024ull * 1024ull)
#define AOTX_MEM_REGION_MAX     8u

#define AOTX_MEM_KIND_NONE      0u
#define AOTX_MEM_KIND_RING      1u   /* the device record ring */
#define AOTX_MEM_KIND_SCRATCH   2u   /* the scratch arena */

typedef struct aotx_mem_region {
    unsigned long long base;   /* first byte of the region */
    unsigned long long bytes;  /* size of the region */
    unsigned int kind;         /* AOTX_MEM_KIND_* */
    unsigned int reserved;     /* zero */
} aotx_mem_region;

typedef struct aotx_mem_table {
    unsigned int count;
    unsigned int reserved;
    aotx_mem_region region[AOTX_MEM_REGION_MAX];
} aotx_mem_table;

/* The device reads the table to find a region and to check a bound. The host glue writes it
 * once, after the map. */
extern __device__ aotx_mem_table aotx_mem_region_table;

/* The map that the host glue keeps. Addresses are plain integers, so this header stays free
 * of the driver header. */
typedef struct aotx_mem_map {
    unsigned long long range;        /* first byte of the reserved virtual range */
    unsigned long long range_bytes;  /* size of the reserved virtual range */
    unsigned long long ring;         /* first byte of the record ring */
    unsigned long long ring_bytes;   /* size of the record ring after the round up */
    unsigned long long scratch;      /* first byte of the scratch arena */
    unsigned long long scratch_bytes; /* size of the scratch arena after the round up */
    unsigned long long handle[2];    /* the physical allocation of each region */
} aotx_mem_map;

/* Reserve the virtual range, map the two regions, and write the region table. The return is
 * zero when the map is complete. */
int aotx_mem_reserve(aotx_mem_map *map);

/* Unmap the regions, release the physical memory, and give the virtual range back. */
void aotx_mem_release(aotx_mem_map *map);

/* Find a region by kind. The return is a null pointer when the table has no such region. */
__device__ __forceinline__ const aotx_mem_region *aotx_mem_find(unsigned int kind)
{
    for (unsigned int i = 0; i < aotx_mem_region_table.count; ++i) {
        if (aotx_mem_region_table.region[i].kind == kind) {
            return &aotx_mem_region_table.region[i];
        }
    }
    return 0;
}

/* Report whether a span of bytes stays inside a region. The bound check is the first
 * defense; the guard gap is the second. */
__device__ __forceinline__ int aotx_mem_holds(const aotx_mem_region *region,
                                              unsigned long long address,
                                              unsigned long long bytes)
{
    if (region == 0 || address < region->base) {
        return 0;
    }
    return (address - region->base) + bytes <= region->bytes;
}

#endif
