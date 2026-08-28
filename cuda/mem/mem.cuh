/* Purpose: Map device memory: regions, guard gaps, slabs, handles.
 * Owns: The region table and the handle table.
 * Launch shape: One thread for each handle lookup.
 * Lifetime: The whole run. */
#ifndef AOTX_MEM_CUH
#define AOTX_MEM_CUH

/* The virtual range holds the regions in this order. The record ring, a guard gap, the
 * scratch arena, a guard gap, the weights region, and a guard gap. A guard gap has no
 * physical memory behind it, so a write past the end of a region faults. */
#define AOTX_MEM_RING_BYTES     (16ull * 1024ull * 1024ull)
#define AOTX_MEM_SCRATCH_BYTES  (64ull * 1024ull * 1024ull)
#define AOTX_MEM_GUARD_BYTES    (2ull * 1024ull * 1024ull)
#define AOTX_MEM_REGION_MAX     8u

/* The weights region takes a virtual range of 8 GB, which holds the three models of this
 * system with room to spare. A virtual range costs no memory: physical memory goes behind
 * a part of the range only when a tensor of that part arrives. */
#define AOTX_MEM_WEIGHTS_BYTES  (8ull * 1024ull * 1024ull * 1024ull)

/* Physical memory goes behind the weights region in pieces of this size. The size is the
 * page size that the virtual memory calls of this device work in. */
#define AOTX_MEM_WEIGHTS_GRAIN  (2ull * 1024ull * 1024ull)
#define AOTX_MEM_WEIGHTS_PIECES (AOTX_MEM_WEIGHTS_BYTES / AOTX_MEM_WEIGHTS_GRAIN)

#define AOTX_MEM_KIND_NONE      0u
#define AOTX_MEM_KIND_RING      1u   /* the device record ring */
#define AOTX_MEM_KIND_SCRATCH   2u   /* the scratch arena */
#define AOTX_MEM_KIND_WEIGHTS   3u   /* the weights of the models */

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

/* Tensors the weights region may hold. The three models of this system hold about 1,000
 * tensors together, and the table has room for more than twice that. */
#define AOTX_MEM_TENSOR_MAX     2048u
#define AOTX_MEM_TENSOR_DIMS    4u

/* One tensor in the weights region. Device code finds a tensor by the mix of its name and
 * the number of its model. No name bytes and no pointers go in the table. */
typedef struct aotx_mem_tensor {
    unsigned long long name;      /* the mix of the tensor name */
    unsigned long long offset;    /* first byte of the tensor in the weights region */
    unsigned long long bytes;     /* bytes of the tensor */
    unsigned long long dims[AOTX_MEM_TENSOR_DIMS];
    unsigned int type;            /* the block type of the tensor, from the model file */
    unsigned int model;           /* the model file the tensor came from */
} aotx_mem_tensor;

/* The table holds AOTX_MEM_TENSOR_MAX tensors. A tensor which comes after that is refused
 * and counted, so the host glue can state that the table is full and stop the load. The
 * count may go above the maximum, and every reader takes the lower of the two. */
typedef struct aotx_mem_tensor_table {
    unsigned int count;
    unsigned int refused;
    aotx_mem_tensor tensor[AOTX_MEM_TENSOR_MAX];
} aotx_mem_tensor_table;

extern __device__ aotx_mem_tensor_table aotx_mem_tensor_list;

/* Mix the bytes of a tensor name. The mix is FNV-1a, and the name stops at a zero byte. */
__device__ unsigned long long aotx_mem_name(const char *name, unsigned int max);

/* Find a tensor of a model by the mix of its name. The return is a null pointer when the
 * table has no such tensor. */
__device__ const aotx_mem_tensor *aotx_mem_tensor_find(unsigned long long name,
                                                       unsigned int model);

/* Put the tensors of one model file in the table, one thread for each tensor. The host glue
 * gives the tensor table of the reader and the place of each tensor in the region. */
__global__ void aotx_mem_tensor_add(const void *infos, const unsigned long long *place,
                                    unsigned int count, unsigned int model);

/* What the boot read of device memory gives the panel that shows the arenas. The display
 * shares the memory of this device, so the free figure is read and never assumed. */
typedef struct aotx_mem_budget {
    unsigned long long total;      /* device memory of this device */
    unsigned long long free_boot;  /* free bytes before the system mapped anything */
    unsigned long long free_now;   /* free bytes at the last read */
    unsigned long long reserved;   /* bytes the regions and the guard gaps hold */
} aotx_mem_budget;

extern __device__ aotx_mem_budget aotx_mem_budget_table;

/* The map that the host glue keeps. Addresses are plain integers, so this header stays free
 * of the driver header. */
typedef struct aotx_mem_map {
    unsigned long long range;        /* first byte of the reserved virtual range */
    unsigned long long range_bytes;  /* size of the reserved virtual range */
    unsigned long long ring;         /* first byte of the record ring */
    unsigned long long ring_bytes;   /* size of the record ring after the round up */
    unsigned long long scratch;      /* first byte of the scratch arena */
    unsigned long long scratch_bytes; /* size of the scratch arena after the round up */
    unsigned long long weights;      /* first byte of the weights region */
    unsigned long long weights_bytes; /* size of the weights region, which is virtual only */
    unsigned long long handle[2];    /* the physical allocation of each region */
} aotx_mem_map;

/* Reserve the virtual range, map the two regions, and write the region table. The return is
 * zero when the map is complete. */
int aotx_mem_reserve(aotx_mem_map *map);

/* Unmap the regions, release the physical memory, and give the virtual range back. */
void aotx_mem_release(aotx_mem_map *map);

/* Put physical memory behind a span of the weights region, so a tensor can stream into it.
 * The offset is from the first byte of the region. A piece which already has memory behind
 * it is left as it is, so a caller may map spans which touch. The return is zero when the
 * whole span has memory behind it. */
int aotx_mem_weights_map(unsigned long long offset, unsigned long long bytes);

/* Give the first byte of the weights region, or zero when the reservation did not open. */
unsigned long long aotx_mem_weights_base(void);

/* Give the bytes of the weights region which have physical memory behind them. */
unsigned long long aotx_mem_weights_held(void);

/* Read free device memory again into the budget table. */
void aotx_mem_budget_read(void);

/* Add the bytes of a reservation that another module holds to the budget table. */
void aotx_mem_budget_add(unsigned long long bytes);

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
