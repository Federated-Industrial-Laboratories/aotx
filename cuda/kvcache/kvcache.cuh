/* Purpose: Hold the key and value pages for each agent.
 * Owns: The page table for each agent and the queue of page requests.
 * Launch shape: One thread for each agent; one thread for each page of the stamp.
 * Lifetime: From agent creation to agent release. */
#ifndef KVCACHE_CUH
#define KVCACHE_CUH

/* Agent slots, and the pages a slot may hold. The pages are 2 MB, which is the granularity
 * the virtual memory calls map. The range is virtual and costs no memory; a page position
 * takes memory of the device when a slot first asks for it. A slot holds up to 160 pages,
 * which is a context of 2,048 tokens of the 36 layer model in the layout of
 * cuda/model/kv_layout.cuh. The range holds 1,024 positions, which is 64 sequences of 128
 * tokens of the same model. */
#define AOTX_KV_AGENTS       64u
#define AOTX_KV_PAGE_BYTES   (2ull * 1024ull * 1024ull)
#define AOTX_KV_RANGE_BYTES  (2048ull * 1024ull * 1024ull)
#define AOTX_KV_PAGES        ((unsigned int)(AOTX_KV_RANGE_BYTES / AOTX_KV_PAGE_BYTES))
#define AOTX_KV_PAGES_EACH   160u

/* Requests the queue holds between two ticks. A power of two, so the position is a mask. */
#define AOTX_KV_QUEUE_MAX    256u

/* The first bytes of a mapped page. The rest of the page keeps what was written in it.
 * The type byte lets a page of another number format join without a change of the table. */
#define AOTX_KV_HEADER_BYTES 16u
#define AOTX_KV_TYPE_FP16    1u

typedef struct aotx_kv_entry {
    unsigned int agent;   /* the agent slot the request is for */
    unsigned int pages;   /* pages to add, or zero to give every page of the slot back */
} aotx_kv_entry;

/* The page table and the request queue. The host glue reads the queue between two ticks,
 * when no kernel runs, so every write of the tick is visible to it. */
typedef struct aotx_kv_table {
    unsigned long long page[AOTX_KV_AGENTS][AOTX_KV_PAGES_EACH]; /* address, or zero */
    unsigned int count[AOTX_KV_AGENTS];  /* pages the slot holds */
    unsigned int mapped_pages;           /* pages mapped over every slot; the panel shows it */
    unsigned int short_of;               /* requests that no free page position could fill */
    unsigned int made;                   /* requests the device made */
    unsigned int served;                 /* requests the host glue answered */
    unsigned int refused;                /* requests that found a full queue or a bad slot */
    aotx_kv_entry queue[AOTX_KV_QUEUE_MAX];
} aotx_kv_table;

extern __device__ aotx_kv_table aotx_kv;

/* Ask for pages for one agent slot. The return is 1 when the request went in the queue. */
__device__ int aotx_kv_request(unsigned int agent, unsigned int pages);

/* Ask for every page of one agent slot to go back. */
__device__ int aotx_kv_release(unsigned int agent);

/* Give the address of one page of a slot, or zero when the slot does not hold it. */
__device__ __forceinline__ unsigned long long aotx_kv_page(unsigned int agent,
                                                           unsigned int index)
{
    if (agent >= AOTX_KV_AGENTS || index >= AOTX_KV_PAGES_EACH) {
        return 0ull;
    }
    return aotx_kv.page[agent][index];
}

/* Write the header of every mapped page. The call is idempotent, and the body of a page
 * that keeps its map keeps its content. */
__global__ void aotx_kv_stamp(void);

/* What the host glue keeps for the reservation. Addresses are plain integers, so this
 * header stays free of the driver header. */
typedef struct aotx_kv_map {
    unsigned long long range;        /* first byte of the reserved virtual range */
    unsigned long long range_bytes;  /* size of the range, the guard gap included */
    unsigned long long handle[AOTX_KV_PAGES]; /* physical memory of each page position */
    unsigned int held[AOTX_KV_PAGES];         /* 1 while the position is mapped */
    unsigned int free_at[AOTX_KV_PAGES];      /* the free positions */
    unsigned int free_count;         /* free positions on the stack */
    unsigned int created;            /* physical allocations made since start */
} aotx_kv_map;

/* Reserve the virtual range and give the device the page size. */
int aotx_kv_open(aotx_kv_map *map);

/* Answer every request in the queue: map pages, unmap pages, write the page table. The
 * stamp kernel runs on the given stream. The call waits for that stream alone, so the
 * stream that draws the display is never waited on. The return is the requests answered. */
int aotx_kv_serve(aotx_kv_map *map, cudaStream_t stream);

/* Unmap every page, release the physical memory, and give the virtual range back. */
void aotx_kv_close(aotx_kv_map *map);

#endif
