/* Purpose: Take page requests from device code and write the header of each mapped page.
 * Owns: The page table and the request queue.
 * Launch shape: One thread for each request; one thread for each page of the stamp.
 * Lifetime: The whole run. */
#include "kvcache/kvcache.cuh"

/* The table is empty until the host glue answers the first request. */
__device__ aotx_kv_table aotx_kv;

/* Put one request in the queue. The queue index moves by a compare and swap. A full queue
 * refuses the request, so no entry of an earlier request is written over. Nothing waits for
 * the host glue. */
static __device__ __forceinline__ int aotx_kv_put(unsigned int agent, unsigned int pages)
{
    if (agent >= AOTX_SLOTS || pages > AOTX_KV_PAGES_EACH) {
        atomicAdd(&aotx_kv.refused, 1u);
        return 0;
    }
    unsigned int at = aotx_kv.made;
    for (;;) {
        if (at - aotx_kv.served >= AOTX_KV_QUEUE_MAX) {
            atomicAdd(&aotx_kv.refused, 1u);
            return 0;
        }
        unsigned int was = atomicCAS(&aotx_kv.made, at, at + 1u);
        if (was == at) {
            break;
        }
        at = was;
    }
    aotx_kv_entry *entry = &aotx_kv.queue[at & (AOTX_KV_QUEUE_MAX - 1u)];
    entry->agent = agent;
    entry->pages = pages;
    __threadfence();
    return 1;
}

__device__ int aotx_kv_request(unsigned int agent, unsigned int pages)
{
    if (pages == 0u) {
        return 0;   /* a request for no page is a release, and it has its own call */
    }
    return aotx_kv_put(agent, pages);
}

__device__ int aotx_kv_release(unsigned int agent)
{
    int made = aotx_kv_put(agent, 0u);
    /* The queue is served in order, so the release takes every page of the slot before an
     * addition that follows it. The count of the slot therefore falls to zero now. A
     * caller that asks for pages between the release and the service would otherwise
     * count the pages it gives back. It would then ask for too few. */
    if (made != 0 && agent < AOTX_SLOTS) {
        aotx_kv.count[agent] = 0u;
    }
    return made;
}

/* One thread for each page position of each slot. A position with no page does nothing. */
__global__ void aotx_kv_stamp(void)
{
    unsigned int agent = blockIdx.x;
    unsigned int index = blockIdx.y*blockDim.x+threadIdx.x;
    if (agent >= AOTX_SLOTS || index >= AOTX_KV_PAGES_EACH) {
        return;
    }
    unsigned long long at = aotx_kv.page[agent][index];
    if (at == 0ull) {
        return;
    }
    unsigned char *header = (unsigned char *)at;
    header[0] = (unsigned char)AOTX_KV_TYPE_FP16;
    for (unsigned int b = 1u; b < AOTX_KV_HEADER_BYTES; ++b) {
        header[b] = 0u;
    }
}
