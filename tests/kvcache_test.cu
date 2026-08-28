/* Purpose: Check the page cache: requests, maps, the page header, release and re-use.
 * Owns: The test fixtures and the counts of the cases.
 * Launch shape: One thread for each agent; one block for each page.
 * Lifetime: One run of the test program. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "boot/check.h"
#include "kvcache/kvcache.cuh"
#include "mem/mem.cuh"

#define AOTX_TEST_PAGES 2u

__device__ unsigned int aotx_kv_test_faults = 0u;

/* The value at one word of one page. The agent, the page and the position all take part,
 * so a wrong index cannot pass. */
__device__ __host__ static unsigned long long aotx_kv_test_word(unsigned int agent,
                                                                unsigned int page,
                                                                unsigned long long at)
{
    return ((unsigned long long)agent << 40) ^ ((unsigned long long)page << 24)
         ^ (at * 0x9E3779B97F4A7C15ull);
}

/* One thread for each agent of the batch. */
__global__ void aotx_kv_test_ask(unsigned int agents, unsigned int pages)
{
    unsigned int agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent >= agents) {
        return;
    }
    if (aotx_kv_request(agent, pages) == 0) {
        atomicAdd(&aotx_kv_test_faults, 1u);
    }
}

__global__ void aotx_kv_test_give(unsigned int agents)
{
    unsigned int agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent >= agents) {
        return;
    }
    if (aotx_kv_release(agent) == 0) {
        atomicAdd(&aotx_kv_test_faults, 1u);
    }
}

/* One block for each page of each agent. The header of the page is left as it is. */
__global__ void aotx_kv_test_fill(unsigned int agents, unsigned int pages)
{
    unsigned int agent = blockIdx.x;
    unsigned int page = blockIdx.y;
    if (agent >= agents || page >= pages) {
        return;
    }
    unsigned long long at = aotx_kv_page(agent, page);
    if (at == 0ull) {
        atomicAdd(&aotx_kv_test_faults, 1u);
        return;
    }
    unsigned long long *words = (unsigned long long *)(at + AOTX_KV_HEADER_BYTES);
    unsigned long long count = (AOTX_KV_PAGE_BYTES - AOTX_KV_HEADER_BYTES) / 8ull;
    for (unsigned long long w = threadIdx.x; w < count; w += blockDim.x) {
        words[w] = aotx_kv_test_word(agent, page, w);
    }
}

__global__ void aotx_kv_test_read(unsigned int agents, unsigned int pages)
{
    unsigned int agent = blockIdx.x;
    unsigned int page = blockIdx.y;
    if (agent >= agents || page >= pages) {
        return;
    }
    unsigned long long at = aotx_kv_page(agent, page);
    if (at == 0ull) {
        atomicAdd(&aotx_kv_test_faults, 1u);
        return;
    }
    const unsigned char *header = (const unsigned char *)at;
    if (threadIdx.x == 0u && header[0] != (unsigned char)AOTX_KV_TYPE_FP16) {
        atomicAdd(&aotx_kv_test_faults, 1u);
    }
    const unsigned long long *words =
        (const unsigned long long *)(at + AOTX_KV_HEADER_BYTES);
    unsigned long long count = (AOTX_KV_PAGE_BYTES - AOTX_KV_HEADER_BYTES) / 8ull;
    for (unsigned long long w = threadIdx.x; w < count; w += blockDim.x) {
        if (words[w] != aotx_kv_test_word(agent, page, w)) {
            atomicAdd(&aotx_kv_test_faults, 1u);
            return;
        }
    }
}

/* A read of the gap at the end of the range, which has no memory behind it. */
__global__ void aotx_kv_test_past(unsigned long long address, unsigned int lanes)
{
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) {
        return;
    }
    const unsigned long long *at = (const unsigned long long *)address;
    if (at[lane] != 0ull) {
        atomicAdd(&aotx_kv_test_faults, 1u);
    }
}

static void aotx_kv_test_context(void)
{
    CUdevice device;
    CUcontext context;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

static unsigned int aotx_kv_test_reset(void)
{
    unsigned int zero = 0u;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_kv_test_faults, &zero, sizeof zero),
                       "cudaMemcpyToSymbol");
    return 0u;
}

static unsigned int aotx_kv_test_took(void)
{
    unsigned int faults = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&faults, aotx_kv_test_faults, sizeof faults),
                       "cudaMemcpyFromSymbol");
    return faults;
}

static void aotx_kv_test_table(aotx_kv_table *table)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_kv, sizeof *table),
                       "cudaMemcpyFromSymbol");
}

/* The child reads the gap at the end of the range and reports whether the read faulted. */
static int aotx_kv_test_child(void)
{
    aotx_kv_map map;
    aotx_kv_test_context();
    if (aotx_kv_open(&map) != 0) {
        return 2;
    }
    aotx_kv_test_past<<<1, 64>>>(map.range + AOTX_KV_RANGE_BYTES, 64u);
    cudaError_t state = cudaDeviceSynchronize();
    if (state != cudaSuccess) {
        printf("child: the read of the gap faulted: %s\n", cudaGetErrorString(state));
        return 7;
    }
    printf("child: the read of the gap did not fault\n");
    return 0;
}

static int aotx_kv_test_fault(void)
{
    char self[4096];
    ssize_t length = readlink("/proc/self/exe", self, sizeof self - 1u);
    if (length <= 0) {
        return -1;
    }
    self[length] = '\0';
    pid_t child = fork();
    if (child < 0) {
        return -1;
    }
    if (child == 0) {
        char *argv[] = { self, (char *)"--fault", NULL };
        execv(self, argv);
        _exit(127);
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child) {
        return -1;
    }
    if (WIFSIGNALED(status)) {
        return 1;
    }
    return (WIFEXITED(status) && WEXITSTATUS(status) != 0) ? 1 : 0;
}

/* One round: ask for pages, take them, write, read back, and give them back. */
static unsigned int aotx_kv_test_round(aotx_kv_map *map, unsigned int agents,
                                       unsigned int *failed)
{
    aotx_kv_table table;
    unsigned int applied = 0u;
    unsigned int threads = 64u;
    unsigned int blocks = (agents + threads - 1u) / threads;
    aotx_kv_test_reset();
    aotx_kv_test_ask<<<blocks, threads>>>(agents, AOTX_TEST_PAGES);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    int served = aotx_kv_serve(map, 0);
    aotx_kv_test_table(&table);

    applied += 2u;
    if (served != (int)agents) {
        printf("kvcache: %d requests of %u were answered\n", served, agents);
        *failed += 1u;
    }
    unsigned int short_of = 0u;
    for (unsigned int a = 0u; a < agents; ++a) {
        if (table.count[a] != AOTX_TEST_PAGES) {
            short_of += 1u;
        }
        for (unsigned int p = 0u; p < AOTX_TEST_PAGES; ++p) {
            if (table.page[a][p] == 0ull) {
                short_of += 1u;
            }
        }
    }
    if (short_of != 0u) {
        printf("kvcache: %u page table entries of %u agents are empty\n", short_of, agents);
        *failed += 1u;
    }

    /* No two agents hold the same page. */
    applied += 1u;
    unsigned int shared = 0u;
    for (unsigned int a = 0u; a < agents; ++a) {
        for (unsigned int p = 0u; p < AOTX_TEST_PAGES; ++p) {
            for (unsigned int b = a; b < agents; ++b) {
                for (unsigned int q = (b == a) ? p + 1u : 0u; q < AOTX_TEST_PAGES; ++q) {
                    if (table.page[a][p] == table.page[b][q]) {
                        shared += 1u;
                    }
                }
            }
        }
    }
    if (shared != 0u) {
        printf("kvcache: %u pages are in two page tables\n", shared);
        *failed += 1u;
    }

    applied += 1u;
    if (table.mapped_pages != agents * AOTX_TEST_PAGES) {
        printf("kvcache: the table states %u mapped pages and %u agents hold %u each\n",
               table.mapped_pages, agents, AOTX_TEST_PAGES);
        *failed += 1u;
    }

    dim3 grid(agents, AOTX_TEST_PAGES, 1);
    aotx_kv_test_fill<<<grid, 256>>>(agents, AOTX_TEST_PAGES);
    aotx_kv_test_read<<<grid, 256>>>(agents, AOTX_TEST_PAGES);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    applied += 1u;
    if (aotx_kv_test_took() != 0u) {
        printf("kvcache: %u pages of %u agents did not read back\n",
               aotx_kv_test_took(), agents);
        *failed += 1u;
    }

    aotx_kv_test_reset();
    aotx_kv_test_give<<<blocks, threads>>>(agents);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_kv_serve(map, 0);
    aotx_kv_test_table(&table);
    applied += 1u;
    unsigned int kept = 0u;
    for (unsigned int a = 0u; a < agents; ++a) {
        kept += table.count[a];
        for (unsigned int p = 0u; p < AOTX_KV_PAGES_EACH; ++p) {
            if (table.page[a][p] != 0ull) {
                kept += 1u;
            }
        }
    }
    if (kept != 0u) {
        printf("kvcache: %u page table entries of %u agents stand after the release\n",
               kept, agents);
        *failed += 1u;
    }
    applied += 1u;
    if (table.mapped_pages != 0u) {
        printf("kvcache: the table states %u mapped pages after the release\n",
               table.mapped_pages);
        *failed += 1u;
    }
    return applied;
}

int main(int argc, char **argv)
{
    aotx_kv_map map;
    unsigned int applied = 0u;
    unsigned int failed = 0u;
    if (argc >= 2 && strcmp(argv[1], "--fault") == 0) {
        return aotx_kv_test_child();
    }

    aotx_kv_test_context();
    if (aotx_kv_open(&map) != 0) {
        printf("kvcache: the range did not open\n");
        return 1;
    }
    unsigned long long range_bytes = map.range_bytes;

    /* Case set 1: one agent, then AOTX_SLOTS agents, each with two pages. Every round
     * gives its pages back, so the round that follows takes the same physical memory
     * again. */
    const unsigned int counts[2] = { 1u, AOTX_SLOTS };
    unsigned int created[3];
    created[0] = map.created;
    for (unsigned int c = 0u; c < 2u; ++c) {
        applied += aotx_kv_test_round(&map, counts[c], &failed);
        created[c + 1u] = map.created;
    }

    /* Case set 2: the second round asked for 64 times two pages, and the first for two.
     * The reservation did not grow, and the physical memory of the first round came back. */
    applied += 2u;
    if (map.range_bytes != range_bytes) {
        printf("kvcache: the reservation grew from %llu to %llu bytes\n",
               range_bytes, map.range_bytes);
        failed += 1u;
    }
    if (created[2] > counts[1] * AOTX_TEST_PAGES) {
        printf("kvcache: %u pages were made for a peak of %u\n",
               created[2], counts[1] * AOTX_TEST_PAGES);
        failed += 1u;
    }

    /* A third round at the peak count makes no new physical memory at all. */
    applied += 1u;
    unsigned int before = map.created;
    applied += aotx_kv_test_round(&map, counts[1], &failed);
    if (map.created != before) {
        printf("kvcache: the third round made %u more pages\n", map.created - before);
        failed += 1u;
    }
    printf("kvcache: pages made %u, reservation %llu MB, peak agents %u\n",
           map.created, map.range_bytes >> 20, counts[1]);

    /* Case set 3: more pages than the range holds. The positions run out, the request is
     * filled as far as it goes, and a count states the requests that no position filled. */
    {
        aotx_kv_table table;
        unsigned int threads = 64u;
        unsigned int blocks = (counts[1] + threads - 1u) / threads;
        aotx_kv_test_table(&table);
        unsigned int short_of = table.short_of;
        aotx_kv_test_reset();
        aotx_kv_test_ask<<<blocks, threads>>>(counts[1], AOTX_KV_PAGES_EACH);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_kv_serve(&map, 0);
        aotx_kv_test_table(&table);
        applied += 2u;
        if (table.short_of <= short_of) {
            printf("kvcache: %u agents asked for %u pages each and nothing was counted\n",
                   counts[1], AOTX_KV_PAGES_EACH);
            failed += 1u;
        }
        if (table.mapped_pages != AOTX_KV_PAGES) {
            printf("kvcache: %u pages of %u are mapped when every position is taken\n",
                   table.mapped_pages, AOTX_KV_PAGES);
            failed += 1u;
        }
        printf("kvcache: %u requests could not be filled, %u pages mapped of %u\n",
               table.short_of - short_of, table.mapped_pages, AOTX_KV_PAGES);
        aotx_kv_test_give<<<blocks, threads>>>(counts[1]);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_kv_serve(&map, 0);
        aotx_kv_test_table(&table);
        applied += 1u;
        if (table.mapped_pages != 0u) {
            printf("kvcache: %u pages stand mapped after every agent gave its pages back\n",
                   table.mapped_pages);
            failed += 1u;
        }
    }

    /* Case set 4: a read of the gap at the end of the range kills the child. The fault
     * reaches the kernel log and kills a context on the display device, so it runs only on
     * request. */
    if (getenv("AOTX_FAULT_TESTS") == NULL) {
        printf("kvcache: skipped 1 fault case; set AOTX_FAULT_TESTS=1 to run it\n");
    } else {
        applied += 1u;
        if (aotx_kv_test_fault() != 1) {
            printf("kvcache: the read of an unmapped page did not fault\n");
            failed += 1u;
        }
    }

    aotx_kv_close(&map);
    printf("kvcache: %u cases applied, %u failed\n", applied, failed);
    return failed == 0u ? 0 : 1;
}
