/* Purpose: Check the region map: the table, the bounds, and the guard gap that faults.
 * Owns: The test fixtures and the counts of the cases.
 * Launch shape: One thread for each lane of the batch.
 * Lifetime: One run of the test program. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "boot/check.h"
#include "mem/mem.cuh"

__device__ unsigned int aotx_mem_test_faults = 0u;

/* Each lane writes a distinct value at a distinct offset inside the region. */
__global__ void aotx_mem_test_fill(unsigned int kind, unsigned int lanes,
                                   unsigned long long stride)
{
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) {
        return;
    }
    const aotx_mem_region *region = aotx_mem_find(kind);
    unsigned long long at = region->base + (unsigned long long)lane * stride;
    if (!aotx_mem_holds(region, at, sizeof(unsigned long long))) {
        atomicAdd(&aotx_mem_test_faults, 1u);
        return;
    }
    *(unsigned long long *)at = 0x5EEDull + (unsigned long long)lane;
}

/* Each lane reads back what it wrote, so a wrong lane index cannot pass. */
__global__ void aotx_mem_test_check(unsigned int kind, unsigned int lanes,
                                    unsigned long long stride)
{
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) {
        return;
    }
    const aotx_mem_region *region = aotx_mem_find(kind);
    unsigned long long at = region->base + (unsigned long long)lane * stride;
    if (*(unsigned long long *)at != 0x5EEDull + (unsigned long long)lane) {
        atomicAdd(&aotx_mem_test_faults, 1u);
    }
}

/* Each lane writes one byte past the end of the region, which has no memory behind it. */
__global__ void aotx_mem_test_past(unsigned int lanes)
{
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) {
        return;
    }
    const aotx_mem_region *region = aotx_mem_find(AOTX_MEM_KIND_RING);
    unsigned char *at = (unsigned char *)(region->base + region->bytes) + lane;
    *at = (unsigned char)(lane + 1u);
}

static void aotx_mem_test_context(void)
{
    CUdevice device;
    CUcontext context;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

/* The child writes past the region and reports whether the write faulted. */
static int aotx_mem_test_child(unsigned int lanes)
{
    aotx_mem_map map;
    aotx_mem_test_context();
    if (aotx_mem_reserve(&map) != 0) {
        return 2;
    }
    unsigned int threads = 64u;
    unsigned int blocks = (lanes + threads - 1u) / threads;
    aotx_mem_test_past<<<blocks, threads>>>(lanes);
    cudaError_t state = cudaDeviceSynchronize();
    if (state != cudaSuccess) {
        printf("child: the write past the region faulted: %s\n", cudaGetErrorString(state));
        return 7;
    }
    printf("child: the write past the region did not fault\n");
    return 0;
}

/* The parent starts a child that faults, and waits for it to die. */
static int aotx_mem_test_fault(unsigned int lanes)
{
    char self[4096];
    char count[32];
    ssize_t length = readlink("/proc/self/exe", self, sizeof self - 1u);
    if (length <= 0) {
        return -1;
    }
    self[length] = '\0';
    snprintf(count, sizeof count, "%u", lanes);
    pid_t child = fork();
    if (child < 0) {
        return -1;
    }
    if (child == 0) {
        char *argv[] = { self, (char *)"--fault", count, NULL };
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

int main(int argc, char **argv)
{
    if (argc >= 3 && strcmp(argv[1], "--fault") == 0) {
        return aotx_mem_test_child((unsigned int)strtoul(argv[2], NULL, 10));
    }

    unsigned int applied = 0u;
    unsigned int failed = 0u;
    aotx_mem_map map;
    aotx_mem_table table;

    aotx_mem_test_context();
    if (aotx_mem_reserve(&map) != 0) {
        printf("mem: the map did not open\n");
        return 1;
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_mem_region_table, sizeof table),
                       "cudaMemcpyFromSymbol");

    applied += 1u;
    if (table.count != 2u) {
        printf("mem: the table holds %u regions\n", table.count);
        failed += 1u;
    }
    applied += 1u;
    if (table.region[0].kind != AOTX_MEM_KIND_RING
        || table.region[0].bytes < AOTX_MEM_RING_BYTES) {
        printf("mem: the first region is not the ring\n");
        failed += 1u;
    }
    applied += 1u;
    if (table.region[1].kind != AOTX_MEM_KIND_SCRATCH
        || table.region[1].bytes < AOTX_MEM_SCRATCH_BYTES) {
        printf("mem: the second region is not the scratch arena\n");
        failed += 1u;
    }
    applied += 1u;
    unsigned long long gap = table.region[1].base
                           - (table.region[0].base + table.region[0].bytes);
    if (gap < AOTX_MEM_GUARD_BYTES) {
        printf("mem: the gap between the regions is %llu bytes\n", gap);
        failed += 1u;
    }

    /* The batch runs at one lane and at 64 lanes, over both regions. */
    const unsigned int counts[2] = { 1u, 64u };
    const unsigned int kinds[2] = { AOTX_MEM_KIND_RING, AOTX_MEM_KIND_SCRATCH };
    for (unsigned int c = 0u; c < 2u; ++c) {
        for (unsigned int k = 0u; k < 2u; ++k) {
            unsigned int lanes = counts[c];
            unsigned int zero = 0u;
            aotx_check_runtime(cudaMemcpyToSymbol(aotx_mem_test_faults, &zero, sizeof zero),
                               "cudaMemcpyToSymbol");
            unsigned int threads = 64u;
            unsigned int blocks = (lanes + threads - 1u) / threads;
            unsigned long long stride = 4096ull;
            aotx_mem_test_fill<<<blocks, threads>>>(kinds[k], lanes, stride);
            aotx_mem_test_check<<<blocks, threads>>>(kinds[k], lanes, stride);
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            unsigned int faults = 0u;
            aotx_check_runtime(cudaMemcpyFromSymbol(&faults, aotx_mem_test_faults,
                                                    sizeof faults),
                               "cudaMemcpyFromSymbol");
            applied += 1u;
            if (faults != 0u) {
                printf("mem: %u lanes of kind %u did not read back\n", faults, kinds[k]);
                failed += 1u;
            }
        }
    }

    /* The guard gap: a write one byte past the ring kills the child. The fault reaches the
     * kernel log and kills a context on the display device, so it runs only on request. */
    if (getenv("AOTX_FAULT_TESTS") == NULL) {
        printf("mem: skipped 2 fault cases; set AOTX_FAULT_TESTS=1 to run them\n");
    }
    for (unsigned int c = 0u; c < 2u && getenv("AOTX_FAULT_TESTS") != NULL; ++c) {
        int died = aotx_mem_test_fault(counts[c]);
        applied += 1u;
        if (died != 1) {
            printf("mem: the write past the ring at %u lanes did not fault\n", counts[c]);
            failed += 1u;
        }
    }

    aotx_mem_release(&map);
    printf("mem: %u cases applied, %u failed\n", applied, failed);
    return failed == 0u ? 0 : 1;
}
