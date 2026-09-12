/* Purpose: Allocate the service tables and map its bounded local transport.
 * Owns: Device allocations, a shared mapping and its descriptor.
 * Launch shape: Host glue only; finite graph nodes own all request processing.
 * Lifetime: Runtime startup through shutdown. */
#include "service/service.cuh"
#include "service/host.h"
#include "media/runtime.cuh"
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <stddef.h>
static aotx_service_state aotx_service_allocations;

void aotx_service_finish(const aotx_seam_rings *r)
{
    if (r->service_map)
        __atomic_store_n(&((aotx_service_ring *)r->service_map)->closed, 1ull, __ATOMIC_RELEASE);
}
void aotx_service_close(aotx_seam_rings *r)
{
    if (!r->service_map && !aotx_service_allocations.enabled) return;
    cudaFree(aotx_service_allocations.jobs); cudaFree(aotx_service_allocations.grants);
    cudaFree(aotx_service_allocations.frames); cudaFree(aotx_service_allocations.ready);
    cudaFree(aotx_service_allocations.uploads);
    aotx_service_allocations = {};
    if (r->service_map) {
        cudaHostUnregister(r->service_map); munmap(r->service_map, r->service_bytes);
        close(r->service_fd); r->service_map = 0; r->service_fd = -1;
    }
    cudaMemcpyToSymbol(aotx_service, &aotx_service_allocations, sizeof aotx_service_allocations);
}
int aotx_service_open(aotx_seam_rings *r, unsigned long long epoch)
{
    r->service_fd = -1;
    r->service_bytes = (sizeof(aotx_service_ring) +
        (unsigned long long)AOTX_SERVICE_CHANNELS * sizeof(aotx_service_mailbox) + 4095ull) & ~4095ull;
    int fd = memfd_create("aotx-service", MFD_CLOEXEC);
    if (fd < 0) return 1;
    if (ftruncate(fd, (off_t)r->service_bytes)) { close(fd); return 1; }
    void *map = mmap(0, r->service_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { close(fd); return 1; }
    memset(map, 0, r->service_bytes);
    if (cudaHostRegister(map, r->service_bytes, cudaHostRegisterMapped) != cudaSuccess) {
        munmap(map, r->service_bytes); close(fd); return 1;
    }
    r->service_map = (unsigned char *)map; r->service_fd = fd;
    aotx_service_ring *p = (aotx_service_ring *)map;
    p->schema = AOTX_SERVICE_SCHEMA; p->channels = AOTX_SERVICE_CHANNELS;
    p->frame_bytes = AOTX_SERVICE_FRAME; p->epoch = epoch;
    void *device = 0;
    aotx_service_state *s = &aotx_service_allocations;
    if (cudaMemcpyFromSymbol(&s->media_count, aotx_media, sizeof s->media_count,
            offsetof(aotx_media_state, profile) + offsetof(aotx_media_profile, objects)) != cudaSuccess ||
        cudaHostGetDevicePointer(&device, map, 0) != cudaSuccess ||
        cudaMalloc(&s->jobs, (size_t)AOTX_SERVICE_REQUESTS * sizeof(aotx_service_job)) != cudaSuccess ||
        cudaMalloc(&s->grants, (size_t)AOTX_SERVICE_PRINCIPALS * sizeof(aotx_service_grant)) != cudaSuccess ||
        cudaMalloc(&s->frames, (size_t)AOTX_SERVICE_CHANNELS * AOTX_SERVICE_FRAME) != cudaSuccess ||
        cudaMalloc(&s->ready, (size_t)AOTX_SERVICE_CHANNELS * sizeof(unsigned)) != cudaSuccess ||
        (s->media_count && cudaMalloc(&s->uploads, (size_t)s->media_count * sizeof(aotx_service_upload)) != cudaSuccess)) {
        aotx_service_close(r); return 1;
    }
    s->ring = (aotx_service_ring *)device;
    s->mailbox = (aotx_service_mailbox *)((unsigned char *)device + sizeof *p);
    s->epoch = epoch; s->enabled = 1;
    s->bytes = (unsigned long long)AOTX_SERVICE_REQUESTS * sizeof(aotx_service_job) +
        (unsigned long long)AOTX_SERVICE_PRINCIPALS * sizeof(aotx_service_grant) +
        (unsigned long long)AOTX_SERVICE_CHANNELS * (AOTX_SERVICE_FRAME + sizeof(unsigned)) +
        (unsigned long long)s->media_count * sizeof(aotx_service_upload);
    if (cudaMemset(s->jobs, 0, (size_t)AOTX_SERVICE_REQUESTS * sizeof(aotx_service_job)) != cudaSuccess ||
        cudaMemset(s->ready, 0, (size_t)AOTX_SERVICE_CHANNELS * sizeof(unsigned)) != cudaSuccess ||
        (s->media_count && cudaMemset(s->uploads, 0, (size_t)s->media_count * sizeof(aotx_service_upload)) != cudaSuccess) ||
        cudaMemcpyToSymbol(aotx_service, s, sizeof *s) != cudaSuccess) {
        aotx_service_close(r); return 1;
    }
    return 0;
}
void aotx_service_capture(void *stream)
{
    if (!aotx_service_allocations.enabled) return;
    cudaStream_t on = (cudaStream_t)stream;
    aotx_service_copy<<<AOTX_SERVICE_CHANNELS, 256, 0, on>>>();
    aotx_service_admit<<<1, 1, 0, on>>>();
    aotx_service_work<<<1, 1, 0, on>>>();
}
void aotx_service_output_capture(void *stream)
{
    if (aotx_service_allocations.enabled)
        aotx_service_reply<<<(AOTX_SLOTS + 31) / 32, 32, 0, (cudaStream_t)stream>>>();
}
