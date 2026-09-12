/* Purpose: Map the bounded image producer ring for the feeder and device.
 * Owns: One shared descriptor, mapping and CUDA registration.
 * Launch shape: Host glue only; CUDA consumes frames in its tick graph.
 * Lifetime: Runtime startup through shutdown. */
#include "media/runtime.cuh"
#include <stddef.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
int aotx_media_ring_open(aotx_seam_rings *r)
{
    r->media_fd = -1;
    r->media_bytes = (sizeof(aotx_media_preamble) +
        (unsigned long long)AOTX_MEDIA_RING_SLOTS * AOTX_MEDIA_FRAME_BYTES + 4095u) & ~4095ull;
    int fd = memfd_create("aotx-media", MFD_CLOEXEC);
    if (fd < 0) return 1;
    if (ftruncate(fd, (off_t)r->media_bytes)) { close(fd); return 1; }
    void *map = mmap(0, r->media_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { close(fd); return 1; }
    memset(map, 0, r->media_bytes);
    if (cudaHostRegister(map, r->media_bytes, cudaHostRegisterMapped) != cudaSuccess) {
        munmap(map, r->media_bytes); close(fd); return 1;
    }
    r->media_map = (unsigned char *)map; r->media_fd = fd;
    aotx_media_preamble *p = (aotx_media_preamble *)map;
    p->magic = AOTX_MEDIA_RING_MAGIC; p->schema = AOTX_MEDIA_SCHEMA;
    p->slots = AOTX_MEDIA_RING_SLOTS; p->frame_bytes = AOTX_MEDIA_FRAME_BYTES;
    void *device = 0;
    if (cudaHostGetDevicePointer(&device, map, 0) != cudaSuccess) {
        aotx_media_ring_close(r); return 1;
    }
    unsigned char *frames = (unsigned char *)device + sizeof *p;
    if (cudaMemcpyToSymbol(aotx_media, &device, sizeof device, offsetof(aotx_media_state, ring)) != cudaSuccess ||
        cudaMemcpyToSymbol(aotx_media, &frames, sizeof frames, offsetof(aotx_media_state, frames)) != cudaSuccess) {
        aotx_media_ring_close(r); return 1;
    }
    return 0;
}
void aotx_media_ring_finish(const aotx_seam_rings *r)
{
    if (r->media_map) __atomic_store_n(&((aotx_media_preamble *)r->media_map)->closed, 1u, __ATOMIC_RELEASE);
}
void aotx_media_ring_close(aotx_seam_rings *r)
{
    if (!r->media_map) return;
    cudaHostUnregister(r->media_map); munmap(r->media_map, r->media_bytes); close(r->media_fd);
    r->media_map = 0; r->media_fd = -1;
}
