/* Purpose: Allocate and register the optional checkpoint transport.
 * Owns: Its memfd, mapping and CUDA registration.
 * Launch shape: Host glue only; checkpoint encoding stays on the device.
 * Lifetime: One runtime; the drain receives the same descriptor. */
#include "cognitive/checkpoint.cuh"
#include "seam/seam.cuh"
#include "boot/check.h"
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

static int aotx_checkpoint_enabled;

int aotx_checkpoint_open(aotx_seam_rings *rings, uint64_t boot) {
    uint64_t bytes = (AOTX_CP_RING_BYTES + 4095) & ~4095ull;
    int fd = memfd_create("aotx-memory-checkpoints", MFD_CLOEXEC);
    if (fd < 0) return 1;
    if (ftruncate(fd, (off_t)bytes)) { close(fd); return 1; }
    void *map = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { close(fd); return 1; }
    if (cudaHostRegister(map, bytes, cudaHostRegisterMapped) != cudaSuccess) {
        munmap(map, bytes); close(fd); return 1;
    }
    aotx_checkpoint_ring *ring = (aotx_checkpoint_ring *)map;
    memset(ring, 0, sizeof(*ring));
    ring->magic = AOTX_CP_MAGIC; ring->layout = AOTX_CP_LAYOUT; ring->boot = boot;
    ring->slots = AOTX_MEMORY_SNAPSHOTS; ring->slot_bytes = AOTX_CP_SLOT_BYTES;
    void *device = NULL;
    aotx_check_runtime(cudaHostGetDevicePointer(&device, map, 0), "cudaHostGetDevicePointer");
    aotx_checkpoint_state state = {};
    state.ring = (aotx_checkpoint_ring *)device;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_checkpoint, &state, sizeof(state)), "cudaMemcpyToSymbol");
    rings->checkpoint_fd = fd; rings->checkpoint_map = (unsigned char *)map;
    rings->checkpoint_bytes = bytes; aotx_checkpoint_enabled = 1;
    return 0;
}
void aotx_checkpoint_close(aotx_seam_rings *rings) {
    if (!rings->checkpoint_map) return;
    aotx_checkpoint_enabled = 0;
    cudaHostUnregister(rings->checkpoint_map);
    munmap(rings->checkpoint_map, rings->checkpoint_bytes);
    close(rings->checkpoint_fd); rings->checkpoint_map = NULL;
}
int aotx_checkpoint_capture(void *stream) {
    if (!aotx_checkpoint_enabled) return 0;
    cudaStream_t on = (cudaStream_t)stream;
    aotx_checkpoint_step<<<1,64,0,on>>>();
    aotx_checkpoint_fill<<<128,256,0,on>>>();
    aotx_checkpoint_copy<<<16,256,0,on>>>();
    aotx_checkpoint_publish<<<1,1,0,on>>>();
    return 0;
}
