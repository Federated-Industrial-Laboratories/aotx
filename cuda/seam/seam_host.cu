/* Purpose: Make the two rings that cross the seam and start the disk side programs.
 * Owns: The ring files, their mappings and their registration.
 * Launch shape: Host glue only; no kernels.
 * Lifetime: From the ring open at start to the close at exit. */
#include <cuda_runtime.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "boot/check.h"
#include "seam/seam.cuh"

#define AOTX_PAGE_BYTES 4096ull

static unsigned long long aotx_seam_page_round(unsigned long long bytes)
{
    return ((bytes + AOTX_PAGE_BYTES - 1ull) / AOTX_PAGE_BYTES) * AOTX_PAGE_BYTES;
}

/* One ring file: memory that has a file descriptor, so another process maps the same bytes.
 * The descriptor stays open across an exec, which is how the disk side attaches. */
static unsigned char *aotx_seam_make(const char *name, unsigned long long bytes, int *out_fd)
{
    int fd = memfd_create(name, 0);
    if (fd < 0) {
        return 0;
    }
    if (ftruncate(fd, (off_t)bytes) != 0) {
        close(fd);
        return 0;
    }
    void *at = mmap(0, (size_t)bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (at == MAP_FAILED) {
        close(fd);
        return 0;
    }
    memset(at, 0, (size_t)bytes);
    aotx_check_runtime(cudaHostRegister(at, (size_t)bytes, cudaHostRegisterMapped),
                       "cudaHostRegister");
    *out_fd = fd;
    return (unsigned char *)at;
}

int aotx_seam_open(aotx_seam_rings *rings, unsigned long long boot_id)
{
    memset(rings, 0, sizeof *rings);
    rings->host_bytes = aotx_seam_page_round(sizeof(aotx_host_ring_preamble)
                                             + AOTX_HOST_RING_DATA_BYTES);
    rings->inbound_bytes = aotx_seam_page_round(sizeof(aotx_inbound_preamble)
                                                + AOTX_INBOUND_SLOTS * AOTX_SLOT_BYTES);
    rings->host_map = aotx_seam_make("aotx-host-ring", rings->host_bytes, &rings->host_fd);
    rings->inbound_map = aotx_seam_make("aotx-inbound-ring", rings->inbound_bytes,
                                        &rings->inbound_fd);
    if (rings->host_map == 0 || rings->inbound_map == 0) {
        return 1;
    }

    /* The preamble is written once, so a reader that attaches late still knows the layout. */
    aotx_host_ring_preamble *host = (aotx_host_ring_preamble *)rings->host_map;
    host->magic = AOTX_WIRE_MAGIC;
    host->layout = (unsigned short)AOTX_WIRE_LAYOUT;
    host->closed = 0u;
    host->boot_id = boot_id;
    host->data_bytes = AOTX_HOST_RING_DATA_BYTES;
    host->preamble_bytes = sizeof(aotx_host_ring_preamble);

    aotx_inbound_preamble *inbound = (aotx_inbound_preamble *)rings->inbound_map;
    inbound->magic = AOTX_WIRE_MAGIC;
    inbound->layout = (unsigned short)AOTX_WIRE_LAYOUT;
    inbound->closed = 0u;
    inbound->slot_count = AOTX_INBOUND_SLOTS;
    inbound->preamble_bytes = sizeof(aotx_inbound_preamble);
    __sync_synchronize();
    return 0;
}

int aotx_seam_bind(const aotx_seam_rings *rings, unsigned long long ring_base,
                   unsigned long long ring_bytes, unsigned long long boot_id)
{
    void *host_device = 0;
    void *inbound_device = 0;
    aotx_check_runtime(cudaHostGetDevicePointer(&host_device, rings->host_map, 0),
                       "cudaHostGetDevicePointer");
    aotx_check_runtime(cudaHostGetDevicePointer(&inbound_device, rings->inbound_map, 0),
                       "cudaHostGetDevicePointer");

    aotx_seam_state state;
    memset(&state, 0, sizeof state);
    state.boot_id = boot_id;
    state.dev.base = (unsigned char *)ring_base;
    state.dev.slot_count = ring_bytes / AOTX_SLOT_BYTES;
    if (state.dev.slot_count > AOTX_DEVICE_RING_SLOTS) {
        state.dev.slot_count = AOTX_DEVICE_RING_SLOTS;
    }
    state.dev.mask = state.dev.slot_count - 1ull;
    state.host.preamble = (unsigned char *)host_device;
    state.host.data = (unsigned char *)host_device + sizeof(aotx_host_ring_preamble);
    state.host.data_bytes = AOTX_HOST_RING_DATA_BYTES;
    state.host.mask = AOTX_HOST_RING_DATA_BYTES - 1ull;
    state.in.preamble = (unsigned char *)inbound_device;
    state.in.slots = (unsigned char *)inbound_device + sizeof(aotx_inbound_preamble);
    state.in.slot_count = AOTX_INBOUND_SLOTS;
    state.in.mask = AOTX_INBOUND_SLOTS - 1ull;
    state.apply.state_hash = AOTX_FNV_BASIS;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &state, sizeof state),
                       "cudaMemcpyToSymbol");
    return 0;
}

void aotx_seam_finish(const aotx_seam_rings *rings)
{
    volatile aotx_host_ring_preamble *host = (volatile aotx_host_ring_preamble *)rings->host_map;
    volatile aotx_inbound_preamble *inbound =
        (volatile aotx_inbound_preamble *)rings->inbound_map;
    __sync_synchronize();
    host->closed = 1u;
    inbound->closed = 1u;
    __sync_synchronize();
}

void aotx_seam_close(aotx_seam_rings *rings)
{
    if (rings->host_map != 0) {
        cudaHostUnregister(rings->host_map);
        munmap(rings->host_map, (size_t)rings->host_bytes);
        close(rings->host_fd);
        rings->host_map = 0;
    }
    if (rings->inbound_map != 0) {
        cudaHostUnregister(rings->inbound_map);
        munmap(rings->inbound_map, (size_t)rings->inbound_bytes);
        close(rings->inbound_fd);
        rings->inbound_map = 0;
    }
}

int aotx_seam_spawn(const char *path, char *const argv[], int *pid)
{
    pid_t child = fork();
    if (child < 0) {
        return 1;
    }
    if (child == 0) {
        execv(path, argv);
        _exit(127);
    }
    *pid = (int)child;
    return 0;
}

int aotx_seam_poll(int pid, int *stopped, int *status)
{
    int state = 0;
    pid_t got = waitpid((pid_t)pid, &state, WNOHANG);
    *stopped = (got == (pid_t)pid) ? 1 : 0;
    *status = WIFEXITED(state) ? WEXITSTATUS(state) : -1;
    return (got < 0) ? 1 : 0;
}

int aotx_seam_wait(int pid)
{
    int state = 0;
    if (waitpid((pid_t)pid, &state, 0) != (pid_t)pid) {
        return -1;
    }
    return WIFEXITED(state) ? WEXITSTATUS(state) : -1;
}
