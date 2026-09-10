/* Purpose: Make the rings and the mirror that cross the seam and start the disk programs.
 * Owns: The ring files, the mirror file, their mappings and their registration.
 * Launch shape: Host glue only; no kernels.
 * Lifetime: From the ring open at start to the close at exit. */
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stddef.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "boot/check.h"
#include "seam/seam.cuh"
#include "ui/mirror.h"

#define AOTX_PAGE_BYTES 4096ull

static unsigned long long aotx_seam_page_round(unsigned long long bytes)
{
    return ((bytes + AOTX_PAGE_BYTES - 1ull) / AOTX_PAGE_BYTES) * AOTX_PAGE_BYTES;
}

/* One ring file: memory that has a file descriptor, so another process maps the same bytes.
 * The descriptor stays open across an exec, which is how the disk side attaches. */
static unsigned char *aotx_seam_make(const char *name, unsigned long long bytes, int *out_fd)
{
    int fd = memfd_create(name, MFD_CLOEXEC);
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
    rings->bulk_bytes = aotx_seam_page_round(sizeof(aotx_host_ring_preamble)
                                             + AOTX_BULK_RING_DATA_BYTES);
    rings->inbound_bytes = aotx_seam_page_round(sizeof(aotx_inbound_preamble)
                                                + AOTX_INBOUND_SLOTS * AOTX_SLOT_BYTES);
    /* The device writes the snapshot with 16-byte stores, so one slot holds a whole number
     * of them and every slot starts on that boundary. */
    unsigned long long slot_bytes = ((sizeof(aotx_mirror_snapshot) + 15ull) / 16ull) * 16ull;
    rings->mirror_bytes = aotx_seam_page_round(sizeof(aotx_mirror_preamble)
                                               + AOTX_MIRROR_SLOTS * slot_bytes);
    rings->host_map = aotx_seam_make("aotx-host-ring", rings->host_bytes, &rings->host_fd);
    rings->bulk_map = aotx_seam_make("aotx-bulk-ring", rings->bulk_bytes, &rings->bulk_fd);
    rings->inbound_map = aotx_seam_make("aotx-inbound-ring", rings->inbound_bytes,
                                        &rings->inbound_fd);
    rings->mirror_map = aotx_seam_make("aotx-mirror", rings->mirror_bytes,
                                       &rings->mirror_fd);
    if (rings->host_map == 0 || rings->inbound_map == 0 || rings->bulk_map == 0
        || rings->mirror_map == 0) {
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

    /* The bulk ring carries the same preamble and the same block protocol. */
    aotx_host_ring_preamble *bulk = (aotx_host_ring_preamble *)rings->bulk_map;
    bulk->magic = AOTX_WIRE_MAGIC;
    bulk->layout = (unsigned short)AOTX_WIRE_LAYOUT;
    bulk->closed = 0u;
    bulk->boot_id = boot_id;
    bulk->data_bytes = AOTX_BULK_RING_DATA_BYTES;
    bulk->preamble_bytes = sizeof(aotx_host_ring_preamble);

    aotx_inbound_preamble *inbound = (aotx_inbound_preamble *)rings->inbound_map;
    inbound->magic = AOTX_WIRE_MAGIC;
    inbound->layout = (unsigned short)AOTX_WIRE_LAYOUT;
    inbound->closed = 0u;
    inbound->slot_count = AOTX_INBOUND_SLOTS;
    inbound->preamble_bytes = sizeof(aotx_inbound_preamble);

    /* The mirror preamble states the shape of a snapshot. The feeder writes the attached
     * count. The mirror node writes the device ring use for a client. */
    aotx_mirror_preamble *mirror = (aotx_mirror_preamble *)rings->mirror_map;
    mirror->magic = AOTX_MIRROR_MAGIC;
    mirror->layout = AOTX_MIRROR_LAYOUT;
    mirror->slots = AOTX_MIRROR_SLOTS;
    mirror->slot_bytes = (unsigned int)slot_bytes;
    mirror->cols = AOTX_MIRROR_COLS;
    mirror->rows = AOTX_MIRROR_ROWS;
    mirror->attached = 0u;
    mirror->device_ring_used = 0ull;
    mirror->device_ring_slots = AOTX_DEVICE_RING_SLOTS;
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

/* The bulk path stays inert while the ring address is zero. A build that does not open the
 * bulk ring refuses every stage request and writes nothing. */
int aotx_seam_bind_bulk(const aotx_seam_rings *rings, unsigned long long stage_base,
                        unsigned long long stage_bytes)
{
    void *bulk_device = 0;
    if (rings->bulk_map == 0 || stage_base == 0ull || stage_bytes == 0ull) {
        return 1;
    }
    aotx_check_runtime(cudaHostGetDevicePointer(&bulk_device, rings->bulk_map, 0),
                       "cudaHostGetDevicePointer");

    aotx_bulk_state state;
    memset(&state, 0, sizeof state);
    state.ring.preamble = (unsigned char *)bulk_device;
    state.ring.data = (unsigned char *)bulk_device + sizeof(aotx_host_ring_preamble);
    state.ring.data_bytes = AOTX_BULK_RING_DATA_BYTES;
    state.ring.mask = AOTX_BULK_RING_DATA_BYTES - 1ull;
    state.stage = (unsigned char *)stage_base;
    state.stage_bytes = stage_bytes;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_bulk, &state, sizeof state),
                       "cudaMemcpyToSymbol");
    return 0;
}

void aotx_seam_set_replaying(int on)
{
    unsigned long long value = (on != 0) ? 1ull : 0ull;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &value, sizeof value,
                                          offsetof(aotx_seam_state, replaying)),
                       "cudaMemcpyToSymbol");
}

void aotx_seam_finish(const aotx_seam_rings *rings)
{
    volatile aotx_host_ring_preamble *host = (volatile aotx_host_ring_preamble *)rings->host_map;
    volatile aotx_host_ring_preamble *bulk = (volatile aotx_host_ring_preamble *)rings->bulk_map;
    volatile aotx_inbound_preamble *inbound =
        (volatile aotx_inbound_preamble *)rings->inbound_map;
    __sync_synchronize();
    host->closed = 1u;
    inbound->closed = 1u;
    if (bulk != 0) {
        bulk->closed = 1u;
    }
    __sync_synchronize();
}

void aotx_seam_close(aotx_seam_rings *rings)
{
    aotx_checkpoint_close(rings);
    if (rings->host_map != 0) {
        cudaHostUnregister(rings->host_map);
        munmap(rings->host_map, (size_t)rings->host_bytes);
        close(rings->host_fd);
        rings->host_map = 0;
    }
    if (rings->bulk_map != 0) {
        cudaHostUnregister(rings->bulk_map);
        munmap(rings->bulk_map, (size_t)rings->bulk_bytes);
        close(rings->bulk_fd);
        rings->bulk_map = 0;
    }
    if (rings->inbound_map != 0) {
        cudaHostUnregister(rings->inbound_map);
        munmap(rings->inbound_map, (size_t)rings->inbound_bytes);
        close(rings->inbound_fd);
        rings->inbound_map = 0;
    }
    if (rings->mirror_map != 0) {
        cudaHostUnregister(rings->mirror_map);
        munmap(rings->mirror_map, (size_t)rings->mirror_bytes);
        close(rings->mirror_fd);
        rings->mirror_map = 0;
    }
}

/* Give the child the descriptors that keep names and no others. A descriptor that is not
 * named gets the close-on-exec flag, so the exec closes it. A ring file starts with that
 * flag, and a pipe does not, so both directions are set here. */
static void aotx_seam_only(const int *keep, unsigned int count)
{
    long limit = sysconf(_SC_OPEN_MAX);
    if (limit < 3 || limit > 65536) {
        limit = 1024;
    }
    for (int fd = 3; fd < (int)limit; ++fd) {
        int flags = fcntl(fd, F_GETFD);
        if (flags < 0) {
            continue;
        }
        unsigned int at = 0u;
        while (at < count && keep[at] != fd) {
            at += 1u;
        }
        if (at < count) {
            fcntl(fd, F_SETFD, flags & ~FD_CLOEXEC);
        } else {
            fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
        }
    }
}

int aotx_seam_spawn(const char *path, char *const argv[], const int *keep,
                    unsigned int keep_count, int *pid)
{
    pid_t child = fork();
    if (child < 0) {
        return 1;
    }
    if (child == 0) {
        aotx_seam_only(keep, keep_count);
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
