/* Purpose: Map a ring memfd, check its preamble, and hold the link to the parent.
 * Owns: One mapping for each aotx_map, and the descriptor that the mapping came from.
 * Threading: One thread; a map is not shared between threads of one program.
 * Lifetime: From the map call to aotx_map_release. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/wire/diskwire.h"

#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

static int power_of_two(uint64_t v)
{
    return v != 0 && (v & (v - 1)) == 0;
}

int aotx_map_fd(int fd, aotx_map *out)
{
    struct stat st;
    void *base;
    if (fd < 0 || fstat(fd, &st) != 0 || st.st_size <= 0) {
        return -1;
    }
    base = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        return -1;
    }
    out->base = (unsigned char *)base;
    out->bytes = (size_t)st.st_size;
    out->fd = fd;
    return 0;
}

void aotx_map_release(aotx_map *m)
{
    if (m->base != NULL) {
        munmap(m->base, m->bytes);
    }
    m->base = NULL;
    m->bytes = 0;
}

int aotx_die_with_parent(void)
{
    return prctl(PR_SET_PDEATHSIG, SIGTERM, 0, 0, 0) == 0 ? 0 : -1;
}

int aotx_host_ring_attach(const aotx_map *m, aotx_host_ring *out)
{
    aotx_host_ring_preamble *p = (aotx_host_ring_preamble *)m->base;
    uint64_t area;
    if (m->bytes < sizeof(*p)) {
        return -1;
    }
    if (p->magic != AOTX_WIRE_MAGIC || p->layout != AOTX_WIRE_LAYOUT) {
        return -1;
    }
    if (p->preamble_bytes < sizeof(*p) || (p->preamble_bytes & 7u) != 0) {
        return -1;
    }
    if (!power_of_two(p->data_bytes) || p->data_bytes < AOTX_BLOCK_HEADER_BYTES) {
        return -1;
    }
    area = p->preamble_bytes + p->data_bytes;
    if (area > (uint64_t)m->bytes) {
        return -1;
    }
    out->pre = p;
    out->data = m->base + p->preamble_bytes;
    out->data_bytes = p->data_bytes;
    out->mask = p->data_bytes - 1;
    return 0;
}

int aotx_inbound_attach(const aotx_map *m, aotx_inbound_ring *out)
{
    aotx_inbound_preamble *p = (aotx_inbound_preamble *)m->base;
    uint64_t area;
    if (m->bytes < sizeof(*p)) {
        return -1;
    }
    if (p->magic != AOTX_WIRE_MAGIC || p->layout != AOTX_WIRE_LAYOUT) {
        return -1;
    }
    if (p->preamble_bytes < sizeof(*p) || (p->preamble_bytes & 7u) != 0) {
        return -1;
    }
    if (!power_of_two(p->slot_count)) {
        return -1;
    }
    area = p->preamble_bytes + p->slot_count * AOTX_SLOT_BYTES;
    if (area > (uint64_t)m->bytes) {
        return -1;
    }
    out->pre = p;
    out->slots = m->base + p->preamble_bytes;
    out->slot_count = p->slot_count;
    out->mask = p->slot_count - 1;
    return 0;
}

static int make_ring(const char *name, uint64_t bytes, aotx_map *m)
{
    int fd = memfd_create(name, 0);
    if (fd < 0) {
        return -1;
    }
    if (ftruncate(fd, (off_t)bytes) != 0) {
        close(fd);
        return -1;
    }
    if (aotx_map_fd(fd, m) != 0) {
        close(fd);
        return -1;
    }
    memset(m->base, 0, m->bytes);
    return 0;
}

int aotx_host_ring_create(uint64_t data_bytes, uint64_t boot_id, aotx_map *m, aotx_host_ring *out)
{
    aotx_host_ring_preamble *p;
    uint64_t preamble = sizeof(aotx_host_ring_preamble);
    if (!power_of_two(data_bytes) || data_bytes < AOTX_BLOCK_HEADER_BYTES) {
        return -1;
    }
    if (make_ring("aotx-host-ring", preamble + data_bytes, m) != 0) {
        return -1;
    }
    p = (aotx_host_ring_preamble *)m->base;
    p->boot_id = boot_id;
    p->data_bytes = data_bytes;
    p->preamble_bytes = preamble;
    p->layout = AOTX_WIRE_LAYOUT;
    /* The magic goes last, so a reader that attaches early sees no half built preamble. */
    __atomic_store_n(&p->magic, AOTX_WIRE_MAGIC, __ATOMIC_RELEASE);
    return aotx_host_ring_attach(m, out);
}

int aotx_inbound_create(uint64_t slot_count, aotx_map *m, aotx_inbound_ring *out)
{
    aotx_inbound_preamble *p;
    uint64_t preamble = sizeof(aotx_inbound_preamble);
    if (!power_of_two(slot_count)) {
        return -1;
    }
    if (make_ring("aotx-inbound-ring", preamble + slot_count * AOTX_SLOT_BYTES, m) != 0) {
        return -1;
    }
    p = (aotx_inbound_preamble *)m->base;
    p->slot_count = slot_count;
    p->preamble_bytes = preamble;
    p->layout = AOTX_WIRE_LAYOUT;
    __atomic_store_n(&p->magic, AOTX_WIRE_MAGIC, __ATOMIC_RELEASE);
    return aotx_inbound_attach(m, out);
}
