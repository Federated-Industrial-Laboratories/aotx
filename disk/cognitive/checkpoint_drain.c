/* Purpose: Consume complete GPU snapshots and acknowledge durable CCIR generations.
 * Owns: The ring consumer position, disk error and writer lifetime.
 * Threading: One drain thread takes the available bounded batch in order.
 * Lifetime: The optional memory mirror of one runtime. */
#include "cognitive/checkpoint_io.h"
#include <string.h>
#include <stdio.h>
#include <time.h>

static uint64_t aotx_cp_clock(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) return 0;
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
int aotx_checkpoint_disk_open(aotx_checkpoint_disk *d, int fd, const char *path) {
    memset(d, 0, sizeof(*d)); d->view.fd = -1; d->path = path;
    if (fd < 0 && !path) return 0;
    if (fd < 0 || !path || aotx_map_fd(fd, &d->map)) return -1;
    if (d->map.bytes < sizeof(aotx_checkpoint_ring)) goto failed;
    aotx_checkpoint_ring *r = (aotx_checkpoint_ring *)d->map.base;
    if (r->magic != AOTX_CP_MAGIC || r->layout != 1 || !r->boot ||
        r->slots != AOTX_MEMORY_SNAPSHOTS || r->slot_bytes != AOTX_CP_SLOT_BYTES ||
        d->map.bytes < AOTX_CP_RING_BYTES) goto failed;
    d->ring = r;
    return 0;
failed:
    aotx_map_release(&d->map); return -1;
}
int aotx_checkpoint_disk_pass(aotx_checkpoint_disk *d) {
    if (!d->ring) return 0;
    aotx_checkpoint_ring *r = d->ring;
    uint64_t head = __atomic_load_n(&r->head, __ATOMIC_ACQUIRE);
    uint64_t consumed = __atomic_load_n(&r->consumed, __ATOMIC_ACQUIRE);
    if (consumed > head || head - consumed > r->slots) {
        __atomic_store_n(&r->error, AOTX_CCIR_INVALID, __ATOMIC_RELEASE);
        return -1;
    }
    if (d->next_retry && aotx_cp_clock() < d->next_retry) return 0;
    int taken = 0;
    while (consumed < head) {
        const unsigned char *slot = (const unsigned char *)(r + 1) + (consumed % r->slots) * r->slot_bytes;
        uint64_t bytes = aotx_cp_get(slot + 16, 8);
        int status = aotx_cp_get(slot, 8) != r->boot || aotx_cp_get(slot + 8, 8) != consumed + 1 ||
            bytes < AOTX_CP_HEADER || bytes > AOTX_CP_BYTES ? AOTX_CCIR_INVALID :
            aotx_checkpoint_file_write(d, slot + AOTX_CP_SLOT_HEADER, bytes);
        if (status) {
            if (__atomic_load_n(&r->error, __ATOMIC_ACQUIRE) != (uint64_t)status)
                fprintf(stderr, "memory mirror: %s\n", aotx_ccir_status_text(status));
            __atomic_store_n(&r->error, (uint64_t)status, __ATOMIC_RELEASE);
            d->next_retry = aotx_cp_clock() + 1000000000ull;
            return taken;
        }
        const unsigned char *image = slot + AOTX_CP_SLOT_HEADER;
        __atomic_store_n(&r->durable_sequence, aotx_cp_get(image + 48, 8), __ATOMIC_RELAXED);
        __atomic_store_n(&r->durable_revision, aotx_cp_get(image + 64, 8), __ATOMIC_RELAXED);
        __atomic_store_n(&r->generation, d->view.generation, __ATOMIC_RELAXED);
        __atomic_store_n(&r->ack_boot, r->boot, __ATOMIC_RELAXED);
        __atomic_store_n(&r->error, 0, __ATOMIC_RELEASE);
        __atomic_store_n(&r->consumed, ++consumed, __ATOMIC_RELEASE);
        ++taken; d->next_retry = 0;
    }
    return taken;
}
void aotx_checkpoint_disk_close(aotx_checkpoint_disk *d) {
    aotx_ccir_close(&d->view); aotx_map_release(&d->map); d->ring = NULL;
}
