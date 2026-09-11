/* Purpose: Publish complete bounded image frames to the mapped device input ring.
 * Owns: One map and the single producer cursor.
 * Threading: The feeder releases frames; the device releases consumed slots.
 * Lifetime: One producer attachment. */
#include "disk/feed/media_io.h"
#include <string.h>
#include <unistd.h>

int aotx_media_producer_open(int fd, aotx_media_producer *out)
{
    memset(out, 0, sizeof *out); out->map.fd = -1;
    if (fd < 0) return 0;
    if (aotx_map_fd(fd, &out->map)) return -1;
    uint64_t need = sizeof(aotx_media_preamble) + (uint64_t)AOTX_MEDIA_RING_SLOTS * AOTX_MEDIA_FRAME_BYTES;
    if (out->map.bytes < need) { aotx_map_release(&out->map); return -1; }
    aotx_media_preamble *p = (aotx_media_preamble *)out->map.base;
    if (p->magic != AOTX_MEDIA_RING_MAGIC || p->schema != AOTX_MEDIA_SCHEMA ||
        p->slots != AOTX_MEDIA_RING_SLOTS || p->frame_bytes != AOTX_MEDIA_FRAME_BYTES ||
        p->reserved[0] || p->reserved[1] ||
        aotx_load_acquire(&p->closed)) { aotx_map_release(&out->map); return -1; }
    out->pre = p; out->frames = out->map.base + sizeof *p; return 0;
}
void aotx_media_producer_close(aotx_media_producer *out)
{
    if (out->pre) aotx_store_release(&out->pre->closed, 1);
    if (out->map.base) aotx_map_release(&out->map);
    out->pre = 0; out->frames = 0;
}
static int aotx_media_producer_ready(aotx_media_producer *out,
                                      const volatile sig_atomic_t *stop, int empty)
{
    if (!out->pre) return -1;
    for (;;) {
        if ((stop && *stop) || aotx_load_acquire(&out->pre->closed)) return -1;
        uint64_t head = aotx_load_acquire(&out->pre->head);
        uint64_t consumed = aotx_load_acquire(&out->pre->consumed);
        if (consumed > head || head - consumed > AOTX_MEDIA_RING_SLOTS || head == UINT64_MAX) return -1;
        if (empty ? head == consumed : head - consumed < AOTX_MEDIA_RING_SLOTS) return 0;
        usleep(1000);
    }
}
int aotx_media_producer_put(aotx_media_producer *out, const unsigned char *frame,
                             const volatile sig_atomic_t *stop)
{
    if (!frame || aotx_media_get(frame + 40, 4) > AOTX_MEDIA_FRAME_DATA ||
        aotx_media_producer_ready(out, stop, 0)) return -1;
    uint64_t head = aotx_load_acquire(&out->pre->head);
    memcpy(out->frames + (head % AOTX_MEDIA_RING_SLOTS) * AOTX_MEDIA_FRAME_BYTES,
        frame, AOTX_MEDIA_FRAME_BYTES);
    aotx_store_release(&out->pre->head, head + 1); return 0;
}
int aotx_media_producer_wait(aotx_media_producer *out, const volatile sig_atomic_t *stop)
{
    int rc = aotx_media_producer_ready(out, stop, 1);
    return rc ? rc : aotx_load_acquire(&out->pre->status) ? 1 : 0;
}
