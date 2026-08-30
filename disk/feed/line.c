/* Purpose: Publish one complete input line as a head and contiguous fragment parts.
 * Owns: The record header of each part while it is published.
 * Threading: One feeder thread; no other record can enter between the parts.
 * Lifetime: One input line. */
#include "disk/feed/line.h"

#include <string.h>

int aotx_line_publish_records(const aotx_inbound_ring *ring,
                              const volatile sig_atomic_t *stop,
                              const aotx_record_header *headers,
                              const void *const *bodies, uint32_t count)
{
    if (count == 0u || count > AOTX_LINE_PARTS_MAX) {
        return 1;
    }
    if (aotx_inbound_wait_many(ring, stop, count) != 0) {
        return -1;
    }
    aotx_inbound_put_many(ring, headers, bodies, count);
    return 0;
}

int aotx_line_publish(const aotx_inbound_ring *ring, const volatile sig_atomic_t *stop,
                      const unsigned char *bytes, uint32_t length)
{
    aotx_record_header headers[AOTX_LINE_PARTS_MAX];
    const void *bodies[AOTX_LINE_PARTS_MAX];
    uint32_t parts;
    uint32_t part;
    if (length > AOTX_INPUT_LINE_BYTES) {
        return 1;
    }
    parts = (length == 0u) ? 1u : (length + AOTX_BODY_BYTES - 1u) / AOTX_BODY_BYTES;
    for (part = 0u; part < parts; part++) {
        aotx_record_header *h = &headers[part];
        uint32_t offset = part * AOTX_BODY_BYTES;
        uint32_t count = length - offset;
        if (count > AOTX_BODY_BYTES) {
            count = AOTX_BODY_BYTES;
        }
        memset(h, 0, sizeof(*h));
        h->writer = AOTX_WRITER_FEEDER;
        h->cls = AOTX_CLASS_A;
        h->type = AOTX_REC_INPUT_LINE;
        h->flags = (part == 0u) ? 0u : AOTX_FLAG_FRAGMENT;
        h->body_len = count;
        bodies[part] = (count != 0u) ? bytes + offset : bytes;
    }
    return aotx_line_publish_records(ring, stop, headers, bodies, parts);
}
