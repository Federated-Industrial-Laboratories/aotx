/* Purpose: Put records in the inbound ring of a check, the way the feeder does.
 * Owns: Nothing; the caller holds the ring and the bodies.
 * Threading: One host thread writes the ring while the pump makes ticks.
 * Lifetime: One run of the test program. */
#ifndef AOTX_TESTS_SEAM_FEED_H
#define AOTX_TESTS_SEAM_FEED_H

#include <string.h>

#include "seam/seam.cuh"

/* Put a run of records in the inbound ring. The device applies them at the tick that
 * follows, up to AOTX_INBOUND_MAX_TICK of them in one tick. A record which crosses this
 * way goes through the apply node. The check therefore exercises the path of the feeder
 * and not a call of its own. */
static void aotx_test_feed_records(aotx_seam_rings *rings, unsigned int type,
                                   unsigned int cls, unsigned int writer,
                                   unsigned int flags, const void *bodies,
                                   unsigned int body_bytes, unsigned int count,
                                   unsigned long long boot_id)
{
    aotx_inbound_preamble *preamble = (aotx_inbound_preamble *)rings->inbound_map;
    unsigned char *slots = rings->inbound_map + preamble->preamble_bytes;
    unsigned long long mask = preamble->slot_count - 1ull;
    unsigned long long head = preamble->head;
    const unsigned char *from = (const unsigned char *)bodies;
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_record_header *header =
            (aotx_record_header *)(slots + ((head + i) & mask) * AOTX_SLOT_BYTES);
        __atomic_store_n(&header->seq, 0ull, __ATOMIC_RELEASE);
        header->magic = AOTX_WIRE_MAGIC;
        header->layout = (unsigned short)AOTX_WIRE_LAYOUT;
        header->header_bytes = (unsigned short)AOTX_HEADER_BYTES;
        header->boot_id = boot_id;
        header->tick = 0ull;
        header->globaltimer = 0ull;
        header->writer = writer;
        header->cls = (unsigned char)cls;
        header->type = (unsigned char)type;
        header->flags = (unsigned short)flags;
        header->body_len = body_bytes;
        memcpy((unsigned char *)header + AOTX_HEADER_BYTES, from + (size_t)i * body_bytes,
               body_bytes);
        __atomic_store_n(&header->seq, head + i + 1ull, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&preamble->head, head + count, __ATOMIC_RELEASE);
}

/* Put a run of tool reply records in the inbound ring, as the feeder does. */
static void aotx_test_feed_replies(aotx_seam_rings *rings, const aotx_tool_reply_body *body,
                                   unsigned int count, unsigned long long boot_id)
{
    aotx_test_feed_records(rings, AOTX_REC_TOOL_REPLY, AOTX_CLASS_A, AOTX_WRITER_FEEDER,
                           0u, body, (unsigned int)sizeof *body, count, boot_id);
}

#endif
