/* Purpose: Read blocks from the host ring and publish records into the inbound ring.
 * Owns: Nothing; the rings live in memory that the device process made.
 * Threading: One thread for each ring; the drain reads one and the feeder writes one.
 * Lifetime: From the attach of a ring to the release of its map. */
#include "disk/wire/diskwire.h"

#include <string.h>

uint64_t aotx_host_ring_head(const aotx_host_ring *r)
{
    return aotx_load_acquire(&r->pre->head);
}

uint64_t aotx_host_ring_cursor(const aotx_host_ring *r)
{
    return aotx_load_acquire(&r->pre->cursor);
}

int aotx_host_ring_closed(const aotx_host_ring *r)
{
    return aotx_load_acquire16(&r->pre->closed) != 0;
}

void aotx_host_ring_advance(const aotx_host_ring *r, uint64_t cursor)
{
    /* The producer reads this field to find the free space. The store comes after the bytes
     * reach the disk, or a crash loses a block that the producer counts as safe. */
    aotx_store_release(&r->pre->cursor, cursor);
}

static int refuse(aotx_take *t, const char *reason)
{
    t->status = AOTX_TAKE_BAD;
    t->reason = reason;
    return t->status;
}

int aotx_host_ring_take(const aotx_host_ring *r, uint64_t cursor, uint64_t head,
                        unsigned char *out, uint32_t out_bytes, aotx_take *t)
{
    const aotx_block_header *live;
    const aotx_block_header *copy;
    uint64_t offset;
    uint64_t first;
    uint64_t second;
    uint32_t len;
    const char *reason = "";

    memset(t, 0, sizeof(*t));
    t->reason = "";
    if (cursor >= head) {
        t->status = AOTX_TAKE_EMPTY;
        return t->status;
    }
    offset = cursor & r->mask;
    if ((offset & 7u) != 0) {
        return refuse(t, "the cursor is not on an eight byte boundary");
    }
    if (r->data_bytes - offset < AOTX_BLOCK_HEADER_BYTES) {
        return refuse(t, "a block header does not fit before the end of the data area");
    }
    live = (const aotx_block_header *)(r->data + offset);

    /* The double-load rule: the sequence is the publish field, so it is read with an
     * acquire load before the copy and again after it. A zero means a block under write. */
    first = aotx_load_acquire(&live->block_seq);
    if (first == 0) {
        t->status = AOTX_TAKE_EMPTY;
        return t->status;
    }
    len = live->byte_len;
    if (len < AOTX_BLOCK_HEADER_BYTES || len > out_bytes) {
        return refuse(t, "the block length is outside the buffer");
    }
    if (offset + len > r->data_bytes) {
        return refuse(t, "the block runs past the end of the data area");
    }
    if (head - cursor < len) {
        return refuse(t, "the block is longer than the published bytes");
    }
    if (live->kind == AOTX_BLOCK_PAD && offset + len != r->data_bytes) {
        return refuse(t, "a pad block does not reach the end of the data area");
    }
    memcpy(out, live, len);
    second = aotx_load_acquire(&live->block_seq);
    if (second != first) {
        t->status = AOTX_TAKE_TORN;
        return t->status;
    }
    if (aotx_block_valid(out, len, &reason) != 0) {
        return refuse(t, reason);
    }
    copy = (const aotx_block_header *)out;
    if (copy->block_seq != first) {
        return refuse(t, "the copied block holds another sequence");
    }
    t->block_seq = copy->block_seq;
    t->tick = copy->tick;
    t->first_seq = copy->first_seq;
    t->boot_id = copy->boot_id;
    t->byte_len = len;
    t->record_count = copy->record_count;
    t->kind = copy->kind;
    t->status = AOTX_TAKE_OK;
    return t->status;
}

uint64_t aotx_inbound_head(const aotx_inbound_ring *r)
{
    return aotx_load_acquire(&r->pre->head);
}

uint64_t aotx_inbound_consumed(const aotx_inbound_ring *r)
{
    return aotx_load_acquire(&r->pre->consumed);
}

int aotx_inbound_closed(const aotx_inbound_ring *r)
{
    return aotx_load_acquire16(&r->pre->closed) != 0;
}

int aotx_inbound_wait(const aotx_inbound_ring *r, const volatile sig_atomic_t *stop)
{
    return aotx_inbound_wait_many(r, stop, 1u);
}

int aotx_inbound_wait_many(const aotx_inbound_ring *r,
                           const volatile sig_atomic_t *stop, uint32_t count)
{
    uint64_t backoff = 0;
    if (count == 0u || count > r->slot_count) {
        return -1;
    }
    for (;;) {
        uint64_t head = aotx_inbound_head(r);
        uint64_t consumed = aotx_inbound_consumed(r);
        if (head - consumed + count <= r->slot_count) {
            return 0;
        }
        if (aotx_inbound_closed(r)) {
            return -1;
        }
        if (stop != NULL && *stop != 0) {
            return -1;
        }
        aotx_pause(&backoff);
    }
}

static void put_at(const aotx_inbound_ring *r, uint64_t sequence,
                   const aotx_record_header *h, const void *body)
{
    unsigned char *slot = r->slots + (sequence & r->mask) * AOTX_SLOT_BYTES;
    aotx_record_header *dst = (aotx_record_header *)slot;
    uint32_t len = h->body_len > AOTX_BODY_BYTES ? AOTX_BODY_BYTES : h->body_len;

    /* The sequence goes to zero first. A reader that looks at the slot during the write
     * then sees an unpublished slot, and never a mix of two records. */
    aotx_store_release(&dst->seq, 0);
    dst->magic = AOTX_WIRE_MAGIC;
    dst->layout = AOTX_WIRE_LAYOUT;
    dst->header_bytes = AOTX_HEADER_BYTES;
    dst->boot_id = h->boot_id;
    dst->tick = h->tick;
    dst->globaltimer = h->globaltimer;
    dst->writer = h->writer;
    dst->cls = h->cls;
    dst->type = h->type;
    dst->flags = h->flags;
    dst->body_len = len;
    dst->reserved[0] = 0;
    dst->reserved[1] = 0;
    dst->reserved[2] = 0;
    if (len > 0) {
        memcpy(slot + AOTX_HEADER_BYTES, body, len);
    }
    memset(slot + AOTX_HEADER_BYTES + len, 0, AOTX_BODY_BYTES - len);
    aotx_store_release(&dst->seq, sequence + 1u);
}

void aotx_inbound_put_many(const aotx_inbound_ring *r, const aotx_record_header *headers,
                           const void *const *bodies, uint32_t count)
{
    uint64_t head = aotx_inbound_head(r);
    uint32_t i;
    for (i = 0u; i < count; i++) {
        put_at(r, head + i, &headers[i], bodies[i]);
    }
    /* The one release store makes the complete group visible to the device. */
    aotx_store_release(&r->pre->head, head + count);
}

void aotx_inbound_put(const aotx_inbound_ring *r, const aotx_record_header *h, const void *body)
{
    const void *bodies[1];
    bodies[0] = body;
    aotx_inbound_put_many(r, h, bodies, 1u);
}
