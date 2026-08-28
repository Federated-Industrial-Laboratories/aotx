/* Purpose: Check that a record header and a block of records hold what the layout states.
 * Owns: Nothing; the caller owns the bytes that the checks read.
 * Threading: One thread; the checks read a private copy of a block.
 * Lifetime: The call.
 *
 * A block that comes over the seam is untrusted input, because a fault on the device can
 * write any bytes. Every length is checked against the block before the bytes are read. */
#include "disk/wire/diskwire.h"

int aotx_record_valid(const aotx_record_header *h)
{
    if (h->magic != AOTX_WIRE_MAGIC || h->layout != AOTX_WIRE_LAYOUT) {
        return 0;
    }
    if (h->header_bytes != AOTX_HEADER_BYTES) {
        return 0;
    }
    if (h->body_len > AOTX_BODY_BYTES) {
        return 0;
    }
    return 1;
}

const aotx_record_header *aotx_block_record(const unsigned char *block, uint32_t index)
{
    return (const aotx_record_header *)(block + AOTX_BLOCK_HEADER_BYTES +
                                        (size_t)index * AOTX_SLOT_BYTES);
}

const unsigned char *aotx_record_body(const aotx_record_header *h)
{
    return (const unsigned char *)h + AOTX_HEADER_BYTES;
}

int aotx_block_valid(const unsigned char *block, uint32_t byte_len, const char **reason)
{
    const aotx_block_header *h = (const aotx_block_header *)block;
    uint32_t i;
    const char *why = "";
    if (byte_len < AOTX_BLOCK_HEADER_BYTES) {
        why = "block is shorter than a block header";
    } else if (h->magic != AOTX_BLOCK_MAGIC) {
        why = "block magic is wrong";
    } else if (h->layout != AOTX_WIRE_LAYOUT) {
        why = "block layout version is wrong";
    } else if (h->byte_len != byte_len) {
        why = "block length does not match the frame";
    } else if (h->block_seq == 0) {
        why = "block sequence is zero";
    } else if (h->kind == AOTX_BLOCK_PAD) {
        if (h->record_count != 0) {
            why = "pad block holds records";
        }
    } else if (h->kind != 0) {
        why = "block kind is unknown";
    } else if ((uint64_t)h->record_count * AOTX_SLOT_BYTES + AOTX_BLOCK_HEADER_BYTES != byte_len) {
        why = "block length does not match the record count";
    }
    if (why[0] != '\0') {
        *reason = why;
        return -1;
    }
    if (h->kind == AOTX_BLOCK_PAD) {
        return 0;
    }
    for (i = 0; i < h->record_count; i++) {
        if (!aotx_record_valid(aotx_block_record(block, i))) {
            *reason = "a record header in the block is wrong";
            return -1;
        }
    }
    *reason = "";
    return 0;
}
