/* Purpose: Find recent bus messages in the record ring and give back their bodies.
 * Owns: Nothing; the record ring holds the messages.
 * Launch shape: One thread for each cursor.
 * Lifetime: The whole run. */
#include "bus/bus.cuh"
#include "seam/seam.cuh"

/* The slot of a sequence, for a reader. The writer helper puts the sequence field to zero
 * first, so a reader must not use it. */
static __device__ __forceinline__ const volatile aotx_record_header *aotx_bus_at(
    unsigned long long seq)
{
    unsigned char *at = aotx_seam.dev.base
                      + ((seq - 1ull) & aotx_seam.dev.mask) * (unsigned long long)AOTX_SLOT_BYTES;
    return (const volatile aotx_record_header *)at;
}

/* Take the kind of the message at one sequence. The sequence in the header is the proof
 * that the slot still holds that record, because the ring gives one slot to one sequence.
 * The sequence is read again after the body, so a slot that a writer took over while it was
 * read is refused. The return is 1 for a message, 0 for another record, and -1 when the
 * slot no longer holds the sequence. */
static __device__ __forceinline__ int aotx_bus_kind_at(unsigned long long seq,
                                                       unsigned int *kind)
{
    const volatile aotx_record_header *header = aotx_bus_at(seq);
    if (header->seq != seq) {
        return -1;
    }
    unsigned int type = header->type;
    const volatile aotx_bus_body *body =
        (const volatile aotx_bus_body *)((const volatile unsigned char *)header
                                         + AOTX_HEADER_BYTES);
    unsigned int found = body->kind;
    if (header->seq != seq) {
        return -1;
    }
    if (type != (unsigned int)AOTX_REC_BUS) {
        return 0;
    }
    *kind = found;
    return 1;
}

/* The scan covers the whole ring, because a message stays until a later record takes its
 * slot. The first slot that no longer holds its sequence is where the ring wrapped, and
 * every sequence below it is gone. */
__device__ unsigned int aotx_bus_recent(unsigned int kind_mask, unsigned int max,
                                        unsigned long long *seqs)
{
    if (seqs == 0 || max == 0u) {
        return 0u;
    }
    unsigned long long tail = aotx_seam.dev.tail;
    unsigned long long span = aotx_seam.dev.slot_count;
    unsigned int found = 0u;
    for (unsigned long long back = 0ull; back < span && found < max; ++back) {
        if (back >= tail) {
            break;   /* the ring holds no sequence below 1 */
        }
        unsigned long long seq = tail - back;
        unsigned int kind = 0u;
        int state = aotx_bus_kind_at(seq, &kind);
        if (state < 0) {
            break;
        }
        if (state == 0 || kind == 0u || kind > 31u
            || ((kind_mask >> kind) & 1u) == 0u) {
            continue;
        }
        seqs[found] = seq;
        found += 1u;
    }
    return found;
}

__device__ const aotx_bus_body *aotx_bus_body_of(unsigned long long seq)
{
    unsigned int kind = 0u;
    if (seq == 0ull || seq > aotx_seam.dev.tail) {
        return 0;
    }
    if (aotx_bus_kind_at(seq, &kind) != 1) {
        return 0;
    }
    return (const aotx_bus_body *)((const unsigned char *)aotx_bus_at(seq)
                                   + AOTX_HEADER_BYTES);
}
