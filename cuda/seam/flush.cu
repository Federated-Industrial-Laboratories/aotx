/* Purpose: Copy the records of one tick into the host ring as one block.
 * Owns: The host ring head, the block sequence and the flushed mark.
 * Launch shape: One block of AOTX_FLUSH_THREADS threads; the barrier orders the publish.
 * Lifetime: One node of every tick. */

/* The block sequence starts at 1 and has no gaps. A reader knows which sequence comes next.
 * A stale block from an earlier pass through the data area holds a smaller sequence, so the
 * reader refuses it. The reader also loads the sequence, reads the block, and loads the
 * sequence again. */
#include "sched/sched.cuh"
#include "seam/seam.cuh"

/* Fill a block header. The block sequence stays at zero, because the publish writes it last. */
static __device__ __forceinline__ void aotx_flush_header(unsigned char *at, unsigned int kind,
                                                         unsigned long long first_seq,
                                                         unsigned int record_count,
                                                         unsigned int byte_len)
{
    aotx_block_header *header = (aotx_block_header *)at;
    header->magic = AOTX_BLOCK_MAGIC;
    header->layout = (unsigned short)AOTX_WIRE_LAYOUT;
    header->kind = (unsigned short)kind;
    header->boot_id = aotx_seam.boot_id;
    header->tick = aotx_time_tick;
    header->first_seq = first_seq;
    header->record_count = record_count;
    header->byte_len = byte_len;
    header->reserved[0] = 0ull;
    header->reserved[1] = 0ull;
}

/* Publish a block: fence the payload out, store the sequence, then the head, then the last
 * sequence. The drain sees a complete block before it sees the head that covers it. */
static __device__ __forceinline__ void aotx_flush_publish(unsigned char *at,
                                                          unsigned long long bytes)
{
    aotx_host_ring_preamble *preamble = (aotx_host_ring_preamble *)aotx_seam.host.preamble;
    unsigned long long seq = aotx_seam.host.block_seq + 1ull;
    __threadfence_system();
    aotx_seam_release_sys(&((aotx_block_header *)at)->block_seq, seq);
    aotx_seam.host.block_seq = seq;
    aotx_seam.host.head += bytes;
    aotx_seam_release_sys(&preamble->head, aotx_seam.host.head);
    aotx_seam_release_sys(&preamble->last_block_seq, seq);
    aotx_sched.blocks += 1ull;
}

__global__ void aotx_seam_flush(void)
{
    __shared__ unsigned long long shared_first;
    __shared__ unsigned long long shared_last;
    __shared__ unsigned long long shared_offset;
    __shared__ unsigned long long shared_count;
    __shared__ unsigned int shared_limit;
    __shared__ unsigned int shared_found;

    if (threadIdx.x == 0u) {
        unsigned long long first = aotx_seam.dev.flushed + 1ull;
        unsigned long long last = aotx_seam.dev.tail;
        shared_first = first;
        shared_last = last;
        shared_found = (last >= first) ? (unsigned int)(last - first + 1ull) : 0u;
        shared_limit = shared_found;
    }
    __syncthreads();

    /* Every record must carry the sequence that its position gives it. A record that does
     * not is a slot that a producer wrote over, so the block stops in front of it. */
    const unsigned int found = shared_found;
    const unsigned long long first = shared_first;
    const unsigned long long mask = aotx_seam.dev.mask;
    unsigned char *base = aotx_seam.dev.base;
    for (unsigned int i = threadIdx.x; i < found; i += AOTX_FLUSH_THREADS) {
        unsigned long long seq = first + (unsigned long long)i;
        const aotx_record_header *record = (const aotx_record_header *)
            (base + ((seq - 1ull) & mask) * (unsigned long long)AOTX_SLOT_BYTES);
        if (record->seq != seq) {
            atomicMin(&shared_limit, i);
        }
    }
    __syncthreads();

    const unsigned int limit = shared_limit;
    if (threadIdx.x == 0u) {
        const aotx_host_ring_preamble *preamble =
            (const aotx_host_ring_preamble *)aotx_seam.host.preamble;
        unsigned long long count = (unsigned long long)limit;
        unsigned long long bytes = aotx_seam_block_bytes(count);
        unsigned long long cursor = aotx_seam_acquire_sys(&preamble->cursor);
        unsigned long long room = aotx_seam_host_free(cursor);
        unsigned long long offset = aotx_seam.host.head & aotx_seam.host.mask;
        unsigned long long to_end = aotx_seam.host.data_bytes - offset;
        unsigned long long need = bytes + ((bytes > to_end) ? to_end : 0ull);

        shared_count = 0ull;
        shared_offset = 0ull;
        if (count == 0ull && limit < found) {
            /* Nothing of this run of records is whole. The loss reaches the drain as a gap
             * in the record sequence, and the next stall record reports it. */
            aotx_seam.dev.flushed = shared_last;
            aotx_seam.dev.overrun += 1ull;
        } else if (count > 0ull && need <= room) {
            if (bytes > to_end) {
                /* A block never wraps. A pad block takes the tail of the data area. */
                unsigned char *pad = aotx_seam.host.data + offset;
                ((volatile aotx_block_header *)pad)->block_seq = 0ull;
                __threadfence_system();
                aotx_flush_header(pad, AOTX_BLOCK_PAD, first, 0u, (unsigned int)to_end);
                aotx_flush_publish(pad, to_end);
                offset = 0ull;
            }
            unsigned char *at = aotx_seam.host.data + offset;
            ((volatile aotx_block_header *)at)->block_seq = 0ull;
            __threadfence_system();
            aotx_flush_header(at, 0u, first, (unsigned int)count, (unsigned int)bytes);
            shared_offset = offset;
            shared_count = count;
        }
    }
    __syncthreads();

    const unsigned long long count = shared_count;
    if (count == 0ull) {
        return;
    }

    /* Every thread copies with 16-byte stores. A slot holds 16 of them. */
    const unsigned long long units = count * (AOTX_SLOT_BYTES / 16u);
    uint4 *to = (uint4 *)(aotx_seam.host.data + shared_offset + AOTX_BLOCK_HEADER_BYTES);
    const unsigned long long first_slot = first - 1ull;
    for (unsigned long long unit = threadIdx.x; unit < units; unit += AOTX_FLUSH_THREADS) {
        unsigned long long slot = (first_slot + (unit >> 4)) & mask;
        const uint4 *from = (const uint4 *)(base + slot * (unsigned long long)AOTX_SLOT_BYTES);
        to[unit] = from[unit & 15ull];
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
        aotx_flush_publish(aotx_seam.host.data + shared_offset,
                           aotx_seam_block_bytes(count));
        if (limit < found) {
            aotx_seam.dev.flushed = shared_last;
            aotx_seam.dev.overrun += 1ull;
        } else {
            aotx_seam.dev.flushed = first + count - 1ull;
        }
    }
}
