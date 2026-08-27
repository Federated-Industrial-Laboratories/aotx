/* Purpose: Stage a large payload, name it in a record, and copy it into the bulk ring.
 * Owns: The staging region, the entry table and the bulk ring head.
 * Launch shape: One thread for each payload; one block of AOTX_FLUSH_THREADS for the copy.
 * Lifetime: The whole run; the entry table lasts one tick. */
#include "sched/sched.cuh"
#include "seam/seam.cuh"

/* The host glue writes the ring and the staging region into this state at bind time. */
__device__ aotx_bulk_state aotx_bulk;

/* Count one payload that found no room. The counter is the backpressure report; a stage
 * call never waits for the drain. */
static __device__ __forceinline__ void *aotx_bulk_refuse(void)
{
    atomicAdd(&aotx_bulk.refused, 1ull);
    return 0;
}

/* The bytes one payload takes in the bulk ring. A block never wraps, so a pad block can
 * take the tail of the data area first. A pad is at most the block size and one header
 * more. The tail rule below can refuse a fit that leaves less than a header. */
static __device__ __forceinline__ unsigned long long aotx_bulk_need(unsigned long long padded)
{
    return 2ull * ((unsigned long long)AOTX_BLOCK_HEADER_BYTES + padded)
         + (unsigned long long)AOTX_BLOCK_HEADER_BYTES;
}

/* A block may take a place only when the space left after it is zero or holds a block
 * header. A payload block rounds to 8 bytes. The tail can come to fewer bytes than a
 * header, where no pad block fits, so a pad block goes in before the payload block. */
static __device__ __forceinline__ int aotx_bulk_fits(unsigned long long bytes,
                                                     unsigned long long to_end)
{
    if (bytes > to_end) {
        return 0;
    }
    unsigned long long left = to_end - bytes;
    return (left == 0ull || left >= (unsigned long long)AOTX_BLOCK_HEADER_BYTES) ? 1 : 0;
}

__device__ void *aotx_bulk_stage(unsigned int kind, unsigned long long length)
{
    if (aotx_bulk.stage == 0 || aotx_bulk.ring.data == 0 || length == 0ull) {
        return aotx_bulk_refuse();
    }
    unsigned long long padded = (length + 7ull) & ~7ull;
    unsigned long long need = aotx_bulk_need(padded);
    if (need > aotx_bulk.ring.data_bytes) {
        return aotx_bulk_refuse();
    }

    /* The room comes from one read of the drain cursor at tick start. A claim that goes
     * past it is refused, and the bytes it claimed stay claimed until the next tick. */
    unsigned long long taken = atomicAdd(&aotx_bulk.reserved, need);
    if (taken + need > aotx_bulk.room) {
        return aotx_bulk_refuse();
    }
    unsigned long long cell = (unsigned long long)AOTX_BULK_PREFIX_BYTES
                           + ((length + 15ull) & ~15ull);
    unsigned long long at = atomicAdd(&aotx_bulk.used, cell);
    if (at + cell > aotx_bulk.stage_bytes) {
        return aotx_bulk_refuse();
    }
    unsigned int index = atomicAdd(&aotx_bulk.count, 1u);
    if (index >= AOTX_BULK_STAGE_MAX) {
        return aotx_bulk_refuse();
    }

    /* The prefix names the entry, so the commit finds the entry from the pointer alone. */
    unsigned char *cell_at = aotx_bulk.stage + at;
    unsigned int *prefix = (unsigned int *)cell_at;
    prefix[0] = AOTX_WIRE_MAGIC;
    prefix[1] = index;
    prefix[2] = 0u;
    prefix[3] = 0u;
    unsigned char *payload = cell_at + AOTX_BULK_PREFIX_BYTES;
    for (unsigned long long b = length; b < padded; ++b) {
        payload[b] = 0u;   /* the block carries whole 8-byte words and no stale byte */
    }
    aotx_bulk_entry *entry = &aotx_bulk.entry[index];
    entry->offset = at + (unsigned long long)AOTX_BULK_PREFIX_BYTES;
    entry->length = length;
    entry->kind = kind;
    entry->handle = aotx_bulk.published + (unsigned long long)index + 1ull;
    entry->tick = aotx_time_tick;
    entry->committed = 0u;
    return payload;
}

__device__ unsigned long long aotx_bulk_commit(void *pointer, unsigned int kind,
                                               unsigned long long length,
                                               unsigned long long tick)
{
    if (pointer == 0 || aotx_bulk.stage == 0) {
        return 0ull;
    }
    unsigned char *payload = (unsigned char *)pointer;
    const unsigned int *prefix = (const unsigned int *)(payload - AOTX_BULK_PREFIX_BYTES);
    if (prefix[0] != AOTX_WIRE_MAGIC || prefix[1] >= AOTX_BULK_STAGE_MAX) {
        return 0ull;
    }
    aotx_bulk_entry *entry = &aotx_bulk.entry[prefix[1]];
    if (entry->offset != (unsigned long long)(payload - aotx_bulk.stage)) {
        return 0ull;
    }

    /* The staging region and the entry table last one tick. A pointer of an earlier tick
     * names a payload that another stage call took, so the commit refuses it. */
    if (entry->tick != aotx_time_tick) {
        atomicAdd(&aotx_bulk.stale, 1ull);
        return 0ull;
    }
    if (length > entry->length) {
        length = entry->length;   /* the stage call fixed the room this payload has */
    }
    unsigned long long padded = (length + 7ull) & ~7ull;
    for (unsigned long long b = length; b < padded; ++b) {
        payload[b] = 0u;
    }
    entry->length = length;
    entry->kind = kind;

    aotx_bulk_body body;
    body.handle = entry->handle;
    body.length = length;
    body.kind = kind;
    body.reserved = 0u;
    unsigned long long seq = aotx_seam_claim(1u);
    aotx_record_header *header = aotx_seam_slot(seq);
    unsigned char *to = aotx_seam_body(header);
    const unsigned char *from = (const unsigned char *)&body;
    for (unsigned int i = 0u; i < (unsigned int)sizeof body; ++i) {
        to[i] = from[i];
    }
    aotx_seam_publish_at(header, seq, AOTX_WRITER_SYSTEM, AOTX_CLASS_B, AOTX_REC_BULK, 0u,
                         (unsigned int)sizeof body, tick);
    __threadfence();
    entry->committed = 1u;
    return entry->handle;
}

/* Fill a bulk block header. The block sequence stays at zero, because the publish writes it
 * last. The handle goes in first_seq, which is the field the record and the block agree on. */
static __device__ __forceinline__ void aotx_bulk_header(unsigned char *at, unsigned int kind,
                                                        unsigned long long handle,
                                                        unsigned int byte_len)
{
    aotx_block_header *header = (aotx_block_header *)at;
    header->magic = AOTX_BLOCK_MAGIC;
    header->layout = (unsigned short)AOTX_WIRE_LAYOUT;
    header->kind = (unsigned short)kind;
    header->boot_id = aotx_seam.boot_id;
    header->tick = aotx_time_tick;
    header->first_seq = handle;
    header->record_count = 0u;
    header->byte_len = byte_len;
    header->reserved[0] = 0ull;
    header->reserved[1] = 0ull;
}

/* Publish one bulk block: fence the payload out, store the sequence, then the head, then
 * the last sequence. The drain sees a complete block before the head that covers it. */
static __device__ __forceinline__ void aotx_bulk_publish(unsigned char *at,
                                                         unsigned long long bytes)
{
    aotx_host_ring_preamble *preamble = (aotx_host_ring_preamble *)aotx_bulk.ring.preamble;
    unsigned long long seq = aotx_bulk.ring.block_seq + 1ull;
    __threadfence_system();
    aotx_seam_release_sys(&((aotx_block_header *)at)->block_seq, seq);
    aotx_bulk.ring.block_seq = seq;
    aotx_bulk.ring.head += bytes;
    aotx_seam_release_sys(&preamble->head, aotx_bulk.ring.head);
    aotx_seam_release_sys(&preamble->last_block_seq, seq);
    aotx_bulk.blocks += 1ull;
}

/* The bulk flush runs after the record flush and copies each committed payload as one
 * block, in the order the stage calls claimed their entries. The handle in the block is the
 * handle the commit put in the record, so the two agree. */
__global__ void aotx_seam_bulk_flush(void)
{
    __shared__ unsigned long long shared_offset;
    __shared__ unsigned long long shared_bytes;

    unsigned int count = aotx_bulk.count;
    if (count > AOTX_BULK_STAGE_MAX) {
        count = AOTX_BULK_STAGE_MAX;
    }
    if (aotx_bulk.ring.data == 0) {
        count = 0u;   /* the bulk path is not open; the tick still ends here */
    }
    for (unsigned int e = 0u; e < count; ++e) {
        const aotx_bulk_entry *entry = &aotx_bulk.entry[e];
        if (entry->committed == 0u) {
            continue;   /* a payload that no record names is not written */
        }
        unsigned long long padded = (entry->length + 7ull) & ~7ull;
        unsigned long long bytes = (unsigned long long)AOTX_BLOCK_HEADER_BYTES + padded;
        if (threadIdx.x == 0u) {
            unsigned long long offset = aotx_bulk.ring.head & aotx_bulk.ring.mask;
            unsigned long long to_end = aotx_bulk.ring.data_bytes - offset;
            if (!aotx_bulk_fits(bytes, to_end)) {
                unsigned char *pad = aotx_bulk.ring.data + offset;
                ((volatile aotx_block_header *)pad)->block_seq = 0ull;
                __threadfence_system();
                aotx_bulk_header(pad, AOTX_BLOCK_PAD, entry->handle, (unsigned int)to_end);
                aotx_bulk_publish(pad, to_end);
                offset = 0ull;
            }
            unsigned char *block = aotx_bulk.ring.data + offset;
            ((volatile aotx_block_header *)block)->block_seq = 0ull;
            __threadfence_system();
            aotx_bulk_header(block, AOTX_BLOCK_BULK, entry->handle, (unsigned int)bytes);
            shared_offset = offset;
            shared_bytes = bytes;
        }
        __syncthreads();

        /* Every thread copies 8-byte words. The stage call rounded the payload up to a
         * whole word, so the last word carries no stale byte. */
        const unsigned long long words = padded >> 3;
        const unsigned long long *from =
            (const unsigned long long *)(aotx_bulk.stage + entry->offset);
        unsigned long long *to = (unsigned long long *)(aotx_bulk.ring.data + shared_offset
                                                        + AOTX_BLOCK_HEADER_BYTES);
        for (unsigned long long w = threadIdx.x; w < words; w += AOTX_FLUSH_THREADS) {
            to[w] = from[w];
        }
        __syncthreads();

        if (threadIdx.x == 0u) {
            aotx_bulk_publish(aotx_bulk.ring.data + shared_offset, shared_bytes);
        }
        __syncthreads();
    }

    /* Every entry of the tick takes a handle. A payload that was staged and never committed
     * leaves a gap, and no handle is given twice. The entry count goes to zero, so a second
     * run of the flush in the same tick writes nothing again. */
    if (threadIdx.x == 0u) {
        aotx_bulk.published += (unsigned long long)count;
        aotx_bulk.count = 0u;

        /* This node is the last of the tick, so the time from the commit record to here is
         * the time the two flush nodes take. The next statistics record carries it. */
        if (aotx_sched.commit_ns != 0ull) {
            aotx_sched.flush_ns = aotx_time_globaltimer() - aotx_sched.commit_ns;
            aotx_sched.commit_ns = 0ull;
        }
    }
}
