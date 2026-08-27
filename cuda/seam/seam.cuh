/* Purpose: Move records across the seam in rings.
 * Owns: The device ring, the host ring layout and the inbound cursor.
 * Launch shape: One thread for each record; one block for the flush.
 * Lifetime: The whole run. */
#ifndef AOTX_SEAM_CUH
#define AOTX_SEAM_CUH

#include "seam/wire.h"
#include "time/time.cuh"

/* The device ring holds this many slots. The size is a power of two, so the position of a
 * sequence is a mask and not a division. */
#define AOTX_DEVICE_RING_SLOTS   65536ull

/* Bounds that keep one tick inside the rings. The tick start kernel holds a tick that cannot
 * meet them. */
#define AOTX_INBOUND_MAX_TICK    1024ull    /* inbound slots applied in one tick */
#define AOTX_TICK_RECORDS_MAX    32768ull   /* records one tick may write */
#define AOTX_APPLY_RECORDS_EACH  2ull       /* records the apply writes for each input */

/* The flush uses one block, because the block barrier is what orders the copy before the
 * publish of the block sequence. */
#define AOTX_FLUSH_THREADS       1024u

/* The apply covers AOTX_INBOUND_MAX_TICK inputs with one thread for each input. */
#define AOTX_APPLY_BLOCKS        8u
#define AOTX_APPLY_THREADS       128u

/* The high bit of held_count in a stall body states that the flush dropped records. The
 * count of held ticks never reaches this bit. A free byte count of zero is a value the ring
 * can hold, so that field cannot carry the mark. */
#define AOTX_STALL_OVERRUN       0x8000000000000000ull

/* FNV-1a over 64 bits. The state hash folds the body of each applied class A record. */
#define AOTX_FNV_BASIS           0xcbf29ce484222325ull
#define AOTX_FNV_PRIME           0x100000001b3ull

typedef struct aotx_seam_device_ring {
    unsigned char *base;           /* first byte of the ring region */
    unsigned long long slot_count; /* slots in the ring, a power of two */
    unsigned long long mask;       /* slot_count minus one */
    unsigned long long tail;       /* the last claimed sequence */
    unsigned long long flushed;    /* the last sequence copied into the host ring */
    unsigned long long overrun;    /* runs of records the flush dropped, because a slot
                                    * held a sequence that its position does not give it */
} aotx_seam_device_ring;

typedef struct aotx_seam_host_ring {
    unsigned char *preamble;       /* device address of the mapped preamble */
    unsigned char *data;           /* device address of the first byte of the data area */
    unsigned long long data_bytes; /* size of the data area, a power of two */
    unsigned long long mask;       /* data_bytes minus one */
    unsigned long long head;       /* bytes written, monotonic; the device copy */
    unsigned long long block_seq;  /* the last published block sequence */
} aotx_seam_host_ring;

typedef struct aotx_seam_inbound_ring {
    unsigned char *preamble;       /* device address of the mapped preamble */
    unsigned char *slots;          /* device address of the first slot */
    unsigned long long slot_count; /* slots in the ring, a power of two */
    unsigned long long mask;       /* slot_count minus one */
    unsigned long long consumed;   /* slots consumed, the device copy */
} aotx_seam_inbound_ring;

/* What the apply builds and the commit reports. */
typedef struct aotx_seam_apply_state {
    unsigned long long state_hash;    /* FNV-1a over applied class A bodies, in order */
    unsigned long long applied_count; /* class A records applied since start */
    unsigned long long wall_ns;       /* wall clock of the last applied tick start record */
    unsigned long long this_tick;     /* inbound slots the apply takes this tick */
    unsigned long long rejected;      /* inbound slots refused by the length check */
} aotx_seam_apply_state;

typedef struct aotx_seam_state {
    aotx_seam_device_ring dev;
    aotx_seam_host_ring host;
    aotx_seam_inbound_ring in;
    aotx_seam_apply_state apply;
    unsigned long long boot_id;
} aotx_seam_state;

extern __device__ aotx_seam_state aotx_seam;

/* The size of each host ring. The data area of the host ring is a power of two. */
#define AOTX_HOST_RING_DATA_BYTES  (64ull * 1024ull * 1024ull)
#define AOTX_INBOUND_SLOTS         4096ull

/* What the host glue keeps for the two rings that cross the seam. The file descriptors stay
 * open across an exec, so a disk side program maps the same memory. */
typedef struct aotx_seam_rings {
    int host_fd;                     /* the host ring file */
    int inbound_fd;                  /* the inbound ring file */
    unsigned char *host_map;         /* host address of the host ring */
    unsigned char *inbound_map;      /* host address of the inbound ring */
    unsigned long long host_bytes;   /* mapped bytes of the host ring */
    unsigned long long inbound_bytes; /* mapped bytes of the inbound ring */
} aotx_seam_rings;

/* Make the two rings, write their preambles, and register them for the device. */
int aotx_seam_open(aotx_seam_rings *rings, unsigned long long boot_id);

/* Give the device the ring addresses and the ring region. */
int aotx_seam_bind(const aotx_seam_rings *rings, unsigned long long ring_base,
                   unsigned long long ring_bytes, unsigned long long boot_id);

/* Mark both rings closed, so a reader stops at the end of the run. */
void aotx_seam_finish(const aotx_seam_rings *rings);

/* Unregister and unmap the rings. */
void aotx_seam_close(aotx_seam_rings *rings);

/* Start a program and give back its process id. */
int aotx_seam_spawn(const char *path, char *const argv[], int *pid);

/* Report whether a program has stopped. The status is its exit code. */
int aotx_seam_poll(int pid, int *stopped, int *status);

/* Wait for a program to stop and give back its exit code. */
int aotx_seam_wait(int pid);

/* An acquire load of a field that the other side of the seam writes. */
__device__ __forceinline__ unsigned long long aotx_seam_acquire_sys(const void *address)
{
    unsigned long long value;
    asm volatile("ld.acquire.sys.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
    return value;
}

/* A release store of a field that the other side of the seam reads. */
__device__ __forceinline__ void aotx_seam_release_sys(void *address, unsigned long long value)
{
    asm volatile("st.release.sys.u64 [%0], %1;" : : "l"(address), "l"(value) : "memory");
}

/* A release store of a field that only device readers see. */
__device__ __forceinline__ void aotx_seam_release_gpu(void *address, unsigned long long value)
{
    asm volatile("st.release.gpu.u64 [%0], %1;" : : "l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ unsigned long long aotx_seam_fnv1a(unsigned long long hash,
                                                              const unsigned char *bytes,
                                                              unsigned int count)
{
    for (unsigned int i = 0; i < count; ++i) {
        hash ^= (unsigned long long)bytes[i];
        hash *= AOTX_FNV_PRIME;
    }
    return hash;
}

/* Claim a run of sequences. The first sequence of a run is 1. */
__device__ __forceinline__ unsigned long long aotx_seam_claim(unsigned int count)
{
    return atomicAdd(&aotx_seam.dev.tail, (unsigned long long)count) + 1ull;
}

/* The slot of a sequence. The sequence field goes to zero first, so a reader that looks at
 * the slot while the record is under write sees an unpublished slot. */
__device__ __forceinline__ aotx_record_header *aotx_seam_slot(unsigned long long seq)
{
    unsigned char *at = aotx_seam.dev.base
                      + ((seq - 1ull) & aotx_seam.dev.mask) * (unsigned long long)AOTX_SLOT_BYTES;
    aotx_record_header *header = (aotx_record_header *)at;
    ((volatile aotx_record_header *)header)->seq = 0ull;
    return header;
}

__device__ __forceinline__ unsigned char *aotx_seam_body(aotx_record_header *header)
{
    return (unsigned char *)header + AOTX_HEADER_BYTES;
}

/* Fill the header and publish the record. The sequence goes last, with release order, so a
 * reader that sees the sequence sees the whole record. */
__device__ __forceinline__ void aotx_seam_publish(aotx_record_header *header,
                                                  unsigned long long seq,
                                                  unsigned int writer, unsigned int cls,
                                                  unsigned int type, unsigned int flags,
                                                  unsigned int body_len)
{
    header->magic = AOTX_WIRE_MAGIC;
    header->layout = (unsigned short)AOTX_WIRE_LAYOUT;
    header->header_bytes = (unsigned short)AOTX_HEADER_BYTES;
    header->boot_id = aotx_seam.boot_id;
    header->tick = aotx_time_tick;
    header->globaltimer = aotx_time_globaltimer();
    header->writer = writer;
    header->cls = (unsigned char)cls;
    header->type = (unsigned char)type;
    header->flags = (unsigned short)flags;
    header->body_len = body_len;
    header->reserved[0] = 0u;
    header->reserved[1] = 0u;
    header->reserved[2] = 0u;
    aotx_seam_release_gpu(&header->seq, seq);
}

/* Write one record whose body is a fixed structure. */
__device__ __forceinline__ unsigned long long aotx_seam_write(unsigned int writer,
                                                              unsigned int cls,
                                                              unsigned int type,
                                                              unsigned int flags,
                                                              const void *body,
                                                              unsigned int body_len)
{
    unsigned long long seq = aotx_seam_claim(1u);
    aotx_record_header *header = aotx_seam_slot(seq);
    const unsigned char *from = (const unsigned char *)body;
    unsigned char *to = aotx_seam_body(header);
    for (unsigned int i = 0; i < body_len; ++i) {
        to[i] = from[i];
    }
    aotx_seam_publish(header, seq, writer, cls, type, flags, body_len);
    return seq;
}

/* A pad record fills a sequence that carries nothing, so the sequence space has no gap. */
__device__ __forceinline__ void aotx_seam_pad(unsigned long long seq)
{
    aotx_record_header *header = aotx_seam_slot(seq);
    aotx_seam_publish(header, seq, AOTX_WRITER_SYSTEM, AOTX_CLASS_B, AOTX_REC_PAD, 0u, 0u);
}

/* The bytes one block takes for a record count. A block never wraps. */
__device__ __forceinline__ unsigned long long aotx_seam_block_bytes(unsigned long long records)
{
    return (unsigned long long)AOTX_BLOCK_HEADER_BYTES
         + records * (unsigned long long)AOTX_SLOT_BYTES;
}

/* Free bytes in the host ring. The cursor is the one field the drain writes. */
__device__ __forceinline__ unsigned long long aotx_seam_host_free(unsigned long long cursor)
{
    return aotx_seam.host.data_bytes - (aotx_seam.host.head - cursor);
}

__global__ void aotx_seam_apply_inbound(void);
__global__ void aotx_seam_flush(void);
__global__ void aotx_seam_note_boot(unsigned long long previous_boot_id,
                                    unsigned long long wall_ns);

#endif
