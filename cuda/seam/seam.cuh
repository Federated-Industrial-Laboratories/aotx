/* Purpose: Move records across the seam in rings.
 * Owns: The device ring, the host ring layout and the inbound cursor.
 * Launch shape: One thread for each record; one block for the flush.
 * Lifetime: The whole run. */
#ifndef AOTX_SEAM_CUH
#define AOTX_SEAM_CUH

#include "profile/profile.cuh"
#include "seam/wire.h"
#include "time/time.cuh"

/* The device ring holds AOTX_DEVICE_RING_SLOTS slots. The size is a power of two, so the
 * position of a sequence is a mask and not a division. */

/* Bounds that keep one tick inside the rings. The tick start kernel holds a tick that cannot
 * meet them. One tick writes at the most one half of the ring. A tick at the bound
 * therefore leaves room for the block that carries it. */
#define AOTX_INBOUND_MAX_TICK    256ull     /* inbound slots applied in one tick */
#define AOTX_TICK_RECORDS_MAX    (AOTX_DEVICE_RING_SLOTS / 2ull)
#define AOTX_APPLY_RECORDS_EACH  2ull       /* records the apply writes for each input */

/* The command layer writes records for an input it accepts: one command record and the
 * console lines of the answer. The tick start keeps room for this many, so a full tick of
 * inputs and their answers stays inside both rings. */
#define AOTX_CLI_RECORDS_EACH    32ull

/* Records a tick writes that no input and no tick load asks for. The set is a stall
 * record, a statistics record, a commit record, and one more. */
#define AOTX_TICK_RECORDS_OWN    4ull

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
    unsigned long long first_seq;     /* the first sequence the apply owns this tick */
} aotx_seam_apply_state;

typedef struct aotx_seam_state {
    aotx_seam_device_ring dev;
    aotx_seam_host_ring host;
    aotx_seam_inbound_ring in;
    aotx_seam_apply_state apply;
    unsigned long long boot_id;
    unsigned long long replaying;  /* 1 while a restore replays the journal, else 0 */
} aotx_seam_state;

extern __device__ aotx_seam_state aotx_seam;

/* The size of each host ring comes from the profile. The data area is a power of two. */

/* What the host glue keeps for the two rings that cross the seam. The file descriptors stay
 * open across an exec, so a disk side program maps the same memory. */
typedef struct aotx_seam_rings {
    int host_fd;                     /* the host ring file */
    int inbound_fd;                  /* the inbound ring file */
    int bulk_fd;                     /* the bulk ring file */
    /* The mirror is the fourth crossing: the snapshot of the cell grid a terminal reads.
     * It carries no consumer field, and a reader of it holds nothing on the device. */
    int mirror_fd;                   /* the mirror file */
    unsigned char *host_map;         /* host address of the host ring */
    unsigned char *inbound_map;      /* host address of the inbound ring */
    unsigned char *bulk_map;         /* host address of the bulk ring */
    unsigned char *mirror_map;       /* host address of the mirror */
    unsigned long long host_bytes;   /* mapped bytes of the host ring */
    unsigned long long inbound_bytes; /* mapped bytes of the inbound ring */
    unsigned long long bulk_bytes;   /* mapped bytes of the bulk ring */
    unsigned long long mirror_bytes; /* mapped bytes of the mirror */
} aotx_seam_rings;

/* Make the rings and the mirror, write their preambles, and register them for the
 * device. */
int aotx_seam_open(aotx_seam_rings *rings, unsigned long long boot_id);

/* Give the device the ring addresses and the ring region. */
int aotx_seam_bind(const aotx_seam_rings *rings, unsigned long long ring_base,
                   unsigned long long ring_bytes, unsigned long long boot_id);

/* Give the device the bulk ring and the staging region that feeds it. The bulk path stays
 * inert until this call: a stage request gives back a null pointer. */
int aotx_seam_bind_bulk(const aotx_seam_rings *rings, unsigned long long stage_base,
                        unsigned long long stage_bytes);

/* State whether a restore replays the journal. The command layer refuses to close the run
 * while the flag stands. */
void aotx_seam_set_replaying(int on);

/* Mark every ring closed, so a reader stops at the end of the run. */
void aotx_seam_finish(const aotx_seam_rings *rings);

/* Unregister and unmap the rings. */
void aotx_seam_close(aotx_seam_rings *rings);

/* Start a program and give back its process id. The child keeps the standard descriptors
 * and the descriptors that keep names. Every other descriptor gets the close-on-exec flag,
 * so a program receives only the rings and the pipes that belong to it. */
int aotx_seam_spawn(const char *path, char *const argv[], const int *keep,
                    unsigned int keep_count, int *pid);

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

/* Give the count of inbound records the apply may take this tick. A run with no replay
 * takes every record that is ready. A replay takes the records of one tick of the journal.
 * The inputs then reach the device at the place in the flow they had before. One thread of
 * the tick start calls this. */
__device__ unsigned int aotx_seam_replay_take(unsigned long long base, unsigned int ready);

/* Ticks of a replay in which the clock of the journal had not reached the next record. */
extern __device__ unsigned long long aotx_seam_replay_holds;

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

/* The body of the record that a sequence names. The call changes nothing, so the writer of
 * a record may read it again after it published it. */
__device__ __forceinline__ unsigned char *aotx_seam_body_of(unsigned long long seq)
{
    unsigned char *at = aotx_seam.dev.base
                      + ((seq - 1ull) & aotx_seam.dev.mask)
                        * (unsigned long long)AOTX_SLOT_BYTES;
    return at + AOTX_HEADER_BYTES;
}

__device__ __forceinline__ unsigned char *aotx_seam_body(aotx_record_header *header)
{
    return (unsigned char *)header + AOTX_HEADER_BYTES;
}

/* Fill the header and publish the record with a given tick. The sequence goes last, with
 * release order, so a reader that sees the sequence sees the whole record. */
__device__ __forceinline__ void aotx_seam_publish_at(aotx_record_header *header,
                                                     unsigned long long seq,
                                                     unsigned int writer, unsigned int cls,
                                                     unsigned int type, unsigned int flags,
                                                     unsigned int body_len,
                                                     unsigned long long tick)
{
    header->magic = AOTX_WIRE_MAGIC;
    header->layout = (unsigned short)AOTX_WIRE_LAYOUT;
    header->header_bytes = (unsigned short)AOTX_HEADER_BYTES;
    header->boot_id = aotx_seam.boot_id;
    header->tick = tick;
    header->globaltimer = aotx_time_globaltimer();
    header->writer = writer;
    header->cls = (unsigned char)cls;
    header->type = (unsigned char)type;
    /* A record written while a replay runs carries the replay flag. The journal already
     * holds the answer to such a record. The disk side therefore derives no request from
     * it, so the feeder executes no tool a second time. */
    header->flags = (unsigned short)(flags | ((aotx_seam.replaying != 0ull)
                                              ? (unsigned int)AOTX_FLAG_REPLAY : 0u));
    header->body_len = body_len;
    header->reserved[0] = 0u;
    header->reserved[1] = 0u;
    header->reserved[2] = 0u;
    aotx_seam_release_gpu(&header->seq, seq);
}

/* Publish a record with the tick that runs. */
__device__ __forceinline__ void aotx_seam_publish(aotx_record_header *header,
                                                  unsigned long long seq,
                                                  unsigned int writer, unsigned int cls,
                                                  unsigned int type, unsigned int flags,
                                                  unsigned int body_len)
{
    aotx_seam_publish_at(header, seq, writer, cls, type, flags, body_len, aotx_time_tick);
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

/* The bulk channel. A large payload never enters a record. The payload is staged in the
 * scratch arena. The bulk flush copies it into the bulk ring as one block at the end of the
 * tick. A BULK record names the payload and carries the handle of the block. */

/* The staging region is the first bytes of the scratch arena. */
#define AOTX_BULK_STAGE_BYTES  (8ull * 1024ull * 1024ull)

/* Payloads one tick may stage. */
#define AOTX_BULK_STAGE_MAX    256u

/* Bytes in front of each staged payload. They hold the entry that the commit needs, so the
 * commit finds its entry from the pointer alone. */
#define AOTX_BULK_PREFIX_BYTES 16u

/* What a payload is. A text export is the only kind of this build. */
#define AOTX_BULK_KIND_TEXT    1u

typedef struct aotx_bulk_entry {
    unsigned long long offset;  /* payload offset in the staging region */
    unsigned long long length;  /* payload bytes */
    unsigned long long handle;  /* the payload sequence that the flush publishes */
    unsigned long long tick;    /* the tick that claimed the entry */
    unsigned int kind;          /* AOTX_BULK_KIND_* */
    unsigned int committed;     /* 1 after the commit wrote the record */
} aotx_bulk_entry;

typedef struct aotx_bulk_state {
    aotx_seam_host_ring ring;      /* the bulk ring; zero base while the path is inert */
    unsigned char *stage;          /* first byte of the staging region, or zero */
    unsigned long long stage_bytes;
    unsigned long long used;       /* staging bytes claimed this tick */
    unsigned long long room;       /* bulk ring bytes free, read at tick start */
    unsigned long long reserved;   /* bulk ring bytes claimed this tick */
    unsigned long long published;  /* payloads published before this tick */
    unsigned long long refused;    /* stage calls that found no room */
    unsigned long long stale;      /* commit calls whose pointer came from an earlier tick */
    unsigned long long blocks;     /* blocks published on the bulk ring, pads included */
    unsigned int count;            /* entries claimed this tick */
    unsigned int reserved0;
    aotx_bulk_entry entry[AOTX_BULK_STAGE_MAX];
} aotx_bulk_state;

extern __device__ aotx_bulk_state aotx_bulk;

/* Claim staging bytes for a payload. The return is the first byte of the payload, or a null
 * pointer when the staging region or the bulk ring has no room. Nothing spins. */
__device__ void *aotx_bulk_stage(unsigned int kind, unsigned long long length);

/* Write the record that names a staged payload. The return is the handle, or 0 when the
 * pointer does not come from a stage call of the tick that runs. The staging region and the
 * entry table last one tick, so a pointer of an earlier tick names another payload. */
__device__ unsigned long long aotx_bulk_commit(void *pointer, unsigned int kind,
                                               unsigned long long length,
                                               unsigned long long tick);

/* Read the bulk ring cursor once, and open the staging region for a new tick. */
__device__ __forceinline__ void aotx_bulk_tick_start(void)
{
    aotx_bulk.used = 0ull;
    aotx_bulk.reserved = 0ull;
    aotx_bulk.count = 0u;
    if (aotx_bulk.ring.data == 0) {
        aotx_bulk.room = 0ull;
        return;
    }
    const aotx_host_ring_preamble *preamble =
        (const aotx_host_ring_preamble *)aotx_bulk.ring.preamble;
    unsigned long long cursor = aotx_seam_acquire_sys(&preamble->cursor);
    aotx_bulk.room = aotx_bulk.ring.data_bytes - (aotx_bulk.ring.head - cursor);
}

__global__ void aotx_seam_apply_inbound(void);
__global__ void aotx_seam_flush(void);
__global__ void aotx_seam_bulk_flush(void);
__global__ void aotx_seam_note_boot(unsigned long long previous_boot_id,
                                    unsigned long long wall_ns);

#endif
