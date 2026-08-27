/* Purpose: Declare the disk-side library that maps the rings and writes segments.
 * Owns: Nothing; the caller owns every structure that this header declares.
 * Threading: One thread for each program; no function here is thread safe.
 * Lifetime: From the map of a ring to the release, and from segment open to close. */
#ifndef AOTX_DISK_WIRE_H
#define AOTX_DISK_WIRE_H

#include <signal.h>
#include <stddef.h>
#include <stdint.h>

#include "cuda/seam/wire.h"

/* The exit codes that the three disk-side programs share. */
#define AOTX_EXIT_OK        0
#define AOTX_EXIT_FAULT     1
#define AOTX_EXIT_LAYOUT    2
#define AOTX_EXIT_NOJOURNAL 3

/* ---- memory order helpers ---- */

/* The producer publishes with a release store. The consumer must pair it with an acquire
 * load, or it can see the payload of a block before the block is complete. */
static inline uint64_t aotx_load_acquire(const volatile uint64_t *p)
{
    return __atomic_load_n((const uint64_t *)p, __ATOMIC_ACQUIRE);
}

static inline void aotx_store_release(volatile uint64_t *p, uint64_t v)
{
    __atomic_store_n((uint64_t *)p, v, __ATOMIC_RELEASE);
}

static inline uint16_t aotx_load_acquire16(const volatile uint16_t *p)
{
    return __atomic_load_n((const uint16_t *)p, __ATOMIC_ACQUIRE);
}

/* The producer of a ring sets the closed flag when it ends. */
static inline void aotx_store_release16(volatile uint16_t *p, uint16_t v)
{
    __atomic_store_n((uint16_t *)p, v, __ATOMIC_RELEASE);
}

/* ---- checksum ---- */

/* The table path is the reference. The seed of a first call is zero. The result of one call
 * is the seed of the next, so one checksum can go over several buffers. */
uint32_t aotx_crc32c_table(const void *data, size_t bytes, uint32_t seed);
uint32_t aotx_crc32c(const void *data, size_t bytes, uint32_t seed);
int aotx_crc32c_has_hardware(void);
uint32_t aotx_crc32c_hardware(const void *data, size_t bytes, uint32_t seed);

/* ---- digest ---- */

/* SHA-256 of FIPS 180-4. The digest is 32 bytes. The text of a digest is 64 hexadecimal
 * characters and one end byte, so a text buffer must hold 65 bytes. */
#define AOTX_SHA256_DIGEST 32

typedef struct aotx_sha256 {
    uint32_t h[8];              /* the eight words of the state */
    uint64_t bytes;             /* the count of bytes that went into the state */
    size_t fill;                /* the bytes of a block that is not complete */
    unsigned char block[64];
} aotx_sha256;

void aotx_sha256_init(aotx_sha256 *s);
void aotx_sha256_update(aotx_sha256 *s, const void *data, size_t bytes);
void aotx_sha256_final(aotx_sha256 *s, unsigned char digest[AOTX_SHA256_DIGEST]);

/* Writes the text of a digest. The buffer must hold 65 bytes. */
void aotx_sha256_text(const unsigned char digest[AOTX_SHA256_DIGEST], char *out);

/* Adds a byte range of an open file to a digest. The caller gives the buffer, so this
 * function makes no allocation. Returns 0, or 1 when the range does not read. */
int aotx_sha256_read(int fd, uint64_t offset, uint64_t bytes, void *buffer, size_t buffer_bytes,
                     aotx_sha256 *state);

/* Hashes a whole file and writes the text of the digest and the count of bytes. The caller
 * gives the buffer. Returns 0, or 1 when the file does not read. */
int aotx_sha256_file(const char *path, char *text, uint64_t *bytes, void *buffer,
                     size_t buffer_bytes);

/* ---- clock and pause ---- */

uint64_t aotx_wall_ns(void);

/* Pauses for a time that doubles at each call, from 50 microseconds to 2 milliseconds. The
 * state starts at zero and goes back to zero after work is done. */
void aotx_pause(uint64_t *state);

/* Asks the kernel for SIGTERM when the parent process ends. */
int aotx_die_with_parent(void);

/* ---- ring maps ---- */

typedef struct aotx_map {
    unsigned char *base; /* the first byte of the mapping */
    size_t bytes;        /* the size of the mapping */
    int fd;              /* the descriptor that the mapping came from */
} aotx_map;

typedef struct aotx_host_ring {
    aotx_host_ring_preamble *pre;
    unsigned char *data;
    uint64_t data_bytes;
    uint64_t mask;
} aotx_host_ring;

typedef struct aotx_inbound_ring {
    aotx_inbound_preamble *pre;
    unsigned char *slots;
    uint64_t slot_count;
    uint64_t mask;
} aotx_inbound_ring;

/* Maps the whole of a descriptor, with the size from fstat. Returns 0 or -1. */
int aotx_map_fd(int fd, aotx_map *out);
void aotx_map_release(aotx_map *m);

/* Attaches to a mapped ring. Returns 0, or -1 when the magic, the layout version, or a
 * size in the preamble is not the one that this build reads. */
int aotx_host_ring_attach(const aotx_map *m, aotx_host_ring *out);
int aotx_inbound_attach(const aotx_map *m, aotx_inbound_ring *out);

/* Makes a ring in a new memfd and writes its preamble. The caller keeps m->fd to give to a
 * child process. The data area and the slot count must be powers of two. */
int aotx_host_ring_create(uint64_t data_bytes, uint64_t boot_id, aotx_map *m, aotx_host_ring *out);
int aotx_inbound_create(uint64_t slot_count, aotx_map *m, aotx_inbound_ring *out);

/* ---- host ring, the side that the drain reads ---- */

#define AOTX_TAKE_OK    0
#define AOTX_TAKE_EMPTY 1
#define AOTX_TAKE_TORN  2
#define AOTX_TAKE_BAD   3

typedef struct aotx_take {
    uint64_t block_seq;
    uint64_t tick;
    uint64_t first_seq;
    uint64_t boot_id;
    uint32_t byte_len;
    uint32_t record_count;
    uint16_t kind;
    int status;
    const char *reason; /* the cause, when the status is AOTX_TAKE_BAD */
} aotx_take;

uint64_t aotx_host_ring_head(const aotx_host_ring *r);
uint64_t aotx_host_ring_cursor(const aotx_host_ring *r);
int aotx_host_ring_closed(const aotx_host_ring *r);

/* Copies the block at the cursor into out and checks it by the double-load rule. The head
 * comes from one acquire load by the caller. The return value is the status. */
int aotx_host_ring_take(const aotx_host_ring *r, uint64_t cursor, uint64_t head,
                        unsigned char *out, uint32_t out_bytes, aotx_take *t);

/* Publishes the new cursor with a release store, after the bytes reach the disk. */
void aotx_host_ring_advance(const aotx_host_ring *r, uint64_t cursor);

/* ---- inbound ring, the side that the feeder writes ---- */

uint64_t aotx_inbound_head(const aotx_inbound_ring *r);
uint64_t aotx_inbound_consumed(const aotx_inbound_ring *r);
int aotx_inbound_closed(const aotx_inbound_ring *r);

/* Waits for one free slot, with backoff. Returns 0 when a slot is free, and -1 when the
 * ring closed or the stop flag went to one. */
int aotx_inbound_wait(const aotx_inbound_ring *r, const volatile sig_atomic_t *stop);

/* Writes one record into the slot at the head and publishes it. The magic, the layout, the
 * header size, and the sequence come from the ring; the caller gives the other fields. */
void aotx_inbound_put(const aotx_inbound_ring *r, const aotx_record_header *h, const void *body);

/* ---- records inside a block ---- */

int aotx_record_valid(const aotx_record_header *h);

/* Checks the block header and the records that follow it. Returns 0 or -1. */
int aotx_block_valid(const unsigned char *block, uint32_t byte_len, const char **reason);

const aotx_record_header *aotx_block_record(const unsigned char *block, uint32_t index);
const unsigned char *aotx_record_body(const aotx_record_header *h);

/* ---- segment files ---- */

#define AOTX_SEGMENT_LIMIT (64u * 1024u * 1024u)
#define AOTX_PATH_BYTES    512

typedef struct aotx_segment_writer {
    int fd;
    uint64_t bytes; /* bytes in the open file */
    uint64_t index; /* the number of the open segment */
    uint64_t limit; /* the size at which the writer opens the next segment */
    char dir[AOTX_PATH_BYTES];
} aotx_segment_writer;

int aotx_segment_open(aotx_segment_writer *w, const char *dir, uint64_t limit);
int aotx_segment_put(aotx_segment_writer *w, const unsigned char *block, uint32_t byte_len);
int aotx_segment_sync(aotx_segment_writer *w);
int aotx_segment_close(aotx_segment_writer *w);

#define AOTX_FRAME_OK   0
#define AOTX_FRAME_END  1
#define AOTX_FRAME_TORN 2

typedef struct aotx_segment_reader {
    int fd;
    uint64_t offset;
} aotx_segment_reader;

int aotx_segment_reader_open(aotx_segment_reader *r, const char *path);
void aotx_segment_reader_close(aotx_segment_reader *r);

/* Reads the next frame and checks its length and its checksum. The return value is
 * AOTX_FRAME_OK, AOTX_FRAME_END at a clean end, or AOTX_FRAME_TORN. */
int aotx_segment_get(aotx_segment_reader *r, unsigned char *out, uint32_t out_bytes, uint32_t *got);

/* Lists the segment file names of one directory in read order. Returns the count or -1. */
#define AOTX_NAME_BYTES 64
int aotx_segment_list(const char *dir, char (*names)[AOTX_NAME_BYTES], int max);

/* Makes one directory, and accepts a directory that is already there. Returns 0 or -1. */
int aotx_make_dir(const char *path);

#endif
