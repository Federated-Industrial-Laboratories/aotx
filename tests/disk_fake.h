/* Purpose: Give the disk-side tests a device that publishes blocks and a case counter.
 * Owns: The staging buffer of one block, and the counters of one test program.
 * Threading: One thread; a test drives the fake device and the reader in turn.
 * Lifetime: The run of one test program. */
#ifndef AOTX_TESTS_DISK_FAKE_H
#define AOTX_TESTS_DISK_FAKE_H

#include "disk/wire/diskwire.h"

#include <ftw.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static int aotx_cases;
static int aotx_fails;

#define CHECK(cond, ...)                                    \
    do {                                                    \
        aotx_cases++;                                       \
        if (!(cond)) {                                      \
            aotx_fails++;                                   \
            printf("FAIL %s:%d ", __FILE__, __LINE__);      \
            printf(__VA_ARGS__);                            \
            printf("\n");                                   \
        }                                                   \
    } while (0)

/* Prints the count of cases and the count of failures. A run with fewer cases than the
 * least count is not a clean run, because a test that applies nothing proves nothing. */
static inline int aotx_report(const char *name, int least)
{
    printf("%s: cases applied %d, failed %d\n", name, aotx_cases, aotx_fails);
    if (aotx_cases < least) {
        printf("%s: too few cases, at least %d were asked for\n", name, least);
        return 1;
    }
    return (aotx_fails == 0) ? 0 : 1;
}

/* ---- temporary directories, and the child programs that a test drives ---- */

static inline int aotx_drop(const char *path, const struct stat *st, int type, struct FTW *w)
{
    (void)st;
    (void)type;
    (void)w;
    return remove(path);
}

static inline void aotx_remove_tree(const char *path)
{
    nftw(path, aotx_drop, 16, FTW_DEPTH | FTW_PHYS);
}

/* Makes a directory that only this test writes, so two tests cannot collide. */
static inline int aotx_temp_dir(char *out, size_t bytes)
{
    const char *root = getenv("TMPDIR");
    snprintf(out, bytes, "%s/aotx-test-XXXXXX", (root != NULL) ? root : "/tmp");
    return (mkdtemp(out) != NULL) ? 0 : -1;
}

/* Starts one program. The input and output descriptors become the standard input and the
 * standard output of the child when they are not negative. Returns the child, or -1. */
static inline int aotx_spawn(char *const argv[], int input_fd, int output_fd)
{
    pid_t child = fork();
    if (child < 0) {
        return -1;
    }
    if (child == 0) {
        if (input_fd >= 0) {
            dup2(input_fd, 0);
        }
        if (output_fd >= 0) {
            dup2(output_fd, 1);
        }
        execv(argv[0], argv);
        _exit(127);
    }
    return (int)child;
}

/* Returns one while the program still runs. The program stays waitable, so a later wait
 * still gives the exit status. */
static inline int aotx_alive(int child)
{
    siginfo_t info;
    memset(&info, 0, sizeof(info));
    if (waitid(P_PID, (id_t)child, &info, WEXITED | WNOHANG | WNOWAIT) != 0) {
        return 0;
    }
    return (info.si_pid == 0) ? 1 : 0;
}

/* Waits for one program and returns its exit status, or -1 when a signal ended it. */
static inline int aotx_wait(int child)
{
    int status = 0;
    if (waitpid((pid_t)child, &status, 0) < 0 || !WIFEXITED(status)) {
        return -1;
    }
    return WEXITSTATUS(status);
}

#define AOTX_FAKE_RECORDS 256
#define AOTX_FAKE_BYTES   (AOTX_BLOCK_HEADER_BYTES + AOTX_FAKE_RECORDS * AOTX_SLOT_BYTES)

typedef struct aotx_fake_device {
    aotx_host_ring *ring;
    uint64_t head;
    uint64_t block_seq;
    uint64_t record_seq;
    uint64_t tick;
    uint64_t boot_id;
    uint32_t writer;   /* the writer that the records of this device carry */
    uint32_t count;
    unsigned char stage[AOTX_FAKE_BYTES];
} aotx_fake_device;

static inline void aotx_fake_start(aotx_fake_device *d, aotx_host_ring *ring, uint64_t boot_id)
{
    memset(d, 0, sizeof(*d));
    d->ring = ring;
    d->boot_id = boot_id;
    d->record_seq = 1;
    d->tick = 1;
}

/* Adds one record to the block that is under construction. */
static inline void aotx_fake_record(aotx_fake_device *d, uint8_t cls, uint8_t type,
                                    const void *body, uint32_t body_len)
{
    unsigned char *slot = d->stage + AOTX_BLOCK_HEADER_BYTES + (size_t)d->count * AOTX_SLOT_BYTES;
    aotx_record_header *h = (aotx_record_header *)slot;
    if (d->count >= AOTX_FAKE_RECORDS) {
        return;
    }
    memset(slot, 0, AOTX_SLOT_BYTES);
    h->magic = AOTX_WIRE_MAGIC;
    h->layout = AOTX_WIRE_LAYOUT;
    h->header_bytes = AOTX_HEADER_BYTES;
    h->boot_id = d->boot_id;
    h->tick = d->tick;
    h->seq = d->record_seq++;
    h->globaltimer = 0;
    h->writer = d->writer;
    h->cls = cls;
    h->type = type;
    h->flags = 0;
    h->body_len = body_len;
    if (body_len > 0) {
        memcpy(slot + AOTX_HEADER_BYTES, body, body_len);
    }
    d->count++;
}

/* Writes one block into the ring by the publish order of the seam. The sequence goes to
 * zero first, the bytes follow, and the true sequence goes last. */
static inline void aotx_fake_write(aotx_fake_device *d, const unsigned char *block,
                                   uint32_t copy_bytes, uint32_t advance, uint64_t seq, int hold)
{
    uint64_t offset = d->head & d->ring->mask;
    unsigned char *at = d->ring->data + offset;
    aotx_block_header *h = (aotx_block_header *)at;
    aotx_store_release(&h->block_seq, 0);
    memcpy(at, block, copy_bytes);
    if (!hold) {
        aotx_store_release(&h->block_seq, seq);
        aotx_store_release(&d->ring->pre->last_block_seq, seq);
    }
    d->head += advance;
    aotx_store_release(&d->ring->pre->head, d->head);
}

/* Fills the tail of the data area with a pad block. A pad goes in when the next block does
 * not fit, or when it would leave a tail that is shorter than a block header. Only the
 * header of a pad block carries data, and a tail of fewer than 64 bytes holds no block. */
static inline void aotx_fake_pad(aotx_fake_device *d, uint32_t byte_len)
{
    unsigned char pad[AOTX_BLOCK_HEADER_BYTES];
    aotx_block_header *h = (aotx_block_header *)pad;
    uint64_t offset = d->head & d->ring->mask;
    uint64_t left = d->ring->data_bytes - offset;
    if (left >= byte_len && (left - byte_len == 0 || left - byte_len >= AOTX_BLOCK_HEADER_BYTES)) {
        return;
    }
    memset(pad, 0, sizeof(pad));
    h->magic = AOTX_BLOCK_MAGIC;
    h->layout = AOTX_WIRE_LAYOUT;
    h->kind = AOTX_BLOCK_PAD;
    h->block_seq = 0;
    h->boot_id = d->boot_id;
    h->tick = d->tick;
    h->first_seq = d->record_seq;
    h->record_count = 0;
    h->byte_len = (uint32_t)left;
    d->block_seq++;
    aotx_fake_write(d, pad, AOTX_BLOCK_HEADER_BYTES, (uint32_t)left, d->block_seq, 0);
}

/* Gives the bytes that a block needs, with the pad block that comes before it. */
static inline uint64_t aotx_fake_need(const aotx_fake_device *d, uint32_t byte_len)
{
    uint64_t left = d->ring->data_bytes - (d->head & d->ring->mask);
    if (left >= byte_len && (left - byte_len == 0 || left - byte_len >= AOTX_BLOCK_HEADER_BYTES)) {
        return byte_len;
    }
    return left + byte_len;
}

/* Waits until the data area has room for the bytes. The device decides this at tick start
 * from the cursor of the drain, and holds the tick when the room is not there. */
static inline void aotx_fake_room(aotx_fake_device *d, uint64_t need)
{
    uint64_t backoff = 0;
    while (d->head + need - aotx_host_ring_cursor(d->ring) > d->ring->data_bytes) {
        aotx_pause(&backoff);
    }
}

/* Closes the block under construction and publishes it. A hold of one leaves the sequence
 * at zero, which is the state of a block that is still under write. */
static inline uint64_t aotx_fake_commit(aotx_fake_device *d, int hold)
{
    aotx_block_header *h = (aotx_block_header *)d->stage;
    uint32_t byte_len = AOTX_BLOCK_HEADER_BYTES + d->count * AOTX_SLOT_BYTES;
    uint64_t seq;
    memset(d->stage, 0, AOTX_BLOCK_HEADER_BYTES);
    h->magic = AOTX_BLOCK_MAGIC;
    h->layout = AOTX_WIRE_LAYOUT;
    h->kind = 0;
    h->boot_id = d->boot_id;
    h->tick = d->tick;
    h->first_seq = d->record_seq - d->count;
    h->record_count = d->count;
    h->byte_len = byte_len;
    if (d->ring != NULL) {
        aotx_fake_room(d, aotx_fake_need(d, byte_len));
        aotx_fake_pad(d, byte_len);
    }
    d->block_seq++;
    seq = d->block_seq;
    if (d->ring != NULL) {
        aotx_fake_write(d, d->stage, byte_len, byte_len, seq, hold);
    } else {
        h->block_seq = seq;
    }
    d->count = 0;
    d->tick++;
    return seq;
}

/* Fills the body of one token of a sequence. Each number gives another slot, another
 * position, another token and another flag set. A test that compares two runs therefore
 * cannot pass with the records out of order. */
static inline void aotx_fake_token(int number, aotx_token_body *t)
{
    static const uint32_t sets[3] = { AOTX_TOKEN_PROMPT, AOTX_TOKEN_SAMPLED,
                                      AOTX_TOKEN_SAMPLED | AOTX_TOKEN_LAST };
    memset(t, 0, sizeof(*t));
    t->slot = (uint32_t)(number % 64);
    t->position = (uint32_t)(number * 3 + 1);
    t->token = (uint32_t)(1000 + number * 7);
    t->flags = sets[number % 3];
    t->seed = 0x5eed000000000000ull + (uint64_t)number;
    t->draw = (uint64_t)number;
    t->role = 1u + (uint32_t)(number % 3);
}

/* Fills the body of one sequence event. */
static inline void aotx_fake_sequence(int number, uint32_t event, aotx_sequence_body *q)
{
    memset(q, 0, sizeof(*q));
    q->slot = (uint32_t)(number % 64);
    q->event = event;
    q->prompt_tokens = (uint32_t)(7 + number);
    q->sampled_tokens = (uint32_t)(11 + 2 * number);
    q->ticks = (uint64_t)(13 + number);
    q->role = 1u + (uint32_t)(number % 3);
}

/* Adds one message record, with the fields that the drain turns into a line. Returns the
 * record sequence, which a later message names in re_seq or in corrects_seq. */
static inline uint64_t aotx_fake_bus(aotx_fake_device *d, uint8_t kind, uint8_t provenance,
                                     uint32_t writer_seq, uint64_t re_seq, uint64_t corrects_seq,
                                     float score, const char *text)
{
    aotx_bus_body body;
    uint64_t seq = d->record_seq;
    uint32_t len = (uint32_t)strlen(text);
    memset(&body, 0, sizeof(body));
    body.kind = kind;
    body.provenance = provenance;
    body.writer_seq = writer_seq;
    body.re_seq = re_seq;
    body.corrects_seq = corrects_seq;
    body.score = score;
    body.text_len = (len > AOTX_BUS_TEXT_BYTES) ? AOTX_BUS_TEXT_BYTES : len;
    memcpy(body.text, text, body.text_len);
    aotx_fake_record(d, AOTX_CLASS_B, AOTX_REC_BUS, &body, sizeof(body));
    return seq;
}

/* Publishes one payload block on a bulk ring. The handle is the block sequence, and the
 * length of the block is the header and the payload, rounded up to eight bytes. */
static inline uint64_t aotx_fake_payload(aotx_fake_device *d, const void *payload, uint32_t len,
                                         int hold)
{
    aotx_block_header *h = (aotx_block_header *)d->stage;
    uint32_t byte_len = AOTX_BLOCK_HEADER_BYTES + ((len + 7u) & ~7u);
    uint64_t seq;
    memset(d->stage, 0, byte_len);
    memcpy(d->stage + AOTX_BLOCK_HEADER_BYTES, payload, len);
    h->magic = AOTX_BLOCK_MAGIC;
    h->layout = AOTX_WIRE_LAYOUT;
    h->kind = AOTX_BLOCK_BULK;
    h->boot_id = d->boot_id;
    h->tick = d->tick;
    h->record_count = 0;
    h->byte_len = byte_len;
    aotx_fake_room(d, aotx_fake_need(d, byte_len));
    aotx_fake_pad(d, byte_len);
    d->block_seq++;
    seq = d->block_seq;
    h->first_seq = seq;
    aotx_fake_write(d, d->stage, byte_len, byte_len, seq, hold);
    d->tick++;
    return seq;
}

/* Publishes the sequence of a block that aotx_fake_payload left unpublished. */
static inline void aotx_fake_release(aotx_fake_device *d, uint64_t offset, uint64_t seq)
{
    aotx_block_header *h = (aotx_block_header *)(d->ring->data + (offset & d->ring->mask));
    aotx_store_release(&h->block_seq, seq);
    aotx_store_release(&d->ring->pre->last_block_seq, seq);
}

#endif
