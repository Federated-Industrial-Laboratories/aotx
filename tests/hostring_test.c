/* Purpose: Check the block acceptor of the drain against a device that fills a host ring.
 * Owns: One host ring for each case.
 * Threading: One thread; the test writes and reads the ring in turn.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#define AOTX_RING_BYTES 16384u
#define AOTX_TAKE_BYTES 8192u

/* Reads every block that the ring holds and checks the sequence and the content. */
static void take_all(aotx_host_ring *ring, uint64_t *cursor, uint64_t *expect,
                     int *blocks, int *pads, int *records)
{
    static unsigned char block[AOTX_TAKE_BYTES];
    for (;;) {
        aotx_take t;
        uint64_t head = aotx_host_ring_head(ring);
        int status = aotx_host_ring_take(ring, *cursor, head, block, sizeof(block), &t);
        if (status == AOTX_TAKE_EMPTY) {
            return;
        }
        CHECK(status == AOTX_TAKE_OK, "the block at %llu is refused: %s",
              (unsigned long long)*cursor, t.reason);
        if (status != AOTX_TAKE_OK) {
            return;
        }
        CHECK(t.block_seq == *expect, "expected block %llu and found %llu",
              (unsigned long long)*expect, (unsigned long long)t.block_seq);
        *expect = t.block_seq + 1;
        if (t.kind == AOTX_BLOCK_PAD) {
            /* A pad block reaches the end of the data area, so the next block is at zero. */
            CHECK(((*cursor + t.byte_len) & ring->mask) == 0, "a pad does not reach the end");
            (*pads)++;
        } else {
            char want[64];
            uint32_t i;
            for (i = 0; i < t.record_count; i++) {
                const aotx_record_header *h = aotx_block_record(block, i);
                snprintf(want, sizeof(want), "block %d record %u", *blocks, i);
                CHECK(h->body_len == (uint32_t)strlen(want) &&
                      memcmp(aotx_record_body(h), want, h->body_len) == 0,
                      "the body of block %d record %u is wrong", *blocks, i);
                (*records)++;
            }
            (*blocks)++;
        }
        *cursor += t.byte_len;
        aotx_host_ring_advance(ring, *cursor);
    }
}

static void batch(int n)
{
    aotx_map map;
    aotx_host_ring ring;
    aotx_fake_device device;
    uint64_t cursor = 0;
    uint64_t expect = 1;
    int blocks = 0;
    int pads = 0;
    int records = 0;
    int i;
    CHECK(aotx_host_ring_create(AOTX_RING_BYTES, 0x0102030405060708ull, &map, &ring) == 0,
          "the ring does not open");
    aotx_fake_start(&device, &ring, 0x0102030405060708ull);
    for (i = 0; i < n; i++) {
        char body[64];
        int count = 1 + (i % 5);
        int r;
        for (r = 0; r < count; r++) {
            snprintf(body, sizeof(body), "block %d record %d", i, r);
            aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_CONSOLE, body, (uint32_t)strlen(body));
        }
        aotx_fake_commit(&device, 0);
        take_all(&ring, &cursor, &expect, &blocks, &pads, &records);
    }
    CHECK(blocks == n, "took %d blocks and %d were written", blocks, n);
    CHECK(cursor == aotx_host_ring_head(&ring), "the cursor does not reach the head");
    CHECK(aotx_host_ring_cursor(&ring) == cursor, "the cursor field does not hold the value");
    if (n > 40) {
        /* A ring of this size cannot hold this many blocks without a wrap. */
        CHECK(pads > 0, "no pad block appeared over a wrap");
    }
    printf("batch %d: blocks %d, pads %d, records %d\n", n, blocks, pads, records);
    aotx_map_release(&map);
}

static void refusals(void)
{
    aotx_map map;
    aotx_host_ring ring;
    aotx_fake_device device;
    static unsigned char block[AOTX_TAKE_BYTES];
    aotx_block_header *live;
    aotx_take t;
    CHECK(aotx_host_ring_create(AOTX_RING_BYTES, 9, &map, &ring) == 0, "the ring does not open");
    aotx_fake_start(&device, &ring, 9);
    aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_CONSOLE, "held", 4);
    aotx_fake_commit(&device, 1);

    CHECK(aotx_host_ring_take(&ring, 0, 0, block, sizeof(block), &t) == AOTX_TAKE_EMPTY,
          "a cursor that reaches the head must find nothing");
    CHECK(aotx_host_ring_take(&ring, 0, aotx_host_ring_head(&ring), block, sizeof(block), &t)
          == AOTX_TAKE_EMPTY, "a block whose sequence stays at zero must not be taken");

    /* The same bytes become a block as soon as the sequence goes out. */
    live = (aotx_block_header *)ring.data;
    aotx_store_release(&live->block_seq, 1);
    CHECK(aotx_host_ring_take(&ring, 0, aotx_host_ring_head(&ring), block, sizeof(block), &t)
          == AOTX_TAKE_OK, "the block must be taken after the sequence goes out");

    live->magic = 0;
    CHECK(aotx_host_ring_take(&ring, 0, aotx_host_ring_head(&ring), block, sizeof(block), &t)
          == AOTX_TAKE_BAD, "a wrong magic must be refused");
    live->magic = AOTX_BLOCK_MAGIC;
    live->byte_len = AOTX_RING_BYTES * 2u;
    CHECK(aotx_host_ring_take(&ring, 0, aotx_host_ring_head(&ring), block, sizeof(block), &t)
          == AOTX_TAKE_BAD, "a length outside the buffer must be refused");
    aotx_map_release(&map);
}

int main(void)
{
    batch(1);
    batch(64);
    refusals();
    return aotx_report("hostring_test", 100);
}
