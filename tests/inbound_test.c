/* Purpose: Check the records that the feeder publishes into the inbound ring.
 * Owns: One inbound ring for each case.
 * Threading: One thread; the test writes the ring and reads it as the device would.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include <signal.h>

#define AOTX_RING_SLOTS 16u

static void put(const aotx_inbound_ring *ring, uint8_t type, uint32_t writer,
                uint16_t flags, const void *body, uint32_t len)
{
    aotx_record_header h;
    memset(&h, 0, sizeof(h));
    h.boot_id = 0x00abcdef01234567ull;
    h.writer = writer;
    h.cls = AOTX_CLASS_A;
    h.type = type;
    h.flags = flags;
    h.body_len = len;
    aotx_inbound_put(ring, &h, body);
}

/* Reads one slot the way a device consumer reads it, and checks every header field. */
static void check_slot(const aotx_inbound_ring *ring, uint64_t index, const char *want,
                       uint32_t writer, uint16_t flags)
{
    const unsigned char *slot = ring->slots + (index & ring->mask) * AOTX_SLOT_BYTES;
    const aotx_record_header *h = (const aotx_record_header *)slot;
    uint32_t want_len = (uint32_t)strlen(want);
    CHECK(h->magic == AOTX_WIRE_MAGIC, "slot %llu has the wrong magic", (unsigned long long)index);
    CHECK(h->layout == AOTX_WIRE_LAYOUT, "slot %llu has the wrong layout",
          (unsigned long long)index);
    CHECK(h->header_bytes == AOTX_HEADER_BYTES, "slot %llu has the wrong header size",
          (unsigned long long)index);
    CHECK(h->seq == index + 1, "slot %llu holds sequence %llu", (unsigned long long)index,
          (unsigned long long)h->seq);
    CHECK(h->writer == writer, "slot %llu has the wrong writer", (unsigned long long)index);
    CHECK(h->cls == AOTX_CLASS_A, "slot %llu has the wrong class", (unsigned long long)index);
    CHECK(h->flags == flags, "slot %llu has the wrong flags", (unsigned long long)index);
    CHECK(h->body_len == want_len, "slot %llu has the wrong body length",
          (unsigned long long)index);
    CHECK(aotx_record_valid(h) == 1, "slot %llu does not validate", (unsigned long long)index);
    CHECK(memcmp(aotx_record_body(h), want, want_len) == 0, "slot %llu has the wrong body",
          (unsigned long long)index);
}

static void batch(int n)
{
    aotx_map map;
    aotx_inbound_ring ring;
    char body[64];
    int i;
    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, &map, &ring) == 0, "the ring does not open");
    for (i = 0; i < n; i++) {
        snprintf(body, sizeof(body), "line %d of %d", i, n);
        CHECK(aotx_inbound_wait(&ring, NULL) == 0, "no free slot at element %d", i);
        put(&ring, AOTX_REC_INPUT_LINE, AOTX_WRITER_FEEDER, 0, body, (uint32_t)strlen(body));
        CHECK(aotx_inbound_head(&ring) == (uint64_t)(i + 1), "the head does not count element %d", i);
        check_slot(&ring, (uint64_t)i, body, AOTX_WRITER_FEEDER, 0);
        /* The device consumes the slot, which is the only field the device writes here. */
        aotx_store_release(&ring.pre->consumed, (uint64_t)(i + 1));
    }
    CHECK(aotx_inbound_consumed(&ring) == (uint64_t)n, "the consumed count is wrong");
    aotx_map_release(&map);
}

static void backpressure(void)
{
    aotx_map map;
    aotx_inbound_ring ring;
    volatile sig_atomic_t stop = 0;
    char body[64];
    unsigned i;
    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, &map, &ring) == 0, "the ring does not open");
    for (i = 0; i < AOTX_RING_SLOTS; i++) {
        snprintf(body, sizeof(body), "held line %u", i);
        CHECK(aotx_inbound_wait(&ring, &stop) == 0, "no free slot at element %u", i);
        put(&ring, AOTX_REC_INPUT_LINE, AOTX_WRITER_RESTORE, AOTX_FLAG_REPLAYED,
            body, (uint32_t)strlen(body));
    }
    CHECK(aotx_inbound_head(&ring) - aotx_inbound_consumed(&ring) == AOTX_RING_SLOTS,
          "the ring is not full");
    stop = 1;
    CHECK(aotx_inbound_wait(&ring, &stop) == -1, "a full ring must not give a slot");
    stop = 0;
    aotx_store_release(&ring.pre->consumed, 1);
    CHECK(aotx_inbound_wait(&ring, &stop) == 0, "a consumed slot must free space");
    aotx_store_release16(&ring.pre->closed, 1);
    aotx_store_release(&ring.pre->consumed, 0);
    aotx_store_release(&ring.pre->head, AOTX_RING_SLOTS);
    CHECK(aotx_inbound_wait(&ring, NULL) == -1, "a closed ring must end the wait");
    for (i = 0; i < AOTX_RING_SLOTS; i++) {
        snprintf(body, sizeof(body), "held line %u", i);
        check_slot(&ring, i, body, AOTX_WRITER_RESTORE, AOTX_FLAG_REPLAYED);
    }
    aotx_map_release(&map);
}

static void replay_origins(unsigned n)
{
    aotx_map map; aotx_inbound_ring ring;
    aotx_record_header headers[64] = {0}; char text[64][64]; const void *bodies[64];
    CHECK(aotx_inbound_create(128, &map, &ring) == 0, "the replay ring does not open");
    for (unsigned i = 0; i < n; ++i) {
        snprintf(text[i], sizeof(text[i]), "replay member %u of %u", i, n); bodies[i] = text[i];
        headers[i].seq = 0x1000000000ull + 7 * i;
        headers[i].cls = AOTX_CLASS_A; headers[i].type = AOTX_REC_INPUT_LINE;
        headers[i].flags = AOTX_FLAG_REPLAYED; headers[i].body_len = (uint32_t)strlen(text[i]);
    }
    for (unsigned pass = 0; pass < 4; ++pass) {
        if (pass == 3) for (unsigned i = 0; i < n; ++i) headers[i].flags = 0;
        uint64_t head = aotx_inbound_head(&ring);
        aotx_inbound_put_many(&ring, headers, bodies, n);
        CHECK(aotx_inbound_head(&ring) == head + n, "the complete replay batch is published");
        for (unsigned i = 0; i < n; ++i) {
            const aotx_record_header *h = (const aotx_record_header *)(ring.slots +
                ((head + i) & ring.mask) * AOTX_SLOT_BYTES);
            uint64_t source = (uint64_t)h->source_seq[0] | ((uint64_t)h->source_seq[1] << 32);
            CHECK(h->seq == head + i + 1, "the new transport sequence is complete");
            CHECK(source == (pass == 3 ? 0 : 0x1000000000ull + 7 * i),
                "replay preserves the original source and fresh input clears it");
            CHECK(h->body_len == strlen(text[i]) && !memcmp(aotx_record_body(h), text[i], h->body_len),
                "the distinct replay body is unchanged");
            headers[i] = *h;
        }
        aotx_store_release(&ring.pre->consumed, head + n);
    }
    aotx_map_release(&map);
}

int main(void)
{
    batch(1);
    batch(64);
    backpressure();
    replay_origins(1); replay_origins(64);
    return aotx_report("inbound_test", 200);
}
