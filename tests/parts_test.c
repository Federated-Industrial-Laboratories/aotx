/* Purpose: Check that one input line enters the ring as contiguous wire parts.
 * Owns: One inbound ring and one source line for each case.
 * Threading: One thread; the test writes and then reads the ring.
 * Lifetime: The run of the test. */
#include "tests/disk_fake.h"

#include "disk/feed/line.h"

#include <pthread.h>
#include <stdatomic.h>

#define AOTX_PARTS_TEXT 4000u
#define AOTX_PARTS_WANT ((AOTX_PARTS_TEXT + AOTX_BODY_BYTES - 1u) / AOTX_BODY_BYTES)

static void fill_text(unsigned char *text, uint32_t length, unsigned int number)
{
    uint32_t i;
    for (i = 0u; i < length; i++) {
        text[i] = (unsigned char)('a' + (i + number * 7u) % 26u);
    }
}

static void check_line(const aotx_inbound_ring *ring, uint64_t first,
                       const unsigned char *text, unsigned int number)
{
    uint32_t part;
    uint32_t offset = 0u;
    for (part = 0u; part < AOTX_PARTS_WANT; part++) {
        const aotx_record_header *h = (const aotx_record_header *)(
            ring->slots + ((first + part) & ring->mask) * AOTX_SLOT_BYTES);
        uint32_t want = AOTX_PARTS_TEXT - offset;
        if (want > AOTX_BODY_BYTES) {
            want = AOTX_BODY_BYTES;
        }
        CHECK(h->seq == first + part + 1u, "line %u part %u is not contiguous", number, part);
        CHECK(h->type == AOTX_REC_INPUT_LINE, "line %u part %u has another type", number, part);
        CHECK(h->writer == AOTX_WRITER_FEEDER, "line %u part %u has another writer", number, part);
        CHECK(h->cls == AOTX_CLASS_A, "line %u part %u has another class", number, part);
        CHECK(h->flags == ((part == 0u) ? 0u : AOTX_FLAG_FRAGMENT),
              "line %u part %u has flags %u", number, part, h->flags);
        CHECK(h->body_len == want, "line %u part %u has length %u, not %u",
              number, part, h->body_len, want);
        CHECK(memcmp(aotx_record_body(h), text + offset, want) == 0,
              "line %u part %u has different bytes", number, part);
        offset += want;
    }
    CHECK(offset == AOTX_PARTS_TEXT, "line %u joins to %u bytes", number, offset);
    CHECK(((const aotx_record_header *)(ring->slots
          + ((first + AOTX_PARTS_WANT - 1u) & ring->mask) * AOTX_SLOT_BYTES))->body_len
          == AOTX_PARTS_TEXT % AOTX_BODY_BYTES,
          "line %u does not have the short last part", number);
}

static void batch(unsigned int lines)
{
    aotx_map map;
    aotx_inbound_ring ring;
    unsigned char text[AOTX_PARTS_TEXT];
    unsigned int i;
    uint64_t slots = (uint64_t)lines * AOTX_PARTS_WANT + 2u;
    uint64_t ring_slots = 1u;
    while (ring_slots < slots) {
        ring_slots <<= 1u;
    }
    CHECK(aotx_inbound_create(ring_slots, &map, &ring) == 0, "the ring does not open");
    for (i = 0u; i < lines; i++) {
        uint64_t first = aotx_inbound_head(&ring);
        fill_text(text, sizeof(text), i);
        CHECK(aotx_line_publish(&ring, NULL, text, sizeof(text)) == 0,
              "line %u did not publish", i);
        CHECK(aotx_inbound_head(&ring) == first + AOTX_PARTS_WANT,
              "line %u published %llu slots", i,
              (unsigned long long)(aotx_inbound_head(&ring) - first));
        check_line(&ring, first, text, i);
    }
    aotx_map_release(&map);
}

static void refusal_and_order(void)
{
    aotx_map map;
    aotx_inbound_ring ring;
    unsigned char text[AOTX_INPUT_LINE_BYTES + 1u];
    aotx_record_header h;
    aotx_key_body key;
    aotx_clock_body clock;
    uint64_t first;
    fill_text(text, sizeof(text), 9u);
    memset(&h, 0, sizeof(h));
    memset(&key, 0, sizeof(key));
    memset(&clock, 0, sizeof(clock));
    CHECK(aotx_inbound_create(64u, &map, &ring) == 0, "the order ring does not open");
    h.writer = AOTX_WRITER_FEEDER;
    h.cls = AOTX_CLASS_A;
    h.type = AOTX_REC_KEY;
    h.body_len = sizeof(key);
    key.key = 65u;
    aotx_inbound_put(&ring, &h, &key);
    first = aotx_inbound_head(&ring);
    CHECK(aotx_line_publish(&ring, NULL, text, AOTX_PARTS_TEXT) == 0,
          "the ordered line did not publish");
    h.type = AOTX_REC_TICK_START;
    h.body_len = sizeof(clock);
    clock.wall_ns = 17u;
    aotx_inbound_put(&ring, &h, &clock);
    CHECK(((const aotx_record_header *)(ring.slots
          + ((first - 1u) & ring.mask) * AOTX_SLOT_BYTES))->type == AOTX_REC_KEY,
          "the key did not stay before the line");
    check_line(&ring, first, text, 9u);
    CHECK(((const aotx_record_header *)(ring.slots
          + ((first + AOTX_PARTS_WANT) & ring.mask) * AOTX_SLOT_BYTES))->type
          == AOTX_REC_TICK_START, "the clock entered the parts of the line");
    first = aotx_inbound_head(&ring);
    CHECK(aotx_line_publish(&ring, NULL, text, sizeof(text)) == 1,
          "a line over the bound was not refused");
    CHECK(aotx_inbound_head(&ring) == first, "a refused line published a part");
    aotx_map_release(&map);
}

typedef struct visibility_state {
    const aotx_inbound_ring *ring;
    atomic_int done;
    atomic_int partial;
} visibility_state;

static void *watch_head(void *argument)
{
    visibility_state *state = (visibility_state *)argument;
    while (atomic_load_explicit(&state->done, memory_order_acquire) == 0) {
        uint64_t head = aotx_inbound_head(state->ring);
        if (head % AOTX_PARTS_WANT != 0u) {
            atomic_store_explicit(&state->partial, 1, memory_order_release);
        }
    }
    return NULL;
}

/* A reader can see the old head or the head after a complete line. It cannot see a head
 * between the parts of one line. */
static void atomic_publication(void)
{
    aotx_map map;
    aotx_inbound_ring ring;
    visibility_state state;
    unsigned char text[AOTX_PARTS_TEXT];
    pthread_t reader;
    unsigned int i;
    CHECK(aotx_inbound_create(2048u, &map, &ring) == 0,
          "the publication ring does not open");
    state.ring = &ring;
    atomic_init(&state.done, 0);
    atomic_init(&state.partial, 0);
    CHECK(pthread_create(&reader, NULL, watch_head, &state) == 0,
          "the head reader does not start");
    for (i = 0u; i < 64u; i++) {
        fill_text(text, sizeof(text), i);
        CHECK(aotx_line_publish(&ring, NULL, text, sizeof(text)) == 0,
              "atomic line %u did not publish", i);
    }
    atomic_store_explicit(&state.done, 1, memory_order_release);
    CHECK(pthread_join(reader, NULL) == 0, "the head reader does not stop");
    CHECK(atomic_load_explicit(&state.partial, memory_order_acquire) == 0,
          "a reader saw a head without all of its parts");
    aotx_map_release(&map);
}

int main(void)
{
    CHECK(AOTX_PARTS_WANT <= AOTX_LINE_PARTS_MAX,
          "the 4000 byte case needs more parts than the wire permits");
    batch(1u);
    batch(64u);
    refusal_and_order();
    atomic_publication();
    return aotx_report("parts_test", 9000);
}
