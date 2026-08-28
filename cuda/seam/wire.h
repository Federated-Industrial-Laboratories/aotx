/* Purpose: Define the byte layouts that cross the seam: records, blocks, ring preambles.
 * Owns: Nothing; layouts and constants only, included by both sides.
 * Launch shape: Not applicable; plain C with no CUDA symbol.
 * Lifetime: The layout version; a change to any struct increments AOTX_WIRE_LAYOUT. */
#ifndef AOTX_SEAM_WIRE_H
#define AOTX_SEAM_WIRE_H

#include <stdint.h>

#define AOTX_WIRE_MAGIC        0x58544F41u   /* "AOTX" in little-endian byte order */
#define AOTX_WIRE_LAYOUT       1u
#define AOTX_LINE_BYTES        64u

/* One record fills one slot: a 64-byte header and a body of AOTX_BODY_BYTES. */
#define AOTX_HEADER_BYTES      64u
#define AOTX_SLOT_BYTES        256u
#define AOTX_BODY_BYTES        (AOTX_SLOT_BYTES - AOTX_HEADER_BYTES)

/* Record classes. Class A is replayed at restore; class B is derived and is not. */
#define AOTX_CLASS_A           1u
#define AOTX_CLASS_B           2u

/* Record types. The body layout of each type is given beside it. */
#define AOTX_REC_PAD           0u   /* no body; fills a slot that carries nothing */
#define AOTX_REC_BOOT          1u   /* class A; body: aotx_boot_body */
#define AOTX_REC_TICK_START    2u   /* class A; body: aotx_clock_body, written by the feeder */
#define AOTX_REC_TICK_COMMIT   3u   /* class A; body: aotx_commit_body, last record of a tick */
#define AOTX_REC_INPUT_LINE    4u   /* class A; body: UTF-8 bytes, body_len gives the count */
#define AOTX_REC_CONSOLE       5u   /* class B; body: UTF-8 bytes to show on the console */
#define AOTX_REC_STALL         6u   /* class B; body: aotx_stall_body */
#define AOTX_REC_STATS         7u   /* class B; body: aotx_stats_body */
#define AOTX_REC_RESTORE       8u   /* class B; body: aotx_restore_body */
#define AOTX_REC_NOTE          9u   /* class B; body: UTF-8 bytes; a bus note */
#define AOTX_REC_KEY           10u  /* class A; body: aotx_key_body, from the window */
#define AOTX_REC_COMMAND       11u  /* class B; body: UTF-8 bytes, a parsed command line */
#define AOTX_REC_BUS           12u  /* class B; body: aotx_bus_body, one bus message */
#define AOTX_REC_BULK          13u  /* class B; body: aotx_bulk_body, names a bulk block */
#define AOTX_REC_TOKEN         14u  /* class A; body: aotx_token_body, one token of a sequence */
#define AOTX_REC_SEQUENCE      15u  /* class B; body: aotx_sequence_body, a sequence event */

/* Record flags. */
#define AOTX_FLAG_REPLAYED     0x0001u  /* the record was applied again at restore */
#define AOTX_FLAG_FRAGMENT     0x0002u  /* the record continues the line of the one before */
#define AOTX_FLAG_FRAGMENT     0x0002u  /* a CONSOLE record that continues the line before it */

/* Writer identities below AOTX_WRITER_AGENT_BASE are system writers. */
#define AOTX_WRITER_SYSTEM     0u
#define AOTX_WRITER_FEEDER     1u
#define AOTX_WRITER_RESTORE    2u
#define AOTX_WRITER_CONSOLE    3u
#define AOTX_WRITER_AGENT_BASE 1024u

/* The record header, exactly 64 bytes. The seq field is the publish field of a slot.
 * Zero means unpublished or under rewrite; a published record has a seq of 1 or more. */
typedef struct aotx_record_header {
    uint32_t magic;        /* AOTX_WIRE_MAGIC */
    uint16_t layout;       /* AOTX_WIRE_LAYOUT */
    uint16_t header_bytes; /* AOTX_HEADER_BYTES */
    uint64_t boot_id;      /* identifies the run that wrote the record */
    uint64_t tick;         /* device time */
    uint64_t seq;          /* position in the device ring, from 1, contiguous */
    uint64_t globaltimer;  /* device clock sample in nanoseconds; lag measurement only */
    uint32_t writer;       /* writer identity, stamped by the append */
    uint8_t  cls;          /* AOTX_CLASS_A or AOTX_CLASS_B */
    uint8_t  type;         /* AOTX_REC_* */
    uint16_t flags;        /* AOTX_FLAG_* */
    uint32_t body_len;     /* bytes of body that carry data, at most AOTX_BODY_BYTES */
    uint32_t reserved[3];  /* zero */
} aotx_record_header;

typedef struct aotx_boot_body {
    uint64_t boot_id;
    uint64_t previous_boot_id;  /* zero on a cold start */
    uint64_t wall_ns;           /* CLOCK_REALTIME at boot, from the host glue */
} aotx_boot_body;

typedef struct aotx_clock_body {
    uint64_t wall_ns;           /* CLOCK_REALTIME when the feeder wrote the record */
} aotx_clock_body;

/* The last record of every tick. The state hash covers every class A record applied so far,
 * in order, so two runs that applied the same inputs carry the same hash. */
typedef struct aotx_commit_body {
    uint64_t state_hash;        /* FNV-1a 64 over applied class A bodies, in order */
    uint64_t applied_count;     /* class A records applied since boot, replayed ones included */
    uint64_t inbound_consumed;  /* inbound slots consumed since boot */
    uint64_t records_this_tick; /* records in the block that ends with this record */
} aotx_commit_body;

typedef struct aotx_stall_body {
    uint64_t host_ring_free;    /* bytes free in the host ring when the tick was held */
    uint64_t held_count;        /* ticks held since boot, this one included */
} aotx_stall_body;

typedef struct aotx_stats_body {
    uint64_t tick_ns;           /* device time the tick took */
    uint64_t records;           /* records written this tick */
    uint64_t inbound;           /* inbound slots consumed this tick */
} aotx_stats_body;

/* One key event from the window. The codes are GLFW codes; the window glue writes this
 * struct, 16 bytes, to the feeder's key pipe and the feeder publishes it unchanged. */
typedef struct aotx_key_body {
    uint32_t key;               /* GLFW key code, or 0 for a character event */
    uint32_t codepoint;         /* Unicode code point for a character event, or 0 */
    uint32_t action;            /* 1 press, 0 release, 2 repeat */
    uint32_t mods;              /* GLFW modifier bits */
} aotx_key_body;

/* Bus message kinds and provenance values, after the bus schema the drain emits. */
#define AOTX_BUS_FINDING       1u
#define AOTX_BUS_RANK          2u
#define AOTX_BUS_QUESTION      3u
#define AOTX_BUS_ANSWER        4u
#define AOTX_BUS_HANDOFF       5u
#define AOTX_BUS_COST          6u
#define AOTX_BUS_NOTE          7u
#define AOTX_PROV_COMPUTED     1u
#define AOTX_PROV_FETCHED      2u
#define AOTX_PROV_RECALLED     3u
#define AOTX_PROV_TESTIMONY    4u
#define AOTX_BUS_TEXT_BYTES    (AOTX_BODY_BYTES - 32u)

/* One bus message. A finding carries a provenance value of 1 to 4; every other kind carries
 * zero. The field re_seq names the record a rank or an answer refers to. The field
 * corrects_seq names the record a correction replaces. The writer's own count is writer_seq. */
typedef struct aotx_bus_body {
    uint8_t  kind;              /* AOTX_BUS_* */
    uint8_t  provenance;        /* AOTX_PROV_* for a finding, else 0 */
    uint16_t reserved0;
    uint32_t writer_seq;        /* per-writer sequence, from 1 */
    uint64_t re_seq;            /* record seq this message refers to, or 0 */
    uint64_t corrects_seq;      /* record seq this message corrects, or 0 */
    float    score;             /* rank score in [0, 1]; 0 for other kinds */
    uint32_t text_len;          /* bytes of text that carry data */
    char     text[AOTX_BUS_TEXT_BYTES];
} aotx_bus_body;

/* A bulk payload lives in a bulk block on the bulk ring, not in a record. */
typedef struct aotx_bulk_body {
    uint64_t handle;            /* the payload count, equal to the bulk block's first_seq */
    uint64_t length;            /* payload bytes */
    uint32_t kind;              /* what the payload is; 1 for a text export */
    uint32_t reserved;
} aotx_bulk_body;

/* One token of a sequence: a prompt token or a sampled one. A sampled token carries the seed
 * and the draw that made it, so a restore applies the token and never samples again. */
#define AOTX_TOKEN_PROMPT      0x0001u
#define AOTX_TOKEN_SAMPLED     0x0002u
#define AOTX_TOKEN_LAST        0x0004u   /* the sequence ends with this token */

typedef struct aotx_token_body {
    uint32_t slot;              /* the sequence slot, equal to the key value cache slot */
    uint32_t token;             /* the token id */
    uint32_t position;          /* the token's position in the sequence, from 0 */
    uint32_t flags;             /* AOTX_TOKEN_* */
    uint64_t seed;              /* the random stream's seed; 0 for a prompt token */
    uint64_t draw;              /* the draw count of the slot's stream at this token */
    uint32_t role;              /* the model role that made the token */
    uint32_t reserved;
} aotx_token_body;

/* A sequence event: open, done, stopped, released. Derived; never replayed. */
#define AOTX_SEQ_OPENED        1u
#define AOTX_SEQ_DONE          2u
#define AOTX_SEQ_STOPPED       3u
#define AOTX_SEQ_RELEASED      4u

typedef struct aotx_sequence_body {
    uint32_t slot;
    uint32_t event;             /* AOTX_SEQ_* */
    uint32_t prompt_tokens;
    uint32_t sampled_tokens;
    uint64_t ticks;             /* ticks from open to this event */
    uint32_t role;
    uint32_t reserved;
} aotx_sequence_body;

typedef struct aotx_restore_body {
    uint64_t restored_boot_id;  /* the journal that was replayed */
    uint64_t last_tick;         /* the last complete tick that was applied */
    uint64_t replayed_count;    /* class A records replayed */
    uint64_t state_hash;        /* the hash after replay */
} aotx_restore_body;

/* The host ring: a byte ring that receives one block for each tick. A block holds the tick's
 * records. A block never wraps. A pad block fills the tail of the data area when the next block
 * does not fit there, and that block starts at offset zero. The device writes head; the drain
 * writes cursor. cursor is the only field the consumer writes. */
#define AOTX_BLOCK_MAGIC       0x4B4C4241u   /* "ABLK" */
#define AOTX_BLOCK_HEADER_BYTES 64u
#define AOTX_BLOCK_PAD         1u
#define AOTX_BLOCK_BULK        2u   /* a bulk payload: header, then length bytes of payload */

/* The bulk ring uses the same preamble and block header. A bulk block's byte_len is the
 * header plus the payload rounded up to 8 bytes; record_count is 0; first_seq is the handle. */
typedef struct aotx_block_header {
    uint32_t magic;          /* AOTX_BLOCK_MAGIC */
    uint16_t layout;         /* AOTX_WIRE_LAYOUT */
    uint16_t kind;           /* 0 for records, AOTX_BLOCK_PAD for a pad block */
    uint64_t block_seq;      /* publish field: zero while under write; from 1 */
    uint64_t boot_id;
    uint64_t tick;
    uint64_t first_seq;      /* seq of the first record in the block */
    uint32_t record_count;
    uint32_t byte_len;       /* bytes of the block including this header */
    uint64_t reserved[2];    /* zero */
} aotx_block_header;

typedef struct aotx_host_ring_preamble {
    uint32_t magic;          /* AOTX_WIRE_MAGIC */
    uint16_t layout;         /* AOTX_WIRE_LAYOUT */
    uint16_t closed;         /* the producer sets 1 when it ends */
    uint64_t boot_id;
    uint64_t data_bytes;     /* size of the data area, a power of two */
    uint64_t preamble_bytes; /* bytes to the data area */
    uint8_t  pad0[AOTX_LINE_BYTES - 32];
    uint64_t head;           /* own line; producer: next write offset, monotonic, not masked */
    uint8_t  pad1[AOTX_LINE_BYTES - 8];
    uint64_t cursor;         /* own line; consumer: bytes drained to disk, monotonic */
    uint8_t  pad2[AOTX_LINE_BYTES - 8];
    uint64_t last_block_seq; /* own line; producer: the last published block sequence */
    uint8_t  pad3[AOTX_LINE_BYTES - 8];
} aotx_host_ring_preamble;

/* The inbound ring: fixed slots of AOTX_SLOT_BYTES. The feeder writes a slot, then
 * release-stores head. The device reads slots below head after an acquire load, then
 * release-stores consumed. consumed is the only field the device writes here. */
typedef struct aotx_inbound_preamble {
    uint32_t magic;          /* AOTX_WIRE_MAGIC */
    uint16_t layout;         /* AOTX_WIRE_LAYOUT */
    uint16_t closed;
    uint64_t slot_count;     /* a power of two */
    uint64_t preamble_bytes;
    uint8_t  pad0[AOTX_LINE_BYTES - 24];
    uint64_t head;           /* own line; feeder: slots published, monotonic */
    uint8_t  pad1[AOTX_LINE_BYTES - 8];
    uint64_t consumed;       /* own line; device: slots consumed, monotonic */
    uint8_t  pad2[AOTX_LINE_BYTES - 8];
} aotx_inbound_preamble;

/* The journal segment on disk: a sequence of frames, one for each block. */
typedef struct aotx_segment_frame {
    uint32_t byte_len;       /* bytes of the block that follow */
    uint32_t crc32c;         /* CRC-32C of the block bytes */
} aotx_segment_frame;

/* Sizes are fixed by this header; a mismatch is a build error on both sides. */
typedef char aotx_wire_check_record[(sizeof(aotx_record_header) == AOTX_HEADER_BYTES) ? 1 : -1];
typedef char aotx_wire_check_block[(sizeof(aotx_block_header) == AOTX_BLOCK_HEADER_BYTES) ? 1 : -1];
typedef char aotx_wire_check_host[(sizeof(aotx_host_ring_preamble) == 4 * AOTX_LINE_BYTES) ? 1 : -1];
typedef char aotx_wire_check_inbound[(sizeof(aotx_inbound_preamble) == 3 * AOTX_LINE_BYTES) ? 1 : -1];
typedef char aotx_wire_check_key[(sizeof(aotx_key_body) == 16) ? 1 : -1];
typedef char aotx_wire_check_bus[(sizeof(aotx_bus_body) == AOTX_BODY_BYTES) ? 1 : -1];
typedef char aotx_wire_check_token[(sizeof(aotx_token_body) == 40) ? 1 : -1];

#endif
