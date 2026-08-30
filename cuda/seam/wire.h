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
#define AOTX_REC_TOOL_REQUEST  16u  /* class B; body: aotx_tool_request_body */
#define AOTX_REC_TOOL_REPLY    17u  /* class A; body: aotx_tool_reply_body, from the feeder */
#define AOTX_REC_MANIFEST      18u  /* class B; body: aotx_manifest_body, one turn of an agent */
#define AOTX_REC_TASK          19u  /* class B; body: aotx_task_body, a task event */
#define AOTX_REC_AGENT         20u  /* class B; body: aotx_agent_body, an agent event */
#define AOTX_REC_SETTING       21u  /* class A; body: aotx_setting_body, one setting */
#define AOTX_REC_CARD          22u  /* class B; body: aotx_card_body, the card and the build */
#define AOTX_REC_IMPORT        23u  /* class A; body: aotx_import_head or aotx_import_part */
#define AOTX_REC_REMOVE        24u  /* class A; body: aotx_remove_body, one module leaves */
#define AOTX_REC_SELECTION     25u  /* class A; body: aotx_selection_body, the turns a prompt took */

/* Record flags. */
#define AOTX_FLAG_REPLAYED     0x0001u  /* the record was applied again at restore */
#define AOTX_FLAG_FRAGMENT     0x0002u  /* the record continues the line of the one before */
#define AOTX_FLAG_REPLAY       0x0004u  /* the device wrote the record while a replay ran */

/* A long input line crosses the inbound ring as an INPUT_LINE and then parts. Each part
 * carries AOTX_FLAG_FRAGMENT and at most AOTX_BODY_BYTES; this is the most parts. */
#define AOTX_LINE_PARTS_MAX    32u

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
    uint32_t source_seq[2]; /* low and high words of the source sequence in a replay ring */
    uint32_t reserved;     /* zero */
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
    uint64_t line_holds;        /* input lines held at an apply boundary */
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
    uint32_t text_len;          /* reply bytes of this token, or zero for a prompt token */
    char     text[AOTX_BODY_BYTES - 40u]; /* detokenized reply bytes */
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

/* Tools. A device tool runs inside the tick; a host tool is a request the feeder answers. */
#define AOTX_TOOL_NONE          0u
#define AOTX_TOOL_MEMORY_RECALL 1u   /* device: the nearest findings to a text */
#define AOTX_TOOL_MEMORY_WRITE  2u   /* device: a finding with provenance and a vector */
#define AOTX_TOOL_FS_READ       3u   /* host: bytes of a file under the allowed root */
#define AOTX_TOOL_FS_LIST       4u   /* host: the entries of a directory under the root */
#define AOTX_TOOL_FS_WRITE      5u   /* host: write a file under the root; the operator permits it */
#define AOTX_TOOL_FS_UPDATE     6u   /* host: replace one text run in a file; the operator permits it */
#define AOTX_TOOL_RUN           7u   /* host: a command line under the root; the operator permits it */
#define AOTX_TOOL_SKILL_USE     8u   /* device: the body of a skill of the catalog */
#define AOTX_TOOL_IMPORT        9u   /* host: the feeder reads a module directory */
#define AOTX_TOOL_MODULE_BASE   16u  /* a tool of the catalog: this plus the import number */

/* The agent of a request that no agent made. The console makes such a request when the
 * operator types a line in a surface the feeder does not read. */
#define AOTX_REQUEST_NO_AGENT   0xffffffffu
#define AOTX_TOOL_ARG_BYTES     (AOTX_BODY_BYTES - 32u)

/* Authorization of a request. A tool that needs it waits for the operator. */
#define AOTX_AUTH_NONE          0u   /* the tool needs no authorization */
#define AOTX_AUTH_PENDING       1u
#define AOTX_AUTH_GRANTED       2u
#define AOTX_AUTH_REFUSED       3u

/* A host tool request. Derived from the agent's reply; the drain hands it to the feeder. */
typedef struct aotx_tool_request_body {
    uint32_t agent;
    uint32_t turn;
    uint32_t tool;              /* AOTX_TOOL_* */
    uint32_t request;           /* the request id, unique in the run */
    uint64_t deadline;          /* the tick after which the request fails */
    uint32_t auth;              /* AOTX_AUTH_* */
    uint32_t arg_len;
    char     arg[AOTX_TOOL_ARG_BYTES];
} aotx_tool_request_body;

/* The reply to a host tool request, in parts of AOTX_TOOL_REPLY_BYTES. Class A: a restore
 * applies the recorded reply and the feeder executes nothing again. */
#define AOTX_TOOL_OK            0u
#define AOTX_TOOL_ERROR         1u
#define AOTX_TOOL_REFUSED       2u
#define AOTX_TOOL_LATE          3u
#define AOTX_TOOL_REPLY_BYTES   (AOTX_BODY_BYTES - 24u)

typedef struct aotx_tool_reply_body {
    uint32_t agent;
    uint32_t request;
    uint32_t status;            /* AOTX_TOOL_* */
    uint32_t part;              /* from 0 */
    uint32_t parts;
    uint32_t len;
    char     bytes[AOTX_TOOL_REPLY_BYTES];
} aotx_tool_reply_body;

/* One completed turn of an agent: the prompt is hashed, the reply is in the sequence. */
#define AOTX_TURN_STOP          0u   /* the reply ended at the stop token */
#define AOTX_TURN_TOOL          1u   /* the reply ended in a tool call */
#define AOTX_TURN_LIMIT         2u   /* the reply reached its limit */

typedef struct aotx_manifest_body {
    uint32_t agent;
    uint32_t turn;
    uint64_t input_hash;        /* FNV-1a 64 over the prompt bytes */
    uint64_t output_hash;       /* FNV-1a 64 over the reply bytes */
    uint32_t output_tokens;
    uint32_t finish;            /* AOTX_TURN_* */
    uint32_t tool;              /* the tool called, or 0 */
    uint32_t request;           /* the request made, or 0 */
} aotx_manifest_body;

/* Task states and events. */
#define AOTX_TASK_PENDING       0u
#define AOTX_TASK_ASSIGNED      1u
#define AOTX_TASK_RUNNING       2u
#define AOTX_TASK_VERIFYING     3u
#define AOTX_TASK_DONE          4u
#define AOTX_TASK_FAILED        5u
#define AOTX_VERIFY_NONE        0u
#define AOTX_VERIFY_SIBLING     1u
#define AOTX_TASK_TEXT_BYTES    (AOTX_BODY_BYTES - 32u)

typedef struct aotx_task_body {
    uint32_t task;
    uint32_t agent;             /* the assignee, or the agent the task was given to */
    uint32_t state;             /* AOTX_TASK_* after the event */
    uint32_t verify;            /* AOTX_VERIFY_* */
    uint32_t attempts;
    uint32_t text_len;
    uint64_t ticks;             /* ticks since the task opened */
    char     text[AOTX_TASK_TEXT_BYTES];  /* the task text, or the result's first bytes */
} aotx_task_body;

/* Agent events. */
#define AOTX_AGENT_SPAWNED      1u
#define AOTX_AGENT_TURN         2u   /* a turn began */
#define AOTX_AGENT_RELEASED     3u

typedef struct aotx_agent_body {
    uint32_t agent;
    uint32_t role;
    uint32_t parent;            /* the agent that made it, or the agent itself for a root */
    uint32_t state;
    uint32_t event;             /* AOTX_AGENT_* */
    uint32_t turn;
    uint64_t ticks;             /* ticks since the agent spawned */
} aotx_agent_body;

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
/* One setting. From the feeder at a fresh boot (the keys the file names) or from the
 * device (a set line). Class A: a restore replays it, so a restored run holds the
 * settings of the run it restores and reads no file. */
#define AOTX_SETTING_WIRE_KEY_BYTES 64u
typedef struct aotx_setting_body {
    int64_t  value;             /* the value times scale */
    uint32_t scale;             /* 1, or 10000 for a value with four decimals */
    uint32_t key_len;           /* bytes of key that carry data */
    char     key[AOTX_SETTING_WIRE_KEY_BYTES];
} aotx_setting_body;

/* The card and the build, written once by the device after the BOOT record. Class B. */
#define AOTX_CARD_NAME_BYTES    64u
#define AOTX_CARD_PROFILE_BYTES 16u
typedef struct aotx_card_body {
    char     name[AOTX_CARD_NAME_BYTES];       /* the device name, end byte included */
    uint64_t memory_total;      /* bytes on the card */
    uint64_t memory_free;       /* bytes free at boot, before the weights */
    uint32_t compute_major;
    uint32_t compute_minor;
    char     profile[AOTX_CARD_PROFILE_BYTES]; /* the profile of the build, end byte included */
    uint32_t arch;              /* the architecture of the build, 86 for sm_86 */
    uint32_t slots;             /* AOTX_SLOTS of the build */
} aotx_card_body;

/* A module goes into the catalog as one import: a head part, then the parts of its
 * files in order. Every part is class A, so a restore rebuilds the catalog from the
 * journal and reads no file. The feeder numbers each import; the parts of one import may
 * stand between other inbound records, so every part names its import. */
#define AOTX_MODULE_SKILL        1u
#define AOTX_MODULE_ROLE         2u
#define AOTX_MODULE_TOOL         3u
#define AOTX_IMPORT_FILES        2u   /* the manifest, then the body */
#define AOTX_IMPORT_NAME_BYTES   64u
#define AOTX_IMPORT_PATH_BYTES   64u  /* the tail of the directory path, for the console */
#define AOTX_IMPORT_TEXT_BYTES   (AOTX_BODY_BYTES - 20u)
typedef struct aotx_import_head {
    uint32_t import;            /* the number of this import in the feeder's run */
    uint32_t part;              /* 0: this is the head */
    uint32_t kind;              /* AOTX_MODULE_* */
    uint32_t files;             /* files that follow: 1 (a manifest) or 2 (and a body) */
    uint32_t file_bytes[AOTX_IMPORT_FILES];  /* bytes of each file */
    uint8_t  digest[32];        /* SHA-256 of a tool's module file; zero for the other kinds */
    char     name[AOTX_IMPORT_NAME_BYTES];   /* the directory name, end byte included */
    char     path[AOTX_IMPORT_PATH_BYTES];   /* the tail of the path, end byte included */
} aotx_import_head;
typedef struct aotx_import_part {
    uint32_t import;
    uint32_t part;              /* from 1, in order */
    uint32_t file;              /* 0 the manifest, 1 the body */
    uint32_t offset;            /* byte offset of this part in its file */
    uint32_t length;            /* bytes of text that carry data */
    char     text[AOTX_IMPORT_TEXT_BYTES];
} aotx_import_part;
typedef struct aotx_remove_body {
    char     name[AOTX_IMPORT_NAME_BYTES];
} aotx_remove_body;

/* The turns the prompt builder put into a prompt. It holds the record sequences of the
 * warm turns it recalled and of the summary, in order. Class A, written by the agent:
 * a replay applies the recorded choice and searches nothing. */
#define AOTX_SELECTION_MAX      20u
typedef struct aotx_selection_body {
    uint32_t agent;
    uint32_t turn;
    uint32_t count;             /* sequences that carry data */
    uint32_t pages;             /* the hot bound the turn took */
    uint64_t summary_seq;       /* the record sequence of the summary, or zero */
    uint64_t seq[AOTX_SELECTION_MAX];
    uint64_t current_seq;       /* this selection record sequence */
} aotx_selection_body;

typedef char aotx_wire_check_record[(sizeof(aotx_record_header) == AOTX_HEADER_BYTES) ? 1 : -1];
typedef char aotx_wire_check_block[(sizeof(aotx_block_header) == AOTX_BLOCK_HEADER_BYTES) ? 1 : -1];
typedef char aotx_wire_check_host[(sizeof(aotx_host_ring_preamble) == 4 * AOTX_LINE_BYTES) ? 1 : -1];
typedef char aotx_wire_check_inbound[(sizeof(aotx_inbound_preamble) == 3 * AOTX_LINE_BYTES) ? 1 : -1];
typedef char aotx_wire_check_key[(sizeof(aotx_key_body) == 16) ? 1 : -1];
typedef char aotx_wire_check_bus[(sizeof(aotx_bus_body) == AOTX_BODY_BYTES) ? 1 : -1];
typedef char aotx_wire_check_token[(sizeof(aotx_token_body) == AOTX_BODY_BYTES) ? 1 : -1];
typedef char aotx_wire_check_request[(sizeof(aotx_tool_request_body) == AOTX_BODY_BYTES) ? 1 : -1];
typedef char aotx_wire_check_reply[(sizeof(aotx_tool_reply_body) == AOTX_BODY_BYTES) ? 1 : -1];
typedef char aotx_wire_check_task[(sizeof(aotx_task_body) == AOTX_BODY_BYTES) ? 1 : -1];
typedef char aotx_wire_check_setting[(sizeof(aotx_setting_body) == 80) ? 1 : -1];
typedef char aotx_wire_check_card[(sizeof(aotx_card_body) == 112) ? 1 : -1];
typedef char aotx_wire_check_import_head[(sizeof(aotx_import_head) == 184) ? 1 : -1];
typedef char aotx_wire_check_import_part[(sizeof(aotx_import_part) == AOTX_BODY_BYTES) ? 1 : -1];
typedef char aotx_wire_check_remove[(sizeof(aotx_remove_body) == 64) ? 1 : -1];
typedef char aotx_wire_check_selection[(sizeof(aotx_selection_body) == 192) ? 1 : -1];

#endif
