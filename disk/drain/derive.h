/* Purpose: Declare the derived files that the drain writes beside the journal segments.
 * Owns: Nothing; the caller holds the state structure.
 * Threading: One thread; the drain calls these functions in block order.
 * Lifetime: From open to close, which is the run of the drain. */
#ifndef AOTX_DISK_DERIVE_H
#define AOTX_DISK_DERIVE_H

#include "disk/wire/diskwire.h"

typedef struct aotx_transcript aotx_transcript;
typedef struct aotx_token_stats aotx_token_stats;
typedef struct aotx_page_stats aotx_page_stats;
#ifdef AOTX_AFFECT
typedef struct aotx_affect_stream aotx_affect_stream;
typedef struct aotx_quality_stream aotx_quality_stream;
#endif

/* The record types that the drain turns into lines. A type that the mask leaves out still
 * reaches the journal, so a type with a high rate costs the drain no line. */
#define AOTX_DERIVE_CONSOLE  1u
#define AOTX_DERIVE_NOTE     2u
#define AOTX_DERIVE_BUS      4u
#define AOTX_DERIVE_BULK     8u
#define AOTX_DERIVE_SEQUENCE 16u
#define AOTX_DERIVE_REQUESTS 32u
#define AOTX_DERIVE_TRANSCRIPT 64u
#define AOTX_DERIVE_TOKENS   128u
#define AOTX_DERIVE_PAGES    256u
#ifdef AOTX_AFFECT
#define AOTX_DERIVE_AFFECT   512u
#define AOTX_DERIVE_QUALITY  1024u
#define AOTX_DERIVE_ALL      2047u
#else
#define AOTX_DERIVE_ALL      511u
#endif

/* The manifest chain is not in the mask. A turn that makes no line makes a gap in the
 * chain, and a chain with a gap proves nothing. */

/* Requests that wait for the operator. The table holds the newest ones, so the memory of
 * the drain has a limit. A request that falls out of the table takes the fields of its
 * line from the record that grants it. */
#define AOTX_PENDING_SLOTS 1024u

/* 64 hexadecimal characters and one end byte. */
#define AOTX_HEX_BYTES 65

/* Four system writers and the agents of one run. */
#define AOTX_AGENT_SLOTS   260
#define AOTX_NAME_MAX      24
#define AOTX_ID_MAX        (AOTX_NAME_MAX + 24)

/* The map from a record sequence to a message id holds the newest entries only, so the
 * memory of the drain has a limit. An older reference is not in the map. */
#define AOTX_REF_SLOTS     65536u

#define AOTX_TEXT_MAX      1280
#define AOTX_BUS_LINE_MAX  4096

/* A JSON escape gives at most six characters for one byte, so this is the largest text a
 * task body can become. */
#define AOTX_TASK_ESCAPED  (AOTX_TASK_TEXT_BYTES * 6 + 8)

/* One message that the drain wrote, so a later message can name it. */
typedef struct aotx_ref {
    uint64_t record_seq; /* the record that the message came from; zero for a free slot */
    uint64_t msg_seq;    /* the sequence of the message in the line file */
    uint32_t writer;     /* the writer of the message */
    uint8_t  rankable;   /* one when a rank can name the message: a finding or a handoff */
} aotx_ref;

/* One request that waits for the operator, held by its identity. The line of a granted
 * request takes the tool, the agent, the turn and the path from this body. It takes the
 * deadline and the tick from the record that grants it. */
typedef struct aotx_pending {
    uint32_t request;   /* the identity, or zero for a free slot */
    aotx_tool_request_body body;
} aotx_pending;

typedef struct aotx_derive {
    int console_fd;
    int bus_fd;
    int requests_fd;          /* the requests file, opened at the first line it takes */
    int manifest_fd;          /* the chain of this boot, opened at the first turn */
    int chain_open;           /* one when a chain file holds bytes that no synchronize took */
    int echo_fd;              /* the operator terminal, which sees every console record */
    unsigned mask;            /* the record types to derive */
    int line_open;            /* one when a console record left a line without its end byte */
    uint64_t next_seq[AOTX_AGENT_SLOTS]; /* the next free sequence of each writer */
    aotx_ref *refs;           /* AOTX_REF_SLOTS entries, or null when the map is off */
    uint64_t tick_start_ns;   /* the wall clock of the newest tick start record */
    uint64_t sync_ns;         /* the wall clock of the last synchronize call */
    uint64_t lines;           /* console records written */
    uint64_t notes;           /* note lines written */
    uint64_t sequences;       /* sequence end lines, which the note count holds too */
    uint64_t messages;        /* message lines written */
    uint64_t unresolved;      /* messages whose reference is not in the map */
    uint64_t refused;         /* messages that the line schema does not accept */
    uint64_t requests;        /* request lines written */
    uint64_t turns;           /* manifest lines written */
    uint64_t events;          /* task and agent lines written */
    uint64_t unheld;          /* granted requests that the pending table did not hold */
    uint64_t replayed;        /* request records written while a replay ran; no line */
    aotx_pending *pending;    /* AOTX_PENDING_SLOTS entries, or null when the mask is off */
    aotx_transcript *transcript; /* per-agent files, or null when not derived */
    aotx_token_stats *token_stats; /* tokens.jsonl, or null when not derived */
    aotx_page_stats *page_stats; /* pages.jsonl, or null when not derived */
#ifdef AOTX_AFFECT
    aotx_affect_stream *affect_stream; /* affect.jsonl, or null when not derived */
    aotx_quality_stream *quality_stream; /* quality.jsonl, or null when not derived */
#endif
    char prev_line[AOTX_HEX_BYTES]; /* the digest of the last line of the chain */
    char bus_dir[AOTX_PATH_BYTES];
    char journal_dir[AOTX_PATH_BYTES];
    char boot_name[24];       /* the boot identity as 16 hexadecimal characters */
    char bus_date[16];
} aotx_derive;

/* Reads a list of type names, such as "console,note,bus,bulk,sequence". The name "none"
 * gives an empty mask. Returns 0, or -1 when a name is not a type. */
int aotx_derive_mask(const char *list, unsigned *out);

/* Opens the console log in the boot directory and the line file in the journal. The derived
 * files are outputs only, and no program reads them back as inputs. */
int aotx_derive_open(aotx_derive *d, const char *journal, const char *boot_dir, unsigned mask);

/* Writes the derived lines of one block. Returns 0 or -1. */
int aotx_derive_block(aotx_derive *d, const unsigned char *block);

/* Synchronizes the derived files when more than one second passed, or when force is one. */
int aotx_derive_sync(aotx_derive *d, int force);

void aotx_derive_close(aotx_derive *d);

/* ---- parts that derive.c and derive_bus.c share ---- */

/* Writes every byte, or returns -1. */
int aotx_derive_put(int fd, const char *data, size_t bytes);

/* Gives the time in ISO 8601 and the day, and opens the file of a new day. Returns 0 or -1. */
int aotx_derive_stamp(aotx_derive *d, char *iso, size_t iso_bytes, uint64_t ns);

/* Gives the place of a writer in the sequence table, or -1 when the writer has no name. */
int aotx_derive_agent(uint32_t writer, char *name, size_t name_bytes);

/* Takes the next free sequence of a writer. The sequence of a writer must go up, so the
 * count of the writer applies only when the file does not hold that number. */
uint64_t aotx_derive_next(aotx_derive *d, int slot, uint64_t writer_seq);

/* Writes the fields that every derived line carries after the message body. */
int aotx_derive_tail(aotx_derive *d, char *out, size_t out_bytes, uint64_t tick,
                     uint64_t boot_id, uint64_t now);

/* Writes the lines of one message record. Returns 0 or -1. */
int aotx_derive_message(aotx_derive *d, const aotx_record_header *h, const unsigned char *body);

/* ---- the request path and the chain of turns (derive_manifest.c) ---- */

/* Takes one request record. A request that waits makes a pending note and stays in the
 * table. Its requests line waits for the record that grants it. A request that the
 * operator refuses makes no requests line. Returns 0 or -1. */
int aotx_derive_request(aotx_derive *d, const aotx_record_header *h, const unsigned char *body);

/* Writes the pending note for one request that entered the table. Returns 0 or -1. */
int aotx_derive_pending(aotx_derive *d, const aotx_record_header *h,
                        const aotx_tool_request_body *request);

/* Writes one line of the chain for one completed turn. Each line carries the digest of the
 * line before it, so a reader can prove that no line was removed. Returns 0 or -1. */
int aotx_derive_turn(aotx_derive *d, const aotx_record_header *h, const unsigned char *body);

/* Synchronizes the requests file and the chain. The drain calls this for each batch of
 * blocks that it takes, as it does for the segments. Returns 0 or -1. */
int aotx_derive_chain_sync(aotx_derive *d);

void aotx_derive_chain_close(aotx_derive *d);

#endif
