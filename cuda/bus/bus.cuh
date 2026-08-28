/* Purpose: Write and read typed messages over records.
 * Owns: The message body layouts, the sequence counter of each writer, the buffer of the
 *       last messages that the panel reads.
 * Launch shape: One thread for each message.
 * Lifetime: The whole run. */
#ifndef BUS_CUH
#define BUS_CUH

#include "seam/wire.h"

/* The sequence table holds one entry for each system writer and one for each agent slot.
 * A writer identity at or above AOTX_BUS_WRITER_MAX has no entry and is refused. */
#define AOTX_BUS_AGENTS      64u
#define AOTX_BUS_WRITER_MAX  (AOTX_WRITER_AGENT_BASE + AOTX_BUS_AGENTS)

typedef struct aotx_bus_state {
    unsigned int writer_seq[AOTX_BUS_WRITER_MAX]; /* messages each writer wrote, from zero */
    unsigned long long appended;                  /* messages that went in the ring */
    unsigned long long refused;                   /* messages the writer or provenance rule
                                                   * refused; nothing was written */
} aotx_bus_state;

extern __device__ aotx_bus_state aotx_bus;

/* Append one bus message as a BUS record. The writer identity is stamped from the argument,
 * never from the text. A finding with provenance 0 is refused and the function returns 0.
 * Returns the record sequence of the message. */
__device__ unsigned long long aotx_bus_append(unsigned int writer, unsigned int kind,
                                              unsigned int provenance, const char *text,
                                              unsigned int length, unsigned long long re_seq,
                                              unsigned long long corrects_seq, float score,
                                              unsigned long long tick);

/* Fill seqs with the most recent BUS records whose kind is in the mask (bit kind set),
 * newest first, at most max entries. Returns the count filled. */
__device__ unsigned int aotx_bus_recent(unsigned int kind_mask, unsigned int max,
                                        unsigned long long *seqs);

/* Give the body of the BUS record with the given sequence, or 0 when it is not in the ring. */
__device__ const aotx_bus_body *aotx_bus_body_of(unsigned long long seq);

/* The message buffer. The bus panel reads its rows from here and not from the record ring.
 * A tick load of thousands of records writes over a bus record in a fraction of a second.
 * A message that an agent put on the bus must stay on the panel. The buffer is device
 * state of the run that makes the messages; the disk holds the records, not the buffer.
 * The buffer holds more messages than the panel shows, so a full panel needs no walk. */
#define AOTX_BUS_LINES   64u

/* One message of the buffer. */
typedef struct aotx_bus_line {
    unsigned long long at;      /* the message number; zero while a writer fills the line */
    unsigned long long seq;     /* the record sequence of the same message */
    unsigned int writer;        /* the writer identity that the append stamped */
    unsigned int text_len;      /* bytes of the text, up to AOTX_BUS_TEXT_BYTES */
    unsigned char kind;         /* AOTX_BUS_* */
    unsigned char provenance;   /* AOTX_PROV_* for a finding, else 0 */
    unsigned char reserved[2];
    char text[AOTX_BUS_TEXT_BYTES];
} aotx_bus_line;

typedef struct aotx_bus_buffer {
    unsigned long long count;              /* messages put in since the start of the run */
    aotx_bus_line line[AOTX_BUS_LINES];
} aotx_bus_buffer;

extern __device__ aotx_bus_buffer aotx_bus_lines;

/* The count of lines is a power of two, so a message number gives its place with a mask. */
typedef char aotx_bus_line_check[((AOTX_BUS_LINES & (AOTX_BUS_LINES - 1u)) == 0u) ? 1 : -1];

/* Give the line of a message number, or a null pointer when the buffer no longer holds it.
 * The reader must look at the number again after it copies the bytes. */
__device__ __forceinline__ const volatile aotx_bus_line *aotx_bus_line_at(
    unsigned long long at)
{
    const volatile aotx_bus_line *line =
        &aotx_bus_lines.line[(at - 1ull) & (AOTX_BUS_LINES - 1u)];
    return (line->at == at) ? line : 0;
}

#endif
