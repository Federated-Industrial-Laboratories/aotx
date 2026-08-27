/* Purpose: Write and read typed messages over records.
 * Owns: The message body layouts and the sequence counter for each writer.
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

#endif
