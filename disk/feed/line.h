/* Purpose: Declare the publication of one complete input line in contiguous parts.
 * Owns: Nothing; the caller owns the bytes and the inbound ring.
 * Threading: One feeder thread; the call publishes all parts before it returns.
 * Lifetime: One input line. */
#ifndef AOTX_FEED_LINE_H
#define AOTX_FEED_LINE_H

#include <signal.h>
#include <stdint.h>

#include "disk/wire/diskwire.h"

#define AOTX_INPUT_LINE_BYTES (AOTX_BODY_BYTES * AOTX_LINE_PARTS_MAX)
#define AOTX_INPUT_LINE_REASON "the line is longer than the input bound"

/* Publishes one INPUT_LINE head and its fragment parts. Returns 0, 1 when the line is
 * over the bound, or -1 when the ring closed or the stop flag was set. */
int aotx_line_publish(const aotx_inbound_ring *ring, const volatile sig_atomic_t *stop,
                      const unsigned char *bytes, uint32_t length);

/* Publishes one record group with one head advance. */
int aotx_line_publish_records(const aotx_inbound_ring *ring,
                              const volatile sig_atomic_t *stop,
                              const aotx_record_header *headers,
                              const void *const *bodies, uint32_t count);

#endif
