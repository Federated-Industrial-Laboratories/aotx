/* Purpose: Publish live memory file commands as bounded journal record batches.
 * Owns: Temporary input bytes; the device owns cognitive state and admission.
 * Threading: One feeder thread owns the inbound producer.
 * Lifetime: One complete file command. */
#ifndef AOTX_FEED_COGNITIVE_IO_H
#define AOTX_FEED_COGNITIVE_IO_H
#include "disk/feed/line.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Return 0 for another command, 1 for a handled command, or -1 for a closed ring. */
int aotx_live_feed_line(const unsigned char *line, uint32_t length,
                         const aotx_inbound_ring *ring, const volatile sig_atomic_t *stop);
#ifdef __cplusplus
}
#endif
#endif
