/* Purpose: Derive typed request audit rows from complete journal transfers.
 * Owns: Nothing; the reader allocates and releases private transfer buffers.
 * Threading: One disk reader in record order; one pending query batch.
 * Lifetime: One journal walk or drain. */
#ifndef AOTX_DRAIN_TRANSCRIPT_LIVE_H
#define AOTX_DRAIN_TRANSCRIPT_LIVE_H
#include "disk/wire/diskwire.h"
typedef struct aotx_transcript_live aotx_transcript_live;
typedef int (*aotx_live_audit_row)(void *context, uint32_t agent, uint64_t tick,
    const unsigned char *input, uint32_t length, const char *description,
    uint32_t description_length, uint32_t status);
int aotx_transcript_live_take(aotx_transcript_live **state, const aotx_record_header *h,
    aotx_live_audit_row emit, void *context);
void aotx_transcript_live_close(aotx_transcript_live *state);
#endif
