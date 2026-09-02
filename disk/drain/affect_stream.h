/* Purpose: Declare the derived affect stream of trace lines and state lines.
 * Owns: Nothing; the open call allocates the private state.
 * Threading: One thread; records are taken in journal order.
 * Lifetime: One drain run. */
#ifndef AOTX_DRAIN_AFFECT_STREAM_H
#define AOTX_DRAIN_AFFECT_STREAM_H

#include "disk/wire/diskwire.h"

typedef struct aotx_affect_stream aotx_affect_stream;

int aotx_affect_stream_open(aotx_affect_stream **out, const char *boot_dir);
int aotx_affect_stream_record(aotx_affect_stream *state,
                              const aotx_record_header *header);
int aotx_affect_stream_sync(aotx_affect_stream *state);
void aotx_affect_stream_close(aotx_affect_stream *state);
uint64_t aotx_affect_stream_lines(const aotx_affect_stream *state);
uint64_t aotx_affect_stream_refused(const aotx_affect_stream *state);

#endif
