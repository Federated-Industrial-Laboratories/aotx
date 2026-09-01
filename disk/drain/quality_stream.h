/* Purpose: Declare the derived conversation quality stream.
 * Owns: Nothing; the open call allocates the private state.
 * Threading: One thread; records are taken in journal order.
 * Lifetime: One drain run. */
#ifndef AOTX_DRAIN_QUALITY_STREAM_H
#define AOTX_DRAIN_QUALITY_STREAM_H

#include "disk/wire/diskwire.h"

typedef struct aotx_quality_stream aotx_quality_stream;

int aotx_quality_stream_open(aotx_quality_stream **out, const char *boot_dir);
int aotx_quality_stream_record(aotx_quality_stream *state,
                               const aotx_record_header *header);
int aotx_quality_stream_sync(aotx_quality_stream *state);
void aotx_quality_stream_close(aotx_quality_stream *state);
uint64_t aotx_quality_stream_lines(const aotx_quality_stream *state);
uint64_t aotx_quality_stream_refused(const aotx_quality_stream *state);

#endif
