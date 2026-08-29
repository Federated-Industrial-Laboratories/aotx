/* Purpose: Declare the transcript files derived from journal records.
 * Owns: Nothing; the open call allocates the private state and the close call releases it.
 * Threading: One thread; records are taken in journal order.
 * Lifetime: One drain or one journal walk. */
#ifndef AOTX_DRAIN_TRANSCRIPT_H
#define AOTX_DRAIN_TRANSCRIPT_H

#include "disk/wire/diskwire.h"

typedef struct aotx_transcript aotx_transcript;

int aotx_transcript_open(aotx_transcript **out, const char *boot_dir);
int aotx_transcript_block(aotx_transcript *t, const unsigned char *block);
int aotx_transcript_sync(aotx_transcript *t);
void aotx_transcript_close(aotx_transcript *t);
uint64_t aotx_transcript_lines(const aotx_transcript *t);

#endif
