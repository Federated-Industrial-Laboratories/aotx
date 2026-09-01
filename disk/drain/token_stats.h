/* Purpose: Declare the derived token statistics stream.
 * Owns: Nothing; the open call allocates the private state.
 * Threading: One thread; records are taken in journal order.
 * Lifetime: One drain run. */
#ifndef AOTX_DRAIN_TOKEN_STATS_H
#define AOTX_DRAIN_TOKEN_STATS_H

#include "disk/wire/diskwire.h"

typedef struct aotx_token_stats aotx_token_stats;

int aotx_token_stats_open(aotx_token_stats **out, const char *boot_dir);
int aotx_token_stats_record(aotx_token_stats *state, const aotx_record_header *header);
int aotx_token_stats_sync(aotx_token_stats *state);
void aotx_token_stats_close(aotx_token_stats *state);
uint64_t aotx_token_stats_lines(const aotx_token_stats *state);
uint64_t aotx_token_stats_refused(const aotx_token_stats *state);

#endif
