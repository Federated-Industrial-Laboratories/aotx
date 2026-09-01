/* Purpose: Declare the derived page statistics stream.
 * Owns: Nothing; the open call allocates the private state.
 * Threading: One thread; records are taken in journal order.
 * Lifetime: One drain run. */
#ifndef AOTX_DRAIN_PAGE_STATS_H
#define AOTX_DRAIN_PAGE_STATS_H

#include "disk/wire/diskwire.h"

typedef struct aotx_page_stats aotx_page_stats;

int aotx_page_stats_open(aotx_page_stats **out, const char *boot_dir);
int aotx_page_stats_record(aotx_page_stats *state, const aotx_record_header *header);
int aotx_page_stats_sync(aotx_page_stats *state);
void aotx_page_stats_close(aotx_page_stats *state);
uint64_t aotx_page_stats_lines(const aotx_page_stats *state);
uint64_t aotx_page_stats_refused(const aotx_page_stats *state);

#endif
