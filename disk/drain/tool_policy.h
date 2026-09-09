/* Purpose: Declare the typed tool policy display stream.
 * Owns: Nothing; the open call allocates the private file state.
 * Threading: One disk thread consumes records in block order.
 * Lifetime: One drain run. */
#ifndef AOTX_DRAIN_TOOL_POLICY_H
#define AOTX_DRAIN_TOOL_POLICY_H

#include "disk/wire/diskwire.h"
typedef struct aotx_tool_policy_stream aotx_tool_policy_stream;
int aotx_tool_policy_stream_open(aotx_tool_policy_stream **out, const char *boot_dir);
int aotx_tool_policy_stream_record(aotx_tool_policy_stream *state, const aotx_record_header *header);
int aotx_tool_policy_stream_sync(aotx_tool_policy_stream *state);
void aotx_tool_policy_stream_close(aotx_tool_policy_stream *state);
uint64_t aotx_tool_policy_stream_lines(const aotx_tool_policy_stream *state);
uint64_t aotx_tool_policy_stream_refused(const aotx_tool_policy_stream *state);
#endif
