/* Purpose: Derive tool status only from typed device policy records.
 * Owns: The tools.jsonl descriptor and its accepted and refused counts.
 * Threading: One disk thread validates and writes records in block order.
 * Lifetime: One drain run; the output belongs to one boot directory. */
#include "disk/drain/tool_policy.h"
#include "cuda/tool/policy.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct aotx_tool_policy_stream { int fd; uint64_t lines; uint64_t refused; };

int aotx_tool_policy_stream_open(aotx_tool_policy_stream **out, const char *boot_dir)
{
    char path[AOTX_PATH_BYTES];
    aotx_tool_policy_stream *state = calloc(1u, sizeof(*state));
    int used = state != NULL ? snprintf(path, sizeof(path), "%s/tools.jsonl", boot_dir) : -1;
    if (state == NULL || used < 0 || (size_t)used >= sizeof(path)) { free(state); return -1; }
    state->fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (state->fd < 0) { free(state); return -1; }
    *out = state;
    return 0;
}

int aotx_tool_policy_stream_record(aotx_tool_policy_stream *state, const aotx_record_header *header)
{
    aotx_tool_policy_body body;
    char line[256];
    uint32_t selected = 0u;
    if (state == NULL) return 0;
    if (header->magic != AOTX_WIRE_MAGIC || header->layout != AOTX_WIRE_LAYOUT
        || header->header_bytes != AOTX_HEADER_BYTES || header->type != AOTX_REC_TOOL_POLICY
        || header->cls != AOTX_CLASS_B || header->writer != AOTX_WRITER_SYSTEM
        || (header->flags & ~AOTX_FLAG_REPLAY) != 0u || header->seq == 0u
        || header->body_len != sizeof(body)) { ++state->refused; return 0; }
    memcpy(&body, aotx_record_body(header), sizeof(body));
    if (body.agent >= 256u || body.defaults > AOTX_TOOL_POLICY_ALL
        || body.choices > AOTX_TOOL_POLICY_CHOICES || body.selected > AOTX_TOOL_POLICY_ALL
        || body.effective > AOTX_TOOL_POLICY_ALL || (body.effective & ~body.selected) != 0u) {
        ++state->refused; return 0;
    }
    for (uint32_t group = 0u; group < AOTX_TOOL_POLICY_GROUPS; ++group) {
        uint32_t value = (body.choices >> (2u * group)) & 3u;
        if (value == 3u) { ++state->refused; return 0; }
        if (value == AOTX_TOOL_POLICY_ON || (value == AOTX_TOOL_POLICY_INHERIT
            && (body.defaults & (1u << group)) != 0u)) selected |= 1u << group;
    }
    if (body.selected != selected) { ++state->refused; return 0; }
    int used = snprintf(line, sizeof(line),
        "{\"tick\":%llu,\"seq\":%llu,\"agent\":%u,\"defaults\":%u,\"choices\":%u,"
        "\"selected\":%u,\"effective\":%u}\n",
        (unsigned long long)header->tick, (unsigned long long)header->seq,
        body.agent, body.defaults, body.choices, body.selected, body.effective);
    if (used < 0 || (size_t)used >= sizeof(line)) return -1;
    size_t done = 0u;
    while (done < (size_t)used) {
        ssize_t bytes = write(state->fd, line + done, (size_t)used - done);
        if (bytes <= 0) return -1;
        done += (size_t)bytes;
    }
    ++state->lines;
    return 0;
}

int aotx_tool_policy_stream_sync(aotx_tool_policy_stream *state)
{ return state == NULL || state->fd < 0 || fsync(state->fd) == 0 ? 0 : -1; }

void aotx_tool_policy_stream_close(aotx_tool_policy_stream *state)
{
    if (state != NULL) { if (state->fd >= 0) { fsync(state->fd); close(state->fd); } free(state); }
}

uint64_t aotx_tool_policy_stream_lines(const aotx_tool_policy_stream *state)
{ return state != NULL ? state->lines : 0u; }
uint64_t aotx_tool_policy_stream_refused(const aotx_tool_policy_stream *state)
{ return state != NULL ? state->refused : 0u; }
