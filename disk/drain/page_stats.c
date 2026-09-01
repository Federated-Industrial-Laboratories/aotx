/* Purpose: Derive one JSON line for each attention page statistics record.
 * Owns: The pages.jsonl descriptor and its counts.
 * Threading: One thread; records are taken in journal order.
 * Lifetime: One drain run. */
#include "disk/drain/page_stats.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct aotx_page_stats { int fd; uint64_t lines; uint64_t refused; };

static int put_all(int fd, const char *text, size_t bytes)
{
    size_t done = 0u;
    while (done < bytes) {
        ssize_t count = write(fd, text + done, bytes - done);
        if (count <= 0) return -1;
        done += (size_t)count;
    }
    return 0;
}

int aotx_page_stats_open(aotx_page_stats **out, const char *boot_dir)
{
    char path[AOTX_PATH_BYTES];
    aotx_page_stats *state = (aotx_page_stats *)calloc(1u, sizeof(*state));
    int used = state != NULL ? snprintf(path, sizeof(path), "%s/pages.jsonl", boot_dir) : -1;
    if (state == NULL || used < 0 || (size_t)used >= sizeof(path)) { free(state); return -1; }
    state->fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (state->fd < 0) { free(state); return -1; }
    *out = state;
    return 0;
}

int aotx_page_stats_record(aotx_page_stats *state, const aotx_record_header *header)
{
    aotx_page_stats_body body;
    char line[256];
    if (state == NULL) return 0;
    if (header->cls != AOTX_CLASS_B || header->body_len != sizeof(body)) {
        state->refused++; return 0;
    }
    memcpy(&body, aotx_record_body(header), sizeof(body));
    if (body.agent >= 64u || body.slots == 0u || body.slots > 4096u
        || body.page >= body.slots || body.residency > 1u
        || body.cadence != 64u
        || !isfinite(body.mass) || body.mass < 0.0f) { state->refused++; return 0; }
    int used = snprintf(line, sizeof(line),
        "{\"tick\":%llu,\"agent\":%u,\"page\":%u,\"residency\":%u,\"slots\":%u,"
        "\"mass\":%.9g}\n",
        (unsigned long long)header->tick, body.agent, body.page, body.residency,
        body.slots, (double)body.mass);
    if (used < 0 || (size_t)used >= sizeof(line)
        || put_all(state->fd, line, (size_t)used) != 0) return -1;
    state->lines++;
    return 0;
}

int aotx_page_stats_sync(aotx_page_stats *state)
{ return state == NULL || state->fd < 0 || fsync(state->fd) == 0 ? 0 : -1; }

void aotx_page_stats_close(aotx_page_stats *state)
{
    if (state != NULL) { if (state->fd >= 0) { fsync(state->fd); close(state->fd); } free(state); }
}

uint64_t aotx_page_stats_lines(const aotx_page_stats *state)
{ return state != NULL ? state->lines : 0u; }

uint64_t aotx_page_stats_refused(const aotx_page_stats *state)
{ return state != NULL ? state->refused : 0u; }
