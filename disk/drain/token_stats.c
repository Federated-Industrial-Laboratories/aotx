/* Purpose: Derive one JSON line for each sampled-token statistics record.
 * Owns: The tokens.jsonl descriptor and its counts.
 * Threading: One thread; records are taken in journal order.
 * Lifetime: One drain run. */
#include "disk/drain/token_stats.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define AOTX_TOKEN_STATS_AGENTS 64u

struct aotx_token_stats {
    int fd;
    uint64_t lines;
    uint64_t refused;
};

static int put_all(int fd, const char *text, size_t bytes)
{
    size_t done = 0u;
    while (done < bytes) {
        ssize_t count = write(fd, text + done, bytes - done);
        if (count <= 0) {
            return -1;
        }
        done += (size_t)count;
    }
    return 0;
}

int aotx_token_stats_open(aotx_token_stats **out, const char *boot_dir)
{
    char path[AOTX_PATH_BYTES];
    aotx_token_stats *state = (aotx_token_stats *)calloc(1u, sizeof(*state));
    int used;
    if (state == NULL) {
        return -1;
    }
    used = snprintf(path, sizeof(path), "%s/tokens.jsonl", boot_dir);
    if (used < 0 || (size_t)used >= sizeof(path)) {
        free(state);
        return -1;
    }
    state->fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (state->fd < 0) {
        free(state);
        return -1;
    }
    *out = state;
    return 0;
}

int aotx_token_stats_record(aotx_token_stats *state, const aotx_record_header *header)
{
    aotx_token_stats_body body;
    char line[256];
    int used;
    if (state == NULL) {
        return 0;
    }
    if (header->cls != AOTX_CLASS_B || header->body_len != sizeof(body)) {
        state->refused++;
        return 0;
    }
    memcpy(&body, aotx_record_body(header), sizeof(body));
    if (body.agent >= AOTX_TOKEN_STATS_AGENTS
        || (body.flags & ~AOTX_TOKEN_STATS_THINK) != 0u
        || !isfinite(body.logprob) || !isfinite(body.entropy) || body.entropy < 0.0f) {
        state->refused++;
        return 0;
    }
    used = snprintf(line, sizeof(line),
                    "{\"tick\":%llu,\"agent\":%u,\"turn\":%u,\"index\":%u,"
                    "\"logprob\":%.9g,\"entropy\":%.9g,\"think\":%s}\n",
                    (unsigned long long)header->tick, body.agent, body.turn, body.index,
                    (double)body.logprob, (double)body.entropy,
                    (body.flags & AOTX_TOKEN_STATS_THINK) != 0u ? "true" : "false");
    if (used < 0 || (size_t)used >= sizeof(line)
        || put_all(state->fd, line, (size_t)used) != 0) {
        return -1;
    }
    state->lines++;
    return 0;
}

int aotx_token_stats_sync(aotx_token_stats *state)
{
    return (state == NULL || state->fd < 0 || fsync(state->fd) == 0) ? 0 : -1;
}

void aotx_token_stats_close(aotx_token_stats *state)
{
    if (state == NULL) {
        return;
    }
    if (state->fd >= 0) {
        fsync(state->fd);
        close(state->fd);
    }
    free(state);
}

uint64_t aotx_token_stats_lines(const aotx_token_stats *state)
{
    return (state != NULL) ? state->lines : 0u;
}

uint64_t aotx_token_stats_refused(const aotx_token_stats *state)
{
    return (state != NULL) ? state->refused : 0u;
}
