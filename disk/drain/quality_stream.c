/* Purpose: Derive one JSON line for each conversation quality record.
 * Owns: The quality.jsonl descriptor and its counts.
 * Threading: One thread; records are taken in journal order.
 * Lifetime: One drain run. */
#include "disk/drain/quality_stream.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct aotx_quality_stream { int fd; uint64_t lines; uint64_t refused; };

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

int aotx_quality_stream_open(aotx_quality_stream **out, const char *boot_dir)
{
    char path[AOTX_PATH_BYTES];
    aotx_quality_stream *state = (aotx_quality_stream *)calloc(1u, sizeof(*state));
    int used = state != NULL ? snprintf(path, sizeof(path), "%s/quality.jsonl", boot_dir) : -1;
    if (state == NULL || used < 0 || (size_t)used >= sizeof(path)) { free(state); return -1; }
    state->fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (state->fd < 0) { free(state); return -1; }
    *out = state;
    return 0;
}

int aotx_quality_stream_record(aotx_quality_stream *state,
                               const aotx_record_header *header)
{
    aotx_quality_body body;
    char prompt[32];
    char turn[32];
    char line[512];
    int used;
    if (state == NULL) return 0;
    if (header->cls != AOTX_CLASS_B || header->body_len != sizeof(body)) {
        state->refused++; return 0;
    }
    memcpy(&body, aotx_record_body(header), sizeof(body));
    if (body.agent >= 64u || !isfinite(body.coherence_prompt)
        || !isfinite(body.coherence_turn) || !isfinite(body.repetition)
        || !isfinite(body.guard[0]) || !isfinite(body.guard[1])
        || body.repetition < 0.0f || body.repetition > 1.0f
        || ((body.flags & 1u) != 0u && (body.coherence_prompt < -1.0f
                                      || body.coherence_prompt > 1.0f))
        || ((body.flags & 2u) != 0u && (body.coherence_turn < -1.0f
                                      || body.coherence_turn > 1.0f))
        || ((body.flags & 1u) == 0u && body.coherence_prompt != 0.0f)
        || ((body.flags & 2u) == 0u && body.coherence_turn != 0.0f)
        || body.tokens > body.limit || body.refusal > 1u
        || (body.flags & ~0x0fu) != 0u || body.reserved != 0u) {
        state->refused++; return 0;
    }
    if ((body.flags & 1u) != 0u) snprintf(prompt, sizeof(prompt), "%.9g", (double)body.coherence_prompt);
    else snprintf(prompt, sizeof(prompt), "null");
    if ((body.flags & 2u) != 0u) snprintf(turn, sizeof(turn), "%.9g", (double)body.coherence_turn);
    else snprintf(turn, sizeof(turn), "null");
    used = snprintf(line, sizeof(line),
        "{\"tick\":%llu,\"agent\":%u,\"turn\":%u,\"coherence_prompt\":%s,"
        "\"coherence_turn\":%s,\"repetition\":%.9g,\"tokens\":%u,\"limit\":%u,"
        "\"limit_hit\":%u,\"refusal\":%u,\"guard\":[%.9g,%.9g],\"flags\":%u}\n",
        (unsigned long long)header->tick, body.agent, body.turn, prompt, turn,
        (double)body.repetition, body.tokens, body.limit,
        (body.flags & 4u) != 0u ? 1u : 0u, body.refusal,
        (double)body.guard[0], (double)body.guard[1], body.flags);
    if (used < 0 || (size_t)used >= sizeof(line)
        || put_all(state->fd, line, (size_t)used) != 0) return -1;
    state->lines++;
    return 0;
}

int aotx_quality_stream_sync(aotx_quality_stream *state)
{ return state == NULL || state->fd < 0 || fsync(state->fd) == 0 ? 0 : -1; }

void aotx_quality_stream_close(aotx_quality_stream *state)
{
    if (state != NULL) { if (state->fd >= 0) { fsync(state->fd); close(state->fd); } free(state); }
}

uint64_t aotx_quality_stream_lines(const aotx_quality_stream *state)
{ return state != NULL ? state->lines : 0u; }

uint64_t aotx_quality_stream_refused(const aotx_quality_stream *state)
{ return state != NULL ? state->refused : 0u; }
