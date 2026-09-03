/* Purpose: Derive one JSON line for each affect trace record and each affect state record.
 * Owns: The affect.jsonl descriptor and its counts.
 * Threading: One thread; records are taken in journal order.
 * Lifetime: One drain run. */
#include "disk/drain/affect_stream.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct aotx_affect_stream { int fd; uint64_t lines; uint64_t refused; };

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

static int finite_body(const aotx_affect_trace_body *body)
{
    unsigned int i;
    for (i = 0u; i < 4u; i++) {
        if (!isfinite(body->prompt[i]) || !isfinite(body->reply[i])) return 0;
    }
    return isfinite(body->guard[0]) && isfinite(body->guard[1])
        && isfinite(body->logprob) && isfinite(body->entropy)
        && isfinite(body->budget_spent) && isfinite(body->entropy_shift)
        && isfinite(body->class_shift);
}

static int add_reasons(char *line, size_t bytes, uint32_t mask)
{
    static const char *names[15] = {
        "stop", "limit", "operator_stop", "role_refused", "tool_ok",
        "tool_error", "tool_refused", "deadline", "task_done", "task_failed",
        "budget", "room_cut", "think_ratio", "low_logprob", "verdict_refute"
    };
    size_t used = 0u;
    unsigned int count = 0u;
    int got = snprintf(line, bytes, "[");
    if (got < 0 || (size_t)got >= bytes) return -1;
    used = (size_t)got;
    for (unsigned int i = 0u; i < 15u; i++) {
        if ((mask & (1u << i)) == 0u) continue;
        got = snprintf(line + used, bytes - used, "%s\"%s\"", count ? "," : "", names[i]);
        if (got < 0 || (size_t)got >= bytes - used) return -1;
        used += (size_t)got;
        count++;
    }
    got = snprintf(line + used, bytes - used, "]");
    return (got < 0 || (size_t)got >= bytes - used) ? -1 : (int)(used + (size_t)got);
}

int aotx_affect_stream_open(aotx_affect_stream **out, const char *boot_dir)
{
    char path[AOTX_PATH_BYTES];
    aotx_affect_stream *state = (aotx_affect_stream *)calloc(1u, sizeof(*state));
    int used = state != NULL ? snprintf(path, sizeof(path), "%s/affect.jsonl", boot_dir) : -1;
    if (state == NULL || used < 0 || (size_t)used >= sizeof(path)) { free(state); return -1; }
    state->fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (state->fd < 0) { free(state); return -1; }
    *out = state;
    return 0;
}

/* The state line of one affect state record. The two parts of the state are fractions of
 * 32768 and the scale is a fraction of 65535. The events are words. The replayed mark
 * marks a record that a restore applied again. */
static int state_line(aotx_affect_stream *state, const aotx_record_header *header)
{
    aotx_affect_body body;
    char reasons[256];
    char line[1024];
    int used;
    if (header->cls != AOTX_CLASS_A || header->body_len != sizeof(body)) {
        state->refused++; return 0;
    }
    memcpy(&body, aotx_record_body(header), sizeof(body));
    if (body.agent >= 64u || (body.reason & ~0x7fffu) != 0u || (body.flags & ~0x0fu) != 0u
        || add_reasons(reasons, sizeof(reasons), body.reason) < 0) {
        state->refused++; return 0;
    }
    used = snprintf(line, sizeof(line),
        "{\"tick\":%llu,\"agent\":%u,\"turn\":%u,\"kind\":\"state\","
        "\"fast\":[%.9g,%.9g,%.9g,%.9g],\"slow\":[%.9g,%.9g,%.9g,%.9g],"
        "\"scale\":%.9g,\"reason\":%s,\"replayed\":%d}\n",
        (unsigned long long)header->tick, body.agent, body.turn,
        (double)body.fast[0] / 32768.0, (double)body.fast[1] / 32768.0,
        (double)body.fast[2] / 32768.0, (double)body.fast[3] / 32768.0,
        (double)body.slow[0] / 32768.0, (double)body.slow[1] / 32768.0,
        (double)body.slow[2] / 32768.0, (double)body.slow[3] / 32768.0,
        (double)body.scale / 65535.0, reasons,
        ((header->flags & AOTX_FLAG_REPLAYED) != 0u) ? 1 : 0);
    if (used < 0 || (size_t)used >= sizeof(line)
        || put_all(state->fd, line, (size_t)used) != 0) return -1;
    state->lines++;
    return 0;
}

int aotx_affect_stream_record(aotx_affect_stream *state,
                              const aotx_record_header *header)
{
    aotx_affect_trace_body body;
    char reasons[256];
    char line[1024];
    int used;
    if (state == NULL) return 0;
    if (header->type == AOTX_REC_AFFECT) return state_line(state, header);
    if (header->cls != AOTX_CLASS_B || header->body_len != sizeof(body)) {
        state->refused++; return 0;
    }
    memcpy(&body, aotx_record_body(header), sizeof(body));
    if (body.agent >= 64u || !finite_body(&body) || body.entropy < 0.0f
        || body.budget_spent < 0.0f || body.class_shift < -1.0f || body.class_shift > 1.0f
        || (body.reason & ~0x7fffu) != 0u || (body.flags & ~0x0fu) != 0u
        || add_reasons(reasons, sizeof(reasons), body.reason) < 0) {
        state->refused++; return 0;
    }
    used = snprintf(line, sizeof(line),
        "{\"tick\":%llu,\"agent\":%u,\"turn\":%u,\"kind\":\"trace\","
        "\"prompt\":[%.9g,%.9g,%.9g,%.9g],\"reply\":[%.9g,%.9g,%.9g,%.9g],"
        "\"guard\":[%.9g,%.9g],\"logprob\":%.9g,\"entropy\":%.9g,"
        "\"rows\":%u,\"think\":%u,\"reason\":%s,"
        "\"effective\":[%.9g,%.9g,%.9g,%.9g],\"flags\":%u,"
        "\"budget_spent\":%.9g,\"entropy_shift\":%.9g,\"class_shift\":%.9g}\n",
        (unsigned long long)header->tick, body.agent, body.turn,
        (double)body.prompt[0], (double)body.prompt[1], (double)body.prompt[2],
        (double)body.prompt[3], (double)body.reply[0], (double)body.reply[1],
        (double)body.reply[2], (double)body.reply[3], (double)body.guard[0],
        (double)body.guard[1], (double)body.logprob, (double)body.entropy,
        body.rows, body.think, reasons, (double)body.effective[0] / 32768.0,
        (double)body.effective[1] / 32768.0, (double)body.effective[2] / 32768.0,
        (double)body.effective[3] / 32768.0, body.flags, (double)body.budget_spent,
        (double)body.entropy_shift, (double)body.class_shift);
    if (used < 0 || (size_t)used >= sizeof(line)
        || put_all(state->fd, line, (size_t)used) != 0) return -1;
    state->lines++;
    return 0;
}

int aotx_affect_stream_sync(aotx_affect_stream *state)
{ return state == NULL || state->fd < 0 || fsync(state->fd) == 0 ? 0 : -1; }

void aotx_affect_stream_close(aotx_affect_stream *state)
{
    if (state != NULL) { if (state->fd >= 0) { fsync(state->fd); close(state->fd); } free(state); }
}

uint64_t aotx_affect_stream_lines(const aotx_affect_stream *state)
{ return state != NULL ? state->lines : 0u; }

uint64_t aotx_affect_stream_refused(const aotx_affect_stream *state)
{ return state != NULL ? state->refused : 0u; }
