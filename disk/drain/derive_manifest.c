/* Purpose: Derive the requests file and the chain of turns from the blocks the drain reads.
 * Owns: The open requests file, the open chain file, and the table of pending requests.
 * Threading: One thread; the drain calls these functions in block order.
 * Lifetime: From the first line that each file takes to the close of the drain. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/drain/derive.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* The first line of a chain has no line before it, so its digest field is 64 zeros. */
#define AOTX_CHAIN_FIRST "0000000000000000000000000000000000000000000000000000000000000000"

#define AOTX_CHAIN_LINE 2048
#define AOTX_CHAIN_READ 4096

/* Gives the name of the state that ended a turn. */
static const char *finish_name(uint32_t finish)
{
    static const char *names[3] = { "stop", "tool", "limit" };
    return (finish <= 2u) ? names[finish] : "other";
}

/* Writes the digest of one line, with the end byte of the line in it. */
static void line_digest(const char *line, size_t bytes, char *out)
{
    unsigned char digest[AOTX_SHA256_DIGEST];
    aotx_sha256 state;
    aotx_sha256_init(&state);
    aotx_sha256_update(&state, line, bytes);
    aotx_sha256_final(&state, digest);
    aotx_sha256_text(digest, out);
}

/* Reads back the chain that the file already holds and takes the digest of its last line.
 * A second run of the drain on one boot must go on from the line that the file ends with.
 * A file that ends with no end byte gets one. The line that a crash cut then stays a line
 * of its own, and the reader of the chain reports the break at it. Returns 0 or -1. */
static int seed_chain(aotx_derive *d, const char *path)
{
    unsigned char buffer[AOTX_CHAIN_READ];
    aotx_sha256 state;
    unsigned char digest[AOTX_SHA256_DIGEST];
    int fd;
    int part = 0;
    ssize_t got;
    snprintf(d->prev_line, sizeof(d->prev_line), "%s", AOTX_CHAIN_FIRST);
    fd = open(path, O_RDONLY);
    if (fd < 0) {
        return 0;
    }
    aotx_sha256_init(&state);
    while ((got = read(fd, buffer, sizeof(buffer))) > 0) {
        unsigned char *at = buffer;
        size_t left = (size_t)got;
        while (left > 0) {
            const unsigned char *end = (const unsigned char *)memchr(at, '\n', left);
            size_t take = (end != NULL) ? (size_t)(end - at) + 1u : left;
            aotx_sha256_update(&state, at, take);
            part = (end == NULL);
            if (end != NULL) {
                aotx_sha256_final(&state, digest);
                aotx_sha256_text(digest, d->prev_line);
                aotx_sha256_init(&state);
            }
            at += take;
            left -= take;
        }
    }
    close(fd);
    if (got < 0) {
        return -1;
    }
    if (part) {
        /* The end byte of the line that the file lost goes in now, and the digest of that
         * line holds it. */
        aotx_sha256_update(&state, "\n", 1);
        aotx_sha256_final(&state, digest);
        aotx_sha256_text(digest, d->prev_line);
        fd = open(path, O_WRONLY | O_APPEND);
        if (fd < 0 || aotx_derive_put(fd, "\n", 1) != 0) {
            if (fd >= 0) {
                close(fd);
            }
            return -1;
        }
        close(fd);
    }
    return 0;
}

/* Opens the chain of this boot at the first turn. The file is one for each boot, and a run
 * with no turn writes none. Returns 0 or -1. */
static int open_chain(aotx_derive *d)
{
    char dir[AOTX_PATH_BYTES + 16];
    char path[AOTX_PATH_BYTES + 48];
    if (d->manifest_fd >= 0) {
        return 0;
    }
    snprintf(dir, sizeof(dir), "%s/manifest", d->journal_dir);
    if (aotx_make_dir(dir) != 0) {
        return -1;
    }
    snprintf(path, sizeof(path), "%s/%s.jsonl", dir, d->boot_name);
    if (seed_chain(d, path) != 0) {
        return -1;
    }
    d->manifest_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    return (d->manifest_fd >= 0) ? 0 : -1;
}

/* Opens the requests file at the first line it takes. The file is one for the journal,
 * because the feeder tails one file over the boots of that journal. Returns 0 or -1. */
static int open_requests(aotx_derive *d)
{
    char path[AOTX_PATH_BYTES + 32];
    if (d->requests_fd >= 0) {
        return 0;
    }
    snprintf(path, sizeof(path), "%s/requests.jsonl", d->journal_dir);
    d->requests_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    return (d->requests_fd >= 0) ? 0 : -1;
}

/* Keeps one request that waits for the operator. The table is direct mapped by the
 * identity, so a new request takes the place of an older one with the same low bits. */
static void hold(aotx_derive *d, const aotx_tool_request_body *r)
{
    aotx_pending *slot;
    if (d->pending == NULL || r->request == 0) {
        return;
    }
    slot = &d->pending[r->request & (AOTX_PENDING_SLOTS - 1u)];
    slot->request = r->request;
    slot->body = *r;
}

/* Gives back the request that the table holds under an identity, or null. */
static const aotx_pending *held(aotx_derive *d, uint32_t request)
{
    aotx_pending *slot;
    if (d->pending == NULL || request == 0) {
        return NULL;
    }
    slot = &d->pending[request & (AOTX_PENDING_SLOTS - 1u)];
    return (slot->request == request) ? slot : NULL;
}

/* Takes the request out of the table, because the operator answered it. */
static void drop(aotx_derive *d, uint32_t request)
{
    aotx_pending *slot;
    if (d->pending == NULL || request == 0) {
        return;
    }
    slot = &d->pending[request & (AOTX_PENDING_SLOTS - 1u)];
    if (slot->request == request) {
        slot->request = 0;
    }
}

/* Writes one line of the requests file. The feeder reads this file and no other, so the
 * line carries every field that an execution needs. The deadline field is the deadline the
 * device holds. The tick field is the tick that deadline counts from.
 *
 * That is the tick of the request for a tool which needs no authorization. It is the tick
 * of the grant for a tool which needs one.
 *
 * The side field states where the tool runs and the number field states the tool number of
 * the record. A tool of the catalog has no name here: the feeder holds the name, the
 * directory and the program under that number. Returns 0 or -1. */
static int put_request(aotx_derive *d, uint64_t tick, const aotx_tool_request_body *r,
                       uint32_t auth)
{
    char line[AOTX_CHAIN_LINE];
    size_t used = aotx_request_line(line, sizeof(line), r, tick, auth);
    if (used == 0) {
        d->refused++;
        return 0;
    }
    if (open_requests(d) != 0) {
        return -1;
    }
    d->requests++;
    d->chain_open = 1;
    return aotx_derive_put(d->requests_fd, line, used);
}

int aotx_derive_request(aotx_derive *d, const aotx_record_header *h, const unsigned char *body)
{
    aotx_tool_request_body r;
    const aotx_pending *from;
    if (h->body_len < sizeof(r) - AOTX_TOOL_ARG_BYTES) {
        /* A body that is shorter than the fixed fields names no request. */
        d->refused++;
        return 0;
    }
    memset(&r, 0, sizeof(r));
    memcpy(&r, body, (h->body_len < sizeof(r)) ? h->body_len : sizeof(r));
    if (r.request == 0) {
        d->refused++;
        return 0;
    }
    /* A record the device wrote while a replay ran names a request that the journal
     * already answers. Or it names one the device presents again when the replay ends. A
     * line from it would make the feeder execute the tool a second time. The table still
     * keeps a request that waits, so the record which grants it later finds its fields. */
    if ((h->flags & AOTX_FLAG_REPLAY) != 0 && r.auth != AOTX_AUTH_PENDING
        && r.auth != AOTX_AUTH_REFUSED) {
        d->replayed++;
        return 0;
    }
    if (r.auth == AOTX_AUTH_NONE) {
        return put_request(d, h->tick, &r, AOTX_AUTH_NONE);
    }
    if (r.auth == AOTX_AUTH_PENDING) {
        /* The line waits for the record that grants the request. A feeder that took the
         * line now would execute a tool that the operator did not authorize. */
        hold(d, &r);
        return 0;
    }
    if (r.auth == AOTX_AUTH_REFUSED) {
        drop(d, r.request);
        return 0;
    }
    if (r.auth != AOTX_AUTH_GRANTED) {
        d->refused++;
        return 0;
    }
    from = held(d, r.request);
    if (from != NULL) {
        /* The tool, the agent, the turn and the path come from the request. The deadline
         * comes from the record that grants it. A request which waits for the operator has
         * no deadline and takes one at the grant. The tick is the tick the operator
         * answered at, which is the tick that deadline counts from. */
        aotx_tool_request_body granted = from->body;
        granted.deadline = r.deadline;
        drop(d, r.request);
        return put_request(d, h->tick, &granted, AOTX_AUTH_GRANTED);
    }
    /* The table lost the request, so the record that grants it must carry the fields. A
     * record that carries none names no file to read. */
    d->unheld++;
    if (r.arg_len == 0) {
        d->refused++;
        return 0;
    }
    return put_request(d, h->tick, &r, AOTX_AUTH_GRANTED);
}

int aotx_derive_turn(aotx_derive *d, const aotx_record_header *h, const unsigned char *body)
{
    aotx_manifest_body m;
    char line[AOTX_CHAIN_LINE];
    int used;
    if (h->body_len < sizeof(m)) {
        /* A body that is shorter than the layout holds no hash, so the line would prove
         * nothing and the chain would carry it. */
        d->refused++;
        return 0;
    }
    memcpy(&m, body, sizeof(m));
    if (open_chain(d) != 0) {
        return -1;
    }
    used = snprintf(line, sizeof(line),
                    "{\"agent\":%u,\"turn\":%u,\"input_hash\":\"%016llx\","
                    "\"output_hash\":\"%016llx\",\"tokens\":%u,\"finish\":\"%s\","
                    "\"tool\":\"%s\",\"request\":%u,\"prev\":\"%s\"}\n",
                    m.agent, m.turn, (unsigned long long)m.input_hash,
                    (unsigned long long)m.output_hash, m.output_tokens, finish_name(m.finish),
                    aotx_tool_name(m.tool), m.request, d->prev_line);
    if (used < 0 || (size_t)used >= sizeof(line)) {
        d->refused++;
        return 0;
    }
    if (aotx_derive_put(d->manifest_fd, line, (size_t)used) != 0) {
        return -1;
    }
    line_digest(line, (size_t)used, d->prev_line);
    d->turns++;
    d->chain_open = 1;
    return 0;
}

int aotx_derive_chain_sync(aotx_derive *d)
{
    if (!d->chain_open) {
        return 0;
    }
    d->chain_open = 0;
    if (d->requests_fd >= 0 && fsync(d->requests_fd) != 0) {
        return -1;
    }
    if (d->manifest_fd >= 0 && fsync(d->manifest_fd) != 0) {
        return -1;
    }
    return 0;
}

void aotx_derive_chain_close(aotx_derive *d)
{
    aotx_derive_chain_sync(d);
    if (d->requests_fd >= 0) {
        close(d->requests_fd);
    }
    if (d->manifest_fd >= 0) {
        close(d->manifest_fd);
    }
    d->requests_fd = -1;
    d->manifest_fd = -1;
    free(d->pending);
    d->pending = NULL;
}
