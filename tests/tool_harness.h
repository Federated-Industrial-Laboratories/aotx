/* Purpose: Give the host tool checks one feeder, one root and one table of replies.
 * Owns: The temporary tree, the inbound ring and the assembled replies of one case.
 * Threading: Two processes; the check reads the ring while the feeder writes it.
 * Lifetime: The run of one case. */
#ifndef AOTX_TESTS_TOOL_HARNESS_H
#define AOTX_TESTS_TOOL_HARNESS_H

#include "disk/feed/fs_tool.h"
#include "tests/disk_fake.h"

#include <fcntl.h>
#include <unistd.h>

#define AOTX_TH_SLOTS     256u
#define AOTX_TH_WAIT_NS   30000000000ull
#define AOTX_TH_REPLY_MAX 200
#define AOTX_TH_ARG_TEXT  1024

typedef struct th_reply {
    uint32_t request;
    uint32_t agent;
    uint32_t status;   /* the status of the last part that came */
    uint32_t parts;    /* the count of parts the reply names */
    uint32_t got;      /* the parts that came */
    uint32_t content;  /* the parts that carry content */
    uint32_t len;      /* the content bytes assembled */
    char reason[AOTX_TOOL_REPLY_BYTES + 1];
    char bytes[AOTX_FS_CAP + 1];
} th_reply;

typedef struct th_replies {
    int count;
    th_reply at[AOTX_TH_REPLY_MAX];
} th_replies;

/* The table holds one whole reply for each request, so it lives beside the program and not
 * on the stack. One case runs at a time, and each case clears it at its start. */
static th_replies th_collected;

typedef struct th_run {
    aotx_map map;
    aotx_inbound_ring ring;
    int child;
    int requests_fd;
    char dir[256];       /* the temporary directory of the case */
    char root[320];      /* the allowed root under it */
    char requests[400];
} th_run;

/* Gives the entry of one request, and makes it when the table does not hold it. */
static th_reply *th_entry(uint32_t request)
{
    th_replies *r = &th_collected;
    int i;
    for (i = 0; i < r->count; i++) {
        if (r->at[i].request == request) {
            return &r->at[i];
        }
    }
    if (r->count >= AOTX_TH_REPLY_MAX) {
        return NULL;
    }
    memset(&r->at[r->count], 0, sizeof(r->at[0]));
    r->at[r->count].request = request;
    return &r->at[r->count++];
}

/* Consumes the slots the feeder published and assembles the replies. A part with the status
 * ok carries content. A part with any other status is the last part of the reply and its
 * bytes are the reason. */
static void th_collect(th_run *t)
{
    aotx_inbound_ring *ring = &t->ring;
    uint64_t consumed = aotx_inbound_consumed(ring);
    uint64_t head = aotx_inbound_head(ring);
    while (consumed < head) {
        const unsigned char *slot = ring->slots + (consumed & ring->mask) * AOTX_SLOT_BYTES;
        const aotx_record_header *h = (const aotx_record_header *)slot;
        if (h->type == AOTX_REC_TOOL_REPLY && h->body_len >= sizeof(aotx_tool_reply_body)) {
            aotx_tool_reply_body body;
            th_reply *e;
            uint32_t take;
            memcpy(&body, aotx_record_body(h), sizeof(body));
            CHECK(h->cls == AOTX_CLASS_A, "a reply record is not authoritative");
            e = th_entry(body.request);
            take = (body.len > AOTX_TOOL_REPLY_BYTES) ? AOTX_TOOL_REPLY_BYTES : body.len;
            if (e != NULL) {
                e->agent = body.agent;
                e->parts = body.parts;
                e->status = body.status;
                e->got++;
                if (body.status == AOTX_TOOL_OK) {
                    if (e->len + take <= AOTX_FS_CAP) {
                        memcpy(e->bytes + e->len, body.bytes, take);
                        e->len += take;
                        e->bytes[e->len] = '\0';
                    }
                    e->content++;
                } else {
                    memcpy(e->reason, body.bytes, take);
                    e->reason[take] = '\0';
                }
            }
        }
        consumed++;
        aotx_store_release(&ring->pre->consumed, consumed);
    }
}

/* Waits until the count of complete replies reaches the figure, or the time runs out. */
static void th_wait(th_run *t, int want)
{
    uint64_t deadline = aotx_wall_ns() + AOTX_TH_WAIT_NS;
    for (;;) {
        uint64_t backoff = 0;
        int done = 0;
        int i;
        th_collect(t);
        for (i = 0; i < th_collected.count; i++) {
            if (th_collected.at[i].parts > 0 &&
                th_collected.at[i].got == th_collected.at[i].parts) {
                done++;
            }
        }
        if (done >= want || aotx_wall_ns() >= deadline) {
            return;
        }
        aotx_pause(&backoff);
    }
}

/* Waits for the first record the feeder publishes. The feeder opens the requests file
 * before it maps the ring, so a record proves that the open is done. */
static void th_wait_start(th_run *t)
{
    uint64_t deadline = aotx_wall_ns() + AOTX_TH_WAIT_NS;
    while (aotx_inbound_head(&t->ring) == 0 && aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        aotx_pause(&backoff);
    }
}

/* Makes the temporary tree of one case: the directory, the allowed root and the name of
 * the requests file. The requests file itself is made after the feeder starts, as the
 * drain makes it at the first request of a run. */
static void th_tree(th_run *t)
{
    memset(t, 0, sizeof(*t));
    t->requests_fd = -1;
    CHECK(aotx_temp_dir(t->dir, sizeof(t->dir)) == 0, "the temporary directory does not open");
    snprintf(t->root, sizeof(t->root), "%s/root", t->dir);
    CHECK(aotx_make_dir(t->root) == 0, "the root does not open");
    snprintf(t->requests, sizeof(t->requests), "%s/requests.jsonl", t->dir);
}

/* Opens a ring and starts one feeder over the tree. The modules directory and the timeout
 * are left out when they are null. A second call starts a second feeder over the same
 * tree, which is what a feeder that follows a crash does. */
static void th_spawn(th_run *t, char *feeder, const char *timeout, const char *modules,
                     int input_fd)
{
    char fd_text[16];
    char *args[12];
    int at = 0;
    memset(&th_collected, 0, sizeof(th_collected));
    CHECK(aotx_inbound_create(AOTX_TH_SLOTS, &t->map, &t->ring) == 0, "the ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", t->map.fd);
    args[at++] = feeder;
    args[at++] = (char *)"--inbound-fd";
    args[at++] = fd_text;
    args[at++] = (char *)"--root";
    args[at++] = t->root;
    args[at++] = (char *)"--requests";
    args[at++] = t->requests;
    if (timeout != NULL) {
        args[at++] = (char *)"--timeout";
        args[at++] = (char *)timeout;
    }
    if (modules != NULL) {
        args[at++] = (char *)"--modules";
        args[at++] = (char *)modules;
    }
    args[at] = NULL;
    t->child = aotx_spawn(args, input_fd, -1);
    CHECK(t->child > 0, "the feeder does not start");
    th_wait_start(t);
    t->requests_fd = open(t->requests, O_WRONLY | O_CREAT | O_APPEND, 0644);
    CHECK(t->requests_fd >= 0, "the requests file does not open");
}

/* Closes the ring and waits for the feeder. The tree stays. */
static void th_close(th_run *t)
{
    if (t->requests_fd >= 0) {
        close(t->requests_fd);
        t->requests_fd = -1;
    }
    aotx_store_release16(&t->ring.pre->closed, 1);
    CHECK(aotx_wait(t->child) == 0, "the feeder does not end with a clean status");
    aotx_map_release(&t->map);
}

/* Closes the ring, waits for the feeder and takes the tree away. */
static void th_stop(th_run *t)
{
    th_close(t);
    aotx_remove_tree(t->dir);
}

/* Writes the argument text of a call with several keys, in the shape the device writes
 * it. A separator byte comes before each pair. An equal sign comes between the key and
 * the value. The text goes into the line with the escape the drain writes. */
static void th_args(char *out, size_t bytes, const char *const *keys,
                    const char *const *values, int count)
{
    size_t used = 0;
    int i;
    out[0] = '\0';
    for (i = 0; i < count; i++) {
        int n = snprintf(out + used, bytes - used, "\\u001f%s=%s", keys[i], values[i]);
        if (n < 0 || (size_t)n >= bytes - used) {
            return;
        }
        used += (size_t)n;
    }
}

/* Gives the argument text of a call with one key and one value. */
static void th_one_arg(char *out, size_t bytes, const char *key, const char *value)
{
    const char *keys[1];
    const char *values[1];
    keys[0] = key;
    values[0] = value;
    th_args(out, bytes, keys, values, 1);
}

/* Writes one line of the requests file, in the shape the drain writes it. The argument
 * text goes in as it stands, so a caller can give an escape of its own. */
static void th_request(th_run *t, uint32_t request, uint32_t agent, const char *tool,
                       uint32_t number, const char *arg)
{
    char line[AOTX_TH_ARG_TEXT + 256];
    int n = snprintf(line, sizeof(line),
                     "{\"request\":%u,\"agent\":%u,\"turn\":1,\"tool\":\"%s\","
                     "\"side\":\"%s\",\"number\":%u,\"arg\":\"%s\","
                     "\"deadline\":500,\"auth\":\"none\",\"tick\":7}\n",
                     request, agent, tool,
                     (number >= AOTX_TOOL_MODULE_BASE) ? "module" : "host", number, arg);
    CHECK(n > 0 && (size_t)n < sizeof(line), "the request line does not fit");
    CHECK(write(t->requests_fd, line, (size_t)n) == n, "the request line does not write");
}

/* Writes one file of the fixture tree, under the root or beside it. */
static void th_file(const char *path, const void *bytes, size_t len)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    CHECK(fd >= 0, "the fixture file %s does not open", path);
    if (fd >= 0) {
        CHECK(write(fd, bytes, len) == (ssize_t)len, "the fixture file %s does not write",
              path);
        close(fd);
    }
}

/* Reads a whole file of the fixture tree back. Returns the byte count, or -1. */
static long th_read(const char *path, char *out, size_t bytes)
{
    ssize_t n;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }
    n = read(fd, out, bytes - 1u);
    close(fd);
    if (n < 0) {
        return -1;
    }
    out[n] = '\0';
    return (long)n;
}

#endif
