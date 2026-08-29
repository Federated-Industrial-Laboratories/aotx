/* Purpose: Run the feeder against two pipes and check the records that reach the ring.
 * Owns: One inbound ring, one line pipe and one key pipe for each case.
 * Threading: Two processes; the test reads the ring while the feeder writes it.
 * Lifetime: The run of the program. */
#include "disk/feed/fs_tool.h"
#include "disk/settings/settings.h"
#include "tests/disk_fake.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#define AOTX_RING_SLOTS      64u
#define AOTX_WAIT_NS    15000000000ull
#define AOTX_PART_NS    200000000ull
#define AOTX_LINES_MAX  200

static char **arguments;

/* The journal reader and the line validator, when the run gives them. */
static const char *reader_program;
static const char *lint_program;

typedef struct taken {
    int lines;
    int clocks;
    int keys;
    char text[AOTX_LINES_MAX][AOTX_BODY_BYTES + 1];
    uint32_t length[AOTX_LINES_MAX];
    aotx_key_body key[AOTX_LINES_MAX];
} taken;

/* Consumes the slots that the feeder published, the way a device consumer would. */
static void consume(aotx_inbound_ring *ring, taken *t)
{
    uint64_t consumed = aotx_inbound_consumed(ring);
    uint64_t head = aotx_inbound_head(ring);
    while (consumed < head) {
        const unsigned char *slot = ring->slots + (consumed & ring->mask) * AOTX_SLOT_BYTES;
        const aotx_record_header *h = (const aotx_record_header *)slot;
        CHECK(aotx_record_valid(h) == 1, "a slot does not validate");
        CHECK(h->seq == consumed + 1, "a slot holds the wrong sequence");
        CHECK(h->cls == AOTX_CLASS_A, "a slot holds the wrong class");
        CHECK(h->writer == AOTX_WRITER_FEEDER, "a slot holds the wrong writer");
        if (h->type == AOTX_REC_INPUT_LINE) {
            if (t->lines < AOTX_LINES_MAX) {
                memcpy(t->text[t->lines], aotx_record_body(h), h->body_len);
                t->text[t->lines][h->body_len] = '\0';
                t->length[t->lines] = h->body_len;
            }
            t->lines++;
        } else if (h->type == AOTX_REC_KEY) {
            CHECK(h->body_len == sizeof(aotx_key_body), "a key record has the wrong body length");
            if (t->keys < AOTX_LINES_MAX && h->body_len == sizeof(aotx_key_body)) {
                memcpy(&t->key[t->keys], aotx_record_body(h), sizeof(aotx_key_body));
            }
            t->keys++;
        } else if (h->type == AOTX_REC_TICK_START) {
            aotx_clock_body clock;
            memcpy(&clock, aotx_record_body(h), sizeof(clock));
            CHECK(h->body_len == sizeof(clock), "a tick start has the wrong body length");
            CHECK(clock.wall_ns > 1000000000000000000ull, "a tick start holds no wall clock");
            t->clocks++;
        }
        consumed++;
        aotx_store_release(&ring->pre->consumed, consumed);
    }
}

static void batch(int n)
{
    aotx_map map;
    aotx_inbound_ring ring;
    taken got;
    char fd_text[16];
    char line[AOTX_BODY_BYTES * 2];
    char key_text[16];
    char *args[6];
    int pipe_fds[2];
    int key_fds[2];
    int child;
    int want;
    int want_keys;
    int i;
    uint64_t deadline;

    memset(&got, 0, sizeof(got));
    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, &map, &ring) == 0, "the ring does not open");
    CHECK(pipe(pipe_fds) == 0, "the line pipe does not open");
    CHECK(pipe(key_fds) == 0, "the key pipe does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    snprintf(key_text, sizeof(key_text), "%d", key_fds[0]);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = (char *)"--keys-fd";
    args[4] = key_text;
    args[5] = NULL;
    child = aotx_spawn(args, pipe_fds[0], -1);
    CHECK(child > 0, "the feeder does not start");
    close(pipe_fds[0]);
    close(key_fds[0]);

    /* Every line carries different content, so a wrong slot index cannot hide. */
    for (i = 0; i < n; i++) {
        int bytes = snprintf(line, sizeof(line), "input line %d of %d\n", i, n);
        CHECK(write(pipe_fds[1], line, (size_t)bytes) == bytes, "the line does not write");
        consume(&ring, &got);
    }
    /* One line longer than a body must arrive as two records. */
    memset(line, 'x', AOTX_BODY_BYTES + 10);
    line[AOTX_BODY_BYTES + 10] = '\n';
    CHECK(write(pipe_fds[1], line, AOTX_BODY_BYTES + 11) == (ssize_t)(AOTX_BODY_BYTES + 11),
          "the long line does not write");
    close(pipe_fds[1]);

    /* Every key frame carries different content, so a wrong frame cannot hide. */
    for (i = 0; i < n; i++) {
        aotx_key_body frame;
        frame.key = (uint32_t)(0x100 + i);
        frame.codepoint = (uint32_t)(0x41 + i);
        frame.action = 1u;
        frame.mods = (uint32_t)(i & 3);
        CHECK(write(key_fds[1], &frame, sizeof(frame)) == (ssize_t)sizeof(frame),
              "the key frame does not write");
        consume(&ring, &got);
    }
    /* One frame that arrives in two reads must give one record and no other. */
    {
        aotx_key_body frame;
        const unsigned char *bytes = (const unsigned char *)&frame;
        frame.key = 0x200u;
        frame.codepoint = 0x7a7au;
        frame.action = 2u;
        frame.mods = 5u;
        CHECK(write(key_fds[1], bytes, 7) == 7, "the first part of the frame does not write");
        deadline = aotx_wall_ns() + AOTX_PART_NS;
        while (aotx_wall_ns() < deadline) {
            uint64_t backoff = 0;
            consume(&ring, &got);
            aotx_pause(&backoff);
        }
        CHECK(got.keys == n, "a part of a frame gave %d records and %d were asked for",
              got.keys, n);
        CHECK(write(key_fds[1], bytes + 7, 9) == 9, "the second part of the frame does not write");
    }
    close(key_fds[1]);

    want_keys = n + 1;
    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while (got.keys < want_keys && aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        consume(&ring, &got);
        if (got.keys < want_keys) {
            aotx_pause(&backoff);
        }
    }
    CHECK(got.keys == want_keys, "the feeder sent %d key records and %d were asked for",
          got.keys, want_keys);
    for (i = 0; i < n && i < got.keys; i++) {
        CHECK(got.key[i].key == (uint32_t)(0x100 + i), "key %d holds the wrong code", i);
        CHECK(got.key[i].codepoint == (uint32_t)(0x41 + i), "key %d holds the wrong point", i);
        CHECK(got.key[i].action == 1u, "key %d holds the wrong action", i);
    }
    if (got.keys == want_keys) {
        CHECK(got.key[n].key == 0x200u, "the frame of two reads holds the wrong code");
        CHECK(got.key[n].codepoint == 0x7a7au, "the frame of two reads holds the wrong point");
        CHECK(got.key[n].mods == 5u, "the frame of two reads holds the wrong modifier bits");
    }

    want = n + 2;
    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while (got.lines < want && aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        consume(&ring, &got);
        if (got.lines < want) {
            aotx_pause(&backoff);
        }
    }
    CHECK(got.lines == want, "the feeder sent %d lines and %d were asked for", got.lines, want);
    for (i = 0; i < n && i < got.lines; i++) {
        char expect[64];
        snprintf(expect, sizeof(expect), "input line %d of %d", i, n);
        CHECK(strcmp(got.text[i], expect) == 0, "line %d holds %s", i, got.text[i]);
    }
    if (got.lines >= want) {
        CHECK(got.length[n] == AOTX_BODY_BYTES, "the first part of the long line is not full");
        CHECK(got.length[n + 1] == 10, "the second part of the long line is the wrong length");
    }

    /* The clock records go on after the end of the input, until the ring closes. */
    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while (got.clocks < 2 && aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        consume(&ring, &got);
        aotx_pause(&backoff);
    }
    CHECK(got.clocks >= 2, "the feeder sent %d tick starts", got.clocks);
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(child) == 0, "the feeder does not end with a clean status");
    printf("batch %d: lines %d, keys %d, clocks %d\n", n, got.lines, got.keys, got.clocks);
    aotx_map_release(&map);
}

/* A preamble that names another layout version must stop the feeder with the layout status. */
static void refuse_layout(void)
{
    aotx_map map;
    aotx_inbound_ring ring;
    char fd_text[16];
    char *args[4];
    int child;
    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, &map, &ring) == 0, "the ring does not open");
    ring.pre->magic = 0;
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the feeder does not start");
    CHECK(aotx_wait(child) == AOTX_EXIT_LAYOUT, "a wrong magic must give status 2");
    aotx_map_release(&map);
}

/* ---- the host tool that reads a file under the allowed root ---- */

#define AOTX_REPLY_MAX 320

/* The fixture file that is longer than the cap. The figure is above the cap by more than
 * one part, so the cut cannot pass by rounding. */
#define AOTX_BIG_BYTES 9000

typedef struct reply {
    uint32_t request;
    uint32_t agent;
    uint32_t status;   /* the status of the last part that came */
    uint32_t parts;    /* the count of parts the reply names */
    uint32_t got;      /* the parts that came */
    uint32_t content;  /* the parts that carry content */
    uint32_t len;      /* the content bytes assembled */
    char reason[AOTX_TOOL_REPLY_BYTES + 1];
    unsigned char bytes[AOTX_FS_CAP];
} reply;

typedef struct replies {
    int count;
    reply at[AOTX_REPLY_MAX];
} replies;

/* The table holds one whole reply for each request, so it lives beside the program and not
 * on the stack. One case runs at a time, and each case clears it at its start. */
static replies collected;

/* Gives the entry of one request, and makes it when the table does not hold it. */
static reply *entry_of(replies *r, uint32_t request)
{
    int i;
    for (i = 0; i < r->count; i++) {
        if (r->at[i].request == request) {
            return &r->at[i];
        }
    }
    if (r->count >= AOTX_REPLY_MAX) {
        return NULL;
    }
    memset(&r->at[r->count], 0, sizeof(r->at[0]));
    r->at[r->count].request = request;
    return &r->at[r->count++];
}

/* Consumes the slots the feeder published and assembles the replies. A part with the status
 * ok carries content. A part with any other status is the last part of the reply and its
 * bytes are the reason. */
static void collect(aotx_inbound_ring *ring, replies *r)
{
    uint64_t consumed = aotx_inbound_consumed(ring);
    uint64_t head = aotx_inbound_head(ring);
    while (consumed < head) {
        const unsigned char *slot = ring->slots + (consumed & ring->mask) * AOTX_SLOT_BYTES;
        const aotx_record_header *h = (const aotx_record_header *)slot;
        if (h->type == AOTX_REC_TOOL_REPLY && h->body_len >= sizeof(aotx_tool_reply_body)) {
            aotx_tool_reply_body body;
            reply *e;
            memcpy(&body, aotx_record_body(h), sizeof(body));
            CHECK(h->cls == AOTX_CLASS_A, "a reply record is not authoritative");
            e = entry_of(r, body.request);
            if (e != NULL) {
                e->agent = body.agent;
                e->parts = body.parts;
                e->status = body.status;
                e->got++;
                if (body.status == AOTX_TOOL_OK) {
                    uint32_t take = body.len;
                    if (take > AOTX_TOOL_REPLY_BYTES) {
                        take = AOTX_TOOL_REPLY_BYTES;
                    }
                    if (e->len + take <= sizeof(e->bytes)) {
                        memcpy(e->bytes + e->len, body.bytes, take);
                        e->len += take;
                    }
                    e->content++;
                } else {
                    uint32_t take = body.len;
                    if (take > AOTX_TOOL_REPLY_BYTES) {
                        take = AOTX_TOOL_REPLY_BYTES;
                    }
                    memcpy(e->reason, body.bytes, take);
                    e->reason[take] = '\0';
                }
            }
        }
        consumed++;
        aotx_store_release(&ring->pre->consumed, consumed);
    }
}

/* Waits for the first record the feeder publishes. The feeder opens the requests file
 * before it maps the ring, so a record proves that the open is done. */
static void wait_for_start(aotx_inbound_ring *ring)
{
    uint64_t deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while (aotx_inbound_head(ring) == 0 && aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        aotx_pause(&backoff);
    }
}

/* Waits until every reply that the table names is complete, or the time runs out. */
static void wait_for(aotx_inbound_ring *ring, replies *r, int want)
{
    uint64_t deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    for (;;) {
        uint64_t backoff = 0;
        int done = 0;
        int i;
        collect(ring, r);
        for (i = 0; i < r->count; i++) {
            if (r->at[i].parts > 0 && r->at[i].got == r->at[i].parts) {
                done++;
            }
        }
        if (done >= want || aotx_wall_ns() >= deadline) {
            return;
        }
        aotx_pause(&backoff);
    }
}

/* Gives the byte that file i holds at place j. Each file holds content that no other file
 * holds, so a wrong file and a wrong part order cannot hide. */
static unsigned char fixture_byte(int i, int j)
{
    return (unsigned char)((i * 31 + j * 17) & 0xff);
}

static int fixture_length(int i)
{
    return i * 37 + 1;
}

/* Writes one file of the fixture. */
static void write_file(const char *path, const unsigned char *bytes, size_t len)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    CHECK(fd >= 0, "the fixture file %s does not open", path);
    if (fd >= 0) {
        CHECK(write(fd, bytes, len) == (ssize_t)len, "the fixture file %s does not write",
              path);
        close(fd);
    }
}

/* Writes one line of the requests file, in the shape the drain writes. */
static void put_request(int fd, uint32_t request, uint32_t agent, const char *tool,
                        const char *arg)
{
    char line[1024];
    int n = snprintf(line, sizeof(line),
                     "{\"request\":%u,\"agent\":%u,\"turn\":1,\"tool\":\"%s\",\"arg\":\"%s\","
                     "\"deadline\":500,\"auth\":\"none\",\"tick\":7}\n",
                     request, agent, tool, arg);
    CHECK(write(fd, line, (size_t)n) == n, "the request line does not write");
}

/* Reads n files under the root, and applies the guards of the root rule beside them. */
static void tools(int n)
{
    aotx_map map;
    aotx_inbound_ring ring;
    replies *got = &collected;
    static unsigned char content[AOTX_BIG_BYTES + 16];
    char dir[256];
    char root[320];
    char path[512];
    char requests[512];
    char fd_text[16];
    char *args[8];
    int requests_fd;
    int child;
    int i;
    int want;

    memset(got, 0, sizeof(*got));
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(root, sizeof(root), "%s/root", dir);
    CHECK(aotx_make_dir(root) == 0, "the root does not open");
    snprintf(path, sizeof(path), "%s/deep", root);
    CHECK(aotx_make_dir(path) == 0, "the directory under the root does not open");

    for (i = 0; i < n; i++) {
        int j;
        for (j = 0; j < fixture_length(i); j++) {
            content[j] = fixture_byte(i, j);
        }
        snprintf(path, sizeof(path), "%s/file-%d.txt", root, i);
        write_file(path, content, (size_t)fixture_length(i));
    }
    /* A file under a directory of the root, to prove that a whole path is walked. */
    for (i = 0; i < 32; i++) {
        content[i] = (unsigned char)('a' + (i % 26));
    }
    snprintf(path, sizeof(path), "%s/deep/under.txt", root);
    write_file(path, content, 32);
    /* A file longer than the cap. */
    for (i = 0; i < AOTX_BIG_BYTES; i++) {
        content[i] = (unsigned char)((i * 13 + 5) & 0xff);
    }
    snprintf(path, sizeof(path), "%s/big.txt", root);
    write_file(path, content, (size_t)AOTX_BIG_BYTES);
    /* A file that no path under the root may reach. */
    snprintf(path, sizeof(path), "%s/outside.txt", dir);
    write_file(path, (const unsigned char *)"the file outside the root", 25);
    /* A link that names that file, and a link to the directory that holds it. */
    snprintf(path, sizeof(path), "%s/link.txt", root);
    CHECK(symlink("../outside.txt", path) == 0, "the link does not open");
    snprintf(path, sizeof(path), "%s/up", root);
    CHECK(symlink("..", path) == 0, "the link to the directory does not open");

    snprintf(requests, sizeof(requests), "%s/requests.jsonl", dir);
    CHECK(aotx_inbound_create(256u, &map, &ring) == 0, "the ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = (char *)"--root";
    args[4] = root;
    args[5] = (char *)"--requests";
    args[6] = requests;
    args[7] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the feeder does not start");
    /* The file is made after the feeder started, as the drain makes it at the first
     * request of a run. The feeder finds no file at its start and reads the file it later
     * finds from the first byte. */
    wait_for_start(&ring);
    requests_fd = open(requests, O_WRONLY | O_CREAT | O_APPEND, 0644);
    CHECK(requests_fd >= 0, "the requests file does not open");

    for (i = 0; i < n; i++) {
        char name[64];
        snprintf(name, sizeof(name), "file-%d.txt", i);
        put_request(requests_fd, (uint32_t)(100 + i), (uint32_t)(i % 64), "fs_read", name);
        collect(&ring, got);
    }
    put_request(requests_fd, 300u, 1u, "fs_read", "deep/under.txt");
    put_request(requests_fd, 301u, 1u, "fs_read", "big.txt");
    put_request(requests_fd, 302u, 1u, "fs_read", "link.txt");
    put_request(requests_fd, 303u, 1u, "fs_read", "../outside.txt");
    put_request(requests_fd, 304u, 1u, "fs_read", "up/outside.txt");
    put_request(requests_fd, 305u, 1u, "fs_read", "not-there.txt");
    put_request(requests_fd, 306u, 1u, "fs_read", "/etc/hostname");
    put_request(requests_fd, 307u, 1u, "fs_read", "deep");
    /* A tool that no host tool of the run names. The feeder answers with an error and
     * executes nothing. */
    put_request(requests_fd, 308u, 1u, "memory_recall", "file-0.txt");
    /* The identity that the first request carries, sent again. One request is executed one
     * time, so this line gives no second reply. */
    put_request(requests_fd, 100u, 0u, "fs_read", "file-0.txt");

    want = n + 9;
    wait_for(&ring, got, want);
    CHECK(got->count == want, "the feeder answered %d requests and %d were asked for",
          got->count, want);

    for (i = 0; i < n; i++) {
        reply *e = entry_of(got, (uint32_t)(100 + i));
        int j;
        int same = 1;
        CHECK(e->status == AOTX_TOOL_OK, "request %d gives the status %u", i, e->status);
        CHECK(e->agent == (uint32_t)(i % 64), "request %d names the agent %u", i, e->agent);
        CHECK((int)e->len == fixture_length(i), "request %d gives %u bytes and %d were"
              " written", i, e->len, fixture_length(i));
        CHECK(e->parts == (uint32_t)((fixture_length(i) + (int)AOTX_TOOL_REPLY_BYTES - 1) /
                                     (int)AOTX_TOOL_REPLY_BYTES),
              "request %d gives %u parts", i, e->parts);
        CHECK(e->got == e->parts, "request %d gives %u parts of %u", i, e->got, e->parts);
        for (j = 0; j < (int)e->len && j < fixture_length(i); j++) {
            if (e->bytes[j] != fixture_byte(i, j)) {
                same = 0;
            }
        }
        CHECK(same == 1, "the parts of request %d do not assemble to the file", i);
    }
    {
        reply *deep = entry_of(got, 300u);
        reply *big = entry_of(got, 301u);
        reply *link = entry_of(got, 302u);
        reply *up = entry_of(got, 303u);
        reply *through = entry_of(got, 304u);
        reply *gone = entry_of(got, 305u);
        reply *absolute = entry_of(got, 306u);
        reply *folder = entry_of(got, 307u);
        reply *other = entry_of(got, 308u);
        int same = 1;
        int j;
        CHECK(deep->status == AOTX_TOOL_OK && deep->len == 32,
              "a file under a directory of the root gives %u bytes", deep->len);
        CHECK(big->len == AOTX_FS_CAP, "the long file gives %u bytes and the cap is %u",
              big->len, AOTX_FS_CAP);
        CHECK(big->parts == (AOTX_FS_CAP + AOTX_TOOL_REPLY_BYTES - 1u) /
                            AOTX_TOOL_REPLY_BYTES + 1u,
              "the long file gives %u parts", big->parts);
        CHECK(big->content == big->parts - 1u, "the long file gives %u parts of content",
              big->content);
        CHECK(big->status == AOTX_TOOL_ERROR, "the last part of a cut file must state the cut");
        CHECK(strstr(big->reason, "longer than the cap") != NULL,
              "the cut reason reads %s", big->reason);
        for (j = 0; j < (int)big->len; j++) {
            if (big->bytes[j] != (unsigned char)((j * 13 + 5) & 0xff)) {
                same = 0;
            }
        }
        CHECK(same == 1, "the bytes of the cut file are not the first bytes of it");
        CHECK(link->status == AOTX_TOOL_REFUSED, "a link must be refused and gives %u",
              link->status);
        CHECK(strstr(link->reason, "symbolic link") != NULL, "the link reason reads %s",
              link->reason);
        CHECK(strstr((const char *)link->bytes, "outside the root") == NULL,
              "a refused read must give no byte of the file");
        CHECK(up->status == AOTX_TOOL_REFUSED, "a path of two dots must be refused and gives"
              " %u", up->status);
        CHECK(strstr(up->reason, "two dots") != NULL, "the reason reads %s", up->reason);
        CHECK(through->status == AOTX_TOOL_REFUSED,
              "a path through a link to a directory must be refused and gives %u",
              through->status);
        CHECK(gone->status == AOTX_TOOL_ERROR, "a file that is not there gives %u",
              gone->status);
        CHECK(strstr(gone->reason, "not there") != NULL, "the reason reads %s", gone->reason);
        CHECK(absolute->status == AOTX_TOOL_REFUSED,
              "a path that starts at the root of the file system gives %u", absolute->status);
        CHECK(folder->status == AOTX_TOOL_REFUSED, "a directory gives %u", folder->status);
        CHECK(other->status == AOTX_TOOL_ERROR, "a tool that is not a host tool gives %u",
              other->status);
        CHECK(strstr(other->reason, "not a host tool") != NULL, "the reason reads %s",
              other->reason);
    }
    /* The second line with the first identity gave no second reply, so the count of parts
     * of that reply did not go up. */
    {
        reply *first = entry_of(got, 100u);
        CHECK(first->got == first->parts, "the request was executed twice: %u parts of %u",
              first->got, first->parts);
    }

    close(requests_fd);
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(child) == 0, "the feeder does not end with a clean status");
    printf("tools %d: replies %d, parts of the long file %u\n", n, got->count,
           entry_of(got, 301u)->parts);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

/* A requests file that is already there is read from its end. A feeder that starts after
 * a restore therefore executes no request of the run before it. The device applies the
 * replies that the journal holds. */
static void from_the_end(void)
{
    aotx_map map;
    aotx_inbound_ring ring;
    replies *got = &collected;
    char dir[256];
    char root[320];
    char path[512];
    char requests[512];
    char fd_text[16];
    char *args[8];
    int requests_fd;
    int child;
    uint64_t deadline;

    memset(got, 0, sizeof(*got));
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(root, sizeof(root), "%s/root", dir);
    CHECK(aotx_make_dir(root) == 0, "the root does not open");
    snprintf(path, sizeof(path), "%s/old.txt", root);
    write_file(path, (const unsigned char *)"the file of the run before", 26);
    snprintf(path, sizeof(path), "%s/new.txt", root);
    write_file(path, (const unsigned char *)"the file of this run", 20);
    snprintf(requests, sizeof(requests), "%s/requests.jsonl", dir);
    requests_fd = open(requests, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
    CHECK(requests_fd >= 0, "the requests file does not open");
    put_request(requests_fd, 700u, 3u, "fs_read", "old.txt");

    CHECK(aotx_inbound_create(64u, &map, &ring) == 0, "the ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = (char *)"--root";
    args[4] = root;
    args[5] = (char *)"--requests";
    args[6] = requests;
    args[7] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the feeder does not start");
    wait_for_start(&ring);
    put_request(requests_fd, 701u, 3u, "fs_read", "new.txt");
    wait_for(&ring, got, 1);
    CHECK(got->count == 1, "the feeder answered %d requests and one was asked for", got->count);
    CHECK(entry_of(got, 701u)->len == 20, "the request of this run gives %u bytes",
          entry_of(got, 701u)->len);
    /* One more period of the clock proves that the line of the run before stays unread. */
    deadline = aotx_wall_ns() + 400000000ull;
    while (aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        collect(&ring, got);
        aotx_pause(&backoff);
    }
    CHECK(got->count == 1, "a line the file already held was executed");
    close(requests_fd);
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(child) == 0, "the feeder does not end with a clean status");
    printf("from the end: replies %d\n", got->count);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

#include "tests/feed_seam.h"
#include "tests/feed_settings.h"
#include "tests/feed_import.h"
#include "tests/feed_refuse.h"
#include "tests/feed_modules.h"

int main(int argc, char **argv)
{
    arguments = argv;
    if (argc < 3) {
        printf("usage: feed_test <feed program> <drain program> [journal program]"
               " [validator]\n");
        return 1;
    }
    reader_program = (argc > 3) ? argv[3] : NULL;
    lint_program = (argc > 4) ? argv[4] : NULL;
    argument_shape();
    request_shape();
    batch(1);
    batch(64);
    refuse_layout();
    tools(1);
    tools(64);
    from_the_end();
    loop(1);
    loop(64);
    settings_arm(1);
    settings_arm(64);
    modules_arm(1);
    modules_arm(64);
    import_line_arm(1);
    import_line_arm(64);
    refused_line_arm(1);
    refused_line_arm(64);
    import_request_arm(1);
    import_request_arm(64);
    table_arm(1);
    table_arm(64);
    if (reader_program != NULL) {
        import_loop(1);
        import_loop(64);
    }
    return aotx_report("feed_test", 650);
}
