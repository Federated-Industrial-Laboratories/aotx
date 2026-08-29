/* Purpose: Run the feeder against two pipes and check the records that reach the ring.
 * Owns: One inbound ring, one line pipe and one key pipe for each case.
 * Threading: Two processes; the test reads the ring while the feeder writes it.
 * Lifetime: The run of the program. */
#include "disk/feed/fs_tool.h"
#include "disk/settings/settings.h"
#include "tests/disk_fake.h"

#include <fcntl.h>
#include <sys/stat.h>

#define AOTX_RING_SLOTS      64u
#define AOTX_WAIT_NS    15000000000ull
#define AOTX_PART_NS    200000000ull
#define AOTX_LINES_MAX  200

static char **arguments;

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

#define AOTX_REPLY_MAX 160

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
    put_request(requests_fd, 308u, 1u, "fs_write", "file-0.txt");
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
        CHECK(other->status == AOTX_TOOL_ERROR, "a tool that is not fs_read gives %u",
              other->status);
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

/* The line that the drain writes is the line the feeder reads. This case runs both
 * programs over one file, so a change to the format of one shows here. The feeder starts
 * first, as it does at a boot, and the drain makes the file at its first request. */
static void loop(int n)
{
    aotx_map map;
    aotx_host_ring ring;
    aotx_fake_device device;
    aotx_map imap;
    aotx_inbound_ring iring;
    replies *got = &collected;
    char dir[256];
    char root[320];
    char path[512];
    char requests[512];
    char fd_text[16];
    char *args[8];
    int feeder;
    int drain;
    int i;

    memset(got, 0, sizeof(*got));
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(root, sizeof(root), "%s/root", dir);
    CHECK(aotx_make_dir(root) == 0, "the root does not open");
    for (i = 0; i < n; i++) {
        char body[64];
        int bytes = snprintf(body, sizeof(body), "the bytes of file %d", i);
        snprintf(path, sizeof(path), "%s/file-%d.txt", root, i);
        write_file(path, (const unsigned char *)body, (size_t)bytes);
    }
    snprintf(requests, sizeof(requests), "%s/requests.jsonl", dir);

    CHECK(aotx_inbound_create(256u, &imap, &iring) == 0, "the inbound ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", imap.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = (char *)"--root";
    args[4] = root;
    args[5] = (char *)"--requests";
    args[6] = requests;
    args[7] = NULL;
    feeder = aotx_spawn(args, -1, -1);
    CHECK(feeder > 0, "the feeder does not start");
    wait_for_start(&iring);

    CHECK(aotx_host_ring_create(262144u, 0x00100b0000000001ull + (uint64_t)n, &map, &ring) == 0,
          "the host ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[2];
    args[1] = (char *)"--ring-fd";
    args[2] = fd_text;
    args[3] = (char *)"--journal";
    args[4] = dir;
    args[5] = NULL;
    drain = aotx_spawn(args, -1, -1);
    CHECK(drain > 0, "the drain does not start");
    aotx_fake_start(&device, &ring, 0x00100b0000000001ull + (uint64_t)n);
    for (i = 0; i < n; i++) {
        aotx_tool_request_body r;
        device.writer = AOTX_WRITER_AGENT_BASE + (uint32_t)(i % 3);
        /* One request needs no authorization; the other waits for the operator and is
         * granted, and the grant carries no path of its own. */
        aotx_fake_request(i, AOTX_AUTH_NONE, &r);
        r.arg_len = (uint32_t)snprintf(r.arg, AOTX_TOOL_ARG_BYTES, "file-%d.txt", i);
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
        aotx_fake_request(2000 + i, AOTX_AUTH_PENDING, &r);
        r.arg_len = (uint32_t)snprintf(r.arg, AOTX_TOOL_ARG_BYTES, "file-%d.txt", i);
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
        aotx_fake_request(2000 + i, AOTX_AUTH_GRANTED, &r);
        r.arg_len = 0;
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
        if ((i + 1) % 32 == 0) {
            aotx_fake_commit(&device, 0);
        }
        collect(&iring, got);
    }
    aotx_fake_commit(&device, 0);
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(drain) == 0, "the drain does not end with a clean status");

    wait_for(&iring, got, 2 * n);
    CHECK(got->count == 2 * n, "the feeder answered %d requests and %d were asked for",
          got->count, 2 * n);
    for (i = 0; i < n; i++) {
        char want[64];
        reply *plain = entry_of(got, (uint32_t)(1000 + i));
        reply *granted = entry_of(got, (uint32_t)(3000 + i));
        int bytes = snprintf(want, sizeof(want), "the bytes of file %d", i);
        CHECK(plain != NULL && granted != NULL, "the reply table is full at request %d", i);
        if (plain == NULL || granted == NULL) {
            continue;
        }
        CHECK(plain->status == AOTX_TOOL_OK && (int)plain->len == bytes,
              "the request that needs no authorization gives %u bytes", plain->len);
        CHECK(memcmp(plain->bytes, want, (size_t)bytes) == 0,
              "the bytes of request %d are not the bytes of the file", i);
        /* The grant carried no path, so the line came from the request that was held. */
        CHECK(granted->status == AOTX_TOOL_OK && (int)granted->len == bytes,
              "the granted request gives %u bytes", granted->len);
        CHECK(memcmp(granted->bytes, want, (size_t)bytes) == 0,
              "the bytes of the granted request %d are not the bytes of the file", i);
    }
    aotx_store_release16(&iring.pre->closed, 1);
    CHECK(aotx_wait(feeder) == 0, "the feeder does not end with a clean status");
    printf("loop %d: replies %d\n", n, got->count);
    aotx_map_release(&map);
    aotx_map_release(&imap);
    aotx_remove_tree(dir);
}

/* ---- the settings that the feeder publishes before every other record ---- */

#define AOTX_SETTINGS_TEXT 8192

typedef struct settings_taken {
    int      count;             /* setting records that came */
    int      lines;             /* input line records that came */
    int      clocks;
    uint64_t first_line_seq;    /* the sequence of the first input line record */
    uint64_t first_clock_seq;   /* the sequence of the first clock record */
    uint64_t last_setting_seq;
    aotx_setting_body body[AOTX_SETTING_NUMBER_COUNT];
} settings_taken;

/* Consumes the slots that the feeder published and keeps the setting records in order. */
static void take_settings(aotx_inbound_ring *ring, settings_taken *t)
{
    uint64_t consumed = aotx_inbound_consumed(ring);
    uint64_t head = aotx_inbound_head(ring);
    while (consumed < head) {
        const unsigned char *slot = ring->slots + (consumed & ring->mask) * AOTX_SLOT_BYTES;
        const aotx_record_header *h = (const aotx_record_header *)slot;
        CHECK(aotx_record_valid(h) == 1, "a slot does not validate");
        if (h->type == AOTX_REC_SETTING) {
            CHECK(h->cls == AOTX_CLASS_A, "a setting record is not authoritative");
            CHECK(h->writer == AOTX_WRITER_FEEDER, "a setting record holds the wrong writer");
            CHECK(h->body_len == sizeof(aotx_setting_body),
                  "a setting record has the wrong body length");
            if (t->count < (int)AOTX_SETTING_NUMBER_COUNT) {
                memcpy(&t->body[t->count], aotx_record_body(h), sizeof(aotx_setting_body));
            }
            t->last_setting_seq = h->seq;
            t->count++;
        } else if (h->type == AOTX_REC_INPUT_LINE) {
            if (t->lines == 0) {
                t->first_line_seq = h->seq;
            }
            t->lines++;
        } else if (h->type == AOTX_REC_TICK_START) {
            if (t->clocks == 0) {
                t->first_clock_seq = h->seq;
            }
            t->clocks++;
        }
        consumed++;
        aotx_store_release(&ring->pre->consumed, consumed);
    }
}

/* Gives the value that line i of the settings file writes for the device key at place k of
 * the device key list. A key that comes again takes another value, so a file where the
 * last line does not win cannot pass. */
static int64_t settings_value(unsigned int key, int occurrence)
{
    int64_t least = aotx_settings_number_least(key);
    int64_t span = aotx_settings_number_most(key) - least + 1;
    return least + ((int64_t)occurrence * 3 + (int64_t)key) % span;
}

/* Runs the feeder with a settings file of n lines. The records must reach the ring before
 * the first line of the standard input. They must also come before the first clock record.
 * They come in the order of the key list. Each holds the value of the last line of its
 * key. */
static void settings_arm(int n)
{
    aotx_map map;
    aotx_inbound_ring ring;
    settings_taken got;
    unsigned int device[AOTX_SETTING_NUMBER_COUNT];
    int last[AOTX_SETTING_NUMBER_COUNT];
    char dir[256];
    char path[320];
    char file_text[AOTX_SETTINGS_TEXT];
    char err_path[400];
    char value[32];
    char fd_text[16];
    char *args[6];
    unsigned int k;
    size_t used;
    int device_count = 0;
    int want;
    int pipe_fds[2];
    int err_fd;
    int saved;
    int child;
    int i;
    uint64_t deadline;

    memset(&got, 0, sizeof(got));
    for (k = 0; k < AOTX_SETTING_NUMBER_COUNT; k++) {
        last[k] = -1;
        if (aotx_settings_number_side(k) == AOTX_SETTING_SIDE_DEVICE) {
            device[device_count++] = k;
        }
    }
    CHECK(device_count > 0, "the key list names no device side number key");
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(path, sizeof(path), "%s/aotx.settings", dir);
    snprintf(err_path, sizeof(err_path), "%s/report.txt", dir);

    /* Line 1 is a comment and line 2 is blank, so the refused line is line 3. */
    used = (size_t)snprintf(file_text, sizeof(file_text),
                            "# the settings of a run of %d lines\n\nno.such.key = 1\n", n);
    for (i = 0; i < n; i++) {
        k = device[i % device_count];
        last[k] = i / device_count;
        aotx_settings_format(settings_value(k, i / device_count),
                             aotx_settings_number_scale(k), value, sizeof(value));
        used += (size_t)snprintf(file_text + used, sizeof(file_text) - used, "  %s  =  %s  \n",
                                 aotx_settings_number_name(k), value);
    }
    /* A boot key and a text key make no record, because the device applies neither. */
    used += (size_t)snprintf(file_text + used, sizeof(file_text) - used,
                             "window.on = 1\ntui.box = unicode\n");
    {
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        CHECK(fd >= 0, "the settings file does not open");
        CHECK(write(fd, file_text, used) == (ssize_t)used, "the settings file does not write");
        close(fd);
    }
    want = (n < device_count) ? n : device_count;

    CHECK(aotx_inbound_create(AOTX_RING_SLOTS, &map, &ring) == 0, "the ring does not open");
    CHECK(pipe(pipe_fds) == 0, "the line pipe does not open");
    /* The line waits in the pipe before the feeder starts. A feeder that reads the
     * standard input first cannot pass this case. */
    CHECK(write(pipe_fds[1], "the first operator line\n", 24) == 24,
          "the operator line does not write");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = (char *)"--settings";
    args[4] = path;
    args[5] = NULL;
    err_fd = open(err_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    saved = dup(2);
    CHECK(err_fd >= 0 && saved >= 0, "the report file does not open");
    fflush(stderr);
    dup2(err_fd, 2);
    child = aotx_spawn(args, pipe_fds[0], -1);
    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    close(err_fd);
    CHECK(child > 0, "the feeder does not start");
    close(pipe_fds[0]);
    close(pipe_fds[1]);

    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while ((got.count < want || got.lines < 1 || got.clocks < 1) &&
           aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        take_settings(&ring, &got);
        aotx_pause(&backoff);
    }
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(child) == 0, "the feeder does not end with a clean status");
    take_settings(&ring, &got);

    CHECK(got.count == want, "the feeder sent %d setting records and %d were asked for",
          got.count, want);
    CHECK(got.lines >= 1 && got.clocks >= 1, "the run gave %d lines and %d clock records",
          got.lines, got.clocks);
    CHECK(got.last_setting_seq < got.first_line_seq, "a setting record came after the first"
          " line of the standard input");
    CHECK(got.last_setting_seq < got.first_clock_seq, "a setting record came after the first"
          " clock record");
    for (i = 0; i < want && i < got.count; i++) {
        const aotx_setting_body *b = &got.body[i];
        char name[AOTX_SETTING_WIRE_KEY_BYTES + 1];
        k = device[i];
        memcpy(name, b->key, b->key_len);
        name[b->key_len] = '\0';
        CHECK(strcmp(name, aotx_settings_number_name(k)) == 0,
              "setting record %d names %s and %s was asked for", i, name,
              aotx_settings_number_name(k));
        CHECK(b->scale == (uint32_t)aotx_settings_number_scale(k),
              "setting record %d holds the scale %u", i, b->scale);
        CHECK(b->value == settings_value(k, last[k]), "setting record %d holds %lld and the"
              " last line of %s gives %lld", i, (long long)b->value,
              aotx_settings_number_name(k), (long long)settings_value(k, last[k]));
    }
    /* The start prints a refused line; the feeder prints nothing, so a refusal reaches the
     * operator one time. */
    {
        char report[4096];
        int fd = open(err_path, O_RDONLY);
        ssize_t bytes = 0;
        CHECK(fd >= 0, "the report file does not read");
        if (fd >= 0) {
            bytes = read(fd, report, sizeof(report) - 1);
            close(fd);
        }
        report[(bytes > 0) ? bytes : 0] = '\0';
        CHECK(strstr(report, "settings:") == NULL,
              "the feeder must print no refusal; the start prints it: %s", report);
    }
    printf("settings %d: records %d, lines in the file %d\n", n, got.count, n + 5);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

int main(int argc, char **argv)
{
    arguments = argv;
    if (argc < 3) {
        printf("usage: feed_test <feed program> <drain program>\n");
        return 1;
    }
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
    return aotx_report("feed_test", 650);
}
