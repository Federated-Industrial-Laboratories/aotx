/* Purpose: Derive one ordered transcript file for each agent of a run.
 * Owns: The open files and the held parts of input, replies, requests, and results.
 * Threading: One thread; the drain or journal reader calls in record order.
 * Lifetime: One drain or one journal walk. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/drain/transcript.h"
#include "disk/drain/transcript_live.h"
#include "disk/feed/line.h"
#include "cognitive/live.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define AOTX_TRANSCRIPT_AGENTS 256u
#define AOTX_TRANSCRIPT_REQUESTS 1024u
#define AOTX_TRANSCRIPT_RESULT 16384u
#define AOTX_TRANSCRIPT_TEXT_MAX ((AOTX_INPUT_LINE_BYTES > AOTX_TRANSCRIPT_RESULT) \
                                ? AOTX_INPUT_LINE_BYTES : AOTX_TRANSCRIPT_RESULT)
#define AOTX_TRANSCRIPT_LINE_MAX (AOTX_TRANSCRIPT_TEXT_MAX * 6u + 512u)

typedef struct aotx_transcript_agent {
    int fd;
    uint32_t turn;
    uint32_t confirmed_turn;
    uint32_t reply_len;
    uint64_t reply_tick;
    int reply_open;
    unsigned char reply[AOTX_INPUT_LINE_BYTES];
} aotx_transcript_agent;

typedef struct aotx_transcript_request {
    uint32_t request;
    uint32_t agent;
    uint32_t turn;
    uint32_t tool;
} aotx_transcript_request;

struct aotx_transcript {
    aotx_transcript_live *live;
    char dir[AOTX_PATH_BYTES];
    aotx_transcript_agent agent[AOTX_TRANSCRIPT_AGENTS];
    aotx_transcript_request request[AOTX_TRANSCRIPT_REQUESTS];
    unsigned char input[AOTX_INPUT_LINE_BYTES];
    uint32_t input_len;
    uint64_t input_tick;
    int input_open;
    aotx_tool_request_body request_head;
    unsigned char request_arg[AOTX_INPUT_LINE_BYTES];
    uint32_t request_len;
    uint64_t request_tick;
    int request_open;
    unsigned char result[AOTX_TRANSCRIPT_RESULT];
    uint32_t result_len;
    uint32_t result_id;
    uint32_t result_agent;
    uint32_t result_status;
    uint64_t result_tick;
    int result_open;
    uint64_t lines;
    uint64_t refused;
};

static int agent_of_writer(uint32_t writer)
{
    if (writer >= AOTX_WRITER_AGENT_BASE
        && writer - AOTX_WRITER_AGENT_BASE < AOTX_TRANSCRIPT_AGENTS) {
        return (int)(writer - AOTX_WRITER_AGENT_BASE);
    }
    return -1;
}

static const char *tool_name(uint32_t tool, char *out, size_t bytes)
{
    static const char *names[10] = { "none", "memory_recall", "memory_write", "fs_read",
                                     "fs_list", "fs_write", "fs_update", "run",
                                     "skill_use", "import" };
    if (tool < 10u) {
        return names[tool];
    }
    if (tool >= AOTX_TOOL_MODULE_BASE) {
        snprintf(out, bytes, "module-%u", tool - AOTX_TOOL_MODULE_BASE);
        return out;
    }
    return "other";
}

static const char *tool_status(uint32_t status)
{
    static const char *names[4] = { "ok", "error", "refused", "late" };
    return (status < 4u) ? names[status] : "other";
}

static int open_agent(aotx_transcript *t, uint32_t agent)
{
    char path[AOTX_PATH_BYTES + 32];
    int fd;
    if (agent >= AOTX_TRANSCRIPT_AGENTS) {
        return -1;
    }
    if (t->agent[agent].fd >= 0) {
        return t->agent[agent].fd;
    }
    snprintf(path, sizeof(path), "%s/%u.jsonl", t->dir, agent);
    fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd >= 0) {
        t->agent[agent].fd = fd;
    }
    return fd;
}

static int put_all(int fd, const char *bytes, size_t count)
{
    size_t done = 0;
    while (done < count) {
        ssize_t n = write(fd, bytes + done, count - done);
        if (n <= 0) {
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

static int put_element(aotx_transcript *t, uint32_t agent, uint64_t tick, const char *kind,
                       const unsigned char *text, uint32_t text_len, const char *tool,
                       uint32_t request, const char *status, uint32_t turn)
{
    char escaped[AOTX_TRANSCRIPT_TEXT_MAX * 6u + 8u];
    char line[AOTX_TRANSCRIPT_LINE_MAX];
    int used;
    int fd = open_agent(t, agent);
    if (fd < 0) {
        return -1;
    }
    if (tool != NULL) {
        used = snprintf(line, sizeof(line),
                        "{\"tick\":%llu,\"kind\":\"%s\",\"tool\":\"%s\","
                        "\"request\":%u,\"status\":\"%s\",\"turn\":%u}\n",
                        (unsigned long long)tick, kind, tool, request, status, turn);
    } else {
        size_t escaped_len = aotx_json_write(escaped, sizeof(escaped), text, text_len);
        used = snprintf(line, sizeof(line),
                        "{\"tick\":%llu,\"kind\":\"%s\",\"text\":\"%.*s\","
                        "\"request\":%u,\"status\":\"%s\",\"turn\":%u}\n",
                        (unsigned long long)tick, kind, (int)escaped_len, escaped,
                        request, status, turn);
    }
    if (used < 0 || (size_t)used >= sizeof(line) || put_all(fd, line, (size_t)used) != 0) {
        return -1;
    }
    t->lines++;
    return 0;
}

static int put_call(aotx_transcript *t, const aotx_tool_request_body *r, uint64_t tick,
                    const unsigned char *argument, uint32_t length, const char *status)
{
    char name[32];
    char escaped[AOTX_INPUT_LINE_BYTES * 6u + 8u];
    char line[AOTX_TRANSCRIPT_LINE_MAX];
    size_t escaped_len;
    int used;
    int fd;
    if (r->agent >= AOTX_TRANSCRIPT_AGENTS) {
        return 0;
    }
    fd = open_agent(t, r->agent);
    if (fd < 0) {
        return -1;
    }
    escaped_len = aotx_json_write(escaped, sizeof(escaped), argument, length);
    used = snprintf(line, sizeof(line),
                    "{\"tick\":%llu,\"kind\":\"call\",\"tool\":\"%s\","
                    "\"text\":\"%.*s\",\"request\":%u,\"status\":\"%s\","
                    "\"turn\":%u}\n",
                    (unsigned long long)tick, tool_name(r->tool, name, sizeof(name)),
                    (int)escaped_len, escaped, r->request, status, r->turn);
    if (used < 0 || (size_t)used >= sizeof(line) || put_all(fd, line, (size_t)used) != 0) {
        return -1;
    }
    t->lines++;
    return 0;
}

static int put_live(void *context, uint32_t agent, uint64_t tick,
                    const unsigned char *input, uint32_t length, const char *description,
                    uint32_t description_length, uint32_t status)
{
    aotx_transcript *t = context;
    uint32_t turn = t->agent[agent].confirmed_turn;
    if (!status) {
        /* Ordinary input can be refused. The last manifest supplies the device turn. */
        turn++;
        t->agent[agent].turn = turn;
        if (put_element(t, agent, tick, "line", input, length, NULL, 0u,
                        "accepted", turn) != 0) return -1;
    }
    return put_element(t, agent, tick, "selection", (const unsigned char *)description,
                       description_length, NULL, 0u, status ? "refused" : "selected",
                       turn);
}

static int parse_target(const unsigned char *line, uint32_t length, uint32_t *agent,
                        const unsigned char **text, uint32_t *text_len)
{
    uint32_t at;
    uint32_t value = 0;
    if (length > 4u && memcmp(line, "say ", 4u) == 0) {
        *agent = 0u;
        *text = line + 4u;
        *text_len = length - 4u;
        return 1;
    }
    if (length <= 5u || memcmp(line, "task ", 5u) != 0) {
        return 0;
    }
    at = 5u;
    if (line[at] < '0' || line[at] > '9') {
        return 0;
    }
    while (at < length && line[at] >= '0' && line[at] <= '9') {
        value = value * 10u + (uint32_t)(line[at] - '0');
        at++;
    }
    if (value >= AOTX_TRANSCRIPT_AGENTS || at >= length || line[at] != ' ') {
        return 0;
    }
    *agent = value;
    *text = line + at + 1u;
    *text_len = length - at - 1u;
    return 1;
}

static int flush_input(aotx_transcript *t)
{
    const unsigned char *text;
    uint32_t text_len;
    uint32_t agent;
    int rc = 0;
    if (t->input_open == 0) {
        return 0;
    }
    if (parse_target(t->input, t->input_len, &agent, &text, &text_len) != 0) {
        uint32_t turn = ++t->agent[agent].turn;
        rc = put_element(t, agent, t->input_tick, "line", text, text_len, NULL, 0u, "", turn);
    }
    t->input_open = 0;
    t->input_len = 0u;
    return rc;
}

static void remember_request(aotx_transcript *t, const aotx_tool_request_body *r)
{
    aotx_transcript_request *at = &t->request[r->request % AOTX_TRANSCRIPT_REQUESTS];
    at->request = r->request;
    at->agent = r->agent;
    at->turn = r->turn;
    at->tool = r->tool;
}

static const aotx_transcript_request *find_request(const aotx_transcript *t, uint32_t id)
{
    const aotx_transcript_request *at = &t->request[id % AOTX_TRANSCRIPT_REQUESTS];
    return (at->request == id) ? at : NULL;
}

static int flush_request(aotx_transcript *t)
{
    aotx_tool_request_body *r = &t->request_head;
    char name[32];
    const char *kind = NULL;
    const char *status = "";
    int rc = 0;
    if (t->request_open == 0) {
        return 0;
    }
    remember_request(t, r);
    if (r->auth == AOTX_AUTH_PENDING || r->auth == AOTX_AUTH_NONE) {
        kind = "call";
        status = "waiting";
    } else if (r->auth == AOTX_AUTH_GRANTED) {
        kind = "grant";
        status = "granted";
    } else if (r->auth == AOTX_AUTH_REFUSED) {
        kind = "refuse";
        status = "refused";
    }
    if (kind != NULL && strcmp(kind, "call") == 0) {
        rc = put_call(t, r, t->request_tick, t->request_arg, t->request_len,
                      (r->auth == AOTX_AUTH_PENDING) ? "waiting" : "open");
    } else if (kind != NULL && r->agent < AOTX_TRANSCRIPT_AGENTS) {
        rc = put_element(t, r->agent, t->request_tick, kind, NULL, 0u,
                         tool_name(r->tool, name, sizeof(name)), r->request, status, r->turn);
    }
    t->request_open = 0;
    t->request_len = 0u;
    return rc;
}

static int flush_reply(aotx_transcript *t, uint32_t agent, uint32_t turn)
{
    aotx_transcript_agent *a;
    int rc;
    if (agent >= AOTX_TRANSCRIPT_AGENTS) {
        return 0;
    }
    a = &t->agent[agent];
    if (a->reply_open == 0) {
        return 0;
    }
    rc = put_element(t, agent, a->reply_tick, "reply", a->reply, a->reply_len, NULL,
                     0u, "", turn);
    a->reply_open = 0;
    a->reply_len = 0u;
    return rc;
}

static int take_input(aotx_transcript *t, const aotx_record_header *h)
{
    uint32_t room;
    if ((h->flags & AOTX_FLAG_FRAGMENT) == 0) {
        if (flush_input(t) != 0) {
            return -1;
        }
        t->input_open = 1;
        t->input_tick = h->tick;
    } else if (t->input_open == 0) {
        t->refused++;
        return 0;
    }
    room = AOTX_INPUT_LINE_BYTES - t->input_len;
    if (h->body_len > room) {
        t->refused++;
        t->input_open = 0;
        t->input_len = 0u;
        return 0;
    }
    memcpy(t->input + t->input_len, aotx_record_body(h), h->body_len);
    t->input_len += h->body_len;
    return 0;
}

static int take_token(aotx_transcript *t, const aotx_record_header *h)
{
    aotx_token_body body;
    uint32_t who;
    aotx_transcript_agent *a;
    uint32_t room;
    if (h->body_len < sizeof(body)) {
        t->refused++;
        return 0;
    }
    memcpy(&body, aotx_record_body(h), sizeof(body));
    if ((body.flags & AOTX_TOKEN_SAMPLED) == 0u) {
        return 0;
    }
    who = body.slot;
    if (who >= AOTX_TRANSCRIPT_AGENTS || body.text_len > sizeof(body.text)) {
        t->refused++;
        return 0;
    }
    a = &t->agent[who];
    if (a->reply_open == 0) {
        a->reply_open = 1;
        a->reply_tick = h->tick;
    }
    if (put_element(t, who, h->tick, "part", (const unsigned char *)body.text,
                    body.text_len, NULL, 0u, "open", a->turn) != 0) {
        return -1;
    }
    room = AOTX_INPUT_LINE_BYTES - a->reply_len;
    if (body.text_len > room) {
        t->refused++;
        a->reply_open = 0;
        a->reply_len = 0u;
        return 0;
    }
    memcpy(a->reply + a->reply_len, body.text, body.text_len);
    a->reply_len += body.text_len;
    return 0;
}

static int take_request(aotx_transcript *t, const aotx_record_header *h)
{
    const unsigned char *body = aotx_record_body(h);
    if ((h->flags & AOTX_FLAG_FRAGMENT) == 0) {
        uint32_t bytes;
        if (flush_request(t) != 0) {
            return -1;
        }
        if (h->body_len < 32u) {
            t->refused++;
            return 0;
        }
        memset(&t->request_head, 0, sizeof(t->request_head));
        memcpy(&t->request_head, body,
               (h->body_len < sizeof(t->request_head)) ? h->body_len : sizeof(t->request_head));
        bytes = t->request_head.arg_len;
        if (bytes > AOTX_TOOL_ARG_BYTES) {
            bytes = AOTX_TOOL_ARG_BYTES;
        }
        memcpy(t->request_arg, t->request_head.arg, bytes);
        t->request_len = bytes;
        t->request_tick = h->tick;
        t->request_open = 1;
        return 0;
    }
    if (t->request_open == 0 || t->request_len + h->body_len > sizeof(t->request_arg)) {
        t->refused++;
        return 0;
    }
    memcpy(t->request_arg + t->request_len, body, h->body_len);
    t->request_len += h->body_len;
    return 0;
}

static int flush_result(aotx_transcript *t)
{
    const aotx_transcript_request *known;
    uint32_t turn = 0u;
    int rc;
    if (t->result_open == 0) {
        return 0;
    }
    known = find_request(t, t->result_id);
    if (known != NULL) {
        turn = known->turn;
    } else if (t->result_agent < AOTX_TRANSCRIPT_AGENTS) {
        turn = t->agent[t->result_agent].turn;
    }
    rc = put_element(t, t->result_agent, t->result_tick, "result", t->result,
                     t->result_len, NULL, t->result_id, tool_status(t->result_status), turn);
    t->result_open = 0;
    t->result_len = 0u;
    return rc;
}

static int take_result(aotx_transcript *t, const aotx_record_header *h)
{
    aotx_tool_reply_body r;
    uint32_t bytes;
    if (h->body_len < 24u) {
        t->refused++;
        return 0;
    }
    memset(&r, 0, sizeof(r));
    memcpy(&r, aotx_record_body(h),
           (h->body_len < sizeof(r)) ? h->body_len : sizeof(r));
    if (t->result_open == 0 || t->result_id != r.request || r.part == 0u) {
        if (flush_result(t) != 0) {
            return -1;
        }
        t->result_open = 1;
        t->result_id = r.request;
        t->result_agent = r.agent;
        t->result_status = r.status;
        t->result_tick = h->tick;
    }
    bytes = (r.len > AOTX_TOOL_REPLY_BYTES) ? AOTX_TOOL_REPLY_BYTES : r.len;
    if (bytes > AOTX_TRANSCRIPT_RESULT - t->result_len) {
        bytes = AOTX_TRANSCRIPT_RESULT - t->result_len;
    }
    memcpy(t->result + t->result_len, r.bytes, bytes);
    t->result_len += bytes;
    t->result_status = r.status;
    if (r.status != AOTX_TOOL_OK || r.part + 1u >= r.parts) {
        return flush_result(t);
    }
    return 0;
}

static int take_manifest(aotx_transcript *t, const aotx_record_header *h)
{
    aotx_manifest_body m;
    aotx_tool_request_body request;
    char name[32];
    if (h->body_len < sizeof(m)) {
        t->refused++;
        return 0;
    }
    memcpy(&m, aotx_record_body(h), sizeof(m));
    if (m.agent >= AOTX_TRANSCRIPT_AGENTS) {
        t->refused++;
        return 0;
    }
    t->agent[m.agent].turn = m.turn;
    t->agent[m.agent].confirmed_turn = m.turn;
    if (flush_reply(t, m.agent, m.turn) != 0) {
        return -1;
    }
    if (m.finish == AOTX_TURN_LIMIT
        && put_element(t, m.agent, h->tick, "bound", NULL, 0u, NULL, 0u,
                       "limit", m.turn) != 0) {
        return -1;
    }
    if (m.finish == AOTX_TURN_STOPPED
        && put_element(t, m.agent, h->tick, "done", NULL, 0u, NULL, 0u,
                       "stopped", m.turn) != 0) {
        return -1;
    }
    if (m.finish == AOTX_TURN_REFUSED
        && put_element(t, m.agent, h->tick, "done", NULL, 0u, NULL, 0u,
                       "prompt_refused", m.turn) != 0) {
        return -1;
    }
    if (m.tool != AOTX_TOOL_NONE) {
        const aotx_transcript_request *known = find_request(t, m.request);
        if (known != NULL) {
            return 0;
        }
        memset(&request, 0, sizeof(request));
        request.agent = m.agent;
        request.turn = m.turn;
        request.tool = m.tool;
        request.request = m.request;
        remember_request(t, &request);
        return put_element(t, m.agent, h->tick, "call", NULL, 0u,
                           tool_name(m.tool, name, sizeof(name)), m.request,
                           (m.finish == AOTX_TURN_TOOL) ? "open" : "refused", m.turn);
    }
    return 0;
}

static int take_task(aotx_transcript *t, const aotx_record_header *h)
{
    aotx_task_body task;
    int who;
    const char *kind;
    const char *status;
    uint32_t len;
    if (h->body_len < sizeof(task) - AOTX_TASK_TEXT_BYTES) {
        t->refused++;
        return 0;
    }
    memset(&task, 0, sizeof(task));
    memcpy(&task, aotx_record_body(h),
           (h->body_len < sizeof(task)) ? h->body_len : sizeof(task));
    if (task.state != AOTX_TASK_DONE && task.state != AOTX_TASK_FAILED) {
        return 0;
    }
    who = agent_of_writer(h->writer);
    if (who < 0) {
        who = (task.agent < AOTX_TRANSCRIPT_AGENTS) ? (int)task.agent : -1;
    }
    if (who < 0) {
        t->refused++;
        return 0;
    }
    kind = (task.verify == AOTX_VERIFY_SIBLING && (uint32_t)who != task.agent)
         ? "verdict" : "done";
    status = (task.state == AOTX_TASK_DONE) ? "done" : "failed";
    len = (task.text_len > AOTX_TASK_TEXT_BYTES) ? AOTX_TASK_TEXT_BYTES : task.text_len;
    return put_element(t, (uint32_t)who, h->tick, kind,
                       (const unsigned char *)task.text, len, NULL, 0u, status,
                       t->agent[who].turn);
}

static int take_selection(aotx_transcript *t, const aotx_record_header *h)
{
    aotx_selection_body s;
    char text[768];
    size_t at;
    uint32_t i;
    if (h->body_len < sizeof(s)) {
        t->refused++;
        return 0;
    }
    memcpy(&s, aotx_record_body(h), sizeof(s));
    if (s.agent >= AOTX_TRANSCRIPT_AGENTS) {
        t->refused++;
        return 0;
    }
    if (s.count > AOTX_SELECTION_MAX) {
        s.count = AOTX_SELECTION_MAX;
    }
    at = (size_t)snprintf(text, sizeof(text), "pages %u summary %llu sequences", s.pages,
                          (unsigned long long)s.summary_seq);
    for (i = 0u; i < s.count && at + 32u < sizeof(text); i++) {
        at += (size_t)snprintf(text + at, sizeof(text) - at, "%s%llu", (i == 0u) ? " " : ",",
                               (unsigned long long)s.seq[i]);
    }
    t->agent[s.agent].turn = s.turn;
    return put_element(t, s.agent, h->tick, "selection", (const unsigned char *)text,
                       (uint32_t)at, NULL, 0u, "selected", s.turn);
}

static int take_summary(aotx_transcript *t, const aotx_record_header *h)
{
    aotx_bus_body b;
    int who = agent_of_writer(h->writer);
    uint32_t len;
    if (who < 0 || h->body_len < sizeof(b)) {
        return 0;
    }
    memcpy(&b, aotx_record_body(h), sizeof(b));
    if (b.kind != AOTX_BUS_FINDING || b.provenance != AOTX_PROV_COMPUTED) {
        return 0;
    }
    len = (b.text_len > AOTX_BUS_TEXT_BYTES) ? AOTX_BUS_TEXT_BYTES : b.text_len;
    return put_element(t, (uint32_t)who, h->tick, "summary",
                       (const unsigned char *)b.text, len, NULL, 0u, "computed",
                       t->agent[who].turn);
}

int aotx_transcript_open(aotx_transcript **out, const char *boot_dir)
{
    aotx_transcript *t;
    unsigned int i;
    char path[AOTX_PATH_BYTES];
    int used = snprintf(path, sizeof(path), "%s/transcript", boot_dir);
    if (used < 0 || (size_t)used >= sizeof(path) || aotx_make_dir(path) != 0) {
        return -1;
    }
    t = (aotx_transcript *)calloc(1u, sizeof(*t));
    if (t == NULL) {
        return -1;
    }
    snprintf(t->dir, sizeof(t->dir), "%s", path);
    for (i = 0u; i < AOTX_TRANSCRIPT_AGENTS; i++) {
        t->agent[i].fd = -1;
    }
    *out = t;
    return 0;
}

int aotx_transcript_block(aotx_transcript *t, const unsigned char *block)
{
    const aotx_block_header *bh = (const aotx_block_header *)block;
    uint32_t i;
    if (bh->kind != 0u) {
        return 0;
    }
    for (i = 0u; i < bh->record_count; i++) {
        const aotx_record_header *h = aotx_block_record(block, i);
        if (h->type != AOTX_REC_INPUT_LINE || (h->flags & AOTX_FLAG_FRAGMENT) == 0) {
            if (flush_input(t) != 0) {
                return -1;
            }
        }
        if (h->type != AOTX_REC_TOOL_REQUEST || (h->flags & AOTX_FLAG_FRAGMENT) == 0) {
            if (flush_request(t) != 0) {
                return -1;
            }
        }
        switch (h->type) {
        case AOTX_LIVE_RECORD:
            if (aotx_transcript_live_take(&t->live, h, put_live, t) != 0) return -1;
            break;
        case AOTX_REC_INPUT_LINE:   if (take_input(t, h) != 0) return -1; break;
        case AOTX_REC_TOKEN:        if (take_token(t, h) != 0) return -1; break;
        case AOTX_REC_TOOL_REQUEST: if (take_request(t, h) != 0) return -1; break;
        case AOTX_REC_TOOL_REPLY:   if (take_result(t, h) != 0) return -1; break;
        case AOTX_REC_MANIFEST:     if (take_manifest(t, h) != 0) return -1; break;
        case AOTX_REC_TASK:         if (take_task(t, h) != 0) return -1; break;
        case AOTX_REC_SELECTION:    if (take_selection(t, h) != 0) return -1; break;
        case AOTX_REC_BUS:          if (take_summary(t, h) != 0) return -1; break;
        default: break;
        }
    }
    return 0;
}

int aotx_transcript_sync(aotx_transcript *t)
{
    unsigned int i;
    for (i = 0u; i < AOTX_TRANSCRIPT_AGENTS; i++) {
        if (t->agent[i].fd >= 0 && fsync(t->agent[i].fd) != 0) {
            return -1;
        }
    }
    return 0;
}

void aotx_transcript_close(aotx_transcript *t)
{
    unsigned int i;
    if (t == NULL) {
        return;
    }
    flush_input(t);
    flush_request(t);
    flush_result(t);
    for (i = 0u; i < AOTX_TRANSCRIPT_AGENTS; i++) {
        flush_reply(t, i, t->agent[i].turn);
        if (t->agent[i].fd >= 0) {
            fsync(t->agent[i].fd);
            close(t->agent[i].fd);
        }
    }
    aotx_transcript_live_close(t->live);
    free(t);
}

uint64_t aotx_transcript_lines(const aotx_transcript *t)
{
    return (t != NULL) ? t->lines : 0u;
}
