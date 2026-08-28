/* Purpose: Turn a message record into one line of the message file, in the message schema.
 * Owns: Nothing; the caller holds the state structure and the map of message ids.
 * Threading: One thread; the drain calls this function in record order.
 * Lifetime: The call. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/drain/derive.h"

#include <stdio.h>
#include <string.h>

/* A record body carries one text. A kind with two text fields takes the part before the
 * first line feed and the part after it. The map gives the message id of a reference. */

#define AOTX_STATUS_DEFAULT "draft"
#define AOTX_PRODUCED_NONE  "not stated"

static const char *kind_name(uint8_t kind)
{
    static const char *names[7] = { "finding", "rank", "question", "answer",
                                    "handoff", "cost", "note" };
    return (kind >= 1u && kind <= 7u) ? names[kind - 1u] : NULL;
}

static const char *provenance_name(uint8_t value)
{
    static const char *names[4] = { "computed", "fetched", "recalled", "testimony" };
    return (value >= 1u && value <= 4u) ? names[value - 1u] : NULL;
}

/* The schema gives a handoff three states. A record that states none is not a statement
 * that the work is complete, so the line takes the first of the three. */
static const char *status_name(const char *text, size_t len)
{
    static const char *names[3] = { "draft", "ready", "blocked" };
    int i;
    for (i = 0; i < 3; i++) {
        if (strlen(names[i]) == len && memcmp(names[i], text, len) == 0) {
            return names[i];
        }
    }
    return AOTX_STATUS_DEFAULT;
}

/* Keeps one message in the map, so a later message can name it. The map holds the newest
 * entries only, and an older reference falls out of it. */
static void keep(aotx_derive *d, uint64_t record_seq, uint64_t msg_seq, uint32_t writer,
                 int rankable)
{
    aotx_ref *slot;
    if (d->refs == NULL || record_seq == 0) {
        return;
    }
    slot = &d->refs[record_seq & (AOTX_REF_SLOTS - 1u)];
    slot->record_seq = record_seq;
    slot->msg_seq = msg_seq;
    slot->writer = writer;
    slot->rankable = (uint8_t)(rankable ? 1 : 0);
}

/* Writes the message id of a record sequence. Returns 1 when the map holds it. */
static int find_id(const aotx_derive *d, uint64_t record_seq, char *out, size_t out_bytes,
                   int *rankable)
{
    const aotx_ref *slot;
    char name[AOTX_NAME_MAX];
    if (d->refs == NULL || record_seq == 0) {
        return 0;
    }
    slot = &d->refs[record_seq & (AOTX_REF_SLOTS - 1u)];
    if (slot->record_seq != record_seq) {
        return 0;
    }
    if (aotx_derive_agent(slot->writer, name, sizeof(name)) < 0) {
        return 0;
    }
    snprintf(out, out_bytes, "%s-%llu", name, (unsigned long long)slot->msg_seq);
    if (rankable != NULL) {
        *rankable = (int)slot->rankable;
    }
    return 1;
}

/* Adds one record sequence to the list of references that the map does not hold. */
static void add_gap(char *gaps, size_t gaps_bytes, uint64_t record_seq)
{
    size_t used = strlen(gaps);
    snprintf(gaps + used, gaps_bytes - used, "%s%llu", (used > 0) ? "," : "",
             (unsigned long long)record_seq);
}

typedef struct message {
    const aotx_bus_body *b;
    char name[AOTX_NAME_MAX];
    char first[AOTX_TEXT_MAX];   /* the text before the first line feed */
    char second[AOTX_TEXT_MAX];  /* the text after the first line feed */
    const unsigned char *tail;   /* the raw bytes of the second part */
    uint32_t tail_len;
    uint64_t seq;
    int slot;
} message;

/* Splits the text of the record and gives back the parts, with the special bytes escaped. */
static void split(message *m, const unsigned char *text, uint32_t len)
{
    uint32_t cut = 0;
    while (cut < len && text[cut] != '\n') {
        cut++;
    }
    aotx_derive_text(m->first, sizeof(m->first), text, cut);
    m->tail = (cut < len) ? text + cut + 1u : text;
    m->tail_len = (cut < len) ? len - cut - 1u : 0u;
    aotx_derive_text(m->second, sizeof(m->second), m->tail, m->tail_len);
}

/* Writes the body fields of one kind. Returns the count of bytes, or -1 when the schema
 * refuses the message. The name of the kind that the line takes goes to type_out. */
static int body_of(aotx_derive *d, const message *m, char *out, size_t out_bytes,
                   const char **type_out, char *gaps, size_t gaps_bytes, int *rankable)
{
    const aotx_bus_body *b = m->b;
    char id[AOTX_ID_MAX];
    int found;
    int target_rankable = 0;
    *type_out = kind_name(b->kind);
    *rankable = 0;
    if (*type_out == NULL || m->first[0] == '\0') {
        return -1;
    }
    found = find_id(d, b->re_seq, id, sizeof(id), &target_rankable);
    switch (b->kind) {
    case AOTX_BUS_FINDING: {
        const char *provenance = provenance_name(b->provenance);
        if (provenance == NULL) {
            return -1;
        }
        *rankable = 1;
        return snprintf(out, out_bytes, "\"id\":\"%s-%llu\",\"claim\":\"%s\",\"provenance\":\"%s\"",
                        m->name, (unsigned long long)m->seq, m->first, provenance);
    }
    case AOTX_BUS_RANK:
        if (!(b->score >= 0.0f && b->score <= 1.0f)) {
            return -1;
        }
        if (found && target_rankable) {
            return snprintf(out, out_bytes, "\"re\":\"%s\",\"score\":%.6f,\"basis\":\"%s\"",
                            id, (double)b->score, m->first);
        }
        /* A rank names a finding or a handoff that the file holds. Where the map does not
         * hold that record, the line states the text and the gap, and takes the note kind. */
        d->unresolved++;
        add_gap(gaps, gaps_bytes, b->re_seq);
        *type_out = "note";
        return snprintf(out, out_bytes, "\"text\":\"%s\",\"kind\":\"rank\",\"score\":%.6f",
                        m->first, (double)b->score);
    case AOTX_BUS_QUESTION:
        return snprintf(out, out_bytes, "\"text\":\"%s\"", m->first);
    case AOTX_BUS_ANSWER:
        if (found) {
            return snprintf(out, out_bytes, "\"re\":\"%s\",\"text\":\"%s\"", id, m->first);
        }
        d->unresolved++;
        add_gap(gaps, gaps_bytes, b->re_seq);
        *type_out = "note";
        return snprintf(out, out_bytes, "\"text\":\"%s\",\"kind\":\"answer\"", m->first);
    case AOTX_BUS_HANDOFF:
        *rankable = 1;
        return snprintf(out, out_bytes, "\"path\":\"%s\",\"status\":\"%s\"", m->first,
                        status_name((const char *)m->tail, m->tail_len));
    case AOTX_BUS_COST:
        return snprintf(out, out_bytes, "\"consumed\":\"%s\",\"produced\":\"%s\"", m->first,
                        (m->second[0] != '\0') ? m->second : AOTX_PRODUCED_NONE);
    default:
        return snprintf(out, out_bytes, "\"text\":\"%s\"", m->first);
    }
}

int aotx_derive_message(aotx_derive *d, const aotx_record_header *h, const unsigned char *body)
{
    aotx_bus_body b;
    message m;
    char line[AOTX_BUS_LINE_MAX];
    char fields[AOTX_BUS_LINE_MAX];
    char relation[AOTX_ID_MAX + 64];
    char gaps[64];
    char reason[AOTX_TEXT_MAX];
    char iso[48];
    const char *type = NULL;
    const char *request = "";
    uint64_t now = aotx_wall_ns();
    uint32_t text_len;
    int rankable = 0;
    int used;
    int count;
    int tail;

    memset(&m, 0, sizeof(m));
    memset(&b, 0, sizeof(b));
    relation[0] = '\0';
    reason[0] = '\0';
    gaps[0] = '\0';
    if (h->body_len < sizeof(b) - AOTX_BUS_TEXT_BYTES) {
        d->refused++;
        return 0;
    }
    memcpy(&b, body, (h->body_len < sizeof(b)) ? h->body_len : sizeof(b));
    text_len = b.text_len;
    if (text_len > AOTX_BUS_TEXT_BYTES) {
        text_len = AOTX_BUS_TEXT_BYTES;
    }
    if ((uint64_t)text_len + sizeof(b) - AOTX_BUS_TEXT_BYTES > h->body_len) {
        text_len = h->body_len - (uint32_t)(sizeof(b) - AOTX_BUS_TEXT_BYTES);
    }
    m.b = &b;
    m.slot = aotx_derive_agent(h->writer, m.name, sizeof(m.name));
    if (m.slot < 0 || b.writer_seq == 0) {
        d->refused++;
        return 0;
    }
    if (aotx_derive_stamp(d, iso, sizeof(iso), now) != 0) {
        return -1;
    }
    split(&m, (const unsigned char *)b.text, text_len);
    m.seq = aotx_derive_next(d, m.slot, b.writer_seq);
    count = body_of(d, &m, fields, sizeof(fields), &type, gaps, sizeof(gaps), &rankable);
    if (count < 0 || (size_t)count >= sizeof(fields)) {
        d->refused++;
        return 0;
    }

    /* A correction names the message that it replaces, and states why. The reason is the
     * text of this message, which is what the writer gives. */
    if (b.corrects_seq != 0) {
        char id[AOTX_ID_MAX];
        if (find_id(d, b.corrects_seq, id, sizeof(id), NULL)) {
            aotx_derive_text(reason, sizeof(reason), (const unsigned char *)b.text, text_len);
            snprintf(relation, sizeof(relation), ",\"corrects\":[\"%s\"]", id);
            request = ",\"req\":[\"msg-relations\"]";
        } else {
            d->unresolved++;
            add_gap(gaps, sizeof(gaps), b.corrects_seq);
        }
    }
    used = snprintf(line, sizeof(line),
                    "{\"v\":1,\"run\":\"aotx\",\"agent\":\"%s\",\"seq\":%llu,\"ts\":\"%s\"%s,"
                    "\"type\":\"%s\",\"body\":{%s%s",
                    m.name, (unsigned long long)m.seq, iso, request, type, fields, relation);
    if (used < 0 || (size_t)used + AOTX_TEXT_MAX + 128 >= sizeof(line)) {
        d->refused++;
        return 0;
    }
    if (relation[0] != '\0') {
        used += snprintf(line + used, sizeof(line) - (size_t)used, ",\"reason\":\"%s\"", reason);
    }
    if (gaps[0] != '\0') {
        used += snprintf(line + used, sizeof(line) - (size_t)used, ",\"unresolved\":\"%s\"", gaps);
    }
    if ((size_t)used + 80 >= sizeof(line)) {
        d->refused++;
        return 0;
    }
    tail = aotx_derive_tail(d, line + used, sizeof(line) - (size_t)used, h->tick, h->boot_id, now);
    if (tail < 0) {
        return -1;
    }
    keep(d, h->seq, m.seq, h->writer, rankable);
    d->messages++;
    return aotx_derive_put(d->bus_fd, line, (size_t)(used + tail));
}
