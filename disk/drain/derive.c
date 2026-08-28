/* Purpose: Derive the console log and the message lines from the blocks that the drain reads.
 * Owns: The open console file and the open message file of one drain.
 * Threading: One thread; the drain calls these functions in block order.
 * Lifetime: From open to close, which is the run of the drain. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/drain/derive.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define AOTX_READ_LINE  8192
#define AOTX_SYNC_NS    1000000000u

int aotx_derive_put(int fd, const char *data, size_t bytes)
{
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = write(fd, data + done, bytes - done);
        if (n <= 0) {
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

static void clock_parts(char *iso, size_t iso_bytes, char *day, size_t day_bytes, uint64_t ns)
{
    time_t seconds = (time_t)(ns / 1000000000u);
    unsigned ms = (unsigned)((ns % 1000000000u) / 1000000u) % 1000u;
    struct tm parts;
    char base[32];
    char zone[8];
    long offset;
    char sign;
    localtime_r(&seconds, &parts);
    strftime(base, sizeof(base), "%Y-%m-%dT%H:%M:%S", &parts);
    strftime(day, day_bytes, "%Y-%m-%d", &parts);
    offset = parts.tm_gmtoff;
    sign = (offset < 0) ? '-' : '+';
    if (offset < 0) {
        offset = -offset;
    }
    snprintf(zone, sizeof(zone), "%c%02d:%02d", sign, (int)(offset / 3600) % 100,
             (int)((offset % 3600) / 60) % 100);
    snprintf(iso, iso_bytes, "%s.%03u%s", base, ms, zone);
}

int aotx_derive_agent(uint32_t writer, char *name, size_t name_bytes)
{
    static const char *system_names[4] = { "system", "feeder", "restore", "console" };
    if (writer < 4u) {
        snprintf(name, name_bytes, "%s", system_names[writer]);
        return (int)writer;
    }
    if (writer >= AOTX_WRITER_AGENT_BASE &&
        writer - AOTX_WRITER_AGENT_BASE < (uint32_t)(AOTX_AGENT_SLOTS - 4)) {
        snprintf(name, name_bytes, "agent-%u", writer - AOTX_WRITER_AGENT_BASE);
        return (int)(4u + writer - AOTX_WRITER_AGENT_BASE);
    }
    return -1;
}

/* Gives the place of a name in the sequence table, or -1. This is the reverse of the name
 * rule, and it reads back the sequences that an earlier run of the drain wrote. */
static int slot_of_name(const char *name, size_t len)
{
    static const char *system_names[4] = { "system", "feeder", "restore", "console" };
    unsigned long number;
    char digits[8];
    int i;
    for (i = 0; i < 4; i++) {
        if (strlen(system_names[i]) == len && memcmp(system_names[i], name, len) == 0) {
            return i;
        }
    }
    if (len < 7 || len > 12 || memcmp(name, "agent-", 6) != 0) {
        return -1;
    }
    memcpy(digits, name + 6, len - 6);
    digits[len - 6] = '\0';
    number = strtoul(digits, NULL, 10);
    if (number >= (unsigned long)(AOTX_AGENT_SLOTS - 4)) {
        return -1;
    }
    return (int)(4u + number);
}

/* Reads back the sequences that the file already holds. A second run of the drain on one day
 * must not write a sequence that the file gives to that writer. */
static void seed_seq(aotx_derive *d, const char *path)
{
    char line[AOTX_READ_LINE];
    FILE *f;
    memset(d->next_seq, 0, sizeof(d->next_seq));
    f = fopen(path, "r");
    if (f == NULL) {
        return;
    }
    while (fgets(line, sizeof(line), f) != NULL) {
        const char *name = strstr(line, "\"agent\":\"");
        const char *at = strstr(line, "\"seq\":");
        const char *end;
        int slot;
        uint64_t value;
        if (name == NULL || at == NULL) {
            continue;
        }
        name += 9;
        end = strchr(name, '"');
        if (end == NULL) {
            continue;
        }
        slot = slot_of_name(name, (size_t)(end - name));
        value = (uint64_t)strtoull(at + 6, NULL, 10);
        if (slot >= 0 && value >= d->next_seq[slot]) {
            d->next_seq[slot] = value + 1;
        }
    }
    fclose(f);
}

uint64_t aotx_derive_next(aotx_derive *d, int slot, uint64_t writer_seq)
{
    uint64_t seq = writer_seq;
    if (seq < d->next_seq[slot]) {
        seq = d->next_seq[slot];
    }
    if (seq < 1) {
        seq = 1;
    }
    d->next_seq[slot] = seq + 1;
    return seq;
}

static int open_bus(aotx_derive *d, const char *day)
{
    char path[AOTX_PATH_BYTES + 32];
    if (d->bus_fd >= 0) {
        close(d->bus_fd);
        d->bus_fd = -1;
    }
    snprintf(path, sizeof(path), "%s/%s-aotx.jsonl", d->bus_dir, day);
    seed_seq(d, path);
    d->bus_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (d->bus_fd < 0) {
        return -1;
    }
    snprintf(d->bus_date, sizeof(d->bus_date), "%s", day);
    return 0;
}

int aotx_derive_stamp(aotx_derive *d, char *iso, size_t iso_bytes, uint64_t ns)
{
    char day[16];
    clock_parts(iso, iso_bytes, day, sizeof(day), ns);
    if (strcmp(day, d->bus_date) != 0) {
        return open_bus(d, day);
    }
    return 0;
}

int aotx_derive_tail(aotx_derive *d, char *out, size_t out_bytes, uint64_t tick,
                     uint64_t boot_id, uint64_t now)
{
    int used = snprintf(out, out_bytes, ",\"tick\":%llu,\"boot\":\"%016llx\",\"lag_ms\":",
                        (unsigned long long)tick, (unsigned long long)boot_id);
    if (used < 0 || (size_t)used + 40 >= out_bytes) {
        return -1;
    }
    if (d->tick_start_ns == 0) {
        used += snprintf(out + used, out_bytes - (size_t)used, "null}}\n");
    } else {
        double lag = ((double)now - (double)d->tick_start_ns) / 1000000.0;
        used += snprintf(out + used, out_bytes - (size_t)used, "%.3f}}\n", lag);
    }
    return used;
}

int aotx_derive_mask(const char *list, unsigned *out)
{
    static const char *names[5] = { "console", "note", "bus", "bulk", "sequence" };
    const char *at = list;
    unsigned mask = 0;
    if (strcmp(list, "none") == 0) {
        *out = 0;
        return 0;
    }
    while (*at != '\0') {
        size_t len = strcspn(at, ",");
        int i;
        int found = 0;
        for (i = 0; i < 5; i++) {
            if (strlen(names[i]) == len && memcmp(names[i], at, len) == 0) {
                mask |= 1u << i;
                found = 1;
            }
        }
        if (!found) {
            return -1;
        }
        at += len;
        if (*at == ',') {
            at++;
        }
    }
    *out = mask;
    return 0;
}

int aotx_derive_open(aotx_derive *d, const char *journal, const char *boot_dir, unsigned mask)
{
    char path[AOTX_PATH_BYTES + 32];
    char iso[48];
    char day[16];
    memset(d, 0, sizeof(*d));
    d->console_fd = -1;
    d->bus_fd = -1;
    d->echo_fd = 1;
    d->mask = mask;
    snprintf(path, sizeof(path), "%s/console.log", boot_dir);
    d->console_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (d->console_fd < 0) {
        return -1;
    }
    snprintf(d->bus_dir, sizeof(d->bus_dir), "%s/bus", journal);
    if (aotx_make_dir(d->bus_dir) != 0) {
        return -1;
    }
    if ((mask & AOTX_DERIVE_BUS) != 0) {
        d->refs = (aotx_ref *)calloc(AOTX_REF_SLOTS, sizeof(aotx_ref));
        if (d->refs == NULL) {
            return -1;
        }
    }
    clock_parts(iso, sizeof(iso), day, sizeof(day), aotx_wall_ns());
    return open_bus(d, day);
}

/* Ends the line that a console record left open. A run leaves whole lines behind, so the
 * end byte of the last line goes in when the drain closes the file. */
static int close_line(aotx_derive *d)
{
    if (!d->line_open) {
        return 0;
    }
    d->line_open = 0;
    if (d->echo_fd >= 0 && aotx_derive_put(d->echo_fd, "\n", 1) != 0) {
        return -1;
    }
    if (d->console_fd >= 0 && aotx_derive_put(d->console_fd, "\n", 1) != 0) {
        return -1;
    }
    return 0;
}

/* Writes the text of one console record to the console log and to the operator terminal.
 * A record with the fragment flag continues the line before it, so its bytes go in with no
 * end byte. A record without the flag ends that line and starts a new one. A reply that
 * comes in one record for each tick therefore reads as one line. */
static int write_console(aotx_derive *d, const aotx_record_header *h, const unsigned char *body,
                         uint32_t len)
{
    char line[AOTX_BODY_BYTES + 2];
    uint32_t i;
    if ((h->flags & AOTX_FLAG_FRAGMENT) == 0 && close_line(d) != 0) {
        return -1;
    }
    for (i = 0; i < len; i++) {
        unsigned char b = body[i];
        /* The drain writes the end byte of a line, so a control byte in a body becomes a
         * space. */
        line[i] = (b < 0x20u || b == 0x7fu) ? ' ' : (char)b;
    }
    d->lines++;
    d->line_open = 1;
    if (len == 0) {
        return 0;
    }
    if (aotx_derive_put(d->echo_fd, line, (size_t)len) != 0) {
        return -1;
    }
    return aotx_derive_put(d->console_fd, line, (size_t)len);
}

/* Writes one note line. The text must hold no character that a JSON string escapes, so a
 * caller that starts from record bytes puts them through aotx_derive_text first. */
static int put_note(aotx_derive *d, const aotx_record_header *h, int slot, const char *name,
                    const char *text)
{
    char line[AOTX_BUS_LINE_MAX];
    char iso[48];
    uint64_t now = aotx_wall_ns();
    int used;
    int tail;
    if (aotx_derive_stamp(d, iso, sizeof(iso), now) != 0) {
        return -1;
    }
    used = snprintf(line, sizeof(line),
                    "{\"v\":1,\"run\":\"aotx\",\"agent\":\"%s\",\"seq\":%llu,"
                    "\"ts\":\"%s\",\"type\":\"note\",\"body\":{\"text\":\"%s\"",
                    name, (unsigned long long)aotx_derive_next(d, slot, 0), iso, text);
    if (used < 0 || (size_t)used + 64 >= sizeof(line)) {
        return -1;
    }
    tail = aotx_derive_tail(d, line + used, sizeof(line) - (size_t)used, h->tick, h->boot_id, now);
    if (tail < 0) {
        return -1;
    }
    d->notes++;
    return aotx_derive_put(d->bus_fd, line, (size_t)(used + tail));
}

/* Writes one note line for a console record or a note record. The writer of the record
 * gives the name of the agent. A note of an agent does not read as a note of the console. */
static int write_note(aotx_derive *d, const aotx_record_header *h, const unsigned char *body)
{
    char text[AOTX_TEXT_MAX];
    char name[AOTX_NAME_MAX];
    int slot = aotx_derive_agent(h->writer, name, sizeof(name));
    if (slot < 0) {
        d->refused++;
        return 0;
    }
    if (!aotx_derive_has_text(body, h->body_len)) {
        /* The schema refuses a text field of white space only, and a reply gives such a
         * record whenever a token detokenizes to a space. */
        d->refused++;
        return 0;
    }
    if (aotx_derive_text(text, sizeof(text), body, h->body_len) == 0) {
        /* The schema refuses a required field that holds nothing. */
        d->refused++;
        return 0;
    }
    return put_note(d, h, slot, name, text);
}

/* Writes one line for the end of a sequence, so a reader of the line file sees where a
 * reply stopped. The line is a note of the system writer, because the event belongs to the
 * run and not to one agent. An open event and a release event make no line, because
 * neither is the end of a reply. */
static int write_sequence(aotx_derive *d, const aotx_record_header *h, const unsigned char *body)
{
    aotx_sequence_body seq;
    char text[AOTX_TEXT_MAX];
    char name[AOTX_NAME_MAX];
    const char *event;
    int slot = aotx_derive_agent(AOTX_WRITER_SYSTEM, name, sizeof(name));
    if (h->body_len < sizeof(seq)) {
        /* A body that is too short holds no counts, so the line would state nothing. */
        d->refused++;
        return 0;
    }
    memcpy(&seq, body, sizeof(seq));
    if (seq.event == AOTX_SEQ_DONE) {
        event = "done";
    } else if (seq.event == AOTX_SEQ_STOPPED) {
        event = "stopped";
    } else {
        return 0;
    }
    snprintf(text, sizeof(text), "sequence %s slot %u role %u prompt %u sampled %u ticks %llu",
             event, seq.slot, seq.role, seq.prompt_tokens, seq.sampled_tokens,
             (unsigned long long)seq.ticks);
    d->sequences++;
    return put_note(d, h, slot, name, text);
}

int aotx_derive_block(aotx_derive *d, const unsigned char *block)
{
    const aotx_block_header *bh = (const aotx_block_header *)block;
    uint32_t i;
    if (bh->kind != 0) {
        return 0;
    }
    for (i = 0; i < bh->record_count; i++) {
        const aotx_record_header *h = aotx_block_record(block, i);
        const unsigned char *body = aotx_record_body(h);
        if (h->type == AOTX_REC_TICK_START && h->body_len >= sizeof(aotx_clock_body)) {
            aotx_clock_body clock;
            memcpy(&clock, body, sizeof(clock));
            d->tick_start_ns = clock.wall_ns;
        } else if (h->type == AOTX_REC_CONSOLE && (d->mask & AOTX_DERIVE_CONSOLE) != 0) {
            if (write_console(d, h, body, h->body_len) != 0 || write_note(d, h, body) != 0) {
                return -1;
            }
        } else if (h->type == AOTX_REC_NOTE && (d->mask & AOTX_DERIVE_NOTE) != 0) {
            if (write_note(d, h, body) != 0) {
                return -1;
            }
        } else if (h->type == AOTX_REC_BUS && (d->mask & AOTX_DERIVE_BUS) != 0) {
            if (aotx_derive_message(d, h, body) != 0) {
                return -1;
            }
        } else if (h->type == AOTX_REC_SEQUENCE && (d->mask & AOTX_DERIVE_SEQUENCE) != 0) {
            /* A token record makes no line. The tokens of a reply stay in the journal
             * segments only. The text of the reply comes to the console log from the
             * console records. The device writes those records as the reply forms. */
            if (write_sequence(d, h, body) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

int aotx_derive_sync(aotx_derive *d, int force)
{
    uint64_t now = aotx_wall_ns();
    if (!force && now - d->sync_ns < AOTX_SYNC_NS) {
        return 0;
    }
    d->sync_ns = now;
    if (d->console_fd >= 0 && fsync(d->console_fd) != 0) {
        return -1;
    }
    if (d->bus_fd >= 0 && fsync(d->bus_fd) != 0) {
        return -1;
    }
    return 0;
}

void aotx_derive_close(aotx_derive *d)
{
    close_line(d);
    aotx_derive_sync(d, 1);
    if (d->console_fd >= 0) {
        close(d->console_fd);
    }
    if (d->bus_fd >= 0) {
        close(d->bus_fd);
    }
    free(d->refs);
    d->refs = NULL;
    d->console_fd = -1;
    d->bus_fd = -1;
}
