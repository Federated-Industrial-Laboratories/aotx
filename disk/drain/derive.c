/* Purpose: Derive the console log and the bus lines from the blocks that the drain reads.
 * Owns: The open console file and the open bus file of one drain.
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

#define AOTX_TEXT_BYTES 1280
#define AOTX_LINE_MAX   2048
#define AOTX_SYNC_NS    1000000000u

static int put_all(int fd, const char *data, size_t bytes)
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

/* Returns the length of a valid UTF-8 sequence at p, or zero. The check refuses an overlong
 * form, a surrogate, and a code point above the last one. Body bytes are untrusted. */
static int utf8_length(const unsigned char *p, uint32_t left)
{
    unsigned char b = p[0];
    int need;
    int i;
    if (b < 0x80u) {
        return 1;
    }
    if (b >= 0xc2u && b <= 0xdfu) {
        need = 1;
    } else if (b >= 0xe0u && b <= 0xefu) {
        need = 2;
    } else if (b >= 0xf0u && b <= 0xf4u) {
        need = 3;
    } else {
        return 0;
    }
    if ((uint32_t)need + 1u > left) {
        return 0;
    }
    for (i = 1; i <= need; i++) {
        if ((p[i] & 0xc0u) != 0x80u) {
            return 0;
        }
    }
    if (b == 0xe0u && p[1] < 0xa0u) {
        return 0;
    }
    if (b == 0xedu && p[1] >= 0xa0u) {
        return 0;
    }
    if (b == 0xf0u && p[1] < 0x90u) {
        return 0;
    }
    if (b == 0xf4u && p[1] >= 0x90u) {
        return 0;
    }
    return need + 1;
}

/* Writes the body as the content of a JSON string, without the quotation marks. A byte that
 * is not part of a valid sequence becomes a question mark, because a bus line must hold
 * valid UTF-8. Returns the count of bytes written. */
static size_t json_text(char *out, size_t out_bytes, const unsigned char *body, uint32_t len)
{
    size_t used = 0;
    uint32_t i = 0;
    while (i < len && used + 8 < out_bytes) {
        unsigned char b = body[i];
        int step;
        if (b == '"' || b == '\\') {
            out[used++] = '\\';
            out[used++] = (char)b;
            i++;
        } else if (b == '\n' || b == '\t' || b == '\r') {
            out[used++] = '\\';
            out[used++] = (b == '\n') ? 'n' : ((b == '\t') ? 't' : 'r');
            i++;
        } else if (b < 0x20u || b == 0x7fu) {
            used += (size_t)snprintf(out + used, out_bytes - used, "\\u%04x", b);
            i++;
        } else if (b < 0x80u) {
            out[used++] = (char)b;
            i++;
        } else {
            step = utf8_length(body + i, len - i);
            if (step == 0) {
                out[used++] = '?';
                i++;
            } else if (used + (size_t)step + 8 >= out_bytes) {
                break;
            } else {
                memcpy(out + used, body + i, (size_t)step);
                used += (size_t)step;
                i += (uint32_t)step;
            }
        }
    }
    out[used] = '\0';
    return used;
}

static void stamp(char *iso, size_t iso_bytes, char *day, size_t day_bytes, uint64_t ns)
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

/* Finds the sequence to continue from. A second run of the drain on one day must not write
 * a sequence that the file already holds. */
static uint64_t last_seq(const char *path)
{
    char line[AOTX_LINE_MAX];
    uint64_t best = 0;
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return 0;
    }
    while (fgets(line, sizeof(line), f) != NULL) {
        const char *at = strstr(line, "\"seq\":");
        if (at != NULL && strstr(line, "\"agent\":\"console\"") != NULL) {
            unsigned long long v = strtoull(at + 6, NULL, 10);
            if ((uint64_t)v > best) {
                best = (uint64_t)v;
            }
        }
    }
    fclose(f);
    return best;
}

static int open_bus(aotx_derive *d, const char *day)
{
    char path[AOTX_PATH_BYTES + 32];
    if (d->bus_fd >= 0) {
        close(d->bus_fd);
        d->bus_fd = -1;
    }
    snprintf(path, sizeof(path), "%s/%s-aotx.jsonl", d->bus_dir, day);
    d->bus_seq = last_seq(path) + 1;
    d->bus_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (d->bus_fd < 0) {
        return -1;
    }
    snprintf(d->bus_date, sizeof(d->bus_date), "%s", day);
    return 0;
}

int aotx_derive_open(aotx_derive *d, const char *journal, const char *boot_dir)
{
    char path[AOTX_PATH_BYTES + 32];
    char iso[48];
    char day[16];
    memset(d, 0, sizeof(*d));
    d->console_fd = -1;
    d->bus_fd = -1;
    d->echo_fd = 1;
    snprintf(path, sizeof(path), "%s/console.log", boot_dir);
    d->console_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (d->console_fd < 0) {
        return -1;
    }
    snprintf(d->bus_dir, sizeof(d->bus_dir), "%s/bus", journal);
    if (aotx_make_dir(d->bus_dir) != 0) {
        return -1;
    }
    stamp(iso, sizeof(iso), day, sizeof(day), aotx_wall_ns());
    return open_bus(d, day);
}

static int write_console(aotx_derive *d, const unsigned char *body, uint32_t len)
{
    char line[AOTX_BODY_BYTES + 2];
    uint32_t i;
    for (i = 0; i < len; i++) {
        unsigned char b = body[i];
        /* One record is one line, so a control byte inside a body becomes a space. */
        line[i] = (b < 0x20u || b == 0x7fu) ? ' ' : (char)b;
    }
    line[len] = '\n';
    d->lines++;
    if (put_all(d->echo_fd, line, (size_t)len + 1) != 0) {
        return -1;
    }
    return put_all(d->console_fd, line, (size_t)len + 1);
}

static int write_note(aotx_derive *d, const aotx_record_header *h, const unsigned char *body)
{
    char text[AOTX_TEXT_BYTES];
    char line[AOTX_LINE_MAX];
    char iso[48];
    char day[16];
    uint64_t now = aotx_wall_ns();
    int used;
    stamp(iso, sizeof(iso), day, sizeof(day), now);
    if (strcmp(day, d->bus_date) != 0 && open_bus(d, day) != 0) {
        return -1;
    }
    json_text(text, sizeof(text), body, h->body_len);
    used = snprintf(line, sizeof(line),
                    "{\"v\":1,\"run\":\"aotx\",\"agent\":\"console\",\"seq\":%llu,"
                    "\"ts\":\"%s\",\"type\":\"note\",\"body\":{\"text\":\"%s\","
                    "\"tick\":%llu,\"boot\":\"%016llx\",\"lag_ms\":",
                    (unsigned long long)d->bus_seq, iso, text,
                    (unsigned long long)h->tick, (unsigned long long)h->boot_id);
    if (used < 0 || (size_t)used + 40 >= sizeof(line)) {
        return -1;
    }
    if (d->tick_start_ns == 0) {
        used += snprintf(line + used, sizeof(line) - (size_t)used, "null}}\n");
    } else {
        double lag = ((double)now - (double)d->tick_start_ns) / 1000000.0;
        used += snprintf(line + used, sizeof(line) - (size_t)used, "%.3f}}\n", lag);
    }
    d->bus_seq++;
    d->notes++;
    return put_all(d->bus_fd, line, (size_t)used);
}

int aotx_derive_block(aotx_derive *d, const unsigned char *block)
{
    const aotx_block_header *bh = (const aotx_block_header *)block;
    uint32_t i;
    if (bh->kind == AOTX_BLOCK_PAD) {
        return 0;
    }
    for (i = 0; i < bh->record_count; i++) {
        const aotx_record_header *h = aotx_block_record(block, i);
        const unsigned char *body = aotx_record_body(h);
        if (h->type == AOTX_REC_TICK_START && h->body_len >= sizeof(aotx_clock_body)) {
            aotx_clock_body clock;
            memcpy(&clock, body, sizeof(clock));
            d->tick_start_ns = clock.wall_ns;
        } else if (h->type == AOTX_REC_CONSOLE) {
            if (write_console(d, body, h->body_len) != 0 || write_note(d, h, body) != 0) {
                return -1;
            }
        } else if (h->type == AOTX_REC_NOTE) {
            if (write_note(d, h, body) != 0) {
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
    aotx_derive_sync(d, 1);
    if (d->console_fd >= 0) {
        close(d->console_fd);
    }
    if (d->bus_fd >= 0) {
        close(d->bus_fd);
    }
    d->console_fd = -1;
    d->bus_fd = -1;
}
