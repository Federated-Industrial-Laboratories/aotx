/* Purpose: Read lines from the terminal and keys from a pipe into the inbound ring.
 * Owns: The partial line buffer, the partial key frame, and the head of the inbound ring.
 * Threading: One thread; the program is the only producer of the ring.
 * Lifetime: From the map of the ring to the exit of the program. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/feed/attach.h"
#include "disk/feed/import.h"
#include "disk/feed/line.h"
#include "disk/feed/modules.h"
#include "disk/feed/model_fetch.h"
#include "disk/feed/run_tool.h"
#include "disk/settings/settings.h"

#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define AOTX_TICK_NS  100000000u /* the tick start record goes out ten times a second */
#define AOTX_READ_MAX 4096

static volatile sig_atomic_t stop_flag;

/* The import state holds the bytes of two files. The children hold the output of every
 * program that runs. The table holds a row for each host tool. All three are large, so
 * they live beside the program and not on the stack of one. */
static aotx_import import_state;
static aotx_children children;
static aotx_modules module_table;

/* The socket that terminal programs attach to. It holds the connection of each terminal
 * and the mapped head of the mirror, so it lives beside the program too. */
static aotx_attach attach_state;
static aotx_fetch_child fetch_child;

static void on_signal(int number)
{
    (void)number;
    stop_flag = 1;
}

typedef struct feed_state {
    aotx_inbound_ring ring;
    aotx_fs_tool tool;
    unsigned char line[AOTX_INPUT_LINE_BYTES];
    unsigned char key[sizeof(aotx_key_body)];
    uint32_t fill;
    int line_long;    /* bytes are dropped through the next line feed */
    uint32_t key_fill; /* bytes of a key frame that a read did not complete */
    int keys_fd;
    int no_stdin;  /* a terminal owns the keyboard; the standard input is not read */
    uint64_t lines;
    uint64_t keys;
    uint64_t clocks;
    uint64_t settings;
    uint64_t refused_lines;
} feed_state;

/* Publishes one record, after a wait for a free slot. Returns 0, or -1 when the ring closed
 * or a signal arrived. */
static int publish(feed_state *s, uint8_t type, const void *body, uint32_t len)
{
    aotx_record_header h;
    memset(&h, 0, sizeof(h));
    /* The inbound preamble carries no boot identity, so the field stays zero. The device
     * stamps its own boot identity when it writes the record to the journal. */
    h.writer = AOTX_WRITER_FEEDER;
    h.cls = AOTX_CLASS_A;
    h.type = type;
    h.body_len = len;
    if (aotx_inbound_wait(&s->ring, &stop_flag) != 0) {
        return -1;
    }
    aotx_inbound_put(&s->ring, &h, body);
    return 0;
}

/* Publishes one line of the input. The feeder takes a line that asks for an import. The
 * import goes out and the line does not, so the device sees no import line from the
 * feeder. A refused directory gives one line of the standard error and one line that
 * states the refusal to the operator. */
static int flush_line(feed_state *s)
{
    char path[AOTX_WALK_BYTES];
    uint32_t len = s->fill;
    s->fill = 0;
    if (s->line_long != 0) {
        s->line_long = 0;
        return 0;
    }
    {
        int taken = aotx_fetch_child_line(&fetch_child, s->line, len, &s->ring, &stop_flag);
        if (taken != 0) {
            s->lines++;
            return (taken < 0) ? -1 : 0;
        }
    }
    if (aotx_import_line(s->line, len, path, sizeof(path))) {
        return aotx_import_take(&import_state, path, &s->ring, &stop_flag);
    }
    s->lines++;
    return aotx_line_publish(&s->ring, &stop_flag, s->line, len);
}

/* Collects bytes through the line feed. A complete line goes out in one contiguous group.
 * A line over the bound is dropped as a whole, so no tail becomes another command. */
static int take_bytes(feed_state *s, const unsigned char *data, size_t bytes)
{
    size_t i;
    for (i = 0; i < bytes; i++) {
        if (data[i] == '\n') {
            if (flush_line(s) != 0) {
                return -1;
            }
        } else if (s->line_long == 0 && s->fill < AOTX_INPUT_LINE_BYTES) {
            s->line[s->fill++] = data[i];
        } else if (s->line_long == 0) {
            s->line_long = 1;
            s->fill = 0;
            s->refused_lines++;
            fprintf(stderr, "feed: the input line is longer than %u bytes and was refused\n",
                    (unsigned int)AOTX_INPUT_LINE_BYTES);
        }
    }
    return 0;
}

/* Takes the bytes of the key pipe. A read gives any count of bytes. The state holds the
 * part of a frame that the read did not complete, and only a whole frame goes out. */
static int take_keys(feed_state *s, const unsigned char *data, size_t bytes)
{
    size_t i = 0;
    while (i < bytes) {
        size_t need = sizeof(s->key) - s->key_fill;
        size_t take = (bytes - i < need) ? bytes - i : need;
        memcpy(s->key + s->key_fill, data + i, take);
        s->key_fill += (uint32_t)take;
        i += take;
        if (s->key_fill == sizeof(s->key)) {
            s->key_fill = 0;
            s->keys++;
            if (publish(s, AOTX_REC_KEY, s->key, (uint32_t)sizeof(s->key)) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

/* Publishes one setting record for each device side number key that the settings file
 * names, in the order of the key list. The caller sends these before the first clock
 * record and before any line of the standard input. The device then holds the settings of
 * the run before one operator line.
 *
 * A file that is not there gives no record and no error. A refused line goes to the
 * standard error and the rest of the file applies. Returns 0, or -1 when the ring
 * closed. */
static int publish_settings(feed_state *s, const char *path)
{
    /* The table is large, so it lives beside the program and not on the stack. */
    static aotx_settings table;
    aotx_setting_body body;
    unsigned int i;
    /* The start reads the same file first and prints every refused line. A refusal
     * therefore reaches the operator one time; the feeder publishes and says nothing. */
    if (aotx_settings_read(path, &table) == 2) {
        return 0;
    }
    for (i = 0; i < AOTX_SETTING_NUMBER_COUNT; i++) {
        const char *name;
        size_t len;
        if (table.number_given[i] == 0u ||
            aotx_settings_number_side(i) != AOTX_SETTING_SIDE_DEVICE) {
            /* A key the file does not name keeps the value the device holds. A boot key
             * and a terminal key make no record. */
            continue;
        }
        name = aotx_settings_number_name(i);
        len = strlen(name);
        memset(&body, 0, sizeof(body));
        body.value = table.number[i];
        body.scale = (uint32_t)aotx_settings_number_scale(i);
        body.key_len = (uint32_t)len;
        memcpy(body.key, name, len);
        if (publish(s, AOTX_REC_SETTING, &body, (uint32_t)sizeof(body)) != 0) {
            return -1;
        }
        s->settings++;
    }
    return 0;
}

static void usage(void)
{
    fprintf(stderr, "usage: aotx_feed --inbound-fd <fd> [--ready-fd <fd>] [--keys-fd <fd>]"
                    " [--root <dir> --requests <file>] [--settings <file>]"
                    " [--modules <dir>] [--timeout <seconds>]"
                    " [--attach <dir>] [--mirror-fd <fd>]\n");
    fprintf(stderr, "  --root      the one directory a file read may reach\n");
    fprintf(stderr, "  --requests  the file of tool requests that the journal gains\n");
    fprintf(stderr, "  --settings  the settings file that the device applies at the start\n");
    fprintf(stderr, "  --modules   the directory of module directories to import at the"
                    " start\n");
    fprintf(stderr, "  --timeout   the seconds a built-in tool may run a command\n");
    fprintf(stderr, "  --attach    the directory that holds the socket of the terminals\n");
    fprintf(stderr, "  --mirror-fd the mirror that each terminal reads\n");
}

/* Gives an attached terminal line to the same path as a standard input line. */
static int take_attached_fetch(void *context, const unsigned char *line, uint32_t bytes)
{
    feed_state *s = (feed_state *)context;
    int taken = aotx_fetch_child_line(&fetch_child, line, bytes, &s->ring, &stop_flag);
    if (taken != 0) {
        s->lines++;
    }
    return taken;
}

static int run(feed_state *s)
{
    uint64_t next_clock = aotx_wall_ns() + AOTX_TICK_NS;
    int at_end = s->no_stdin;
    int keys_at_end = (s->keys_fd < 0);
    while (stop_flag == 0) {
        struct pollfd fds[2 + AOTX_ATTACH_MAX + 1u];
        unsigned int attached;
        uint64_t now = aotx_wall_ns();
        int wait_ms;
        int ready;
        if (aotx_inbound_closed(&s->ring)) {
            return AOTX_EXIT_OK;
        }
        if (aotx_fetch_child_poll(&fetch_child, &s->ring, &stop_flag) != 0) {
            return AOTX_EXIT_OK;
        }
        /* The requests file is read at each turn of the loop, so a request waits at most
         * one clock period for its reply. */
        if (aotx_fs_tool_poll(&s->tool, &import_state, &s->ring, &stop_flag) < 0) {
            return AOTX_EXIT_OK;
        }
        if (now >= next_clock) {
            aotx_clock_body clock;
            clock.wall_ns = now;
            if (publish(s, AOTX_REC_TICK_START, &clock, sizeof(clock)) != 0) {
                return AOTX_EXIT_OK;
            }
            s->clocks++;
            next_clock += AOTX_TICK_NS;
            if (next_clock < now) {
                next_clock = now + AOTX_TICK_NS;
            }
            continue;
        }
        wait_ms = (int)((next_clock - now + 999999u) / 1000000u);
        fds[0].fd = at_end ? -1 : 0;
        fds[1].fd = keys_at_end ? -1 : s->keys_fd;
        fds[0].events = POLLIN;
        fds[1].events = POLLIN;
        fds[0].revents = 0;
        fds[1].revents = 0;
        attached = aotx_attach_poll_set(&attach_state, fds + 2,
                                        (unsigned int)(sizeof(fds) / sizeof(fds[0])) - 2u);
        ready = poll(fds, (nfds_t)(2u + attached), wait_ms);
        if (ready < 0) {
            /* A signal breaks the wait. The loop reads the stop flag at the top. */
            continue;
        }
        if (ready > 0 && (fds[0].revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
            unsigned char buffer[AOTX_READ_MAX];
            ssize_t n = read(0, buffer, sizeof(buffer));
            if (n > 0) {
                if (take_bytes(s, buffer, (size_t)n) != 0) {
                    return AOTX_EXIT_OK;
                }
            } else if (n == 0) {
                /* The end of the input is not the end of the run. The clock records go on
                 * until the ring closes or a signal arrives. */
                at_end = 1;
                if ((s->fill > 0 || s->line_long != 0) && flush_line(s) != 0) {
                    return AOTX_EXIT_OK;
                }
            }
        }
        if (ready > 0 && (fds[1].revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
            unsigned char buffer[AOTX_READ_MAX];
            ssize_t n = read(s->keys_fd, buffer, sizeof(buffer));
            if (n > 0) {
                if (take_keys(s, buffer, (size_t)n) != 0) {
                    return AOTX_EXIT_OK;
                }
            } else if (n == 0) {
                keys_at_end = 1;
            }
        }
        /* The socket wakes this loop. A key or a line of a terminal thus reaches the
         * ring within a millisecond, and not at the clock period. */
        if (ready > 0
            && aotx_attach_take(&attach_state, fds + 2, attached, &s->ring, &stop_flag) != 0) {
            return AOTX_EXIT_OK;
        }
    }
    return AOTX_EXIT_OK;
}

int main(int argc, char **argv)
{
    aotx_map map;
    feed_state s;
    struct sigaction act;
    const char *root = NULL;
    const char *requests = NULL;
    const char *settings = NULL;
    const char *modules = NULL;
    const char *attach = NULL;
    uint32_t timeout = 0;
    int inbound_fd = -1;
    int keys_fd = -1;
    int no_stdin = 0;
    int mirror_fd = -1;
    int ready_fd = -1;
    int i;
    int rc;
    aotx_settings feed_settings;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--inbound-fd") == 0 && i + 1 < argc) {
            inbound_fd = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ready-fd") == 0 && i + 1 < argc) {
            ready_fd = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--keys-fd") == 0 && i + 1 < argc) {
            keys_fd = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-stdin") == 0) {
            no_stdin = 1;
        } else if (strcmp(argv[i], "--root") == 0 && i + 1 < argc) {
            root = argv[++i];
        } else if (strcmp(argv[i], "--requests") == 0 && i + 1 < argc) {
            requests = argv[++i];
        } else if (strcmp(argv[i], "--settings") == 0 && i + 1 < argc) {
            settings = argv[++i];
        } else if (strcmp(argv[i], "--modules") == 0 && i + 1 < argc) {
            modules = argv[++i];
        } else if (strcmp(argv[i], "--timeout") == 0 && i + 1 < argc) {
            timeout = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--attach") == 0 && i + 1 < argc) {
            attach = argv[++i];
        } else if (strcmp(argv[i], "--mirror-fd") == 0 && i + 1 < argc) {
            mirror_fd = atoi(argv[++i]);
        } else {
            usage();
            return AOTX_EXIT_FAULT;
        }
    }
    if (inbound_fd < 0 || (root == NULL) != (requests == NULL)) {
        /* A root with no requests file reads nothing, and a requests file with no root has
         * no boundary to read under. */
        usage();
        return AOTX_EXIT_FAULT;
    }

    memset(&s, 0, sizeof(s));
    s.keys_fd = keys_fd;
    s.no_stdin = no_stdin;
    aotx_settings_defaults(&feed_settings);
    if (settings != NULL) {
        (void)aotx_settings_read(settings, &feed_settings);
    }
    aotx_fetch_child_init(&fetch_child, feed_settings.text[AOTX_SET_MODELS_DIR]);
    if (aotx_modules_open(&module_table, requests) != 0) {
        fprintf(stderr, "feed: the module table does not open\n");
        return AOTX_EXIT_FAULT;
    }
    /* The import count goes on from the highest number the table holds. A feeder that
     * starts after a crash thus gives no module a number the journal already holds. */
    import_state.table = &module_table;
    import_state.number = aotx_modules_high(&module_table);
    s.tool.kids = &children;
    s.tool.table = &module_table;
    s.tool.timeout = timeout;
    if (aotx_fs_tool_open(&s.tool, root, requests) != 0) {
        fprintf(stderr, "feed: the allowed root does not open\n");
        return AOTX_EXIT_FAULT;
    }
    memset(&act, 0, sizeof(act));
    act.sa_handler = on_signal;
    sigaction(SIGTERM, &act, NULL);
    sigaction(SIGINT, &act, NULL);
    if (aotx_die_with_parent() != 0) {
        fprintf(stderr, "feed: the parent death signal is not set\n");
    }
    if (aotx_map_fd(inbound_fd, &map) != 0) {
        fprintf(stderr, "feed: the ring descriptor does not map\n");
        return AOTX_EXIT_FAULT;
    }
    if (aotx_inbound_attach(&map, &s.ring) != 0) {
        fprintf(stderr, "feed: the ring preamble does not match this layout version\n");
        return AOTX_EXIT_LAYOUT;
    }
    if (aotx_attach_open(&attach_state, attach, mirror_fd) != 0) {
        aotx_modules_close(&module_table);
        aotx_fs_tool_close(&s.tool);
        aotx_map_release(&map);
        return AOTX_EXIT_FAULT;
    }
    attach_state.line_take = take_attached_fetch;
    attach_state.line_context = &s;
    if (ready_fd >= 0) {
        const char mark = 'R';
        if (write(ready_fd, &mark, 1u) != 1) {
            fprintf(stderr, "feed: the start status does not go out\n");
            aotx_attach_close(&attach_state);
            aotx_modules_close(&module_table);
            aotx_fs_tool_close(&s.tool);
            aotx_map_release(&map);
            close(ready_fd);
            return AOTX_EXIT_FAULT;
        }
        close(ready_fd);
    }

    if (settings != NULL && publish_settings(&s, settings) != 0) {
        fprintf(stderr, "feed: the ring closed before the settings went out\n");
        aotx_modules_close(&module_table);
        aotx_fs_tool_close(&s.tool);
        aotx_map_release(&map);
        return AOTX_EXIT_OK;
    }

    /* The modules go out after the settings and before the first line of the standard
     * input. The device thus holds the catalog of the run before one operator line. A
     * restore gives no module directory, because the journal holds every import. */
    if (modules != NULL && aotx_import_tree(&import_state, modules, &s.ring, &stop_flag) != 0) {
        fprintf(stderr, "feed: the ring closed before the modules went out\n");
        aotx_modules_close(&module_table);
        aotx_fs_tool_close(&s.tool);
        aotx_map_release(&map);
        return AOTX_EXIT_OK;
    }

    rc = run(&s);
    fprintf(stderr, "feed: terminals %llu joined, %llu left, keys %llu, lines %llu,"
                    " refused %llu\n",
            (unsigned long long)attach_state.joined, (unsigned long long)attach_state.left,
            (unsigned long long)attach_state.keys, (unsigned long long)attach_state.lines,
            (unsigned long long)attach_state.refused);
    fprintf(stderr, "feed: lines %llu, keys %llu, clocks %llu, settings %llu,"
                    " refused lines %llu\n",
            (unsigned long long)s.lines, (unsigned long long)s.keys,
            (unsigned long long)s.clocks, (unsigned long long)s.settings,
            (unsigned long long)s.refused_lines);
    fprintf(stderr, "feed: requests %llu, replies %llu, refused %llu, errors %llu,"
                    " already answered %llu, import requests %llu\n",
            (unsigned long long)s.tool.taken, (unsigned long long)s.tool.replies,
            (unsigned long long)s.tool.refusals, (unsigned long long)s.tool.errors,
            (unsigned long long)s.tool.again, (unsigned long long)s.tool.imports);
    fprintf(stderr, "feed: imports %llu, import records %llu, modules refused %llu,"
                    " refusal lines %llu\n",
            (unsigned long long)import_state.imports,
            (unsigned long long)import_state.records,
            (unsigned long long)import_state.refusals,
            (unsigned long long)import_state.lines);
    fprintf(stderr, "feed: programs %llu, ended %llu, ended by the timeout %llu,"
                    " tool rows %u\n",
            (unsigned long long)children.started, (unsigned long long)children.ended,
            (unsigned long long)children.killed, module_table.count);
    aotx_attach_close(&attach_state);
    aotx_fetch_child_close(&fetch_child);
    aotx_modules_close(&module_table);
    aotx_fs_tool_close(&s.tool);
    aotx_map_release(&map);
    return rc;
}
