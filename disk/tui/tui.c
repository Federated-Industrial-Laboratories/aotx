/* Purpose: Run the terminal program: the options, the poll loop and the draw of each frame.
 * Owns: The whole state of the program, which lives beside it and not on a stack.
 * Threading: One thread; the program waits in one poll and holds no lock.
 * Lifetime: From the start of the program to its exit. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/tui/tui.h"

#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* The frames a second the terminal draws. The mirror is published at its own rate; the
 * terminal reads what is there and never waits for a frame. */
#define AOTX_TUI_FRAME_MS  33

/* The wait between one try to attach and the next, after a start. */
#define AOTX_TUI_ATTACH_MS 200

/* The bytes one read takes from the terminal, and the keys they may decode to. */
#define AOTX_TUI_READ      1024u

/* The state is large, so the program holds it beside itself. */
static aotx_tui aotx_state;
static aotx_tui_key aotx_key_run[AOTX_TUI_READ];

static void usage(void)
{
    fprintf(stderr, "usage: aotx_tui [--attach <journal>]... [--journal <dir>]"
                    " [--settings <file>] [--no-splash]\n");
    fprintf(stderr, "  --attach    a journal directory; repeat for several systems\n");
    fprintf(stderr, "  --journal   the journal directory a start uses\n");
    fprintf(stderr, "  --settings  the settings file to show and to start with\n");
    fprintf(stderr, "  --no-splash open the frame with no splash\n");
}

/* Finds the boot program beside this one, from the path of this program. */
static void find_program(aotx_tui *tui)
{
    char self[AOTX_PATH_BYTES];
    ssize_t got = readlink("/proc/self/exe", self, sizeof(self) - 1u);
    char *end;
    if (got <= 0) {
        snprintf(tui->program, sizeof(tui->program), "aotx_boot");
        return;
    }
    self[got] = '\0';
    end = strrchr(self, '/');
    if (end == NULL) {
        snprintf(tui->program, sizeof(tui->program), "aotx_boot");
        return;
    }
    *end = '\0';
    aotx_tui_join(tui->program, sizeof(tui->program), self, "aotx_boot");
}

/* Reads the settings file and takes the keys of the terminal. */
static void read_settings(aotx_tui *tui)
{
    aotx_settings_read(tui->settings_path, &tui->settings);
    tui->color = (strcmp(tui->settings.text[AOTX_SET_TUI_COLOR], "16") == 0) ? 1 : 0;
    tui->utf8_box = (strcmp(tui->settings.text[AOTX_SET_TUI_BOX], "utf8") == 0) ? 1 : 0;
    tui->keys.escape_ms = (unsigned int)tui->settings.number[AOTX_SET_TUI_ESCAPE_MS];
}

/* Reads the splash art for the size the terminal has now. */
static void read_splash(aotx_tui *tui)
{
    char dir[AOTX_PATH_BYTES];
    char self[AOTX_PATH_BYTES];
    const char *from = getenv("AOTX_SPLASH_DIR");
    ssize_t got;
    if (tui->no_splash != 0) {
        memset(&tui->splash, 0, sizeof(tui->splash));
        snprintf(tui->splash.reason, sizeof(tui->splash.reason), "the splash is off");
        return;
    }
    if (from != NULL && from[0] != '\0') {
        snprintf(dir, sizeof(dir), "%.*s", (int)sizeof(dir) - 1, from);
    } else {
        char *end;
        got = readlink("/proc/self/exe", self, sizeof(self) - 1u);
        self[(got > 0) ? got : 0] = '\0';
        end = strrchr(self, '/');
        if (end != NULL) {
            *end = '\0';
        }
        aotx_tui_join(dir, sizeof(dir), self, "../share/splash");
    }
    aotx_splash_read(&tui->splash, dir, tui->settings.text[AOTX_SET_TUI_SPLASH],
                     aotx_paint_view_cols(&tui->paint), aotx_paint_view_rows(&tui->paint));
    if (tui->splash.held == 0 && tui->splash.reason[0] != '\0') {
        snprintf(tui->says, sizeof(tui->says), "%.200s", tui->splash.reason);
    }
}

/* Takes one key while no screen is open. */
static void console_key(aotx_tui *tui, const aotx_tui_key *key)
{
    if (key->code == 0 && key->codepoint == 0x0cu
        && (key->mods & AOTX_TUI_MOD_CONTROL) != 0) {
        tui->paint.full = 1;
        aotx_paint_follow(&tui->paint);
        return;
    }
    if (key->code == 0 && key->codepoint == 0x03u
        && (key->mods & AOTX_TUI_MOD_CONTROL) != 0) {
        tui->screen = 9u;  /* the Quit screen */
        tui->cursor = 0;
        tui->top = 0;
        return;
    }
    if (key->code == AOTX_TUI_KEY_ESCAPE) {
        tui->screen = 1u;  /* the Menu screen */
        tui->cursor = 0;
        tui->top = 0;
        return;
    }
    if ((key->mods & AOTX_TUI_MOD_ALT) != 0) {
        switch (key->code) {
        case AOTX_TUI_KEY_UP:    aotx_paint_pan(&tui->paint, -1, 0); return;
        case AOTX_TUI_KEY_DOWN:  aotx_paint_pan(&tui->paint, 1, 0); return;
        case AOTX_TUI_KEY_LEFT:  aotx_paint_pan(&tui->paint, 0, -1); return;
        case AOTX_TUI_KEY_RIGHT: aotx_paint_pan(&tui->paint, 0, 1); return;
        default: break;
        }
    }
    if (tui->session.fd < 0) {
        if (key->code != 0 || key->codepoint >= 32u) {
            snprintf(tui->says, sizeof(tui->says), "no system runs, F9 to start");
        }
        return;
    }
    /* The console keys go as the window's frames, so the device editor sees no difference
     * between the two surfaces. A byte above the font is dropped, as the window drops it. */
    if (key->code == 0 && (key->codepoint < 32u || key->codepoint > 126u)) {
        return;
    }
    if (aotx_session_key(&tui->session, key) != 0) {
        snprintf(tui->says, sizeof(tui->says), "the key did not go out");
    } else if (key->code == AOTX_TUI_KEY_ENTER) {
        aotx_paint_follow(&tui->paint);
    }
}

/* Takes one key of the whole program. */
static void take_key(aotx_tui *tui, const aotx_tui_key *key)
{
    unsigned int screen = aotx_screen_of_key(key);
    tui->socket_closed = 0;
    tui->says[0] = '\0';
    if (screen != AOTX_TUI_SCREEN_NONE) {
        /* The key of the open screen closes it, so one key opens and closes. */
        tui->screen = (tui->screen == screen) ? AOTX_TUI_SCREEN_NONE : screen;
        tui->cursor = 0;
        tui->top = 0;
        tui->editing = 0;
        return;
    }
    if (aotx_screen_key(tui, key) != 0) {
        return;
    }
    console_key(tui, key);
}

/* Reads the mirror and keeps the snapshot when a whole one came back. */
static void take_frame(aotx_tui *tui, uint64_t now)
{
    aotx_mirror_snapshot shot;
    if (tui->session.mirror == NULL) {
        return;
    }
    if (aotx_mirror_take(tui->session.mirror, &shot) == 0) {
        return;
    }
    if (tui->have_shot != 0 && shot.head.sequence == tui->shot_sequence) {
        return;
    }
    /* A gap of more than one frame means the terminal lost frames. The next write is
     * then a whole frame, and not a difference against a picture that is old. */
    if (tui->have_shot != 0 && shot.head.sequence > tui->shot_sequence + 1u) {
        tui->paint.full = 1;
    }
    if (tui->have_shot == 0) {
        aotx_splash_dissolve_open(&tui->splash, tui->splash.cols, tui->splash.rows,
                                  AOTX_TUI_DISSOLVE);
        tui->dissolve = 1;
    }
    memcpy(&tui->shot, &shot, sizeof(shot));
    tui->shot_sequence = shot.head.sequence;
    tui->shot_ns = now;
    tui->have_shot = 1;
    tui->frames++;
}

/* Tries to attach to the system of the journal directory. */
static void try_attach(aotx_tui *tui)
{
    if (tui->session.fd >= 0 || tui->journal[0] == '\0') {
        return;
    }
    if (aotx_session_attach(&tui->session, tui->journal) == 0) {
        snprintf(tui->state, sizeof(tui->state), "connected");
        tui->socket_closed = 0;
        tui->says[0] = '\0';
        return;
    }
    {
        uint64_t seconds = aotx_wall_ns() / 1000000000ull;
        int phase = aotx_session_phase(tui->journal, seconds, tui->state,
                                       sizeof(tui->state));
        if (phase == 0) {
            snprintf(tui->state, sizeof(tui->state), "no system runs, F9 to start");
        } else if (phase < 0) {
            snprintf(tui->state, sizeof(tui->state), "the system state does not read");
        }
    }
}

static int run(aotx_tui *tui)
{
    uint64_t next_frame = aotx_wall_ns();
    uint64_t next_attach = aotx_wall_ns();
    while (tui->quit == 0) {
        struct pollfd fds[3];
        uint64_t now = aotx_wall_ns();
        int wait_ms = AOTX_TUI_FRAME_MS;
        int escape_ms;
        int count = 2;
        int ready;
        int resized = 0;
        int child = 0;

        fds[0].fd = tui->term.in_fd;
        fds[0].events = POLLIN;
        fds[0].revents = 0;
        fds[1].fd = tui->term.wake[0];
        fds[1].events = POLLIN;
        fds[1].revents = 0;
        if (tui->session.fd >= 0) {
            fds[2].fd = tui->session.fd;
            fds[2].events = POLLIN;
            fds[2].revents = 0;
            count = 3;
        }
        escape_ms = aotx_keys_timeout(&tui->keys, now);
        if (escape_ms >= 0 && escape_ms < wait_ms) {
            wait_ms = escape_ms;
        }
        ready = poll(fds, (nfds_t)count, wait_ms);
        now = aotx_wall_ns();
        if (ready > 0 && (fds[0].revents & (POLLIN | POLLHUP)) != 0) {
            unsigned char buffer[AOTX_TUI_READ];
            ssize_t got = read(tui->term.in_fd, buffer, sizeof(buffer));
            if (got > 0) {
                unsigned int keys = aotx_keys_take(&tui->keys, buffer, (size_t)got, now,
                                                   aotx_key_run, AOTX_TUI_READ);
                unsigned int i;
                for (i = 0; i < keys; i++) {
                    take_key(tui, &aotx_key_run[i]);
                }
            } else if (got == 0) {
                /* The terminal went away. The program leaves the system running. */
                tui->quit = 1;
                tui->quit_reason = "the input closed";
            }
        }
        {
            aotx_tui_key late;
            if (aotx_keys_wait(&tui->keys, now, &late) != 0) {
                take_key(tui, &late);
            }
        }
        if (ready > 0 && count == 3 && (fds[2].revents & (POLLIN | POLLHUP)) != 0) {
            int state = aotx_session_take(&tui->session);
            if (state < 0) {
                aotx_session_detach(&tui->session);
                tui->have_shot = 0;
                tui->paint.full = 1;
                tui->socket_closed = 1;
                snprintf(tui->says, sizeof(tui->says), "the system closed the socket");
            } else if (state > 0) {
                snprintf(tui->says, sizeof(tui->says), "%.200s",
                         tui->session.reason);
            }
        }
        if (aotx_term_signals(&tui->term, &resized, &child) != 0) {
            if (tui->term.ended != 0) {
                tui->quit = 1;
                tui->quit_reason = "a signal ended the program";
            }
        }
        if (resized != 0) {
            if (aotx_term_size(&tui->term) != 0) {
                aotx_paint_size(&tui->paint, tui->term.cols, tui->term.rows);
                read_splash(tui);
                aotx_term_clear(&tui->term);
            }
        }
        aotx_models_poll(tui);
        if (tui->session.fd < 0 && now >= next_attach) {
            try_attach(tui);
            next_attach = now + ((uint64_t)AOTX_TUI_ATTACH_MS * 1000000ull);
        }
        take_frame(tui, now);
        if (now >= next_frame) {
            aotx_frame_draw(tui);
            aotx_paint_flush(&tui->paint, &tui->term, tui->color);
            next_frame = now + ((uint64_t)AOTX_TUI_FRAME_MS * 1000000ull);
        }
    }
    return AOTX_EXIT_OK;
}

int main(int argc, char **argv)
{
    aotx_tui *tui = &aotx_state;
    const char *attach[AOTX_TUI_SYSTEMS];
    unsigned int attaches = 0u;
    int i;
    int rc;

    memset(tui, 0, sizeof(*tui));
    tui->session.fd = -1;
    tui->session.mirror_fd = -1;
    tui->session.boot_pid = -1;
    tui->model_pid = -1;
    tui->model_fd = -1;
    tui->screen = AOTX_TUI_SCREEN_NONE;
    for (i = 0; i < (int)(AOTX_TUI_SYSTEMS - 1u); i++) {
        tui->other[i].session.fd = -1;
        tui->other[i].session.mirror_fd = -1;
        tui->other[i].session.boot_pid = -1;
    }
    snprintf(tui->settings_path, sizeof(tui->settings_path), "%s",
             AOTX_SETTINGS_FILE_DEFAULT);
    snprintf(tui->state, sizeof(tui->state), "no system runs, F9 to start");

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--attach") == 0 && i + 1 < argc) {
            if (attaches >= AOTX_TUI_SYSTEMS) {
                fprintf(stderr, "aotx_tui: at most %u systems can attach\n",
                        (unsigned int)AOTX_TUI_SYSTEMS);
                return AOTX_EXIT_FAULT;
            }
            attach[attaches++] = argv[++i];
        } else if (strcmp(argv[i], "--journal") == 0 && i + 1 < argc) {
            snprintf(tui->journal, sizeof(tui->journal), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--settings") == 0 && i + 1 < argc) {
            snprintf(tui->settings_path, sizeof(tui->settings_path), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--no-splash") == 0) {
            tui->no_splash = 1;
        } else {
            usage();
            return AOTX_EXIT_FAULT;
        }
    }
    aotx_settings_defaults(&tui->settings);
    read_settings(tui);
    find_program(tui);
    if (aotx_session_version(tui->program, tui->version, sizeof(tui->version)) != 0) {
        snprintf(tui->version, sizeof(tui->version), "the build does not answer");
    }
    if (attaches != 0u) {
        snprintf(tui->journal, sizeof(tui->journal), "%s", attach[0]);
    }
    if (tui->journal[0] == '\0') {
        snprintf(tui->journal, sizeof(tui->journal), "%.*s",
                 (int)sizeof(tui->journal) - 1, tui->settings.text[AOTX_SET_JOURNAL_DIR]);
    }
    if (aotx_term_open(&tui->term, 0, 1) != 0) {
        fprintf(stderr, "aotx_tui: the standard input is not a terminal\n");
        return AOTX_EXIT_FAULT;
    }
    aotx_paint_size(&tui->paint, tui->term.cols, tui->term.rows);
    read_splash(tui);
    if (attaches != 0u) {
        try_attach(tui);
        for (i = 1; i < (int)attaches; i++) {
            aotx_tui_system *system = &tui->other[tui->other_count];
            if (aotx_session_attach(&system->session, attach[i]) == 0) {
                system->card = tui->other_count + 1u;
                tui->other_count++;
            } else {
                snprintf(tui->says, sizeof(tui->says), "card %d: %.180s", i + 1,
                         system->session.reason);
            }
        }
    }
    rc = run(tui);
    aotx_term_close(&tui->term);
    aotx_models_close(tui);
    aotx_session_detach(&tui->session);
    for (i = 0; i < (int)tui->other_count; i++) {
        aotx_session_detach(&tui->other[i].session);
    }
    fprintf(stderr, "\naotx_tui: frames %llu, cells %llu, keys %llu, lines %llu,"
                    " sequences dropped %llu, end: %s\n",
            (unsigned long long)tui->paint.frames, (unsigned long long)tui->paint.cells,
            (unsigned long long)tui->session.keys, (unsigned long long)tui->session.lines,
            (unsigned long long)tui->keys.dropped,
            (tui->quit_reason != NULL) ? tui->quit_reason : "quit");
    return rc;
}
