/* Purpose: Hold the terminal itself: the raw mode, the screen, the cursor and the signals.
 * Owns: The saved terminal state, the output buffer and the self pipe of the signals.
 * Threading: One thread; the signal handler writes one byte and nothing else.
 * Lifetime: From the open of the terminal to its close. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/tui/tui.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

/* The bytes the handler writes. The reader turns each one into a flag. */
#define AOTX_TERM_RESIZE 'W'
#define AOTX_TERM_CHILD  'C'

/* The handler holds nothing but the write end of the pipe, because a handler may use no
 * other state of the program. */
static volatile sig_atomic_t aotx_term_wake = -1;

static void on_signal(int number)
{
    char byte = (number == SIGCHLD) ? AOTX_TERM_CHILD : AOTX_TERM_RESIZE;
    int fd = (int)aotx_term_wake;
    if (fd >= 0) {
        ssize_t sent = write(fd, &byte, 1);
        (void)sent;
    }
}

void aotx_term_put(aotx_term *t, const char *bytes, size_t count)
{
    size_t at = 0;
    while (at < count) {
        size_t room = sizeof(t->out) - t->fill;
        size_t take = (count - at < room) ? count - at : room;
        memcpy(t->out + t->fill, bytes + at, take);
        t->fill += take;
        at += take;
        if (t->fill == sizeof(t->out)) {
            aotx_term_flush(t);
        }
    }
}

void aotx_term_text(aotx_term *t, const char *text)
{
    aotx_term_put(t, text, strlen(text));
}

void aotx_term_number(aotx_term *t, unsigned int value)
{
    char digits[12];
    unsigned int at = sizeof(digits);
    if (value == 0) {
        aotx_term_put(t, "0", 1);
        return;
    }
    while (value > 0 && at > 0) {
        digits[--at] = (char)('0' + (value % 10u));
        value /= 10u;
    }
    aotx_term_put(t, digits + at, sizeof(digits) - at);
}

void aotx_term_flush(aotx_term *t)
{
    size_t at = 0;
    while (at < t->fill) {
        ssize_t sent = write(t->out_fd, t->out + at, t->fill - at);
        if (sent > 0) {
            at += (size_t)sent;
            t->written += (uint64_t)sent;
            continue;
        }
        if (sent < 0 && (errno == EINTR || errno == EAGAIN)) {
            continue;
        }
        break;
    }
    t->fill = 0;
}

void aotx_term_move(aotx_term *t, unsigned int row, unsigned int col)
{
    if (t->placed != 0 && t->row == row && t->col == col) {
        return;
    }
    aotx_term_put(t, "\033[", 2);
    aotx_term_number(t, row);
    aotx_term_put(t, ";", 1);
    aotx_term_number(t, col);
    aotx_term_put(t, "H", 1);
    t->row = row;
    t->col = col;
    t->placed = 1;
}

/* The four renditions. Every change starts from the plain rendition, so no attribute of an
 * earlier cell stays behind. With a color the plain rendition takes the default color, the
 * bright one takes yellow and the dim one takes gray. */
void aotx_term_rendition(aotx_term *t, unsigned int rendition)
{
    if (t->rendition == rendition) {
        return;
    }
    switch (rendition) {
    case AOTX_TUI_BRIGHT:
        aotx_term_text(t, t->color ? "\033[0;1;33m" : "\033[0;1m");
        break;
    case AOTX_TUI_DIM:
        aotx_term_text(t, t->color ? "\033[0;2;37m" : "\033[0;2m");
        break;
    case AOTX_TUI_REVERSE:
        aotx_term_text(t, "\033[0;7m");
        break;
    default:
        aotx_term_text(t, "\033[0m");
        break;
    }
    t->rendition = rendition;
}

void aotx_term_cursor(aotx_term *t, int on)
{
    aotx_term_text(t, on ? "\033[?25h" : "\033[?25l");
}

void aotx_term_clear(aotx_term *t)
{
    aotx_term_text(t, "\033[H\033[2J");
    t->row = 1;
    t->col = 1;
    t->placed = 1;
}

int aotx_term_size(aotx_term *t)
{
    struct winsize size;
    unsigned int cols = AOTX_TUI_COLS_MIN;
    unsigned int rows = AOTX_TUI_ROWS_MIN;
    if (ioctl(t->out_fd, TIOCGWINSZ, &size) == 0 && size.ws_col > 0 && size.ws_row > 0) {
        cols = size.ws_col;
        rows = size.ws_row;
    }
    if (cols > AOTX_TUI_COLS_MAX) {
        cols = AOTX_TUI_COLS_MAX;
    }
    if (rows > AOTX_TUI_ROWS_MAX) {
        rows = AOTX_TUI_ROWS_MAX;
    }
    if (cols == t->cols && rows == t->rows) {
        return 0;
    }
    t->cols = cols;
    t->rows = rows;
    return 1;
}

int aotx_term_signals(aotx_term *t, int *resized, int *child_ended)
{
    char buffer[64];
    int found = 0;
    for (;;) {
        ssize_t got = read(t->wake[0], buffer, sizeof(buffer));
        ssize_t i;
        if (got <= 0) {
            break;
        }
        found = 1;
        for (i = 0; i < got; i++) {
            if (buffer[i] == AOTX_TERM_CHILD) {
                *child_ended = 1;
            } else {
                *resized = 1;
            }
        }
        if ((size_t)got < sizeof(buffer)) {
            break;
        }
    }
    return found;
}

int aotx_term_open(aotx_term *t, int in_fd, int out_fd)
{
    struct termios raw;
    struct sigaction act;
    t->in_fd = in_fd;
    t->out_fd = out_fd;
    t->fill = 0;
    t->rendition = ~0u;
    t->placed = 0;
    t->cols = 0;
    t->rows = 0;
    t->wake[0] = -1;
    t->wake[1] = -1;
    if (tcgetattr(in_fd, &t->saved) != 0) {
        return -1;
    }
    if (pipe2(t->wake, O_NONBLOCK | O_CLOEXEC) != 0) {
        return -1;
    }
    aotx_term_wake = t->wake[1];
    memset(&act, 0, sizeof(act));
    act.sa_handler = on_signal;
    act.sa_flags = SA_RESTART;
    sigaction(SIGWINCH, &act, NULL);
    sigaction(SIGCHLD, &act, NULL);

    raw = t->saved;
    /* The program reads bytes and no line, gives no echo, and takes the signal keys as
     * bytes of its own. The read gives what is there and waits for nothing, because the
     * poll decides when to read. */
    raw.c_iflag &= (tcflag_t)~(IXON | ICRNL | BRKINT | INPCK | ISTRIP);
    raw.c_oflag &= (tcflag_t)~OPOST;
    raw.c_lflag &= (tcflag_t)~(ECHO | ICANON | ISIG | IEXTEN);
    raw.c_cflag |= CS8;
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(in_fd, TCSAFLUSH, &raw) != 0) {
        return -1;
    }
    t->raw = 1;
    /* The alternate screen is a private mode, and a console older than one version of the
     * kernel ignores it. The clear that follows therefore stands on its own, and the whole
     * frame is drawn after it. */
    aotx_term_text(t, "\033[?1049h");
    t->alternate = 1;
    aotx_term_text(t, "\033[?7l");
    aotx_term_cursor(t, 0);
    aotx_term_clear(t);
    aotx_term_flush(t);
    aotx_term_size(t);
    return 0;
}

void aotx_term_close(aotx_term *t)
{
    if (t->raw == 0) {
        return;
    }
    aotx_term_rendition(t, AOTX_TUI_PLAIN);
    /* The clear happens before the alternate screen goes away. A terminal that took no
     * alternate screen is thus left empty, and not with the last frame on it. */
    aotx_term_clear(t);
    aotx_term_text(t, "\033[?7h");
    aotx_term_cursor(t, 1);
    if (t->alternate != 0) {
        aotx_term_text(t, "\033[?1049l");
        t->alternate = 0;
    }
    aotx_term_flush(t);
    tcsetattr(t->in_fd, TCSAFLUSH, &t->saved);
    t->raw = 0;
    aotx_term_wake = -1;
    if (t->wake[0] >= 0) {
        close(t->wake[0]);
        close(t->wake[1]);
        t->wake[0] = -1;
        t->wake[1] = -1;
    }
}
