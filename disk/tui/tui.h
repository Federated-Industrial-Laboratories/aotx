/* Purpose: Declare the state and the calls of the terminal program, one concern a file.
 * Owns: Nothing; the caller owns every structure that the calls fill.
 * Threading: One thread; the program waits in one poll and holds no lock.
 * Lifetime: From the start of the program to its exit. */
#ifndef AOTX_TUI_H
#define AOTX_TUI_H

#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <termios.h>

#include "cuda/ui/mirror.h"
#include "disk/settings/settings.h"
#include "disk/tui/actions.h"
#include "disk/wire/diskwire.h"

/* The smallest terminal the program claims. Every screen lays out inside it. */
#define AOTX_TUI_COLS_MIN   80u
#define AOTX_TUI_ROWS_MIN   24u

/* The largest terminal the program draws. A wider one keeps the columns after this
 * bound empty. */
#define AOTX_TUI_COLS_MAX   400u
#define AOTX_TUI_ROWS_MAX   200u
#define AOTX_TUI_CELLS      (AOTX_TUI_COLS_MAX * AOTX_TUI_ROWS_MAX)
#define AOTX_TUI_SYSTEMS    8u

/* The four renditions. The grid carries the first three in its attribute byte; the frame
 * uses the fourth for the label of the screen that is open. */
#define AOTX_TUI_PLAIN      0u
#define AOTX_TUI_DIM        1u
#define AOTX_TUI_BRIGHT     2u
#define AOTX_TUI_REVERSE    3u

/* The key codes are the codes the window sends, so the device editor sees no difference
 * between the two surfaces. */
#define AOTX_TUI_KEY_ESCAPE   256u
#define AOTX_TUI_KEY_ENTER    257u
#define AOTX_TUI_KEY_TAB      258u
#define AOTX_TUI_KEY_BACK     259u
#define AOTX_TUI_KEY_INSERT   260u
#define AOTX_TUI_KEY_DELETE   261u
#define AOTX_TUI_KEY_RIGHT    262u
#define AOTX_TUI_KEY_LEFT     263u
#define AOTX_TUI_KEY_DOWN     264u
#define AOTX_TUI_KEY_UP       265u
#define AOTX_TUI_KEY_PAGE_UP  266u
#define AOTX_TUI_KEY_PAGE_DN  267u
#define AOTX_TUI_KEY_HOME     268u
#define AOTX_TUI_KEY_END      269u
#define AOTX_TUI_KEY_F1       290u
#define AOTX_TUI_KEY_F12      301u

/* The modifier bits the window sends. */
#define AOTX_TUI_MOD_SHIFT    0x0001u
#define AOTX_TUI_MOD_CONTROL  0x0002u
#define AOTX_TUI_MOD_ALT      0x0004u

/* One key the decoder gives. A character carries a code of zero and its code point. */
typedef struct aotx_tui_key {
    unsigned int code;
    unsigned int codepoint;
    unsigned int mods;
} aotx_tui_key;

/* ---- keys.c, the decoder ---- */

/* The bytes of one sequence. The longest form the table holds is nine bytes; the buffer
 * holds more, and a sequence that fills it is dropped whole. */
#define AOTX_TUI_SEQUENCE   32u

#define AOTX_TUI_ESCAPE_DEFAULT 25u

typedef struct aotx_keys {
    unsigned char part[AOTX_TUI_SEQUENCE];
    unsigned int  fill;
    uint64_t      escape_ns;   /* the time a lone escape arrived, or 0 */
    unsigned int  escape_ms;   /* the setting for a lone escape */
    uint64_t      dropped;     /* sequences the decoder did not know */
} aotx_keys;

/* Decodes the bytes into keys. Returns the count of keys written to `out`. */
unsigned int aotx_keys_take(aotx_keys *k, const unsigned char *bytes, size_t count,
                            uint64_t now_ns, aotx_tui_key *out, unsigned int most);

/* Gives the Escape key when the wait passed with no further byte. Returns 1 or 0. */
unsigned int aotx_keys_wait(aotx_keys *k, uint64_t now_ns, aotx_tui_key *out);

/* The milliseconds a wait may last, or -1 when nothing is held. */
int aotx_keys_timeout(const aotx_keys *k, uint64_t now_ns);

/* ---- term.c, the terminal itself ---- */

#define AOTX_TUI_OUT_BYTES  65536u

typedef struct aotx_term {
    int            in_fd;
    int            out_fd;
    struct termios saved;
    int            raw;          /* the raw mode is set */
    int            alternate;    /* the alternate screen was asked for */
    unsigned int   cols, rows;
    unsigned int   rendition;    /* the rendition the terminal holds */
    int            color;        /* the renditions carry a color as well */
    unsigned int   row, col;     /* the cell the cursor stands at, from 1 */
    int            placed;       /* the cursor position is known */
    int            wake[2];      /* the self pipe of the signals */
    char           out[AOTX_TUI_OUT_BYTES];
    size_t         fill;
    uint64_t       written;      /* bytes written to the terminal */
} aotx_term;

/* Opens the terminal: raw mode, the alternate screen, no auto-wrap, no cursor. Returns 0,
 * or -1 when the descriptors are not a terminal. */
int aotx_term_open(aotx_term *t, int in_fd, int out_fd);

/* Gives the terminal back what it had: the renditions, the cursor, the screen. */
void aotx_term_close(aotx_term *t);

/* Reads the size. Returns 1 when the size changed. */
int aotx_term_size(aotx_term *t);

/* The signal state, from the self pipe. Each call clears what it reports. */
int aotx_term_signals(aotx_term *t, int *resized, int *child_ended);

void aotx_term_put(aotx_term *t, const char *bytes, size_t count);
void aotx_term_text(aotx_term *t, const char *text);
void aotx_term_number(aotx_term *t, unsigned int value);
void aotx_term_move(aotx_term *t, unsigned int row, unsigned int col);
void aotx_term_rendition(aotx_term *t, unsigned int rendition);
void aotx_term_cursor(aotx_term *t, int on);
void aotx_term_clear(aotx_term *t);
void aotx_term_flush(aotx_term *t);

/* ---- raster.c, the picture and the diff ---- */

/* One cell of the terminal's own picture. The code point is wider than the glyph index of
 * the grid, because the splash art holds code points of the braille block. */
typedef struct aotx_tui_cell {
    uint16_t code;
    uint8_t  attribute;
    uint8_t  reserved;
} aotx_tui_cell;

typedef struct aotx_paint {
    unsigned int     cols, rows;
    aotx_tui_cell    want[AOTX_TUI_CELLS];
    aotx_tui_cell    drawn[AOTX_TUI_CELLS];
    int              full;        /* the next write draws every cell */
    unsigned int     cursor_row, cursor_col;   /* the cell of the cursor, from 0 */
    int              cursor_on;
    unsigned int     pan_row, pan_col;         /* the first cell of the grid on view */
    int              follow_cursor;            /* the console cursor moves the view */
    uint64_t         cells;       /* cells written since the start */
    uint64_t         frames;
} aotx_paint;

/* Sets the size of the picture and asks for a whole frame. */
void aotx_paint_size(aotx_paint *p, unsigned int cols, unsigned int rows);

/* Fills the picture with spaces. */
void aotx_paint_clear(aotx_paint *p);

void aotx_paint_put(aotx_paint *p, unsigned int row, unsigned int col,
                    unsigned int code, unsigned int attribute);
void aotx_paint_text(aotx_paint *p, unsigned int row, unsigned int col,
                     const char *text, unsigned int attribute);
void aotx_paint_fill(aotx_paint *p, unsigned int row, unsigned int col,
                     unsigned int cols, unsigned int code, unsigned int attribute);

/* The rows and the columns of the work area, which is the frame with the status line and
 * the key bar taken off. */
unsigned int aotx_paint_view_rows(const aotx_paint *p);
unsigned int aotx_paint_view_cols(const aotx_paint *p);

/* Draws a box with the characters the settings name. `utf8` is 0 for the ascii box. */
void aotx_paint_box(aotx_paint *p, unsigned int row, unsigned int col,
                    unsigned int rows, unsigned int cols, int utf8);

/* Copies the panel that the pan names into the work area. The work area is every row
 * between the status line and the key bar. */
void aotx_paint_picture(aotx_paint *p, const aotx_mirror_snapshot *shot);

/* Moves the pan so that the named panel is on view. */
void aotx_paint_panel(aotx_paint *p, const aotx_mirror_snapshot *shot, unsigned int panel);

/* Resumes the follow of the console cursor. */
void aotx_paint_follow(aotx_paint *p);

/* Moves the pan by the step given, inside the grid. */
void aotx_paint_pan(aotx_paint *p, int rows, int cols);

/* Takes one snapshot from the mirror by the seqlock rule. Returns 1 when a whole snapshot
 * came back, and 0 when no slot held one. */
int aotx_mirror_take(const unsigned char *mirror, aotx_mirror_snapshot *out);

/* Writes the cells that changed. One cursor move covers each run of changed cells. */
void aotx_paint_flush(aotx_paint *p, aotx_term *t, int color);

/* ---- splash.c ---- */

#define AOTX_TUI_SPLASH_SIZES  5u
#define AOTX_TUI_DISSOLVE      24u

typedef struct aotx_splash {
    unsigned int cols, rows;                 /* the size of the art that was read */
    uint16_t     code[AOTX_TUI_CELLS];       /* the code point of each cell of the art */
    int          braille;                    /* the art holds braille code points */
    int          held;                       /* art was read */
    unsigned int step;                       /* the dissolve step that was drawn */
    unsigned int order[AOTX_TUI_CELLS];      /* the cell order of the dissolve */
    unsigned int order_count;
    char         name[64];                   /* the file the art came from */
    char         reason[128];                /* why no art was read */
} aotx_splash;

/* Reads the largest art that fits the work area. `mode` is the value of tui.splash.
 * Returns 1 when art was read, and 0 when the name alone is shown. */
int aotx_splash_read(aotx_splash *s, const char *dir, const char *mode,
                     unsigned int cols, unsigned int rows);

/* Draws the art in the middle of the work area with one line under it. */
void aotx_splash_draw(const aotx_splash *s, aotx_paint *p, unsigned int top,
                      unsigned int rows, const char *state);

/* Opens the dissolve of `frames` steps in a fixed order from one seed. */
void aotx_splash_dissolve_open(aotx_splash *s, unsigned int cols, unsigned int rows,
                               unsigned int frames);

/* Draws one step of the dissolve over the picture below. Returns 1 while the dissolve
 * runs and 0 when it ended. */
int aotx_splash_dissolve_step(aotx_splash *s, aotx_paint *p, unsigned int top,
                              unsigned int rows, unsigned int frames);

/* ---- session.c, the mirror and the child ---- */

typedef struct aotx_session {
    int            fd;             /* the socket to the feeder, or -1 */
    int            mirror_fd;      /* the mirror, or -1 */
    unsigned char *mirror;         /* the mapped mirror, or NULL */
    size_t         mirror_bytes;
    int            boot_pid;       /* the boot that this program started, or -1 */
    char           journal[AOTX_PATH_BYTES];
    char           settings[AOTX_PATH_BYTES];
    char           reason[160];    /* why the last call did not do its work */
    uint64_t       keys;           /* key frames sent */
    uint64_t       lines;          /* lines sent */
} aotx_session;

typedef struct aotx_tui_system {
    aotx_session session;
    unsigned int card;       /* the command-line order, from zero */
} aotx_tui_system;

/* Connects to <journal>/aotx.sock and takes the mirror descriptor. Returns 0, or -1 with
 * the reason. */
int aotx_session_attach(aotx_session *s, const char *journal);

/* Closes the socket and the mirror. The boot that runs is left running. */
void aotx_session_detach(aotx_session *s);

/* Sends one key frame. Returns 0 or -1. */
int aotx_session_key(aotx_session *s, const aotx_tui_key *key);

/* Sends one line. Returns 0 or -1. */
int aotx_session_line(aotx_session *s, const char *line);

/* Reads what the feeder sent. A reason frame goes to `reason`. Returns 1 when a reason
 * came back, 0 when nothing came, and -1 when the socket closed. */
int aotx_session_take(aotx_session *s);

/* Starts a system with the settings file given. The output of the boot goes to
 * <journal>/boot.log. Returns 0, or -1 with the reason. */
int aotx_session_start(aotx_session *s, const char *program, const char *settings,
                       const char *journal, int restore);

/* Reads the state of the boot that this program started. Returns 1 while it runs, 0 when
 * it ended, and -1 when none was started. `status` takes the exit status. */
int aotx_session_boot_state(aotx_session *s, int *status);

/* Reads <journal>/phase. Active states give text with elapsed seconds and return 1. An
 * absent or closed state returns 0. A state that does not read returns -1. */
int aotx_session_phase(const char *journal, uint64_t now_seconds, char *out, size_t bytes);

/* Reads the last lines of the boot log into `out`, at most `rows` lines of `cols`
 * bytes. Returns the count of lines. */
unsigned int aotx_session_boot_log(const aotx_session *s, char *out, unsigned int rows,
                                   unsigned int cols);

/* Reads the first line that `<program> --version` prints. Returns 0, or -1. */
int aotx_session_version(const char *program, char *out, size_t bytes);

/* ---- screens.c, the frame and the screens ---- */

#define AOTX_TUI_SCREEN_NONE  (~0u)
#define AOTX_TUI_ROWS_LIST    64u
#define AOTX_TUI_LINE_BYTES   256u
#define AOTX_TUI_EDIT_BYTES   (AOTX_BODY_BYTES * AOTX_LINE_PARTS_MAX + 1u)

/* One line of a manifest on the disk, which the store screens read. */
#define AOTX_MANIFEST_LINE_BYTES 1024u

typedef struct aotx_tui {
    aotx_term    term;
    aotx_paint   paint;
    aotx_keys    keys;
    aotx_session session;
    aotx_tui_system other[AOTX_TUI_SYSTEMS - 1u];
    aotx_splash  splash;
    aotx_settings settings;

    aotx_mirror_snapshot shot;  /* the last whole snapshot the mirror gave */
    int          have_shot;
    uint64_t     shot_sequence;
    uint64_t     shot_ns;       /* the time the last snapshot came */
    uint64_t     frames;        /* snapshots taken */

    unsigned int screen;        /* the open screen, or AOTX_TUI_SCREEN_NONE */
    unsigned int cursor;        /* the row a screen has under the cursor */
    unsigned int top;           /* the first row a screen shows */
    unsigned int dissolve;      /* the dissolve step that runs, or 0 */
    uint64_t     rows_ns;       /* the time the rows of a screen were filled */
    unsigned int rows_screen;   /* the screen those rows belong to */
    unsigned int rows_count;    /* the rows that fill gave */
    int          quit;
    int          color;         /* tui.color names 16 colors */
    int          utf8_box;      /* tui.box names the utf8 box */
    int          no_splash;
    int          socket_closed; /* keep the close report until a key or an attach */
    unsigned int other_count;   /* attached systems beside the selected one */
    unsigned int card;          /* selected card, from zero */

    char         settings_path[AOTX_PATH_BYTES];
    char         journal[AOTX_PATH_BYTES];
    char         program[AOTX_PATH_BYTES];   /* the boot beside this program */
    char         version[AOTX_TUI_LINE_BYTES]; /* what the build says of itself */
    char         state[192];    /* the line under the splash */
    char         says[AOTX_TUI_LINE_BYTES];  /* the last line the frame shows */
    char         picker_dir[AOTX_PATH_BYTES]; /* the directory the picker shows */
    char         edit[AOTX_TUI_EDIT_BYTES];  /* the field a screen edits */
    unsigned int edit_fill;
    int          editing;
    unsigned int session_agent;  /* the agent shown by the Session screen */
    unsigned int session_scroll; /* lines above the newest transcript line */
    unsigned int session_mode;   /* zero prompt, one pages, two spawn */
    int          session_result; /* a result is open beyond its first lines */
} aotx_tui;

/* Joins a directory and a name into a path. Returns 0, or -1 when the result does not
 * fit, which leaves the path empty and never a path that is cut short. */
int aotx_tui_join(char *out, size_t bytes, const char *dir, const char *name);

/* The screen numbers follow the rows of aotx_tui_screens. */
unsigned int aotx_screen_of_key(const aotx_tui_key *key);
const char *aotx_screen_name(unsigned int screen);

/* Draws the whole frame: the status line, the work area and the key bar. */
void aotx_frame_draw(aotx_tui *tui);

/* Draws the status line. */
void aotx_frame_status(aotx_tui *tui);

/* Draws the key bar with the open screen bright. */
void aotx_frame_keybar(aotx_tui *tui);

/* Draws the open screen inside the work area. */
void aotx_screen_draw(aotx_tui *tui, unsigned int top, unsigned int rows);

/* Gives one key to the open screen. Returns 1 when the screen took it. */
int aotx_screen_key(aotx_tui *tui, const aotx_tui_key *key);

/* Sends the line of an action, or writes the state it names when no system runs. */
void aotx_screen_send(aotx_tui *tui, const char *line);

/* The line of an action of the open screen, or NULL. */
const char *aotx_screen_action(unsigned int screen, const aotx_tui_key *key);

/* The screens that read the disk. Each one fills the rows it shows. */
unsigned int aotx_rows_models(aotx_tui *tui, char *out, unsigned int rows, unsigned int cols);
unsigned int aotx_rows_modules(aotx_tui *tui, int skills, char *out, unsigned int rows,
                               unsigned int cols);

/* The path of the module of one row of the Tools or the Skills screen. Returns 0, or -1
 * when the row holds no module. */
int aotx_module_path(unsigned int index, char *out, size_t bytes);
unsigned int aotx_rows_settings(aotx_tui *tui, char *out, unsigned int rows, unsigned int cols);
unsigned int aotx_rows_agents(aotx_tui *tui, char *out, unsigned int rows, unsigned int cols);
unsigned int aotx_rows_help(aotx_tui *tui, char *out, unsigned int rows, unsigned int cols);
unsigned int aotx_rows_system(aotx_tui *tui, char *out, unsigned int rows, unsigned int cols);
unsigned int aotx_rows_picker(aotx_tui *tui, char *out, unsigned int rows, unsigned int cols);

/* The System screen holds its own keys: a start, a restore and a stop. Returns 1 when the
 * screen took the key. */
int aotx_screen_system_key(aotx_tui *tui, const aotx_tui_key *key);

/* The Session screen has a multi-line editor and its own keys. */
void aotx_screen_session_draw(aotx_tui *tui, unsigned int top, unsigned int rows);
int aotx_screen_session_key(aotx_tui *tui, const aotx_tui_key *key);

/* Selects the card before or after the current card. Returns 1 when it changed. */
int aotx_tui_select_card(aotx_tui *tui, int step);

/* The picker holds the directory it shows. */
void aotx_picker_open(aotx_tui *tui, const char *dir);
int aotx_picker_enter(aotx_tui *tui, char *out, size_t bytes);

#endif
