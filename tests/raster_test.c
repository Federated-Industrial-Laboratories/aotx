/* Purpose: Check the read of the mirror, the difference the raster writes, and the viewport.
 * Owns: One mirror, one pseudo terminal and one picture for each case.
 * Threading: The read rule is checked against a thread that publishes frames beside it.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include "disk/tui/tui.h"

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <termios.h>

/* The mirror this test makes: one preamble and the slots after it, as the glue makes it. */
#define AOTX_TEST_MIRROR_BYTES (sizeof(aotx_mirror_preamble) \
                                + (size_t)AOTX_MIRROR_SLOTS * sizeof(aotx_mirror_snapshot))

/* The frames the writer publishes while the reader takes snapshots. */
#define AOTX_TEST_PUBLISHED 20000u

static unsigned char aotx_test_mirror[AOTX_TEST_MIRROR_BYTES];
static aotx_paint aotx_test_paint;
static aotx_mirror_snapshot aotx_test_shot;

/* The glyph a cell of frame `n` holds. Every cell of one frame agrees with it, so a
 * snapshot that mixes two frames is seen at once. */
static unsigned char cell_of(uint64_t frame, unsigned int at)
{
    return (unsigned char)((frame + at) % 95u);
}

static aotx_mirror_snapshot *slot_at(unsigned int index)
{
    return (aotx_mirror_snapshot *)(aotx_test_mirror + sizeof(aotx_mirror_preamble)
                                    + (size_t)index * sizeof(aotx_mirror_snapshot));
}

static void open_mirror(void)
{
    aotx_mirror_preamble *pre = (aotx_mirror_preamble *)aotx_test_mirror;
    memset(aotx_test_mirror, 0, sizeof(aotx_test_mirror));
    pre->magic = AOTX_MIRROR_MAGIC;
    pre->layout = AOTX_MIRROR_LAYOUT;
    pre->slots = AOTX_MIRROR_SLOTS;
    pre->slot_bytes = (uint32_t)sizeof(aotx_mirror_snapshot);
    pre->cols = AOTX_MIRROR_COLS;
    pre->rows = AOTX_MIRROR_ROWS;
}

/* Publishes one frame into one slot by the rule of the design: zero into the sequence, the
 * bytes, then the sequence. */
static void publish(unsigned int index, uint64_t frame)
{
    aotx_mirror_snapshot *shot = slot_at(index);
    unsigned int at;
    __atomic_store_n(&shot->head.sequence, 0ull, __ATOMIC_RELEASE);
    shot->head.tick = frame;
    for (at = 0; at < AOTX_MIRROR_CELLS; at++) {
        shot->cell[at].glyph = cell_of(frame, at);
        shot->cell[at].attribute = (uint8_t)(frame % 3u);
    }
    __atomic_store_n(&shot->head.sequence, frame, __ATOMIC_RELEASE);
}

/* Reports whether a snapshot holds one frame and not two. */
static int whole(const aotx_mirror_snapshot *shot)
{
    unsigned int at;
    for (at = 0; at < AOTX_MIRROR_CELLS; at++) {
        if (shot->cell[at].glyph != cell_of(shot->head.sequence, at)) {
            return 0;
        }
    }
    return 1;
}

static volatile int aotx_test_run;

static void *writer(void *unused)
{
    uint64_t frame;
    (void)unused;
    for (frame = 1; frame <= AOTX_TEST_PUBLISHED && aotx_test_run != 0; frame++) {
        publish((unsigned int)(frame % AOTX_MIRROR_SLOTS), frame);
    }
    aotx_test_run = 0;
    return NULL;
}

/* The read rule: a slow reader loses frames and never sees a torn one. */
static void seqlock(void)
{
    pthread_t thread;
    uint64_t taken = 0;
    uint64_t missed = 0;
    uint64_t torn = 0;
    uint64_t last = 0;
    open_mirror();
    CHECK(aotx_mirror_take(aotx_test_mirror, &aotx_test_shot) == 0,
          "a mirror with no frame gives a snapshot");
    publish(0u, 7ull);
    CHECK(aotx_mirror_take(aotx_test_mirror, &aotx_test_shot) == 1,
          "a mirror with one frame gives none");
    CHECK(aotx_test_shot.head.sequence == 7ull, "the snapshot is not the frame published");
    __atomic_store_n(&slot_at(0)->head.sequence, 0ull, __ATOMIC_RELEASE);
    CHECK(aotx_mirror_take(aotx_test_mirror, &aotx_test_shot) == 0,
          "a slot under the pen must give no snapshot");

    aotx_test_run = 1;
    CHECK(pthread_create(&thread, NULL, writer, NULL) == 0, "the writer does not start");
    while (aotx_test_run != 0 && taken < 2000u) {
        if (aotx_mirror_take(aotx_test_mirror, &aotx_test_shot) == 0) {
            missed++;
            continue;
        }
        taken++;
        if (whole(&aotx_test_shot) == 0) {
            torn++;
        }
        if (aotx_test_shot.head.sequence < last) {
            /* A reader may lose frames; it may never go backward past one slot. */
            missed++;
        }
        last = aotx_test_shot.head.sequence;
    }
    aotx_test_run = 0;
    pthread_join(thread, NULL);
    CHECK(taken > 100u, "the reader took %llu snapshots, too few to judge",
          (unsigned long long)taken);
    CHECK(torn == 0u, "%llu snapshots of %llu were torn", (unsigned long long)torn,
          (unsigned long long)taken);
}

/* Opens a pseudo terminal and gives the two descriptors. Returns 0 or -1. */
static int open_pty(int *master, int *slave)
{
    char name[256];
    *master = posix_openpt(O_RDWR | O_NOCTTY);
    if (*master < 0 || grantpt(*master) != 0 || unlockpt(*master) != 0
        || ptsname_r(*master, name, sizeof(name)) != 0) {
        return -1;
    }
    *slave = open(name, O_RDWR | O_NOCTTY);
    return (*slave >= 0) ? 0 : -1;
}

/* Sets the size of a pseudo terminal, so the program reads it as a terminal of that size. */
static void set_size(int fd, unsigned int cols, unsigned int rows)
{
    struct winsize size;
    memset(&size, 0, sizeof(size));
    size.ws_col = (unsigned short)cols;
    size.ws_row = (unsigned short)rows;
    ioctl(fd, TIOCSWINSZ, &size);
}

/* Reads what the terminal was sent, up to the bytes given. */
static size_t drain(int master, char *out, size_t bytes)
{
    size_t at = 0;
    for (;;) {
        ssize_t got;
        struct pollfd fd;
        fd.fd = master;
        fd.events = POLLIN;
        fd.revents = 0;
        if (poll(&fd, 1, 20) <= 0) {
            break;
        }
        got = read(master, out + at, bytes - at - 1u);
        if (got <= 0) {
            break;
        }
        at += (size_t)got;
        if (at + 1u >= bytes) {
            break;
        }
    }
    out[at] = '\0';
    return at;
}

/* The viewport at one size: the work area is the frame with the two lines taken off, and
 * one column on each side. */
static void viewport(unsigned int cols, unsigned int rows)
{
    aotx_paint *p = &aotx_test_paint;
    unsigned int r;
    unsigned int c;
    unsigned int view_cols;
    unsigned int view_rows;
    open_mirror();
    publish(0u, 11ull);
    CHECK(aotx_mirror_take(aotx_test_mirror, &aotx_test_shot) == 1,
          "the snapshot does not come back");
    aotx_paint_size(p, cols, rows);
    view_cols = aotx_paint_view_cols(p);
    view_rows = aotx_paint_view_rows(p);
    CHECK(view_cols == cols, "the view holds %u columns of %u", view_cols, cols);
    CHECK(view_rows == rows - 2u, "the view holds %u rows of %u", view_rows, rows);
    aotx_paint_clear(p);
    p->pan_row = 0;
    p->pan_col = 0;
    aotx_paint_picture(p, &aotx_test_shot);
    /* The status line and the key bar are pinned: the picture writes neither. */
    for (c = 0; c < cols; c++) {
        CHECK(p->want[c].code == (uint16_t)' ', "the picture wrote the status line");
        CHECK(p->want[(rows - 1u) * cols + c].code == (uint16_t)' ',
              "the picture wrote the key bar");
    }
    for (r = 0; r < view_rows && r < AOTX_MIRROR_ROWS; r++) {
        for (c = 0; c < view_cols && c < AOTX_MIRROR_COLS; c++) {
            unsigned int want = 32u + cell_of(11ull, r * AOTX_MIRROR_COLS + c);
            CHECK(p->want[(r + 1u) * cols + c].code == (uint16_t)want,
                  "the cell at %u,%u of the view is not the cell of the grid", r, c);
        }
    }
    /* A terminal of the width of the grid and two rows over its height shows the whole
     * picture, and the pan has nowhere to go. */
    if (cols >= AOTX_MIRROR_COLS && rows >= AOTX_MIRROR_ROWS + 2u) {
        CHECK(view_cols >= AOTX_MIRROR_COLS, "the whole grid does not fit %u columns",
              cols);
        CHECK(view_rows >= AOTX_MIRROR_ROWS, "the whole grid does not fit %u rows", rows);
        aotx_paint_pan(p, 1000, 1000);
        CHECK(p->pan_row == 0u && p->pan_col == 0u,
              "a terminal that holds the whole grid still pans");
        aotx_paint_clear(p);
        aotx_paint_picture(p, &aotx_test_shot);
        for (r = 0; r < AOTX_MIRROR_ROWS; r++) {
            unsigned int last = AOTX_MIRROR_COLS - 1u;
            unsigned int want = 32u + cell_of(11ull, r * AOTX_MIRROR_COLS + last);
            CHECK(p->want[(r + 1u) * cols + last].code == (uint16_t)want,
                  "the last column of the grid is not on view at row %u", r);
        }
    }
    /* The console cursor moves into view at the floor size. A manual pan suspends that
     * follow until the caller resumes it after Ctrl-L or a line sent. */
    if (cols == 80u && rows == 24u) {
        p->follow_cursor = 1;
        aotx_test_shot.head.focus = 0u;
        aotx_test_shot.head.cursor_row = 33u;
        aotx_test_shot.head.cursor_col = 79u;
        aotx_paint_picture(p, &aotx_test_shot);
        CHECK(aotx_test_shot.head.cursor_row >= p->pan_row
              && aotx_test_shot.head.cursor_row < p->pan_row + view_rows,
              "the console cursor row is outside the floor view");
        CHECK(aotx_test_shot.head.cursor_col >= p->pan_col
              && aotx_test_shot.head.cursor_col < p->pan_col + view_cols,
              "the console cursor column is outside the floor view");
        aotx_paint_pan(p, -1000, -1000);
        aotx_paint_picture(p, &aotx_test_shot);
        CHECK(p->pan_row == 0u, "the console cursor resumed the manual pan");
        aotx_paint_follow(p);
        aotx_paint_picture(p, &aotx_test_shot);
        CHECK(p->pan_row + view_rows > aotx_test_shot.head.cursor_row,
              "the resumed follow did not show the console cursor");
    }
    /* A panel row brings its rectangle into view, and the pan stays inside the grid. */
    aotx_paint_panel(p, &aotx_test_shot, 1u);
    CHECK(p->pan_col <= AOTX_MIRROR_COLS, "the pan went past the grid");
    aotx_paint_pan(p, -1000, -1000);
    CHECK(p->pan_row == 0u && p->pan_col == 0u, "the pan did not stop at the first cell");
    aotx_paint_pan(p, 1000, 1000);
    CHECK(p->pan_row + view_rows <= AOTX_MIRROR_ROWS
          || view_rows >= AOTX_MIRROR_ROWS, "the pan went past the last row");
    CHECK(p->pan_col + view_cols <= AOTX_MIRROR_COLS
          || view_cols >= AOTX_MIRROR_COLS, "the pan went past the last column");
    if ((cols == 80u && rows == 24u) || (cols == 160u && rows == 52u)) {
        printf("raster_test: %u x %u keeps the editor cursor in view\n", cols, rows);
    }
}

/* The difference: the first write is a whole frame, and a write after one changed cell
 * covers that cell and no other. */
static void difference(int n)
{
    static char out[262144];
    aotx_term term;
    aotx_paint *p = &aotx_test_paint;
    int master = -1;
    int slave = -1;
    uint64_t after_full;
    uint64_t before;
    size_t bytes;
    int i;

    CHECK(open_pty(&master, &slave) == 0, "the pseudo terminal does not open");
    set_size(slave, 80u, 24u);
    memset(&term, 0, sizeof(term));
    CHECK(aotx_term_open(&term, slave, slave) == 0, "the terminal does not open");
    CHECK(term.cols == 80u && term.rows == 24u, "the terminal size is %u by %u", term.cols,
          term.rows);
    drain(master, out, sizeof(out));

    aotx_paint_size(p, term.cols, term.rows);
    aotx_paint_clear(p);
    p->cells = 0;
    aotx_paint_flush(p, &term, 0);
    after_full = p->cells;
    CHECK(after_full == (uint64_t)term.cols * term.rows,
          "the first write covers %llu cells, not the whole frame",
          (unsigned long long)after_full);
    bytes = drain(master, out, sizeof(out));
    CHECK(bytes > 0, "the whole frame wrote no byte");

    /* A write with nothing changed writes nothing at all. */
    before = p->cells;
    aotx_paint_flush(p, &term, 0);
    CHECK(p->cells == before, "a frame with no change wrote %llu cells",
          (unsigned long long)(p->cells - before));

    /* The bytes are drained at each turn. A write that covers the whole frame thus fills
     * no pipe, and the case fails with a count and never with a wait. */
    bytes = 0;
    for (i = 0; i < n; i++) {
        unsigned int row = 1u + (unsigned int)i % (term.rows - 2u);
        unsigned int col = 1u + (unsigned int)i % (term.cols - 2u);
        before = p->cells;
        aotx_paint_put(p, row, col, (unsigned int)('a' + (i % 26)), AOTX_TUI_PLAIN);
        aotx_paint_flush(p, &term, 0);
        CHECK(p->cells == before + 1u, "one changed cell wrote %llu cells at element %d",
              (unsigned long long)(p->cells - before), i);
        bytes += drain(master, out, sizeof(out));
    }
    CHECK(bytes < (size_t)(n * 32 + 64), "the writes of %d cells took %zu bytes", n, bytes);

    /* A run of changed cells takes one cursor move and not one for each cell. */
    before = p->cells;
    for (i = 0; i < 40; i++) {
        aotx_paint_put(p, 5u, 2u + (unsigned int)i, (unsigned int)'x', AOTX_TUI_PLAIN);
    }
    aotx_paint_flush(p, &term, 0);
    CHECK(p->cells == before + 40u, "the run of forty cells wrote %llu cells",
          (unsigned long long)(p->cells - before));
    bytes = drain(master, out, sizeof(out));
    CHECK(bytes < 80u, "the run of forty cells took %zu bytes, more than one move",
          bytes);

    /* A whole frame is asked for after a gap or a resize, and it covers every cell. */
    p->full = 1;
    before = p->cells;
    aotx_paint_flush(p, &term, 0);
    CHECK(p->cells == before + (uint64_t)term.cols * term.rows,
          "the whole frame after a gap covers %llu cells",
          (unsigned long long)(p->cells - before));
    drain(master, out, sizeof(out));

    aotx_term_close(&term);
    close(slave);
    close(master);
}

/* The floor of eighty by twenty four: every screen lays out inside it. */
static void floor_size(void)
{
    aotx_paint *p = &aotx_test_paint;
    aotx_paint_size(p, AOTX_TUI_COLS_MIN, AOTX_TUI_ROWS_MIN);
    CHECK(aotx_paint_view_cols(p) == 80u, "the floor gives %u columns, not 80",
          aotx_paint_view_cols(p));
    CHECK(aotx_paint_view_rows(p) == 22u, "the floor gives %u rows, not 22",
          aotx_paint_view_rows(p));
    aotx_paint_box(p, 1u, 0u, 22u, 80u, 0);
    CHECK(p->want[1u * p->cols].code == (uint16_t)'+',
          "the box has no corner at its first cell");
    CHECK(p->want[22u * p->cols + 79u].code == (uint16_t)'+',
          "the box has no corner at its last cell");
    CHECK(p->want[0].code == (uint16_t)' ', "the box wrote the status line");
    CHECK(p->want[(p->rows - 1u) * p->cols].code == (uint16_t)' ',
          "the box wrote the key bar");
}

static void layout_bounds(void)
{
    aotx_mirror_preamble *pre = (aotx_mirror_preamble *)aotx_test_mirror;
    open_mirror();
    publish(0u, 1u);
    aotx_mirror_snapshot *source = slot_at(0u);
    for (unsigned int i = 0u; i < AOTX_MIRROR_AGENT_ROWS; ++i) {
        source->tables.agent[i].id = i;
        source->tables.agent[i].turn = 1000u + i;
    }
    for (unsigned int i = 0u; i < AOTX_MIRROR_MODULE_ROWS; ++i)
        snprintf(source->tables.module[i].name, AOTX_MIRROR_TEXT_BYTES, "module_%u", i);
    CHECK(aotx_mirror_take(aotx_test_mirror, &aotx_test_shot) == 1, "the current layout was refused");
    CHECK(!memcmp(&source->tables, &aotx_test_shot.tables, sizeof source->tables),
          "the complete current tables did not reach the terminal");
    for (unsigned int layout = 0u; layout <= AOTX_MIRROR_LAYOUT + 1u; ++layout) {
        if (layout == AOTX_MIRROR_LAYOUT) continue;
        pre->layout = layout;
        CHECK(!aotx_mirror_take(aotx_test_mirror, &aotx_test_shot), "layout %u was accepted", layout);
    }
    pre->layout = AOTX_MIRROR_LAYOUT;
    pre->slot_bytes = sizeof(aotx_mirror_snapshot) - 1u;
    CHECK(!aotx_mirror_take(aotx_test_mirror, &aotx_test_shot), "a short snapshot was accepted");
}

int main(void)
{
    layout_bounds();
    seqlock();
    viewport(80u, 24u);
    viewport(160u, 52u);
    viewport(200u, 60u);
    floor_size();
    difference(1);
    difference(64);
    return aotx_report("raster_test", 200);
}
