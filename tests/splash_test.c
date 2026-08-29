/* Purpose: Check the splash art files, the fallback and the order of the dissolve.
 * Owns: One splash state and one picture for each case.
 * Threading: One thread.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include "disk/tui/tui.h"

/* The sizes the tool writes, which the program reads and never rescales. */
static const unsigned int aotx_sizes[][2] = {
    { 160u, 40u }, { 120u, 30u }, { 80u, 20u }, { 60u, 12u }, { 40u, 8u }
};

#define AOTX_SIZES (sizeof(aotx_sizes) / sizeof(aotx_sizes[0]))

static aotx_splash aotx_test_splash;
static aotx_splash aotx_second_splash;
static aotx_paint aotx_test_paint;

/* Counts the code points of one line of a file that holds text of the wide set. */
static unsigned int code_points(const char *line, unsigned int bytes)
{
    unsigned int at = 0;
    unsigned int count = 0;
    while (at < bytes) {
        unsigned char first = (unsigned char)line[at];
        if (first < 0x80u) {
            at += 1u;
        } else if ((first & 0xe0u) == 0xc0u) {
            at += 2u;
        } else if ((first & 0xf0u) == 0xe0u) {
            at += 3u;
        } else {
            at += 1u;
        }
        count++;
    }
    return count;
}

/* Each file has the width and the height its name says. */
static void files(const char *dir)
{
    unsigned int i;
    unsigned int form;
    for (i = 0; i < AOTX_SIZES; i++) {
        for (form = 0; form < 2u; form++) {
            char path[AOTX_PATH_BYTES];
            char line[AOTX_TUI_COLS_MAX * 4u + 8u];
            FILE *file;
            unsigned int rows = 0;
            snprintf(path, sizeof(path), "%s/%ux%u.%s", dir, aotx_sizes[i][0],
                     aotx_sizes[i][1], (form == 0) ? "braille" : "ascii");
            file = fopen(path, "r");
            CHECK(file != NULL, "the art %s is not there", path);
            if (file == NULL) {
                continue;
            }
            while (fgets(line, (int)sizeof(line), file) != NULL) {
                unsigned int bytes = (unsigned int)strlen(line);
                unsigned int wide;
                while (bytes > 0 && (line[bytes - 1u] == '\n' || line[bytes - 1u] == '\r')) {
                    bytes--;
                }
                wide = code_points(line, bytes);
                CHECK(wide == aotx_sizes[i][0], "%s row %u holds %u cells, not %u", path,
                      rows, wide, aotx_sizes[i][0]);
                rows++;
            }
            fclose(file);
            CHECK(rows == aotx_sizes[i][1], "%s holds %u rows, not %u", path, rows,
                  aotx_sizes[i][1]);
        }
    }
}

/* The program picks the largest art that fits, and the braille form when the mode says so. */
static void pick(const char *dir)
{
    unsigned int i;
    for (i = 0; i < AOTX_SIZES; i++) {
        unsigned int cols = aotx_sizes[i][0];
        unsigned int rows = aotx_sizes[i][1] + 1u;
        CHECK(aotx_splash_read(&aotx_test_splash, dir, "braille", cols, rows) == 1,
              "no art was read at %u by %u", cols, rows);
        CHECK(aotx_test_splash.cols == cols, "the art at %u columns is %u wide", cols,
              aotx_test_splash.cols);
        CHECK(aotx_test_splash.rows == aotx_sizes[i][1], "the art at %u rows is %u tall",
              rows, aotx_test_splash.rows);
        CHECK(aotx_test_splash.braille == 1, "the braille form did not come back");
        /* One column short of the size is one size down, and never the same art scaled. */
        if (i + 1u < AOTX_SIZES) {
            CHECK(aotx_splash_read(&aotx_test_splash, dir, "ascii", cols - 1u, rows) == 1,
                  "no art was read one column under %u", cols);
            CHECK(aotx_test_splash.cols == aotx_sizes[i + 1u][0],
                  "the art one column under %u is %u wide", cols, aotx_test_splash.cols);
            CHECK(aotx_test_splash.braille == 0, "the ascii form did not come back");
        }
    }
}

/* With no art the name stands alone and the reason is kept. */
static void fallback(const char *dir)
{
    char empty[128];
    CHECK(aotx_temp_dir(empty, sizeof(empty)) == 0, "the temporary directory does not open");
    CHECK(aotx_splash_read(&aotx_test_splash, empty, "auto", 160u, 50u) == 0,
          "art came back from a directory that holds none");
    CHECK(aotx_test_splash.held == 0, "the splash says it holds art");
    CHECK(aotx_test_splash.reason[0] != '\0', "the splash gives no reason");
    aotx_paint_size(&aotx_test_paint, 80u, 24u);
    aotx_paint_clear(&aotx_test_paint);
    aotx_splash_draw(&aotx_test_splash, &aotx_test_paint, 1u, 22u, "no system runs");
    CHECK(aotx_test_paint.want[12u * 80u + 37u].code != (uint16_t)' ',
          "the name is not in the middle of the work area");
    aotx_remove_tree(empty);

    /* The mode off gives no art at all, whatever the directory holds. */
    CHECK(aotx_splash_read(&aotx_test_splash, dir, "off", 160u, 50u) == 0,
          "the mode off gave art");
    CHECK(aotx_test_splash.held == 0, "the mode off holds art");

    /* A terminal that fits no size gets the name and no art. */
    CHECK(aotx_splash_read(&aotx_test_splash, dir, "braille", 30u, 6u) == 0,
          "art came back for a terminal that fits none");
    CHECK(aotx_test_splash.held == 0, "the small terminal holds art");
}

/* The dissolve runs its frames and its order is the same at every run. */
static void dissolve(const char *dir)
{
    unsigned int step;
    unsigned int i;
    int going = 1;
    CHECK(aotx_splash_read(&aotx_test_splash, dir, "braille", 80u, 24u) == 1,
          "no art was read for the dissolve");
    CHECK(aotx_splash_read(&aotx_second_splash, dir, "braille", 80u, 24u) == 1,
          "no art was read for the second dissolve");
    aotx_splash_dissolve_open(&aotx_test_splash, aotx_test_splash.cols,
                              aotx_test_splash.rows, AOTX_TUI_DISSOLVE);
    aotx_splash_dissolve_open(&aotx_second_splash, aotx_second_splash.cols,
                              aotx_second_splash.rows, AOTX_TUI_DISSOLVE);
    CHECK(aotx_test_splash.order_count
          == aotx_test_splash.cols * aotx_test_splash.rows,
          "the dissolve holds %u cells, not the cells of the art",
          aotx_test_splash.order_count);
    for (i = 0; i < aotx_test_splash.order_count; i++) {
        CHECK(aotx_test_splash.order[i] == aotx_second_splash.order[i],
              "the order of the dissolve is not the same at cell %u", i);
    }
    /* Every cell is in the order one time, so no cell is drawn twice and none is lost. */
    {
        static unsigned char seen[AOTX_TUI_CELLS];
        memset(seen, 0, sizeof(seen));
        for (i = 0; i < aotx_test_splash.order_count; i++) {
            unsigned int cell = aotx_test_splash.order[i];
            CHECK(cell < aotx_test_splash.order_count, "the order names the cell %u",
                  cell);
            if (cell < aotx_test_splash.order_count) {
                CHECK(seen[cell] == 0, "the cell %u is in the order two times", cell);
                seen[cell] = 1;
            }
        }
    }
    aotx_paint_size(&aotx_test_paint, 80u, 24u);
    for (step = 0; step < AOTX_TUI_DISSOLVE; step++) {
        aotx_paint_clear(&aotx_test_paint);
        going = aotx_splash_dissolve_step(&aotx_test_splash, &aotx_test_paint, 1u, 22u,
                                          AOTX_TUI_DISSOLVE);
        CHECK(aotx_test_splash.step == step + 1u, "the dissolve is at step %u, not %u",
              aotx_test_splash.step, step + 1u);
        CHECK(going == ((step + 1u < AOTX_TUI_DISSOLVE) ? 1 : 0),
              "the dissolve does not end at the frame it names");
    }
    CHECK(aotx_splash_dissolve_step(&aotx_test_splash, &aotx_test_paint, 1u, 22u,
                                    AOTX_TUI_DISSOLVE) == 0,
          "the dissolve goes on after its frames");
}

int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : "share/splash";
    files(dir);
    pick(dir);
    fallback(dir);
    dissolve(dir);
    return aotx_report("splash_test", 200);
}
