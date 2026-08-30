/* Purpose: Check the key decoder against the table of the sources it was written from.
 * Owns: One decoder state for each case.
 * Threading: One thread.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include "disk/tui/tui.h"

#define AOTX_TEST_KEYS 4096u

/* One row of the table: the bytes a terminal sends, the key they mean, and the modifier
 * the key carries. The name states the terminal the form comes from. */
typedef struct aotx_test_row {
    const char  *bytes;
    unsigned int code;
    unsigned int mods;
    const char  *name;
} aotx_test_row;

static const aotx_test_row aotx_table[] = {
    /* The arrows in the state the program leaves the terminal in. */
    { "\033[A", AOTX_TUI_KEY_UP,    0, "up, every terminal" },
    { "\033[B", AOTX_TUI_KEY_DOWN,  0, "down, every terminal" },
    { "\033[C", AOTX_TUI_KEY_RIGHT, 0, "right, every terminal" },
    { "\033[D", AOTX_TUI_KEY_LEFT,  0, "left, every terminal" },
    /* The arrows of a terminal that a program left in application mode. */
    { "\033OA", AOTX_TUI_KEY_UP,    0, "up, application mode" },
    { "\033OB", AOTX_TUI_KEY_DOWN,  0, "down, application mode" },
    { "\033OC", AOTX_TUI_KEY_RIGHT, 0, "right, application mode" },
    { "\033OD", AOTX_TUI_KEY_LEFT,  0, "left, application mode" },
    /* Home and End have two shapes: the letter form and the number form. */
    { "\033[H",  AOTX_TUI_KEY_HOME, 0, "home, xterm and vte and kitty" },
    { "\033[F",  AOTX_TUI_KEY_END,  0, "end, xterm and vte and kitty" },
    { "\033OH",  AOTX_TUI_KEY_HOME, 0, "home, application mode" },
    { "\033OF",  AOTX_TUI_KEY_END,  0, "end, application mode" },
    { "\033[1~", AOTX_TUI_KEY_HOME, 0, "home, console and multiplexer" },
    { "\033[4~", AOTX_TUI_KEY_END,  0, "end, console and multiplexer" },
    /* The keys of the number form. */
    { "\033[2~", AOTX_TUI_KEY_INSERT,  0, "insert" },
    { "\033[3~", AOTX_TUI_KEY_DELETE,  0, "delete" },
    { "\033[5~", AOTX_TUI_KEY_PAGE_UP, 0, "page up" },
    { "\033[6~", AOTX_TUI_KEY_PAGE_DN, 0, "page down" },
    /* The first four function keys have three shapes. */
    { "\033OP", AOTX_TUI_KEY_F1,      0, "f1, xterm and vte and kitty" },
    { "\033OQ", AOTX_TUI_KEY_F1 + 1u, 0, "f2, xterm and vte and kitty" },
    { "\033OR", AOTX_TUI_KEY_F1 + 2u, 0, "f3, xterm and vte and kitty" },
    { "\033OS", AOTX_TUI_KEY_F1 + 3u, 0, "f4, xterm and vte and kitty" },
    { "\033[11~", AOTX_TUI_KEY_F1,      0, "f1, old xterm" },
    { "\033[12~", AOTX_TUI_KEY_F1 + 1u, 0, "f2, old xterm" },
    { "\033[13~", AOTX_TUI_KEY_F1 + 2u, 0, "f3, old xterm" },
    { "\033[14~", AOTX_TUI_KEY_F1 + 3u, 0, "f4, old xterm" },
    { "\033[[A", AOTX_TUI_KEY_F1,      0, "f1, console" },
    { "\033[[B", AOTX_TUI_KEY_F1 + 1u, 0, "f2, console" },
    { "\033[[C", AOTX_TUI_KEY_F1 + 2u, 0, "f3, console" },
    { "\033[[D", AOTX_TUI_KEY_F1 + 3u, 0, "f4, console" },
    { "\033[[E", AOTX_TUI_KEY_F1 + 4u, 0, "f5, console" },
    /* The function keys from five to twelve. */
    { "\033[15~", AOTX_TUI_KEY_F1 + 4u,  0, "f5" },
    { "\033[17~", AOTX_TUI_KEY_F1 + 5u,  0, "f6" },
    { "\033[18~", AOTX_TUI_KEY_F1 + 6u,  0, "f7" },
    { "\033[19~", AOTX_TUI_KEY_F1 + 7u,  0, "f8" },
    { "\033[20~", AOTX_TUI_KEY_F1 + 8u,  0, "f9" },
    { "\033[21~", AOTX_TUI_KEY_F1 + 9u,  0, "f10" },
    { "\033[23~", AOTX_TUI_KEY_F1 + 10u, 0, "f11" },
    { "\033[24~", AOTX_TUI_KEY_F1 + 11u, 0, "f12" },
    /* A modifier goes in the parameter before the final byte. */
    { "\033[1;5A",  AOTX_TUI_KEY_UP, AOTX_TUI_MOD_CONTROL, "up with control" },
    { "\033[1;3D",  AOTX_TUI_KEY_LEFT, AOTX_TUI_MOD_ALT, "left with alt" },
    { "\033[1;2P",  AOTX_TUI_KEY_F1, AOTX_TUI_MOD_SHIFT, "f1 with shift" },
    { "\033[15;3~", AOTX_TUI_KEY_F1 + 4u, AOTX_TUI_MOD_ALT, "f5 with alt" },
    { "\033[1;4C",  AOTX_TUI_KEY_RIGHT, AOTX_TUI_MOD_SHIFT | AOTX_TUI_MOD_ALT,
      "right with shift and alt" },
    /* The named control bytes carry the codes the window sends. */
    { "\r",   AOTX_TUI_KEY_ENTER, 0, "enter" },
    { "\n",   AOTX_TUI_KEY_ENTER, 0, "enter as a line feed" },
    { "\t",   AOTX_TUI_KEY_TAB,   0, "tab" },
    { "\177", AOTX_TUI_KEY_BACK,  0, "backspace" },
    { "\010", AOTX_TUI_KEY_BACK,  0, "backspace as the control byte" },
};

#define AOTX_TABLE_ROWS (sizeof(aotx_table) / sizeof(aotx_table[0]))

/* The sequences the decoder must drop whole and never give as text. */
static const char *aotx_dropped[] = {
    "\033[1u",        /* the form for a key with no legacy encoding */
    "\033[?1;2c",     /* a report with a private parameter */
    "\033[200~",      /* a paste mark, which this program never asks for */
    "\033[>4;2m",     /* a private form with a parameter this build does not read */
    "\033Oz",         /* a final byte that no key of the table holds */
    "\033[[Z",        /* the console form with a letter that is not one of five */
};

#define AOTX_DROPPED_ROWS (sizeof(aotx_dropped) / sizeof(aotx_dropped[0]))

static aotx_tui_key aotx_out[AOTX_TEST_KEYS];

/* Decodes one text in one read and gives the keys. */
static unsigned int decode(aotx_keys *k, const char *bytes)
{
    return aotx_keys_take(k, (const unsigned char *)bytes, strlen(bytes), 1000000000ull,
                          aotx_out, AOTX_TEST_KEYS);
}

/* Each row of the table decodes to the key it names, on its own. */
static void one_by_one(void)
{
    unsigned int i;
    for (i = 0; i < AOTX_TABLE_ROWS; i++) {
        aotx_keys k;
        unsigned int got;
        memset(&k, 0, sizeof(k));
        got = decode(&k, aotx_table[i].bytes);
        CHECK(got == 1u, "%s gives %u keys, not one", aotx_table[i].name, got);
        if (got != 1u) {
            continue;
        }
        CHECK(aotx_out[0].code == aotx_table[i].code, "%s gives the code %u, not %u",
              aotx_table[i].name, aotx_out[0].code, aotx_table[i].code);
        CHECK(aotx_out[0].mods == aotx_table[i].mods, "%s gives the modifier %u, not %u",
              aotx_table[i].name, aotx_out[0].mods, aotx_table[i].mods);
        CHECK(k.fill == 0u, "%s holds bytes after the decode", aotx_table[i].name);
    }
}

/* A batch of n sequences in one read decodes to n keys in order. */
static void batch(int n)
{
    aotx_keys k;
    char bytes[AOTX_TEST_KEYS];
    unsigned int at = 0;
    unsigned int got;
    int i;
    memset(&k, 0, sizeof(k));
    for (i = 0; i < n; i++) {
        const aotx_test_row *row = &aotx_table[(unsigned int)i % AOTX_TABLE_ROWS];
        size_t bytes_of = strlen(row->bytes);
        memcpy(bytes + at, row->bytes, bytes_of);
        at += (unsigned int)bytes_of;
    }
    got = aotx_keys_take(&k, (const unsigned char *)bytes, at, 1000000000ull, aotx_out,
                         AOTX_TEST_KEYS);
    CHECK(got == (unsigned int)n, "a read of %d sequences gives %u keys", n, got);
    for (i = 0; i < n && (unsigned int)i < got; i++) {
        const aotx_test_row *row = &aotx_table[(unsigned int)i % AOTX_TABLE_ROWS];
        CHECK(aotx_out[i].code == row->code, "key %d of the batch is %u, not %u", i,
              aotx_out[i].code, row->code);
        CHECK(aotx_out[i].mods == row->mods, "key %d of the batch has the wrong modifier",
              i);
    }
    CHECK(k.fill == 0u, "the batch holds bytes after the decode");
}

/* Every line feed in one read is one Enter key. Text after the first one therefore starts
 * a new line and does not join the line before it. */
static void two_lines(void)
{
    static const char text[] = "skills\nsay what is a tick\n";
    aotx_keys k;
    unsigned int got;
    unsigned int enters = 0u;
    unsigned int first = ~0u;
    unsigned int second = ~0u;
    memset(&k, 0, sizeof(k));
    got = decode(&k, text);
    for (unsigned int i = 0u; i < got; ++i) {
        if (aotx_out[i].code == AOTX_TUI_KEY_ENTER) {
            if (enters == 0u) {
                first = i;
            } else if (enters == 1u) {
                second = i;
            }
            enters++;
        }
    }
    CHECK(enters == 2u, "two lines in one read give %u Enter keys", enters);
    CHECK(first == 6u, "the first Enter key is at %u, not after skills", first);
    CHECK(second == got - 1u, "the second Enter key is not the last key");
    CHECK(k.fill == 0u, "the two lines leave bytes in the decoder");
}

/* A lone escape is the Escape key after the wait, and not before it. */
static void lone_escape(void)
{
    aotx_keys k;
    aotx_tui_key late;
    uint64_t now = 1000000000ull;
    uint64_t wait = (uint64_t)AOTX_TUI_ESCAPE_DEFAULT * 1000000ull;
    memset(&k, 0, sizeof(k));
    CHECK(aotx_keys_take(&k, (const unsigned char *)"\033", 1u, now, aotx_out,
                         AOTX_TEST_KEYS) == 0u, "a lone escape gives a key at once");
    CHECK(k.fill == 1u, "the lone escape is not held");
    CHECK(aotx_keys_timeout(&k, now) > 0, "the wait gives no time to the poll");
    CHECK(aotx_keys_wait(&k, now + wait - 1000000ull, &late) == 0u,
          "the escape came before its wait ended");
    CHECK(aotx_keys_wait(&k, now + wait, &late) == 1u, "the escape did not come");
    CHECK(late.code == AOTX_TUI_KEY_ESCAPE, "the key after the wait is not Escape");
    CHECK(k.fill == 0u, "the escape stays held after it was given");
    CHECK(aotx_keys_timeout(&k, now) == -1, "an empty decoder asks the poll to wait");

    /* An escape that a sequence follows inside the wait is not the Escape key. */
    memset(&k, 0, sizeof(k));
    CHECK(aotx_keys_take(&k, (const unsigned char *)"\033", 1u, now, aotx_out,
                         AOTX_TEST_KEYS) == 0u, "a lone escape gives a key at once");
    CHECK(aotx_keys_take(&k, (const unsigned char *)"[A", 2u, now + 1000000ull, aotx_out,
                         AOTX_TEST_KEYS) == 1u, "the sequence did not complete");
    CHECK(aotx_out[0].code == AOTX_TUI_KEY_UP, "the completed sequence is not the up key");

    /* The setting changes the wait at both ends of its range. */
    for (unsigned int ms = 5u; ms <= 500u; ms += 495u) {
        memset(&k, 0, sizeof(k));
        k.escape_ms = ms;
        CHECK(aotx_keys_take(&k, (const unsigned char *)"\033", 1u, now, aotx_out,
                             AOTX_TEST_KEYS) == 0u, "the set wait gives a key at once");
        CHECK(aotx_keys_wait(&k, now + (uint64_t)ms * 1000000ull - 1u, &late) == 0u,
              "the set wait of %u ms ended early", ms);
        CHECK(aotx_keys_wait(&k, now + (uint64_t)ms * 1000000ull, &late) == 1u,
              "the set wait of %u ms did not end", ms);
    }
}

/* A sequence that arrives in parts, one byte at a time, decodes to one key. */
static void in_parts(void)
{
    unsigned int i;
    for (i = 0; i < AOTX_TABLE_ROWS; i++) {
        aotx_keys k;
        const char *bytes = aotx_table[i].bytes;
        size_t count = strlen(bytes);
        size_t at;
        unsigned int got = 0;
        memset(&k, 0, sizeof(k));
        for (at = 0; at < count; at++) {
            got += aotx_keys_take(&k, (const unsigned char *)bytes + at, 1u,
                                  1000000000ull + at, aotx_out + got,
                                  AOTX_TEST_KEYS - got);
        }
        CHECK(got == 1u, "%s in parts gives %u keys, not one", aotx_table[i].name, got);
        if (got == 1u) {
            CHECK(aotx_out[0].code == aotx_table[i].code,
                  "%s in parts gives the wrong code", aotx_table[i].name);
        }
    }
}

/* A sequence this build does not know is dropped whole and never given as text. */
static void dropped(void)
{
    unsigned int i;
    for (i = 0; i < AOTX_DROPPED_ROWS; i++) {
        aotx_keys k;
        char bytes[64];
        unsigned int got;
        memset(&k, 0, sizeof(k));
        snprintf(bytes, sizeof(bytes), "%sa", aotx_dropped[i]);
        got = decode(&k, bytes);
        CHECK(got == 1u, "%s gives %u keys, not the one letter after it", aotx_dropped[i],
              got);
        if (got >= 1u) {
            CHECK(aotx_out[got - 1u].code == 0u
                  && aotx_out[got - 1u].codepoint == (unsigned int)'a',
                  "the letter after %s did not come", aotx_dropped[i]);
        }
        CHECK(k.dropped >= 1u, "%s was not counted as dropped", aotx_dropped[i]);
    }
}

/* The printing bytes go out as code points, and a byte above the font is dropped. */
static void text(void)
{
    aotx_keys k;
    unsigned int got;
    unsigned char high[3];
    memset(&k, 0, sizeof(k));
    got = decode(&k, "ab z");
    CHECK(got == 4u, "four letters give %u keys", got);
    CHECK(aotx_out[0].codepoint == (unsigned int)'a' && aotx_out[0].code == 0u,
          "the first letter is not a character key");
    CHECK(aotx_out[3].codepoint == (unsigned int)'z', "the last letter is wrong");
    memset(&k, 0, sizeof(k));
    got = decode(&k, "\014");
    CHECK(got == 1u && aotx_out[0].codepoint == 0x0cu
          && (aotx_out[0].mods & AOTX_TUI_MOD_CONTROL) != 0,
          "the control byte does not carry the control modifier");
    high[0] = 0xc3u;
    high[1] = 0xa9u;
    high[2] = 0;
    memset(&k, 0, sizeof(k));
    got = aotx_keys_take(&k, high, 2u, 1000000000ull, aotx_out, AOTX_TEST_KEYS);
    CHECK(got == 0u, "a byte above the font gives %u keys, not none", got);
    CHECK(k.dropped == 2u, "the bytes above the font are not counted");
}

/* A stream of bytes that never completes a sequence is dropped and never overruns. The
 * escape and the bracket never come back as text; the bytes after the drop are text and
 * are the only keys the decoder gives. */
static void overrun(void)
{
    aotx_keys k;
    char bytes[AOTX_TUI_SEQUENCE * 4u];
    unsigned int got;
    unsigned int i;
    memset(&k, 0, sizeof(k));
    memset(bytes, '1', sizeof(bytes));
    bytes[0] = '\033';
    bytes[1] = '[';
    got = aotx_keys_take(&k, (const unsigned char *)bytes, sizeof(bytes), 1000000000ull,
                         aotx_out, AOTX_TEST_KEYS);
    CHECK(k.fill < AOTX_TUI_SEQUENCE, "the decoder holds a buffer that is full");
    CHECK(k.dropped >= 1u, "the sequence that never ends is not counted");
    CHECK(got < sizeof(bytes), "the decoder gave a key for every byte of the sequence");
    for (i = 0; i < got; i++) {
        CHECK(aotx_out[i].code == 0u && aotx_out[i].codepoint == (unsigned int)'1',
              "key %u of the stream is not the byte after the sequence", i);
    }
}

int main(void)
{
    one_by_one();
    batch(1);
    batch(64);
    two_lines();
    lone_escape();
    in_parts();
    dropped();
    text();
    overrun();
    return aotx_report("keys_test", 300);
}
