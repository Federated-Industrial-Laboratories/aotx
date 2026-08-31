/* Purpose: Decode the byte sequences of a terminal into the key codes the window sends.
 * Owns: The bytes of a sequence that a read did not complete.
 * Threading: One thread; the decoder holds no state between programs.
 * Lifetime: The whole run. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/tui/tui.h"

#include <string.h>

#define AOTX_ESC 0x1b

/* The parameter bytes and the intermediate bytes of a control sequence, from the
 * standard. A parameter byte is 03/00 to 03/15. An intermediate byte is 02/00 to
 * 02/15. */
#define AOTX_PARAM_FIRST 0x30
#define AOTX_PARAM_LAST  0x3f
#define AOTX_INTER_FIRST 0x20
#define AOTX_INTER_LAST  0x2f

typedef struct aotx_keys_out {
    aotx_tui_key *key;
    unsigned int  count;
    unsigned int  most;
} aotx_keys_out;

static void emit(aotx_keys_out *out, unsigned int code, unsigned int codepoint,
                 unsigned int mods)
{
    if (out->count >= out->most) {
        return;
    }
    out->key[out->count].code = code;
    out->key[out->count].codepoint = codepoint;
    out->key[out->count].mods = mods;
    out->count++;
}

/* The modifier of a sequence. The parameter is one more than a mask whose bits are shift,
 * alt and control, in that order. */
static unsigned int mods_of(unsigned int parameter)
{
    unsigned int mask;
    unsigned int mods = 0;
    if (parameter < 2u) {
        return 0;
    }
    mask = parameter - 1u;
    if ((mask & 1u) != 0) {
        mods |= AOTX_TUI_MOD_SHIFT;
    }
    if ((mask & 2u) != 0) {
        mods |= AOTX_TUI_MOD_ALT;
    }
    if ((mask & 4u) != 0) {
        mods |= AOTX_TUI_MOD_CONTROL;
    }
    return mods;
}

/* The key of a final byte that is a letter. Home and End have this form on every terminal
 * of the table but the console and the multiplexer, which use the number form below. */
static unsigned int key_of_letter(unsigned char final)
{
    switch (final) {
    case 'A': return AOTX_TUI_KEY_UP;
    case 'B': return AOTX_TUI_KEY_DOWN;
    case 'C': return AOTX_TUI_KEY_RIGHT;
    case 'D': return AOTX_TUI_KEY_LEFT;
    case 'H': return AOTX_TUI_KEY_HOME;
    case 'F': return AOTX_TUI_KEY_END;
    case 'P': return AOTX_TUI_KEY_F1;
    case 'Q': return AOTX_TUI_KEY_F1 + 1u;
    case 'R': return AOTX_TUI_KEY_F1 + 2u;
    case 'S': return AOTX_TUI_KEY_F1 + 3u;
    default:  return 0;
    }
}

/* The key of the first parameter of a sequence that ends with a tilde. */
static unsigned int key_of_number(unsigned int parameter)
{
    if (parameter >= 11u && parameter <= 14u) {
        return AOTX_TUI_KEY_F1 + (parameter - 11u);
    }
    if (parameter == 15u) {
        return AOTX_TUI_KEY_F1 + 4u;
    }
    if (parameter >= 17u && parameter <= 21u) {
        return AOTX_TUI_KEY_F1 + 5u + (parameter - 17u);
    }
    if (parameter == 23u || parameter == 24u) {
        return AOTX_TUI_KEY_F1 + 10u + (parameter - 23u);
    }
    switch (parameter) {
    case 1u: return AOTX_TUI_KEY_HOME;
    case 2u: return AOTX_TUI_KEY_INSERT;
    case 3u: return AOTX_TUI_KEY_DELETE;
    case 4u: return AOTX_TUI_KEY_END;
    case 5u: return AOTX_TUI_KEY_PAGE_UP;
    case 6u: return AOTX_TUI_KEY_PAGE_DN;
    default: return 0;
    }
}

/* Reads the two parameters of a sequence. Returns 0 when the bytes hold a private
 * parameter, which this program never asks for and drops. */
static int parameters(const unsigned char *bytes, unsigned int count,
                      unsigned int *first, unsigned int *second)
{
    unsigned int at = 0;
    unsigned int which = 0;
    unsigned int value = 0;
    int digits = 0;
    *first = 1u;
    *second = 1u;
    if (count > 0 && (bytes[0] == '?' || bytes[0] == '<' || bytes[0] == '='
                      || bytes[0] == '>')) {
        return 0;
    }
    for (at = 0; at <= count; at++) {
        if (at < count && bytes[at] >= '0' && bytes[at] <= '9') {
            value = (value * 10u) + (unsigned int)(bytes[at] - '0');
            digits = 1;
            if (value > 100000u) {
                return 0;
            }
            continue;
        }
        if (at == count || bytes[at] == ';' || bytes[at] == ':') {
            if (digits != 0) {
                if (which == 0) {
                    *first = value;
                } else if (which == 1) {
                    *second = value;
                }
            }
            which++;
            value = 0;
            digits = 0;
            continue;
        }
        return 0;
    }
    return 1;
}

/* Decodes one whole control sequence. The bytes hold the parameters and the final byte
 * follows them. A sequence this build does not know is dropped whole. */
static void decode_csi(aotx_keys *k, aotx_keys_out *out, const unsigned char *bytes,
                       unsigned int count, unsigned char final)
{
    unsigned int first;
    unsigned int second;
    unsigned int code;
    if (parameters(bytes, count, &first, &second) == 0) {
        k->dropped++;
        return;
    }
    code = (final == '~') ? key_of_number(first) : key_of_letter(final);
    if (code == 0) {
        k->dropped++;
        return;
    }
    emit(out, code, 0, mods_of(second));
}

/* One byte that is not part of a sequence. The named control bytes carry the codes the
 * window sends. Every other control byte goes out as its own code point with the control
 * bit. The program thus acts on it, and the console drops it. */
static void decode_byte(aotx_keys *k, aotx_keys_out *out, unsigned char byte)
{
    switch (byte) {
    case '\r':
    case '\n':
        emit(out, AOTX_TUI_KEY_ENTER, 0, 0);
        return;
    case '\t':
        emit(out, AOTX_TUI_KEY_TAB, 0, 0);
        return;
    case 0x08:
    case 0x7f:
        emit(out, AOTX_TUI_KEY_BACK, 0, 0);
        return;
    default:
        break;
    }
    if (byte >= 0x20 && byte < 0x7f) {
        emit(out, 0, byte, 0);
        return;
    }
    if (byte < 0x20) {
        emit(out, 0, byte, AOTX_TUI_MOD_CONTROL);
        return;
    }
    /* A byte above the font is dropped, as the window drops it. */
    k->dropped++;
}

/* Takes bytes off the front of the held sequence. */
static void shift(aotx_keys *k, unsigned int count)
{
    if (count >= k->fill) {
        k->fill = 0;
        return;
    }
    memmove(k->part, k->part + count, k->fill - count);
    k->fill -= count;
}

/* Decodes what the held bytes hold. Returns 1 when the bytes at the front are a sequence
 * that is not complete, so the caller waits for more. */
static int decode(aotx_keys *k, aotx_keys_out *out)
{
    while (k->fill > 0) {
        unsigned int at;
        unsigned char byte = k->part[0];
        if (byte != AOTX_ESC) {
            decode_byte(k, out, byte);
            shift(k, 1u);
            continue;
        }
        if (k->fill < 2u) {
            return 1;
        }
        if (k->part[1] == 'O') {
            unsigned int code;
            if (k->fill < 3u) {
                return 1;
            }
            code = key_of_letter(k->part[2]);
            if (code == 0) {
                k->dropped++;
            } else {
                emit(out, code, 0, 0);
            }
            shift(k, 3u);
            continue;
        }
        /* A terminal sends Alt with the legacy key as an escape prefix. Keep Enter as a
         * named key and carry the modifier, so the device editor can insert a line break. */
        if (k->part[1] == '\r' || k->part[1] == '\n') {
            emit(out, AOTX_TUI_KEY_ENTER, 0, AOTX_TUI_MOD_ALT);
            shift(k, 2u);
            continue;
        }
        if (k->part[1] != '[') {
            /* A sequence this build does not know is dropped whole and never given as
             * text. */
            k->dropped++;
            shift(k, 2u);
            continue;
        }
        if (k->fill < 3u) {
            return 1;
        }
        /* The console sends the first five function keys as an escape, two open brackets
         * and one letter. The second bracket is a final byte, so a strict reader would end
         * the sequence at it; this case comes before the general rule. */
        if (k->part[2] == '[') {
            if (k->fill < 4u) {
                return 1;
            }
            if (k->part[3] >= 'A' && k->part[3] <= 'E') {
                emit(out, AOTX_TUI_KEY_F1 + (unsigned int)(k->part[3] - 'A'), 0, 0);
            } else {
                k->dropped++;
            }
            shift(k, 4u);
            continue;
        }
        at = 2u;
        while (at < k->fill && k->part[at] >= AOTX_PARAM_FIRST
               && k->part[at] <= AOTX_PARAM_LAST) {
            at++;
        }
        while (at < k->fill && k->part[at] >= AOTX_INTER_FIRST
               && k->part[at] <= AOTX_INTER_LAST) {
            at++;
        }
        if (at == k->fill) {
            return 1;
        }
        decode_csi(k, out, k->part + 2, at - 2u, k->part[at]);
        shift(k, at + 1u);
    }
    return 0;
}

unsigned int aotx_keys_take(aotx_keys *k, const unsigned char *bytes, size_t count,
                            uint64_t now_ns, aotx_tui_key *out, unsigned int most)
{
    aotx_keys_out sink;
    size_t at = 0;
    sink.key = out;
    sink.count = 0;
    sink.most = most;
    while (at < count) {
        unsigned int room = AOTX_TUI_SEQUENCE - k->fill;
        size_t take = (count - at < room) ? count - at : room;
        memcpy(k->part + k->fill, bytes + at, take);
        k->fill += (unsigned int)take;
        at += take;
        if (decode(k, &sink) != 0 && k->fill == AOTX_TUI_SEQUENCE) {
            /* A sequence that fills the buffer is longer than any form of the table. The
             * bytes go away whole, so no part of them is read as text. */
            k->dropped++;
            k->fill = 0;
        }
    }
    k->escape_ns = (k->fill > 0) ? ((k->escape_ns != 0) ? k->escape_ns : now_ns) : 0;
    return sink.count;
}

unsigned int aotx_keys_wait(aotx_keys *k, uint64_t now_ns, aotx_tui_key *out)
{
    unsigned int ms = (k->escape_ms != 0u) ? k->escape_ms : AOTX_TUI_ESCAPE_DEFAULT;
    uint64_t wait = (uint64_t)ms * 1000000ull;
    if (k->fill == 0 || k->escape_ns == 0 || now_ns < k->escape_ns + wait) {
        return 0;
    }
    if (k->fill == 1u && k->part[0] == AOTX_ESC) {
        out->code = AOTX_TUI_KEY_ESCAPE;
        out->codepoint = 0;
        out->mods = 0;
        k->fill = 0;
        k->escape_ns = 0;
        return 1;
    }
    /* A sequence that stopped in the middle is not a key. The bytes go away, and the
     * escape they started is not given as text. */
    k->dropped++;
    k->fill = 0;
    k->escape_ns = 0;
    return 0;
}

int aotx_keys_timeout(const aotx_keys *k, uint64_t now_ns)
{
    unsigned int ms = (k->escape_ms != 0u) ? k->escape_ms : AOTX_TUI_ESCAPE_DEFAULT;
    uint64_t wait = (uint64_t)ms * 1000000ull;
    uint64_t due;
    if (k->fill == 0 || k->escape_ns == 0) {
        return -1;
    }
    due = k->escape_ns + wait;
    if (now_ns >= due) {
        return 0;
    }
    return (int)((due - now_ns + 999999ull) / 1000000ull);
}
