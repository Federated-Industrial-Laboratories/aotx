/* Purpose: Write untrusted body bytes as the content of a JSON string.
 * Owns: Nothing; the caller owns the buffer that takes the result.
 * Threading: One thread; the function holds no state.
 * Lifetime: The call. */
#include "disk/wire/diskwire.h"

#include <stdio.h>
#include <string.h>

/* A body can hold any byte, because a fault on the device can write anything. A line of
 * the message file must hold valid UTF-8. A byte that is not part of a valid sequence
 * becomes a question mark. A byte that JSON refuses becomes an escape. */

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

/* Reports whether a code point is white space by the rule the line schema applies. The set
 * holds the ASCII controls that count as space, the space itself, and the space code points
 * above ASCII. A text of these only is a text field that holds nothing. */
static int is_space(uint32_t code)
{
    if (code >= 0x09u && code <= 0x0du) {
        return 1;
    }
    if (code >= 0x1cu && code <= 0x20u) {
        return 1;
    }
    if (code == 0x85u || code == 0xa0u || code == 0x1680u) {
        return 1;
    }
    if (code >= 0x2000u && code <= 0x200au) {
        return 1;
    }
    if (code == 0x2028u || code == 0x2029u || code == 0x202fu || code == 0x205fu) {
        return 1;
    }
    return code == 0x3000u;
}

/* Reads the code point of a valid sequence of the given length. */
static uint32_t utf8_code(const unsigned char *p, int length)
{
    static const unsigned char first[5] = { 0u, 0x7fu, 0x1fu, 0x0fu, 0x07u };
    uint32_t code = (uint32_t)(p[0] & first[length]);
    int i;
    for (i = 1; i < length; i++) {
        code = (code << 6) | (uint32_t)(p[i] & 0x3fu);
    }
    return code;
}

int aotx_json_has_text(const unsigned char *body, uint32_t len)
{
    uint32_t i = 0;
    while (i < len) {
        int step = utf8_length(body + i, len - i);
        if (step == 0) {
            /* A byte that is not part of a valid sequence becomes a question mark, which
             * is text. */
            return 1;
        }
        if (!is_space(utf8_code(body + i, step))) {
            return 1;
        }
        i += (uint32_t)step;
    }
    return 0;
}

/* A byte that is not part of a valid sequence becomes a question mark, because a message
 * line must hold valid UTF-8. */
size_t aotx_json_write(char *out, size_t out_bytes, const unsigned char *body, uint32_t len)
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
