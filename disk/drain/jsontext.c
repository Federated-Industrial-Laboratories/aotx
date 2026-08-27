/* Purpose: Write untrusted body bytes as the content of a JSON string.
 * Owns: Nothing; the caller owns the buffer that takes the result.
 * Threading: One thread; the function holds no state.
 * Lifetime: The call. */
#include "disk/drain/derive.h"

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

/* A byte that is not part of a valid sequence becomes a question mark, because a message
 * line must hold valid UTF-8. */
size_t aotx_derive_text(char *out, size_t out_bytes, const unsigned char *body, uint32_t len)
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
