/* Purpose: Read back the fields of one line that the drain wrote.
 * Owns: Nothing; the caller owns the buffer that takes a value.
 * Threading: One thread; neither function holds state.
 * Lifetime: The call. */
#include "disk/wire/diskwire.h"

#include <string.h>

/* The drain writes these lines, so the escape set is the one the drain writes. A reader of
 * a line that another program wrote gets a question mark for a byte it does not know. */

int aotx_json_number(const char *line, const char *key, uint64_t *out)
{
    const char *at = strstr(line, key);
    if (at == NULL) {
        return 0;
    }
    at += strlen(key);
    if (*at < '0' || *at > '9') {
        return 0;
    }
    *out = 0;
    while (*at >= '0' && *at <= '9') {
        *out = *out * 10u + (uint64_t)(*at - '0');
        at++;
    }
    return 1;
}

/* Gives the value of one hexadecimal character, or -1. */
static int hex_of(char c)
{
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

int aotx_json_text(const char *line, const char *key, char *out, size_t out_bytes)
{
    const char *at = strstr(line, key);
    size_t used = 0;
    if (at == NULL || out_bytes == 0) {
        return 0;
    }
    at += strlen(key);
    while (*at != '\0' && *at != '"' && used + 1 < out_bytes) {
        if (*at != '\\') {
            out[used++] = *at++;
            continue;
        }
        at++;
        if (*at == '"' || *at == '\\' || *at == '/') {
            out[used++] = *at++;
        } else if (*at == 'n') {
            out[used++] = '\n';
            at++;
        } else if (*at == 't') {
            out[used++] = '\t';
            at++;
        } else if (*at == 'r') {
            out[used++] = '\r';
            at++;
        } else if (*at == 'b' || *at == 'f') {
            out[used++] = (*at == 'b') ? '\b' : '\f';
            at++;
        } else if (*at == 'u') {
            int a = (at[1] != '\0') ? hex_of(at[1]) : -1;
            int b = (a >= 0 && at[2] != '\0') ? hex_of(at[2]) : -1;
            int c = (b >= 0 && at[3] != '\0') ? hex_of(at[3]) : -1;
            int e = (c >= 0 && at[4] != '\0') ? hex_of(at[4]) : -1;
            int code = (e >= 0) ? (((a * 16 + b) * 16 + c) * 16 + e) : -1;
            if (e < 0) {
                out[used] = '\0';
                return 0;
            }
            /* A code point above ASCII becomes a question mark, which no file name holds
             * by accident. */
            out[used++] = (code > 0 && code < 0x80) ? (char)code : '?';
            at += 5;
        } else {
            out[used++] = '?';
            if (*at != '\0') {
                at++;
            }
        }
    }
    out[used] = '\0';
    return (*at == '"') ? 1 : 0;
}
