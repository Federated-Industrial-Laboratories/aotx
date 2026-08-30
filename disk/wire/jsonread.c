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

static const char *json_space(const char *at)
{
    while (*at == ' ' || *at == '\t' || *at == '\r' || *at == '\n') {
        at++;
    }
    return at;
}

static const char *json_string(const char *at)
{
    if (*at++ != '"') {
        return NULL;
    }
    while (*at != '\0' && *at != '"') {
        unsigned char byte = (unsigned char)*at++;
        if (byte < 0x20u) {
            return NULL;
        }
        if (byte != '\\') {
            continue;
        }
        if (*at == '"' || *at == '\\' || *at == '/' || *at == 'b' || *at == 'f'
            || *at == 'n' || *at == 'r' || *at == 't') {
            at++;
            continue;
        }
        if (*at++ != 'u') {
            return NULL;
        }
        for (int i = 0; i < 4; ++i) {
            if (hex_of(*at++) < 0) {
                return NULL;
            }
        }
    }
    return (*at == '"') ? at + 1 : NULL;
}

static const char *json_number_value(const char *at)
{
    if (*at == '-') {
        at++;
    }
    if (*at == '0') {
        at++;
    } else {
        if (*at < '1' || *at > '9') {
            return NULL;
        }
        while (*at >= '0' && *at <= '9') {
            at++;
        }
    }
    if (*at == '.') {
        at++;
        if (*at < '0' || *at > '9') {
            return NULL;
        }
        while (*at >= '0' && *at <= '9') {
            at++;
        }
    }
    if (*at == 'e' || *at == 'E') {
        at++;
        if (*at == '+' || *at == '-') {
            at++;
        }
        if (*at < '0' || *at > '9') {
            return NULL;
        }
        while (*at >= '0' && *at <= '9') {
            at++;
        }
    }
    return at;
}

static const char *json_value(const char *at, unsigned int depth);

static const char *json_array(const char *at, unsigned int depth)
{
    at = json_space(at + 1);
    if (*at == ']') {
        return at + 1;
    }
    for (;;) {
        at = json_value(at, depth + 1u);
        if (at == NULL) {
            return NULL;
        }
        at = json_space(at);
        if (*at == ']') {
            return at + 1;
        }
        if (*at != ',') {
            return NULL;
        }
        at = json_space(at + 1);
    }
}

static const char *json_object(const char *at, unsigned int depth)
{
    at = json_space(at + 1);
    if (*at == '}') {
        return at + 1;
    }
    for (;;) {
        at = json_string(at);
        if (at == NULL) {
            return NULL;
        }
        at = json_space(at);
        if (*at != ':') {
            return NULL;
        }
        at = json_value(json_space(at + 1), depth + 1u);
        if (at == NULL) {
            return NULL;
        }
        at = json_space(at);
        if (*at == '}') {
            return at + 1;
        }
        if (*at != ',') {
            return NULL;
        }
        at = json_space(at + 1);
    }
}

static const char *json_value(const char *at, unsigned int depth)
{
    if (depth > 16u) {
        return NULL;
    }
    if (*at == '"') {
        return json_string(at);
    }
    if (*at == '{') {
        return json_object(at, depth);
    }
    if (*at == '[') {
        return json_array(at, depth);
    }
    if (strncmp(at, "true", 4u) == 0) {
        return at + 4;
    }
    if (strncmp(at, "false", 5u) == 0) {
        return at + 5;
    }
    if (strncmp(at, "null", 4u) == 0) {
        return at + 4;
    }
    return json_number_value(at);
}

int aotx_json_whole(const char *line)
{
    const char *end;
    if (line == NULL) {
        return 0;
    }
    end = json_value(json_space(line), 0u);
    return (end != NULL && *json_space(end) == '\0') ? 1 : 0;
}
