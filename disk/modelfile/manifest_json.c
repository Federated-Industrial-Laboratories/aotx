/* Purpose: Decode bounded manifest strings without a key search inside values.
 * Owns: No storage; output bytes belong to the caller.
 * Threading: One caller for each cursor.
 * Lifetime: One call. */
#include "disk/modelfile/manifest_json.h"
#include <string.h>

static void space(aotx_manifest_json *j)
{
    while (j->at < j->end && (*j->at == ' ' || *j->at == '\t' ||
           *j->at == '\r' || *j->at == '\n')) ++j->at;
}

int aotx_manifest_json_take(aotx_manifest_json *j, unsigned char byte)
{
    space(j);
    if (j->at == j->end || *j->at != byte) return -1;
    ++j->at;
    return 0;
}

int aotx_manifest_json_end(aotx_manifest_json *j)
{
    space(j);
    return j->at == j->end ? 0 : -1;
}

static int hex(aotx_manifest_json *j, uint32_t *out)
{
    uint32_t value = 0;
    if ((size_t)(j->end - j->at) < 4u) return -1;
    for (unsigned i = 0; i < 4u; ++i) {
        unsigned char c = *j->at++;
        unsigned digit;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10u;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10u;
        else return -1;
        value = value * 16u + digit;
    }
    *out = value;
    return 0;
}

static int put(unsigned char *out, size_t room, size_t *used, unsigned char c)
{
    if (*used == room) return -1;
    out[(*used)++] = c;
    return 0;
}

int aotx_manifest_json_string(aotx_manifest_json *j, unsigned char *out,
                              size_t room, size_t *length)
{
    size_t used = 0;
    if (aotx_manifest_json_take(j, '"') != 0) return -1;
    while (j->at < j->end) {
        unsigned char c = *j->at++;
        uint32_t cp;
        if (c == '"') { *length = used; return 0; }
        if (c < 0x20u) return -1;
        if (c == '\\') {
            if (j->at == j->end) return -1;
            c = *j->at++;
            switch (c) {
            case '"': case '\\': case '/': break;
            case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;
            case 'n': c = '\n'; break;
            case 'r': c = '\r'; break;
            case 't': c = '\t'; break;
            case 'u':
                if (hex(j, &cp) != 0) return -1;
                if (cp >= 0xd800u && cp <= 0xdbffu) {
                    uint32_t low;
                    if ((size_t)(j->end - j->at) < 6u || j->at[0] != '\\' ||
                        j->at[1] != 'u') return -1;
                    j->at += 2;
                    if (hex(j, &low) != 0 || low < 0xdc00u || low > 0xdfffu) return -1;
                    cp = 0x10000u + ((cp - 0xd800u) << 10) + low - 0xdc00u;
                } else if (cp >= 0xdc00u && cp <= 0xdfffu) return -1;
                if (cp < 0x80u) c = (unsigned char)cp;
                else {
                    unsigned n = cp < 0x800u ? 2u : cp < 0x10000u ? 3u : 4u;
                    unsigned char first = n == 2u ? 0xc0u : n == 3u ? 0xe0u : 0xf0u;
                    if (put(out, room, &used, first | (cp >> (6u * (n - 1u))))) return -1;
                    while (--n) {
                        if (put(out, room, &used, 0x80u | ((cp >> (6u * (n - 1u))) & 63u))) return -1;
                    }
                    continue;
                }
                break;
            default: return -1;
            }
        } else if (c >= 0x80u) {
            unsigned n = c >= 0xc2u && c <= 0xdfu ? 1u :
                         c >= 0xe0u && c <= 0xefu ? 2u :
                         c >= 0xf0u && c <= 0xf4u ? 3u : 0u;
            unsigned char first;
            if (n == 0u || (size_t)(j->end - j->at) < n) return -1;
            first = *j->at;
            if ((c == 0xe0u && first < 0xa0u) || (c == 0xedu && first >= 0xa0u) ||
                (c == 0xf0u && first < 0x90u) || (c == 0xf4u && first >= 0x90u)) return -1;
            if (put(out, room, &used, c)) return -1;
            while (n--) {
                c = *j->at++;
                if ((c & 0xc0u) != 0x80u || put(out, room, &used, c)) return -1;
            }
            continue;
        }
        if (put(out, room, &used, c)) return -1;
    }
    return -1;
}

int aotx_manifest_json_number(aotx_manifest_json *j, uint64_t *out)
{
    uint64_t value = 0;
    const unsigned char *start;
    space(j);
    start = j->at;
    while (j->at < j->end && *j->at >= '0' && *j->at <= '9') {
        unsigned digit = *j->at++ - '0';
        if (value > (UINT64_MAX - digit) / 10u) return -1;
        value = value * 10u + digit;
    }
    if (j->at == start || (j->at - start > 1 && *start == '0')) return -1;
    if (j->at < j->end && *j->at != ',' && *j->at != '}' && *j->at != ']' &&
        *j->at != ' ' && *j->at != '\t' && *j->at != '\r' && *j->at != '\n') return -1;
    *out = value;
    return 0;
}

int aotx_manifest_json_quote(char **at, size_t *room, const unsigned char *text,
                             size_t length)
{
    static const char digits[] = "0123456789abcdef";
    if (*room < 3u) return -1;
    *(*at)++ = '"'; --*room;
    for (size_t i = 0; i < length; ++i) {
        unsigned char c = text[i];
        size_t n = c < 0x20u ? 6u : c == '"' || c == '\\' ? 2u : 1u;
        if (*room <= n) return -1;
        if (n == 6u) {
            memcpy(*at, "\\u00", 4u);
            (*at)[4] = digits[c >> 4]; (*at)[5] = digits[c & 15u];
        } else if (n == 2u) { (*at)[0] = '\\'; (*at)[1] = (char)c; }
        else **at = (char)c;
        *at += n; *room -= n;
    }
    if (*room < 2u) return -1;
    *(*at)++ = '"'; --*room; **at = '\0';
    return 0;
}
