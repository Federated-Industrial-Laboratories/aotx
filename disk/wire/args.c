/* Purpose: Write and read the argument text of one tool call.
 * Owns: Nothing; the caller owns the buffer and the argument table.
 * Threading: One thread; neither function holds state between calls.
 * Lifetime: The call.
 *
 * The request body carries one text field. The device writes the argument keys of the tool
 * call into it, and the feeder reads them back. Both directions live here, so one file
 * holds the shape and a change to one direction cannot pass the other. */
#include "disk/wire/diskwire.h"

#include <stdio.h>
#include <string.h>

/* The reason of a refusal that names a key of the call. The text lives beside the program,
 * because a caller reads the reason after the function returns. */
static char key_reason[128];

uint32_t aotx_args_join(char *out, uint32_t bytes, const char *const *keys,
                        const char *const *values, uint32_t count)
{
    uint32_t at = 0;
    uint32_t k;
    if (out == NULL || bytes == 0) {
        return 0;
    }
    out[0] = '\0';
    for (k = 0; k < count; k++) {
        uint32_t key_len = (uint32_t)strlen(keys[k]);
        uint32_t value_len = (uint32_t)strlen(values[k]);
        uint32_t need = 1u + key_len + 1u + value_len;
        if (at + need + 1u > bytes) {
            out[0] = '\0';
            return 0;
        }
        out[at++] = AOTX_ARG_SEPARATOR;
        memcpy(out + at, keys[k], key_len);
        at += key_len;
        out[at++] = '=';
        memcpy(out + at, values[k], value_len);
        at += value_len;
    }
    out[at] = '\0';
    return at;
}

/* Reports whether the tool names the key. */
static int named(const char *key, const char *const *keys, uint32_t count)
{
    uint32_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(key, keys[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

const char *aotx_args_value(const aotx_args *a, const char *key)
{
    uint32_t i;
    for (i = 0; i < a->count; i++) {
        if (strcmp(a->key[i], key) == 0) {
            return a->value[i];
        }
    }
    return NULL;
}

int aotx_args_split(aotx_args *a, const char *arg, const char *const *keys, uint32_t count,
                    const char **reason)
{
    size_t len = strlen(arg);
    char *at;
    memset(a, 0, sizeof(*a));
    if (len >= sizeof(a->work)) {
        *reason = "the arguments are longer than a request body holds";
        return 0;
    }
    memcpy(a->work, arg, len + 1u);
    if (count == 0) {
        *reason = "the tool takes no argument";
        return 0;
    }
    /* A text that does not start with the separator byte is one value, which is the shape
     * of a call with one argument. The separator marks the key and value shape, so a value
     * that holds an equal sign cannot be read as a key. */
    if (a->work[0] != AOTX_ARG_SEPARATOR) {
        snprintf(a->key[0], sizeof(a->key[0]), "%s", keys[0]);
        a->value[0] = a->work;
        a->count = 1;
        return 1;
    }
    at = a->work + 1;
    while (at != NULL) {
        char *end = strchr(at, AOTX_ARG_SEPARATOR);
        char *equal;
        if (end != NULL) {
            *end = '\0';
        }
        if (at[0] != '\0') {
            if (a->count >= AOTX_ARG_MAX) {
                *reason = "the call holds more arguments than a tool takes";
                return 0;
            }
            equal = strchr(at, '=');
            if (equal == NULL) {
                *reason = "an argument holds no key and no equal sign";
                return 0;
            }
            *equal = '\0';
            if (strlen(at) >= AOTX_ARG_KEY_BYTES) {
                *reason = "an argument key is too long";
                return 0;
            }
            if (!named(at, keys, count)) {
                snprintf(key_reason, sizeof(key_reason),
                         "the argument key %.64s is not a key of this tool", at);
                *reason = key_reason;
                return 0;
            }
            snprintf(a->key[a->count], sizeof(a->key[0]), "%s", at);
            a->value[a->count] = equal + 1;
            a->count++;
        }
        at = (end != NULL) ? end + 1 : NULL;
    }
    return 1;
}
