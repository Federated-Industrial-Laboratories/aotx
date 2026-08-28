/* Purpose: Take one tool call out of the reply of a turn.
 * Owns: Nothing; the caller holds the call it fills.
 * Launch shape: A device function; one call for each reply of the tick.
 * Lifetime: Each call.
 *
 * The shape is the shape the chat template of the model file defines. It is the call tags,
 * one JSON object with a name and an arguments object, and string values only. The machine
 * takes white space between every piece and refuses every other shape. It keeps no array
 * of its own. A name is compared with the text where it stands, so the frame is
 * registers. */
#include "tool/tool_state.cuh"

/* The tags of a call. */
#define AOTX_PARSE_HEAD  "<tool_call>"
#define AOTX_PARSE_TAIL  "</tool_call>"

/* The members the parser knows. A key that is not one of them refuses the call. */
#define AOTX_PARSE_TEXT  1u
#define AOTX_PARSE_PATH  2u
#define AOTX_PARSE_PROV  4u

/* Step over space, tab, carriage return and line feed. */
__device__ __forceinline__ static void aotx_parse_space(const unsigned char *text,
                                                        unsigned int length,
                                                        unsigned int *at)
{
    while (*at < length) {
        unsigned char byte = text[*at];
        if (byte != (unsigned char)' ' && byte != (unsigned char)'\t'
            && byte != (unsigned char)'\r' && byte != (unsigned char)'\n') {
            return;
        }
        *at += 1u;
    }
}

/* Take one byte when it is the byte that is wanted. */
__device__ __forceinline__ static int aotx_parse_byte(const unsigned char *text,
                                                      unsigned int length,
                                                      unsigned int *at, char want)
{
    if (*at >= length || text[*at] != (unsigned char)want) {
        return 0;
    }
    *at += 1u;
    return 1;
}

/* Take a run of bytes when it stands at the position. */
__device__ __forceinline__ static int aotx_parse_word(const unsigned char *text,
                                                      unsigned int length,
                                                      unsigned int *at, const char *word)
{
    unsigned int i = 0u;
    while (word[i] != '\0') {
        if (*at + i >= length || text[*at + i] != (unsigned char)word[i]) {
            return 0;
        }
        i += 1u;
    }
    *at += i;
    return 1;
}

/* Find the first place a run of bytes stands, or the length when it is not there. */
__device__ __forceinline__ static unsigned int aotx_parse_find(const unsigned char *text,
                                                               unsigned int length,
                                                               const char *word)
{
    unsigned int span = 0u;
    while (word[span] != '\0') {
        span += 1u;
    }
    if (span == 0u || span > length) {
        return length;
    }
    for (unsigned int i = 0u; i + span <= length; ++i) {
        unsigned int k = 0u;
        while (k < span && text[i + k] == (unsigned char)word[k]) {
            k += 1u;
        }
        if (k == span) {
            return i;
        }
    }
    return length;
}

/* Take a JSON string and compare it with a word. The return is 1 when the string is that
 * word exactly. The position moves only when the return is 1. */
__device__ __forceinline__ static int aotx_parse_is(const unsigned char *text,
                                                    unsigned int length, unsigned int *at,
                                                    const char *word)
{
    unsigned int walk = *at;
    if (aotx_parse_byte(text, length, &walk, '"') == 0) {
        return 0;
    }
    if (aotx_parse_word(text, length, &walk, word) == 0) {
        return 0;
    }
    if (aotx_parse_byte(text, length, &walk, '"') == 0) {
        return 0;
    }
    *at = walk;
    return 1;
}

/* Read one hexadecimal figure, or 16 when the byte is not one. */
__device__ __forceinline__ static unsigned int aotx_parse_hex(unsigned char byte)
{
    if (byte >= (unsigned char)'0' && byte <= (unsigned char)'9') {
        return (unsigned int)(byte - (unsigned char)'0');
    }
    if (byte >= (unsigned char)'a' && byte <= (unsigned char)'f') {
        return (unsigned int)(byte - (unsigned char)'a') + 10u;
    }
    if (byte >= (unsigned char)'A' && byte <= (unsigned char)'F') {
        return (unsigned int)(byte - (unsigned char)'A') + 10u;
    }
    return 16u;
}

/* Take a JSON string into the argument of the call. The escapes of the schema are taken;
 * a value that does not fit refuses the call. The return is 1 when the string was read. */
__device__ __forceinline__ static int aotx_parse_string(const unsigned char *text,
                                                        unsigned int length,
                                                        unsigned int *at, char *out,
                                                        unsigned int max, unsigned int *made)
{
    if (aotx_parse_byte(text, length, at, '"') == 0) {
        return 0;
    }
    unsigned int held = 0u;
    while (*at < length) {
        unsigned char byte = text[*at];
        *at += 1u;
        if (byte == (unsigned char)'"') {
            *made = held;
            return 1;
        }
        unsigned int point = 0u;
        if (byte == (unsigned char)'\\') {
            if (*at >= length) {
                return 0;
            }
            unsigned char mark = text[*at];
            *at += 1u;
            if (mark == (unsigned char)'u') {
                if (*at + 4u > length) {
                    return 0;
                }
                for (unsigned int i = 0u; i < 4u; ++i) {
                    unsigned int figure = aotx_parse_hex(text[*at + i]);
                    if (figure > 15u) {
                        return 0;
                    }
                    point = point * 16u + figure;
                }
                *at += 4u;
                if (held + 4u > max) {
                    return 0;
                }
                held += aotx_text_encode(point, (unsigned char *)out + held);
                continue;
            }
            if (mark == (unsigned char)'n') {
                byte = (unsigned char)'\n';
            } else if (mark == (unsigned char)'t') {
                byte = (unsigned char)'\t';
            } else if (mark == (unsigned char)'r') {
                byte = (unsigned char)'\r';
            } else if (mark == (unsigned char)'b') {
                byte = 8u;
            } else if (mark == (unsigned char)'f') {
                byte = 12u;
            } else if (mark == (unsigned char)'"' || mark == (unsigned char)'\\'
                       || mark == (unsigned char)'/') {
                byte = mark;
            } else {
                return 0;
            }
        }
        if (held >= max) {
            return 0;
        }
        out[held] = (char)byte;
        held += 1u;
    }
    return 0;
}

/* Take the arguments object. Each member is a key the tool table names and a string value.
 * The keys that were read go in seen. */
__device__ __forceinline__ static int aotx_parse_arguments(const unsigned char *text,
                                                           unsigned int length,
                                                           unsigned int *at,
                                                           aotx_tool_call *call,
                                                           unsigned int *seen)
{
    if (aotx_parse_byte(text, length, at, '{') == 0) {
        return 0;
    }
    aotx_parse_space(text, length, at);
    if (aotx_parse_byte(text, length, at, '}') != 0) {
        return 1;
    }
    for (;;) {
        aotx_parse_space(text, length, at);
        unsigned int which = 0u;
        if (aotx_parse_is(text, length, at, "text") != 0) {
            which = AOTX_PARSE_TEXT;
        } else if (aotx_parse_is(text, length, at, "path") != 0) {
            which = AOTX_PARSE_PATH;
        } else if (aotx_parse_is(text, length, at, "provenance") != 0) {
            which = AOTX_PARSE_PROV;
        } else {
            return 0;
        }
        if ((*seen & which) != 0u) {
            return 0;
        }
        *seen |= which;
        aotx_parse_space(text, length, at);
        if (aotx_parse_byte(text, length, at, ':') == 0) {
            return 0;
        }
        aotx_parse_space(text, length, at);
        if (which == AOTX_PARSE_PROV) {
            if (aotx_parse_is(text, length, at, "computed") != 0) {
                call->provenance = AOTX_PROV_COMPUTED;
            } else if (aotx_parse_is(text, length, at, "fetched") != 0) {
                call->provenance = AOTX_PROV_FETCHED;
            } else if (aotx_parse_is(text, length, at, "recalled") != 0) {
                call->provenance = AOTX_PROV_RECALLED;
            } else if (aotx_parse_is(text, length, at, "testimony") != 0) {
                call->provenance = AOTX_PROV_TESTIMONY;
            } else {
                return 0;
            }
        } else {
            unsigned int made = 0u;
            if (aotx_parse_string(text, length, at, call->arg, AOTX_TOOL_ARG_BYTES,
                                  &made) == 0) {
                return 0;
            }
            call->arg_len = made;
        }
        aotx_parse_space(text, length, at);
        if (aotx_parse_byte(text, length, at, ',') != 0) {
            continue;
        }
        return aotx_parse_byte(text, length, at, '}');
    }
}

/* The shape, from the call tag to the closing tag. The call holds the pieces it read, and
 * the caller clears them when this function refuses the shape. */
__device__ __forceinline__ static int aotx_tool_take(const unsigned char *reply,
                                                     unsigned int length,
                                                     aotx_tool_call *call)
{
    if (reply == 0 || length == 0u) {
        return 0;
    }
    unsigned int head = aotx_parse_find(reply, length, AOTX_PARSE_HEAD);
    if (head >= length) {
        return 0;
    }
    unsigned int at = head;
    if (aotx_parse_word(reply, length, &at, AOTX_PARSE_HEAD) == 0) {
        return 0;
    }
    aotx_parse_space(reply, length, &at);
    if (aotx_parse_byte(reply, length, &at, '{') == 0) {
        return 0;
    }
    unsigned int seen = 0u;
    unsigned int named = 0u;
    unsigned int args = 0u;
    for (;;) {
        aotx_parse_space(reply, length, &at);
        if (aotx_parse_is(reply, length, &at, "name") != 0) {
            if (named != 0u) {
                return 0;
            }
            named = 1u;
            aotx_parse_space(reply, length, &at);
            if (aotx_parse_byte(reply, length, &at, ':') == 0) {
                return 0;
            }
            aotx_parse_space(reply, length, &at);
            if (aotx_parse_is(reply, length, &at, "memory_recall") != 0) {
                call->tool = AOTX_TOOL_MEMORY_RECALL;
            } else if (aotx_parse_is(reply, length, &at, "memory_write") != 0) {
                call->tool = AOTX_TOOL_MEMORY_WRITE;
            } else if (aotx_parse_is(reply, length, &at, "fs_read") != 0) {
                call->tool = AOTX_TOOL_FS_READ;
            } else {
                return 0;
            }
        } else if (aotx_parse_is(reply, length, &at, "arguments") != 0) {
            if (args != 0u) {
                return 0;
            }
            args = 1u;
            aotx_parse_space(reply, length, &at);
            if (aotx_parse_byte(reply, length, &at, ':') == 0) {
                return 0;
            }
            aotx_parse_space(reply, length, &at);
            if (aotx_parse_arguments(reply, length, &at, call, &seen) == 0) {
                return 0;
            }
        } else {
            return 0;
        }
        aotx_parse_space(reply, length, &at);
        if (aotx_parse_byte(reply, length, &at, ',') != 0) {
            continue;
        }
        if (aotx_parse_byte(reply, length, &at, '}') == 0) {
            return 0;
        }
        break;
    }
    aotx_parse_space(reply, length, &at);
    if (aotx_parse_word(reply, length, &at, AOTX_PARSE_TAIL) == 0) {
        return 0;
    }
    if (named == 0u || args == 0u) {
        return 0;
    }

    /* Each tool takes the keys the tool table names, and no other key. An argument of no
     * length is not an argument. */
    unsigned int want = 0u;
    if (call->tool == AOTX_TOOL_MEMORY_RECALL) {
        want = AOTX_PARSE_TEXT;
    } else if (call->tool == AOTX_TOOL_MEMORY_WRITE) {
        want = AOTX_PARSE_TEXT | AOTX_PARSE_PROV;
    } else {
        want = AOTX_PARSE_PATH;
    }
    return (seen == want && call->arg_len != 0u) ? 1 : 0;
}

__device__ int aotx_tool_parse(const unsigned char *reply, unsigned int length,
                               aotx_tool_call *call)
{
    if (call == 0) {
        return 0;
    }
    call->tool = AOTX_TOOL_NONE;
    call->provenance = 0u;
    call->arg_len = 0u;
    if (aotx_tool_take(reply, length, call) != 0) {
        return 1;
    }
    /* A shape the machine refused leaves no piece behind, so a caller which reads the call
     * after a refusal finds no tool and no argument. */
    call->tool = AOTX_TOOL_NONE;
    call->provenance = 0u;
    call->arg_len = 0u;
    return 0;
}
