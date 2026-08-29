/* Purpose: Take one tool call out of the reply of a turn.
 * Owns: Nothing; the caller holds the call it fills.
 * Launch shape: A device function; one call for each reply of the tick.
 * Lifetime: Each call.
 *
 * The shape is the shape the chat template of the model file defines. It is the call tags,
 * one JSON object with a name and an arguments object, and string values only. The machine
 * takes white space between every piece and refuses every other shape. It keeps no array
 * of its own. A name is compared with the entries of the catalog where it stands. A key is
 * compared with the argument keys of that entry, so the frame is registers. */
#include "catalog/catalog.cuh"
#include "tool/tool_state.cuh"

/* The tags of a call. */
#define AOTX_PARSE_HEAD  "<tool_call>"
#define AOTX_PARSE_TAIL  "</tool_call>"

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

/* Take a JSON string and give where its bytes stand. An escape refuses the take, because
 * a name and an argument key hold letters, figures and the low line alone. */
__device__ __forceinline__ static int aotx_parse_span(const unsigned char *text,
                                                      unsigned int length, unsigned int *at,
                                                      unsigned int *start, unsigned int *end)
{
    unsigned int walk = *at;
    if (aotx_parse_byte(text, length, &walk, '"') == 0) {
        return 0;
    }
    *start = walk;
    while (walk < length && text[walk] != (unsigned char)'"') {
        if (text[walk] == (unsigned char)'\\') {
            return 0;
        }
        walk += 1u;
    }
    if (walk >= length) {
        return 0;
    }
    *end = walk;
    *at = walk + 1u;
    return 1;
}

/* Report whether the bytes from start to end are a run of the catalog arena. */
__device__ __forceinline__ static int aotx_parse_run_is(const unsigned char *text,
                                                        unsigned int start, unsigned int end,
                                                        aotx_catalog_run run)
{
    if (end - start != run.length) {
        return 0;
    }
    for (unsigned int i = 0u; i < run.length; ++i) {
        if (text[start + i] != aotx_catalog_arena[run.at + i]) {
            return 0;
        }
    }
    return 1;
}

/* Report whether the bytes from start to end are the word. */
__device__ __forceinline__ static int aotx_parse_bytes_are(const unsigned char *text,
                                                           unsigned int start,
                                                           unsigned int end,
                                                           const char *word)
{
    unsigned int at = start;
    unsigned int i = 0u;
    while (at < end && word[i] != '\0' && text[at] == (unsigned char)word[i]) {
        at += 1u;
        i += 1u;
    }
    return (at == end && word[i] == '\0') ? 1 : 0;
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

/* Step over one JSON object, from its opening brace to the brace that closes it. A string
 * inside it is stepped over with its escapes, so a brace inside a string does not count.
 * The machine keeps a depth in a register and no array. */
__device__ __forceinline__ static int aotx_parse_skip(const unsigned char *text,
                                                      unsigned int length, unsigned int *at)
{
    if (aotx_parse_byte(text, length, at, '{') == 0) {
        return 0;
    }
    unsigned int depth = 1u;
    while (*at < length && depth > 0u) {
        unsigned char byte = text[*at];
        *at += 1u;
        if (byte == (unsigned char)'"') {
            while (*at < length && text[*at] != (unsigned char)'"') {
                *at += ((text[*at] == (unsigned char)'\\') && (*at + 1u < length)) ? 2u : 1u;
            }
            if (*at >= length) {
                return 0;
            }
            *at += 1u;
        } else if (byte == (unsigned char)'{') {
            depth += 1u;
        } else if (byte == (unsigned char)'}') {
            depth -= 1u;
        }
    }
    return (depth == 0u) ? 1 : 0;
}

/* Put one value that is a word of the schema in the pack of a call. The return is 1 when
 * the pack holds it. */
__device__ __forceinline__ static int aotx_parse_keep(aotx_tool_call *call,
                                                      unsigned int which, const char *word,
                                                      unsigned int room)
{
    unsigned int made = 0u;
    while (word[made] != '\0') {
        made += 1u;
    }
    if (call->pack_len + made > room) {
        return 0;
    }

    for (unsigned int i = 0u; i < made; ++i) {
        call->pack[call->pack_len + i] = word[i];
    }
    call->at[which] = call->pack_len;
    call->length[which] = made;
    call->pack_len += made;
    call->values += 1u;
    return 1;
}

/* Take the arguments object. Each member is one argument key of the entry the name found,
 * and a string value. The keys that were read go in seen, one bit for each key. */
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
    if (call->entry >= AOTX_MODULE_SLOTS) {
        return 0;
    }
    const aotx_catalog_tool *tool = &aotx_catalog.entry[call->entry].tool;
    /* The values of a call go out as one line of key=value pairs with the unit separator
     * byte between two pairs. The keys and the separators take room of their own, so the
     * values together fit the bound of the line less that room. */
    unsigned int room = AOTX_TOOL_ARG_BYTES;
    for (unsigned int k = 0u; k < tool->arguments; ++k) {
        unsigned int cost = tool->key[k].length + 2u;
        room = (room > cost) ? (room - cost) : 0u;
    }
    for (;;) {
        aotx_parse_space(text, length, at);
        unsigned int start = 0u;
        unsigned int end = 0u;
        if (aotx_parse_span(text, length, at, &start, &end) == 0) {
            return 0;
        }
        unsigned int which = tool->arguments;
        for (unsigned int k = 0u; k < tool->arguments; ++k) {
            if (aotx_parse_run_is(text, start, end, tool->key[k]) != 0) {
                which = k;
                break;
            }
        }
        if (which >= tool->arguments || (*seen & (1u << which)) != 0u) {
            return 0;
        }
        *seen |= 1u << which;
        aotx_parse_space(text, length, at);
        if (aotx_parse_byte(text, length, at, ':') == 0) {
            return 0;
        }
        aotx_parse_space(text, length, at);
        /* The source of a note is one of four words and not free text. Every other key
         * takes a string value. Each value goes in the pack of the call, so the request
         * writes a line that carries every key. */
        if (tool->built_in == AOTX_TOOL_MEMORY_WRITE
            && aotx_parse_bytes_are(text, start, end, "provenance") != 0) {
            const char *word = 0;
            if (aotx_parse_is(text, length, at, "computed") != 0) {
                call->provenance = AOTX_PROV_COMPUTED;
                word = "computed";
            } else if (aotx_parse_is(text, length, at, "fetched") != 0) {
                call->provenance = AOTX_PROV_FETCHED;
                word = "fetched";
            } else if (aotx_parse_is(text, length, at, "recalled") != 0) {
                call->provenance = AOTX_PROV_RECALLED;
                word = "recalled";
            } else if (aotx_parse_is(text, length, at, "testimony") != 0) {
                call->provenance = AOTX_PROV_TESTIMONY;
                word = "testimony";
            } else {
                return 0;
            }
            if (aotx_parse_keep(call, which, word, room) == 0) {
                return 0;
            }
        } else {
            unsigned int made = 0u;
            if (call->pack_len >= room
                || aotx_parse_string(text, length, at, call->pack + call->pack_len,
                                     room - call->pack_len, &made) == 0) {
                return 0;
            }
            /* The unit separator byte parts two pairs of the argument line, so no value
             * may carry it. */
            for (unsigned int i = 0u; i < made; ++i) {
                if (call->pack[call->pack_len + i] == AOTX_TOOL_UNIT) {
                    return 0;
                }
            }
            call->at[which] = call->pack_len;
            call->length[which] = made;
            call->pack_len += made;
            call->values += 1u;
            call->key = which;
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
    /* The arguments of a call are read against the argument keys of its entry, so the
     * name must be in hand first. A call that gives the arguments first is stepped over
     * here and read after the loop, so the order of the two members is free. */
    unsigned int held = 0u;
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
            unsigned int start = 0u;
            unsigned int end = 0u;
            if (aotx_parse_span(reply, length, &at, &start, &end) == 0) {
                return 0;
            }
            /* The name is compared with the entries of the catalog where it stands, so
             * the parser keeps no list of names of its own. */
            call->entry = aotx_catalog_find((const char *)reply + start, end - start,
                                            AOTX_MODULE_TOOL);
            if (call->entry >= AOTX_MODULE_SLOTS) {
                return 0;
            }
            call->tool = aotx_catalog.entry[call->entry].tool.built_in;
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
            if (named == 0u) {
                held = at;
                if (aotx_parse_skip(reply, length, &at) == 0) {
                    return 0;
                }
            } else if (aotx_parse_arguments(reply, length, &at, call, &seen) == 0) {
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
    /* The call gave the arguments in front of the name. The name is in hand now, so the
     * members of that object are read against the argument keys of its entry. */
    if (held != 0u && aotx_parse_arguments(reply, length, &held, call, &seen) == 0) {
        return 0;
    }

    /* Each tool takes the keys its manifest names, and no other key. An argument of no
     * length is not an argument. */
    unsigned int keys = aotx_catalog.entry[call->entry].tool.arguments;
    unsigned int want = (keys >= 32u) ? 0xffffffffu : ((1u << keys) - 1u);
    if (seen != want || call->values == 0u || call->key >= AOTX_CATALOG_ARGS) {
        return 0;
    }
    /* The value of the call is the run of the key that carried free text. A built-in tool
     * reads that value and the tokenizer of the tool path takes it. */
    unsigned int made = call->length[call->key];
    for (unsigned int i = 0u; i < made; ++i) {
        call->arg[i] = call->pack[call->at[call->key] + i];
    }
    call->arg_len = made;
    return (made != 0u) ? 1 : 0;
}

/* Give a call back the state of a call that read nothing. */
__device__ __forceinline__ static void aotx_tool_call_clear(aotx_tool_call *call)
{
    call->entry = AOTX_MODULE_SLOTS;
    call->tool = AOTX_TOOL_NONE;
    call->key = AOTX_CATALOG_ARGS;
    call->provenance = 0u;
    call->arg_len = 0u;
    call->values = 0u;
    call->pack_len = 0u;
    for (unsigned int i = 0u; i < AOTX_CATALOG_ARGS; ++i) {
        call->at[i] = 0u;
        call->length[i] = 0u;
    }
}

__device__ int aotx_tool_parse(const unsigned char *reply, unsigned int length,
                               aotx_tool_call *call)
{
    if (call == 0) {
        return 0;
    }
    aotx_tool_call_clear(call);
    if (aotx_tool_take(reply, length, call) != 0) {
        return 1;
    }
    /* A shape the machine refused leaves no piece behind, so a caller which reads the call
     * after a refusal finds no tool and no argument. */
    aotx_tool_call_clear(call);
    return 0;
}
