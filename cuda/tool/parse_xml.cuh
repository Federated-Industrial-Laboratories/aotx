/* Purpose: Read the selected function and parameter form without a local array.
 * Owns: Nothing; values go into the caller's bounded pack.
 * Launch shape: One device thread for each reply.
 * Lifetime: One parser call. */
#ifndef AOTX_TOOL_PARSE_XML_CUH
#define AOTX_TOOL_PARSE_XML_CUH

__device__ __forceinline__ static int aotx_parse_xml(const unsigned char *text,
    unsigned int length, unsigned int *at, const aotx_call_format *form,
    aotx_tool_call *call)
{
    if (form->length[AOTX_CALL_NAME_HEAD] == 0u
        || form->length[AOTX_CALL_NAME_TAIL] == 0u
        || form->length[AOTX_CALL_NAME_CLOSE] == 0u
        || form->length[AOTX_CALL_ARG_HEAD] == 0u
        || form->length[AOTX_CALL_ARG_TAIL] == 0u
        || aotx_parse_part(text, length, at, form, AOTX_CALL_NAME_HEAD) == 0) {
        return 0;
    }
    unsigned int start = *at;
    unsigned int end = aotx_parse_find(text, length, start, form, AOTX_CALL_NAME_TAIL);
    if (end == length) {
        return 0;
    }
    call->entry = aotx_catalog_find((const char *)text + start, end - start,
                                    AOTX_MODULE_TOOL);
    if (call->entry >= AOTX_MODULE_SLOTS) {
        return 0;
    }
    const aotx_catalog_tool *tool = &aotx_catalog.entry[call->entry].tool;
    call->tool = tool->built_in;
    *at = end + form->length[AOTX_CALL_NAME_TAIL];
    unsigned int room = AOTX_TOOL_ARG_BYTES;
    for (unsigned int k = 0u; k < tool->arguments; ++k) {
        unsigned int cost = tool->key[k].length + 2u;
        room = room > cost ? room - cost : 0u;
    }
    unsigned int seen = 0u;
    for (;;) {
        aotx_parse_space(text, length, at);
        if (aotx_parse_part(text, length, at, form, AOTX_CALL_NAME_CLOSE) != 0) {
            return seen == ((1u << tool->arguments) - 1u);
        }
        if (aotx_parse_part(text, length, at, form, AOTX_CALL_ARG_HEAD) == 0) {
            return 0;
        }
        start = *at;
        end = aotx_parse_find(text, length, start, form, AOTX_CALL_NAME_TAIL);
        if (end == length) {
            return 0;
        }
        unsigned int which = tool->arguments;
        for (unsigned int k = 0u; k < tool->arguments; ++k) {
            if (aotx_parse_run_is(text, start, end, tool->key[k]) != 0) {
                which = k;
                break;
            }
        }
        if (which >= tool->arguments || (seen & (1u << which)) != 0u) {
            return 0;
        }
        seen |= 1u << which;
        start = end + form->length[AOTX_CALL_NAME_TAIL];
        /* The source adds exactly one line feed on each side of a parameter value. */
        unsigned int framed = start < length && text[start] == (unsigned char)'\n';
        start += framed;
        end = aotx_parse_find(text, length, start, form, AOTX_CALL_ARG_TAIL);
        while (end < length && framed != 0u
               && (end == start || text[end - 1u] != (unsigned char)'\n')) {
            end = aotx_parse_find(text, length, end + 1u, form, AOTX_CALL_ARG_TAIL);
        }
        if (end == length) {
            return 0;
        }
        *at = end + form->length[AOTX_CALL_ARG_TAIL];
        unsigned int made = end - start - framed;
        for (unsigned int i = 0u; i < made; ++i) {
            if (text[start + i] == (unsigned char)AOTX_TOOL_UNIT) {
                return 0;
            }
        }
        if (made > room - call->pack_len) {
            call->over = 1u;
        } else {
            for (unsigned int i = 0u; i < made; ++i) {
                call->pack[call->pack_len + i] = (char)text[start + i];
            }
            if (aotx_parse_value(call, which, made) == 0) {
                return 0;
            }
        }
    }
}

#endif
