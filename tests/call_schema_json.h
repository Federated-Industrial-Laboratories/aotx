/* Purpose: Check the structure and values of advertised tool schemas.
 * Owns: Bounded JSON cursors and decoded property values.
 * Threading: One host test thread.
 * Lifetime: One tool list check. */
#ifndef AOTX_TEST_CALL_SCHEMA_JSON_H
#define AOTX_TEST_CALL_SCHEMA_JSON_H

extern "C" {
#include "disk/modelfile/manifest_json.h"
}

static int aotx_schema_string(aotx_manifest_json *j, char *out, size_t room)
{
    size_t length = 0u;
    if (aotx_manifest_json_string(j, (unsigned char *)out, room - 1u, &length)) return 0;
    out[length] = '\0';
    return 1;
}

static int aotx_schema_word(aotx_manifest_json *j, const char *word)
{
    char text[256];
    return aotx_schema_string(j, text, sizeof text) && !strcmp(text, word);
}

static int aotx_schema_key(aotx_manifest_json *j, const char *word)
{
    return aotx_schema_word(j, word) && !aotx_manifest_json_take(j, ':');
}

/* A property is decoded as an object. A string containing an enum cannot satisfy this check. */
static int aotx_schema_property(aotx_manifest_json *j, int provenance)
{
    unsigned int seen = 0u;
    if (aotx_manifest_json_take(j, '{')) return 0;
    do {
        char key[256];
        unsigned int bit = 0u;
        if (!aotx_schema_string(j, key, sizeof key) || aotx_manifest_json_take(j, ':')) return 0;
        if (!strcmp(key, "type")) {
            bit = 1u;
            if (!aotx_schema_word(j, "string")) return 0;
        } else if (!strcmp(key, "enum") && provenance) {
            static const char *values[] = {"computed", "fetched", "recalled", "testimony"};
            unsigned int mask = 0u;
            bit = 2u;
            if (aotx_manifest_json_take(j, '[')) return 0;
            for (unsigned int i = 0u; i < 4u; ++i) {
                char value[256];
                if (i != 0u && aotx_manifest_json_take(j, ',')) return 0;
                if (!aotx_schema_string(j, value, sizeof value)) return 0;
                unsigned int v = 0u;
                while (v < 4u && strcmp(value, values[v])) ++v;
                if (v == 4u || (mask & (1u << v))) return 0;
                mask |= 1u << v;
            }
            if (mask != 15u || aotx_manifest_json_take(j, ']')) return 0;
        } else if (!strcmp(key, "description") && provenance) {
            char text[256];
            bit = 4u;
            if (!aotx_schema_string(j, text, sizeof text)
                || !strstr(text, "computed: derived here")
                || !strstr(text, "fetched: external source")
                || !strstr(text, "recalled: unverified model memory")
                || !strstr(text, "testimony: report from a person or agent")
                || !strstr(text, "For a statement from the operator, use testimony.")) return 0;
        } else return 0;
        if (seen & bit) return 0;
        seen |= bit;
    } while (!aotx_manifest_json_take(j, ','));
    return !aotx_manifest_json_take(j, '}') && seen == (provenance ? 7u : 1u);
}

static int aotx_schema_tool(aotx_manifest_json *j, char *name, size_t room)
{
    char description[512], keys[AOTX_CATALOG_ARGS][256];
    unsigned int count = 0u, required = 0u;
    if (aotx_manifest_json_take(j, '{') || !aotx_schema_key(j, "name")
        || !aotx_schema_string(j, name, room) || aotx_manifest_json_take(j, ',')
        || !aotx_schema_key(j, "description")
        || !aotx_schema_string(j, description, sizeof description)
        || aotx_manifest_json_take(j, ',') || !aotx_schema_key(j, "parameters")
        || aotx_manifest_json_take(j, '{') || !aotx_schema_key(j, "type")
        || !aotx_schema_word(j, "object") || aotx_manifest_json_take(j, ',')
        || !aotx_schema_key(j, "properties") || aotx_manifest_json_take(j, '{')) return 0;
    do {
        if (count == AOTX_CATALOG_ARGS
            || !aotx_schema_string(j, keys[count], sizeof keys[count])
            || aotx_manifest_json_take(j, ':')) return 0;
        for (unsigned int k = 0u; k < count; ++k)
            if (!strcmp(keys[k], keys[count])) return 0;
        if (!aotx_schema_property(j, !strcmp(name, "memory_write")
                                     && !strcmp(keys[count], "provenance"))) return 0;
        ++count;
    } while (!aotx_manifest_json_take(j, ','));
    if (aotx_manifest_json_take(j, '}') || aotx_manifest_json_take(j, ',')
        || !aotx_schema_key(j, "required") || aotx_manifest_json_take(j, '[')) return 0;
    for (unsigned int k = 0u; k < count; ++k) {
        char key[256];
        if (k != 0u && aotx_manifest_json_take(j, ',')) return 0;
        if (!aotx_schema_string(j, key, sizeof key)) return 0;
        unsigned int index = 0u;
        while (index < count && strcmp(key, keys[index])) ++index;
        if (index == count || (required & (1u << index))) return 0;
        required |= 1u << index;
    }
    if (!strcmp(name, "memory_write")) {
        if (count != 2u || strcmp(keys[0], "provenance") || strcmp(keys[1], "text")) return 0;
    } else if (!strcmp(name, "external_note")) {
        if (count != 1u || strcmp(keys[0], "provenance")) return 0;
    }
    return !aotx_manifest_json_take(j, ']') && !aotx_manifest_json_take(j, '}')
        && !aotx_manifest_json_take(j, '}');
}

/* Framing bytes come from the selected row; every byte between them must be complete JSON. */
static int aotx_schema_list(const unsigned char *text, unsigned int length,
    const aotx_call_format *format, unsigned int *mask, unsigned int *count)
{
    unsigned int head = format->length[AOTX_CALL_TOOLS_HEAD];
    unsigned int tail = format->length[AOTX_CALL_TOOLS_TAIL];
    unsigned int instruction = format->length[AOTX_CALL_INSTRUCTION];
    const unsigned char *spans = format->bytes;
    *mask = *count = 0u;
    if (length < head + tail + instruction || length > AOTX_CATALOG_LIST_BYTES
        || memcmp(text, spans + format->offset[AOTX_CALL_TOOLS_HEAD], head)) return 0;
    if (format->kind == AOTX_CALL_LLAMA_JSON) {
        if (memcmp(text + head, spans + format->offset[AOTX_CALL_INSTRUCTION], instruction)) return 0;
        head += instruction;
    } else {
        if (memcmp(text + length - instruction,
                   spans + format->offset[AOTX_CALL_INSTRUCTION], instruction)) return 0;
        tail += instruction;
    }
    unsigned int end = length - tail;
    if (memcmp(text + end, spans + format->offset[AOTX_CALL_TOOLS_TAIL],
               format->length[AOTX_CALL_TOOLS_TAIL])) return 0;
    aotx_manifest_json json = {text + head, text + end};
    while (aotx_manifest_json_end(&json)) {
        char name[256];
        static const char *names[] = {
            "memory_recall", "memory_write", "fs_read", "skill_use", "external_note"
        };
        if (!aotx_schema_tool(&json, name, sizeof name)) return 0;
        for (unsigned int i = 0u; i < 5u; ++i) {
            if (!strcmp(name, names[i])) {
                if (*mask & (1u << i)) return 0;
                *mask |= 1u << i;
            }
        }
        ++*count;
    }
    return 1;
}

#endif
