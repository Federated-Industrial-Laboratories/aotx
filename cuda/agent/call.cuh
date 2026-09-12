/* Purpose: Render calls and results with the selected model form.
 * Owns: No storage; source bytes can occupy a circular transcript arena.
 * Launch shape: One thread for each prompt in a batch.
 * Lifetime: One prompt build. */
#ifndef AOTX_AGENT_CALL_CUH
#define AOTX_AGENT_CALL_CUH

#include "catalog/catalog.cuh"
#include "model/call_format.cuh"
#include "agent/result.cuh"

/* Name and key spans of a stored call refer to its transcript arena, not the catalog. */
typedef struct aotx_call_schema {
    unsigned int name_at;
    unsigned int name_len;
    unsigned int arguments;
    aotx_catalog_run key[AOTX_CATALOG_ARGS];
} aotx_call_schema;

/* Quote all JSON control bytes. A source span must fit its circular buffer. */
__device__ __forceinline__ unsigned int aotx_call_text(
    unsigned char *out, unsigned int at, const unsigned char *source,
    unsigned int base, unsigned int length, unsigned int capacity, int json)
{
    if (capacity == 0u || base >= capacity || length > capacity) return AOTX_SAY_BYTES + 1u;
    if (out == 0 && !json) {
        return at <= AOTX_SAY_BYTES && length <= AOTX_SAY_BYTES - at
             ? at + length : AOTX_SAY_BYTES + 1u;
    }
    for (unsigned int i = 0u; i < length; ++i) {
        unsigned char byte = source[(base + i) % capacity];
        unsigned int need = json && byte < 0x20u ? 6u
                          : json && (byte == '"' || byte == '\\') ? 2u : 1u;
        if (at > AOTX_SAY_BYTES || need > AOTX_SAY_BYTES - at) return AOTX_SAY_BYTES + 1u;
        if (out == 0) { at += need; continue; }
        if (need == 6u) {
            const char *hex = "0123456789abcdef";
            out[at++] = '\\'; out[at++] = 'u'; out[at++] = '0'; out[at++] = '0';
            out[at++] = (unsigned char)hex[byte >> 4u];
            out[at++] = (unsigned char)hex[byte & 15u];
        } else {
            if (need == 2u) out[at++] = '\\';
            out[at++] = byte;
        }
    }
    return at;
}

__device__ __forceinline__ unsigned int aotx_call_literal(
    unsigned char *out, unsigned int at, const char *text)
{
    unsigned int length = 0u;
    while (text[length] != '\0') ++length;
    if (at > AOTX_SAY_BYTES || length > AOTX_SAY_BYTES - at) return AOTX_SAY_BYTES + 1u;
    if (out == 0) return at + length;
    return aotx_wrap_run(out, at, AOTX_SAY_BYTES, (const unsigned char *)text, length);
}

/* A null output measures the same grammar without a scratch buffer or a second serializer. */
__device__ __forceinline__ unsigned int aotx_call_render(
    unsigned char *out, unsigned int at, unsigned int entry,
    const unsigned char *source, unsigned int base, unsigned int capacity,
    const unsigned int *offset, const unsigned int *length,
    const aotx_call_schema *schema = 0, unsigned role = AOTX_MODEL_ROLES)
{
    const aotx_call_format *format = aotx_call_format_active(role);
    if (format->kind == AOTX_CALL_NONE || format->kind >= AOTX_CALL_FORMAT_KINDS
        || (schema == 0 && entry >= AOTX_MODULE_SLOTS) || base >= capacity) return AOTX_SAY_BYTES + 1u;
    const aotx_catalog_entry *row = schema == 0 ? &aotx_catalog.entry[entry] : 0;
    unsigned int arguments = schema != 0 ? schema->arguments : row->tool.arguments;
    if (arguments > AOTX_CATALOG_ARGS) return AOTX_SAY_BYTES + 1u;
    int xml = format->kind == AOTX_CALL_QWEN_XML;
    at = aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_HEAD);
    if (xml) {
        at = aotx_call_literal(out, at, "\n");
        at = aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_NAME_HEAD);
    } else {
        if (format->length[AOTX_CALL_HEAD] != 0u) at = aotx_call_literal(out, at, "\n");
        at = aotx_call_literal(out, at, "{\"name\": \"");
    }
    if (schema != 0) {
        at = aotx_call_text(out, at, source, schema->name_at, schema->name_len, capacity, 0);
    } else {
        at = aotx_call_text(out, at, (const unsigned char *)row->name, 0u,
                            row->name_len, sizeof row->name, 0);
    }
    if (xml) {
        at = aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_NAME_TAIL);
        at = aotx_call_literal(out, at, "\n");
    } else {
        at = aotx_call_literal(out, at, "\", \"");
        at = aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_ARG_KEY);
        at = aotx_call_literal(out, at, "\": {");
    }
    for (unsigned int k = 0u; k < arguments; ++k) {
        if (offset[k] > capacity || length[k] > capacity - offset[k]) return AOTX_SAY_BYTES + 1u;
        if (xml) {
            at = aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_ARG_HEAD);
        } else {
            if (k != 0u) at = aotx_call_literal(out, at, ", ");
            at = aotx_call_literal(out, at, "\"");
        }
        aotx_catalog_run key = schema != 0 ? schema->key[k] : row->tool.key[k];
        if (schema != 0) {
            at = aotx_call_text(out, at, source, key.at, key.length, capacity, 0);
        } else {
            at = aotx_call_text(out, at, aotx_catalog_arena, key.at, key.length,
                                AOTX_CATALOGUE_BYTES, 0);
        }
        at = aotx_call_literal(out, at, xml ? ">\n" : "\": \"");
        at = aotx_call_text(out, at, source, (base + offset[k]) % capacity, length[k], capacity, !xml);
        if (xml) {
            at = aotx_call_literal(out, at, "\n");
            at = aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_ARG_TAIL);
            at = aotx_call_literal(out, at, "\n");
        } else {
            at = aotx_call_literal(out, at, "\"");
        }
    }
    if (xml) {
        at = aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_NAME_CLOSE);
        at = aotx_call_literal(out, at, "\n");
    } else {
        at = aotx_call_literal(out, at, "}}");
        if (format->length[AOTX_CALL_TAIL] != 0u) at = aotx_call_literal(out, at, "\n");
    }
    return aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_TAIL);
}

/* A result is a separate tool turn, never assistant text. */
__device__ __forceinline__ unsigned int aotx_call_result(
    unsigned char *out, unsigned int at, const unsigned char *source,
    unsigned int base, unsigned int length, unsigned int capacity, unsigned role = AOTX_MODEL_ROLES)
{
    const aotx_call_format *format = aotx_call_format_active(role);
    const aotx_wrap *wrap = aotx_wrap_active(role);
    if (format->kind >= AOTX_CALL_FORMAT_KINDS) return AOTX_SAY_BYTES + 1u;
    int plain = format->kind == AOTX_CALL_NONE;
    int json = !plain && format->result_json != 0u;
    unsigned int generation = wrap->length[AOTX_WRAP_GENERATION_HEAD]
                            + wrap->length[AOTX_WRAP_THINK_OPEN] + wrap->length[AOTX_WRAP_THINK_CLOSE];
    unsigned int framing = plain ? wrap->length[AOTX_WRAP_USER_HEAD]
                                 + wrap->length[AOTX_WRAP_USER_TAIL] + 14u
                                 : format->length[AOTX_CALL_RESULT_HEAD]
                                 + format->length[AOTX_CALL_RESULT_TAIL] + 2u * (unsigned int)json;
    if (at > AOTX_SAY_BYTES || framing + generation > AOTX_SAY_BYTES - at)
        return AOTX_SAY_BYTES + 1u;
    unsigned int budget = AOTX_SAY_BYTES - at - framing - generation;
    unsigned int needed = aotx_result_bytes(source, base, length, capacity, json);
    if (needed == ~0u) return AOTX_SAY_BYTES + 1u;
    unsigned int kept = length;
    if (needed > budget) {
        unsigned int suffix = (unsigned int)sizeof(AOTX_RESULT_CUT_TEXT) - 1u;
        if (budget < suffix) return AOTX_SAY_BYTES + 1u;
        kept = aotx_result_prefix(source, base, length, capacity, json, budget - suffix);
    }
    if (plain) {
        at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_USER_HEAD);
        at = aotx_call_literal(out, at, "[tool result]\n");
    } else {
        at = aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_RESULT_HEAD);
    }
    if (json) at = aotx_call_literal(out, at, "\"");
    at = aotx_call_text(out, at, source, base, kept, capacity, json);
    if (kept != length) at = aotx_call_literal(out, at, AOTX_RESULT_CUT_TEXT);
    if (json) at = aotx_call_literal(out, at, "\"");
    return plain ? aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_USER_TAIL)
                 : aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_RESULT_TAIL);
}

#endif
