/* Purpose: Select bounded tool call forms from exact model template identities.
 * Owns: Constant protocol text; the caller owns each result row.
 * Threading: One caller for each model file.
 * Lifetime: One model metadata read. */
#include "disk/modelfile/call_format.h"
#include "disk/modelfile/modelfile.h"
#include "disk/wire/diskwire.h"
#include <string.h>

int aotx_call_format_valid(const aotx_call_format *format)
{
    size_t used = 0u;
    if (format == NULL || format->kind >= AOTX_CALL_FORMAT_KINDS ||
        format->result_json != (format->kind == AOTX_CALL_LLAMA_JSON)) return 0;
    for (unsigned i = 0u; i < AOTX_CALL_FORMAT_SPANS; ++i) {
        size_t length = format->length[i];
        if (length > AOTX_CALL_FORMAT_SPAN_BYTES ||
            format->offset[i] > AOTX_CALL_FORMAT_BYTES ||
            length > AOTX_CALL_FORMAT_BYTES - format->offset[i] ||
            length > AOTX_CALL_FORMAT_BYTES - used) return 0;
        if (format->kind == AOTX_CALL_NONE && length != 0u) return 0;
        used += length;
    }
    return 1;
}

int aotx_call_format_make(unsigned int kind, aotx_call_format *out)
{
    static const char *const hermes[AOTX_CALL_FORMAT_SPANS] = {
        "",
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
        "</tools>\n",
        "\nFor each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
        "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
        "<tool_call>", "</tool_call>", "", "", "", "", "", "arguments",
        "<|im_start|>user\n\n<tool_response>\n", "\n</tool_response><|im_end|>\n"
    };
    static const char *const llama[AOTX_CALL_FORMAT_SPANS] = {
        "Environment: ipython\n", "", "",
        "You have access to the following functions. To call a function, please respond with JSON for a function call."
        "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}."
        "Do not use variables.\n\n",
        "", "", "<|python_tag|>", "", "", "", "", "parameters",
        "<|start_header_id|>ipython<|end_header_id|>\n\n", "<|eot_id|>"
    };
    static const char *const xml[AOTX_CALL_FORMAT_SPANS] = {
        "", "# Tools\n\nYou have access to the following functions:\n\n<tools>\n", "</tools>\n",
        "\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
        "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\nthat can span\nmultiple lines\n"
        "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        "</IMPORTANT>",
        "<tool_call>", "</tool_call>", "<function=", ">", "</function>",
        "<parameter=", "</parameter>", "",
        "<|im_start|>user\n<tool_response>\n", "\n</tool_response><|im_end|>\n"
    };
    size_t used = 0u;
    if (out == NULL) return -1;
    memset(out, 0, sizeof *out);
    if (kind >= AOTX_CALL_FORMAT_KINDS) return -1;
    if (kind == AOTX_CALL_NONE) return 0;
    const char *const *spans = kind == AOTX_CALL_HERMES ? hermes
                            : kind == AOTX_CALL_LLAMA_JSON ? llama : xml;
    for (unsigned i = 0u; i < AOTX_CALL_FORMAT_SPANS; ++i) {
        size_t length = strlen(spans[i]);
        if (length > AOTX_CALL_FORMAT_SPAN_BYTES || length > AOTX_CALL_FORMAT_BYTES - used) {
            memset(out, 0, sizeof *out);
            return -1;
        }
        out->offset[i] = (uint16_t)used;
        out->length[i] = (uint16_t)length;
        memcpy(out->bytes + used, spans[i], length);
        used += length;
    }
    out->kind = kind;
    out->result_json = kind == AOTX_CALL_LLAMA_JSON;
    return aotx_call_format_valid(out) ? 0 : -1;
}

int aotx_call_format_read(const aotx_modelfile *file, aotx_call_format *out)
{
    static const struct { size_t length; const char *digest; unsigned kind; } known[] = {
        {4100u, "57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361", AOTX_CALL_HERMES},
        {4168u, "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8", AOTX_CALL_HERMES},
        {4761u, "8428c815ac94d82064e35ff1e841dcbe260e7e53a8d0bd3b94afa2eefa9bccab", AOTX_CALL_HERMES},
        {4116u, "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5", AOTX_CALL_HERMES},
        {2507u, "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f", AOTX_CALL_HERMES},
        {3827u, "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4", AOTX_CALL_LLAMA_JSON},
        {7816u, "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67", AOTX_CALL_QWEN_XML}
    };
    const char *text;
    size_t length;
    if (out == NULL) return -1;
    memset(out, 0, sizeof *out);
    if (file == NULL || aotx_modelfile_string(file, "tokenizer.chat_template", &text, &length) != 0)
        return 0;
    for (unsigned i = 0u; i < sizeof known / sizeof known[0]; ++i) {
        if (length != known[i].length) continue;
        aotx_sha256 hash;
        unsigned char digest[AOTX_SHA256_DIGEST];
        char hex[AOTX_SHA256_HEX];
        aotx_sha256_init(&hash);
        aotx_sha256_update(&hash, text, length);
        aotx_sha256_final(&hash, digest);
        aotx_sha256_text(digest, hex);
        if (strcmp(hex, known[i].digest) == 0) return aotx_call_format_make(known[i].kind, out);
        return 0;
    }
    return 0;
}
