/* Purpose: Check exact call form selection and bounded protocol rows in batches.
 * Owns: Test metadata and output rows.
 * Threading: One test thread.
 * Lifetime: The test process. */
#include "tests/disk_fake.h"
#include "tests/wrap_templates.h"
#include "tests/call_templates.h"
#include "disk/modelfile/call_format.h"
#include "disk/modelfile/gguf.h"
#include "disk/modelfile/manifest.h"

static int same_span(const aotx_call_format *format, unsigned span, const char *text)
{
    size_t length = strlen(text);
    return format->length[span] == length &&
           memcmp(format->bytes + format->offset[span], text, length) == 0;
}

static void row_cases(void)
{
    aotx_call_format row, changed, zero = {0};
    CHECK(sizeof row == 2112u, "the row size differs");
    CHECK(aotx_call_format_make(AOTX_CALL_NONE, &row) == 0 &&
          memcmp(&row, &zero, sizeof row) == 0, "no-protocol row contains data");
    CHECK(aotx_call_format_valid(&zero), "zero row is invalid");
    CHECK(!aotx_call_format_valid(NULL), "null row passes");
    CHECK(aotx_call_format_make(AOTX_CALL_HERMES, NULL) != 0, "null output passes");
    for (unsigned kind = AOTX_CALL_HERMES; kind < AOTX_CALL_FORMAT_KINDS; ++kind) {
        CHECK(aotx_call_format_make(kind, &row) == 0 && aotx_call_format_valid(&row),
              "protocol %u does not fit", kind);
        CHECK(row.kind == kind && row.result_json == (kind == AOTX_CALL_LLAMA_JSON),
              "protocol kind or result encoding differs");
        size_t used = 0u;
        for (unsigned span = 0u; span < AOTX_CALL_FORMAT_SPANS; ++span) {
            CHECK(row.offset[span] == used, "span order differs");
            used += row.length[span];
            changed = row; changed.offset[span] = UINT16_MAX;
            CHECK(!aotx_call_format_valid(&changed), "out-of-range offset passes");
            changed = row; changed.length[span] = AOTX_CALL_FORMAT_SPAN_BYTES + 1u;
            CHECK(!aotx_call_format_valid(&changed), "long span passes");
            changed = row; changed.offset[span] = AOTX_CALL_FORMAT_BYTES; changed.length[span] = 1u;
            CHECK(!aotx_call_format_valid(&changed), "span crosses row boundary");
        }
        changed = row; changed.result_json = 2u;
        CHECK(!aotx_call_format_valid(&changed), "invalid result encoding passes");
        changed = row; changed.result_json ^= 1u;
        CHECK(!aotx_call_format_valid(&changed), "wrong protocol result encoding passes");
        changed = row; changed.kind = AOTX_CALL_FORMAT_KINDS;
        CHECK(!aotx_call_format_valid(&changed), "unknown kind passes");
    }
    changed = zero; changed.length[0] = 1u;
    CHECK(!aotx_call_format_valid(&changed), "no-protocol row advertises text");
    changed = zero; changed.kind = AOTX_CALL_HERMES;
    changed.length[0] = AOTX_CALL_FORMAT_SPAN_BYTES;
    changed.length[1] = AOTX_CALL_FORMAT_SPAN_BYTES;
    CHECK(aotx_call_format_valid(&changed), "exact aggregate bound is refused");
    changed.length[2] = 1u;
    CHECK(!aotx_call_format_valid(&changed), "overlapping spans hide aggregate overflow");
    memset(&row, 0xa5, sizeof row);
    CHECK(aotx_call_format_make(AOTX_CALL_FORMAT_KINDS, &row) != 0 &&
          memcmp(&row, &zero, sizeof row) == 0, "invalid construction leaves stale bytes");

    aotx_call_format_make(AOTX_CALL_HERMES, &row);
    CHECK(same_span(&row, AOTX_CALL_TOOLS_HEAD,
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"),
        "legacy tool heading differs");
    CHECK(same_span(&row, AOTX_CALL_TOOLS_TAIL, "</tools>\n") &&
          same_span(&row, AOTX_CALL_INSTRUCTION,
        "\nFor each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
        "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"),
        "legacy instructions differ");
    CHECK(same_span(&row, AOTX_CALL_HEAD, "<tool_call>") &&
          same_span(&row, AOTX_CALL_TAIL, "</tool_call>") &&
          same_span(&row, AOTX_CALL_ARG_KEY, "arguments"), "tagged JSON grammar differs");
    CHECK(same_span(&row, AOTX_CALL_RESULT_HEAD, "<|im_start|>user\n\n<tool_response>\n") &&
          same_span(&row, AOTX_CALL_RESULT_TAIL, "\n</tool_response><|im_end|>\n"),
          "legacy result framing differs");
    aotx_call_format_make(AOTX_CALL_LLAMA_JSON, &row);
    CHECK(same_span(&row, AOTX_CALL_SYSTEM_PREFIX, "Environment: ipython\n") &&
          same_span(&row, AOTX_CALL_ARG_KEY, "parameters") &&
          same_span(&row, AOTX_CALL_NAME_HEAD, "<|python_tag|>") &&
          same_span(&row, AOTX_CALL_HEAD, "") && same_span(&row, AOTX_CALL_TAIL, ""),
          "Llama bare JSON grammar differs");
    CHECK(same_span(&row, AOTX_CALL_INSTRUCTION,
        "You have access to the following functions. To call a function, please respond with JSON for a function call."
        "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}."
        "Do not use variables.\n\n"), "Llama system instructions differ");
    CHECK(same_span(&row, AOTX_CALL_RESULT_HEAD, "<|start_header_id|>ipython<|end_header_id|>\n\n") &&
          same_span(&row, AOTX_CALL_RESULT_TAIL, "<|eot_id|>"), "Llama result framing differs");
    aotx_call_format_make(AOTX_CALL_QWEN_XML, &row);
    CHECK(same_span(&row, AOTX_CALL_NAME_HEAD, "<function=") &&
          same_span(&row, AOTX_CALL_NAME_TAIL, ">") &&
          same_span(&row, AOTX_CALL_NAME_CLOSE, "</function>") &&
          same_span(&row, AOTX_CALL_ARG_HEAD, "<parameter=") &&
          same_span(&row, AOTX_CALL_ARG_TAIL, "</parameter>") &&
          same_span(&row, AOTX_CALL_ARG_KEY, ""), "XML grammar differs");
    CHECK(same_span(&row, AOTX_CALL_TOOLS_HEAD,
          "# Tools\n\nYou have access to the following functions:\n\n<tools>\n") &&
          same_span(&row, AOTX_CALL_RESULT_HEAD, "<|im_start|>user\n<tool_response>\n"),
          "XML tool or result heading differs");
}

static void select_cases(const char *text, unsigned kind, unsigned batch)
{
    aotx_call_format rows[64], expected, zero = {0};
    aotx_meta meta[2] = {0};
    aotx_modelfile file = {0};
    aotx_manifest_entry entry = {0};
    aotx_wrap wrap;
    char altered[8192];
    size_t length = strlen(text);
    CHECK(length < sizeof altered, "fixture is too long");
    if (length >= sizeof altered) return;
    meta[0].key = "tokenizer.chat_template"; meta[0].type = AOTX_GGUF_STRING;
    meta[0].text = (char *)text; meta[0].text_bytes = length;
    meta[1].key = "general.architecture"; meta[1].type = AOTX_GGUF_STRING;
    meta[1].text = "llama"; meta[1].text_bytes = 5u;
    file.meta = meta; file.meta_count = 2u;
    entry.wrap_present = 1u; entry.wrap.end_count = 1u;
    entry.wrap.bytes[0] = 'S'; entry.wrap.length[0] = 1u;
    CHECK(aotx_call_format_make(kind, &expected) == 0, "expected protocol does not fit");
    memset(rows, 0xa5, sizeof rows);
    for (unsigned i = 0u; i < batch; ++i) {
        CHECK(aotx_call_format_read(&file, &rows[i]) == 0 &&
              memcmp(&rows[i], &expected, sizeof expected) == 0, "exact template selects wrong row");
        CHECK(aotx_wrap_read(&file, &entry, &wrap) == 0 && wrap.kind == 0u,
              "manifest wrap was changed");
        CHECK(aotx_call_format_read(&file, &rows[i]) == 0 && rows[i].kind == kind,
              "manifest wrap suppresses template selection");
        const size_t positions[] = { 0u, length / 2u, length - 1u };
        for (unsigned k = 0u; k < 3u; ++k) {
            memcpy(altered, text, length); altered[positions[k]] ^= 1;
            meta[0].text = altered;
            CHECK(aotx_call_format_read(&file, &rows[i]) == 0 &&
                  memcmp(&rows[i], &zero, sizeof zero) == 0,
                  "same-length template mutation selects a protocol");
        }
        memcpy(altered, text, length); altered[length] = ' ';
        meta[0].text_bytes = length + 1u;
        CHECK(aotx_call_format_read(&file, &rows[i]) == 0 && rows[i].kind == AOTX_CALL_NONE,
              "template suffix selects a protocol");
        meta[0].text = (char *)text; meta[0].text_bytes = length;
    }
    meta[0].type = AOTX_GGUF_U32;
    CHECK(aotx_call_format_read(&file, &rows[0]) == 0 && rows[0].kind == AOTX_CALL_NONE,
          "non-string template selects a protocol");
    file.meta_count = 0u;
    CHECK(aotx_call_format_read(&file, &rows[0]) == 0 && rows[0].kind == AOTX_CALL_NONE,
          "missing template selects a protocol");
    CHECK(aotx_call_format_read(NULL, &rows[0]) == 0 && rows[0].kind == AOTX_CALL_NONE,
          "missing file selects a protocol");
    CHECK(aotx_call_format_read(&file, NULL) != 0, "null output accepts a read");
}

int main(void)
{
    const unsigned native_kinds[] = { AOTX_CALL_HERMES, AOTX_CALL_QWEN_XML,
                                     AOTX_CALL_NONE, AOTX_CALL_NONE };
    for (unsigned batch = 1u; batch <= 64u; batch *= 64u) {
        for (unsigned i = 0u; i < batch; ++i) row_cases();
        for (unsigned i = 0u; i < sizeof wrap_templates / sizeof wrap_templates[0]; ++i)
            select_cases(wrap_templates[i], i == 4u ? AOTX_CALL_LLAMA_JSON : AOTX_CALL_HERMES, batch);
        for (unsigned i = 0u; i < sizeof call_templates / sizeof call_templates[0]; ++i)
            select_cases(call_templates[i], native_kinds[i], batch);
        select_cases("<tool_call>{\"name\":\"run\",\"arguments\":{}}</tool_call>", AOTX_CALL_NONE, batch);
        printf("call_format_test: batch %u\n", batch);
    }
    return aotx_report("call_format_test", 1000);
}
