/* Purpose: Check manifest precedence, bounded spans, and exact template reduction.
 * Owns: Test entries, model metadata, and output buffers.
 * Threading: One test thread.
 * Lifetime: The test process. */
#include "tests/disk_fake.h"
#include "tests/wrap_templates.h"
#include "tests/call_templates.h"
#include "disk/modelfile/manifest.h"
#include "disk/modelfile/gguf.h"

#define BASE "\"name\":\"test\",\"role\":\"say\",\"path\":\"model.gguf\"," \
    "\"source\":\"source\",\"revision\":\"main\",\"license\":\"license\"," \
    "\"bytes\":12,\"sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\""
#define SPANS "\"system_head\":\"S\\n\",\"system_tail\":\"s\",\"user_head\":\"U\"," \
    "\"user_tail\":\"u\",\"assistant_head\":\"A\",\"assistant_tail\":\"a\"," \
    "\"generation_head\":\"G\",\"think_open\":\"<\",\"think_close\":\">\""
#define BLOCK "\"wrap\":{" SPANS ",\"end_ids\":[7,8],\"prefix_length\":1}"

static int same_span(const aotx_wrap *w, unsigned index, const char *text)
{
    return w->length[index] == strlen(text) &&
           memcmp(w->bytes + w->offset[index], text, strlen(text)) == 0;
}

static void manifest_cases(void)
{
    aotx_manifest_entry entry, again;
    aotx_wrap wrap;
    char line[AOTX_MANIFEST_LINE];
    const char *bad[] = {
        "{" BASE ",\"name\":\"other\"}",
        "{" BASE ",\"\\u006eame\":\"other\"}",
        "{" BASE ",\"probe_numerator\":1x}",
        "{" BASE ",\"probe_numerator\":01}",
        "{" BASE ",\"probe_numerator\":1.0}",
        "{" BASE ",\"probe_numerator\":1e0}",
        "{" BASE ",\"probe_numerator\":-1}",
        "{" BASE ",\"probe_numerator\":4294967296}",
        "{" BASE ",\"probe_numerator\":18446744073709551616}",
        "{" BASE ",\"probe_denominator\":0}",
        "{" BASE ",\"probe_numerator\":3,\"probe_denominator\":3}",
        "{" BASE ",\"wrap\":{" SPANS ",\"end_ids\":[]}}",
        "{" BASE ",\"wrap\":{" SPANS ",\"end_ids\":[1,1]}}",
        "{" BASE ",\"wrap\":{" SPANS ",\"end_ids\":[1,2,3,4,5,6,7,8,9]}}",
        "{" BASE ",\"wrap\":{" SPANS ",\"end_ids\":[4294967296]}}",
        "{" BASE ",\"wrap\":{" SPANS ",\"end_ids\":[1],\"prefix_length\":3}}",
        "{" BASE ",\"wrap\":{" SPANS ",\"end_ids\":[1],\"system_head\":\"x\"}}",
        "{" BASE ",\"wrap\":{\"end_ids\":[1]}}",
        "{" BASE ",\"wrap\":\"bad\"}",
        "{" BASE ",\"wrap\":{" SPANS ",\"end_ids\":[1],}}",
        "{" BASE ",}", "{" BASE "}junk",
        "{\"name\":\"bad\\q\"}", "{\"name\":\"bad\\uD800\"}",
        "{\"name\":\"bad\\uDC00\"}", "{\"name\":\"bad\\u12\"}",
        "{\"name\":\"bad\\u0000key\"}", "{\"name\":\"bad\nvalue\"}"
    };
    CHECK(aotx_manifest_line("{" BASE "}", &entry) == 0, "default entry does not read");
    CHECK(entry.probe_numerator == 2 && entry.probe_denominator == 3, "probe default differs");
    CHECK(aotx_manifest_line("{" BASE "," BLOCK ",\"probe_numerator\":0,\"probe_denominator\":5}", &entry) == 0,
          "explicit entry does not read");
    CHECK(aotx_wrap_read(NULL, &entry, &wrap) == 0, "explicit block reads model metadata");
    CHECK(wrap.kind == 0 && !wrap.usable && wrap.think_open_id == UINT32_MAX &&
          wrap.think_close_id == UINT32_MAX, "derived state was not cleared");
    CHECK(same_span(&wrap, AOTX_WRAP_SYSTEM_HEAD, "S\n"), "escaped span differs");
    CHECK(aotx_manifest_write_line(line, sizeof(line), &entry) == 0 &&
          aotx_manifest_line(line, &again) == 0, "manifest round trip fails");
    CHECK(again.probe_numerator == 0 && again.probe_denominator == 5 &&
          again.wrap_present && memcmp(&entry.wrap, &again.wrap, sizeof(wrap)) == 0,
          "explicit fields did not survive the write");
    snprintf(entry.source, sizeof(entry.source), "a \\\"wrap\\\": key");
    CHECK(aotx_manifest_write_line(line, sizeof(line), &entry) == 0 &&
          aotx_manifest_line(line, &again) == 0 && strcmp(entry.source, again.source) == 0,
          "a key inside a string changed the parse");
    CHECK(aotx_manifest_write_line(line, 8, &entry) != 0, "a short writer buffer succeeds");
    for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i)
        CHECK(aotx_manifest_line(bad[i], &again) != 0, "bad manifest %u reads", i);
    for (unsigned n = 0; n < strlen("{" BASE "," BLOCK "}"); ++n) {
        memcpy(line, "{" BASE "," BLOCK "}", n); line[n] = 0;
        CHECK(aotx_manifest_line(line, &again) != 0, "truncated manifest reads at %u", n);
    }
    memset(line, ' ', sizeof(line) - 1u); line[sizeof(line) - 1u] = 0;
    CHECK(aotx_manifest_line(line, &again) != 0, "oversized whitespace reads");
    {
        char large[66];
        memset(large, 'x', 65); large[65] = 0;
        const char *format = "{" BASE ",\"wrap\":{\"system_head\":\"%s\","
            "\"system_tail\":\"\",\"user_head\":\"\",\"user_tail\":\"\",\"assistant_head\":\"\","
            "\"assistant_tail\":\"\",\"generation_head\":\"\",\"think_open\":\"\",\"think_close\":\"\",\"end_ids\":[1]}}";
        snprintf(line, sizeof(line), format, large);
        CHECK(aotx_manifest_line(line, &again) != 0, "a 65-byte span reads");
        large[64] = 0;
        snprintf(line, sizeof(line), format, large);
        CHECK(aotx_manifest_line(line, &again) == 0 && again.wrap.length[0] == 64u,
              "a 64-byte span does not read");
        const char *bad_string[] = {"\\q", "\\uD800", "\\uDC00", "\\u12", "\n", "\xc0\x80"};
        for (unsigned i = 0; i < sizeof(bad_string) / sizeof(bad_string[0]); ++i) {
            snprintf(line, sizeof(line), format, bad_string[i]);
            CHECK(aotx_manifest_line(line, &again) != 0, "bad span string %u reads", i);
        }
        snprintf(line, sizeof(line), "{" BASE ",\"wrap\":{\"system_head\":\"\\uD83D\\uDE00\","
            "\"system_tail\":\"\",\"user_head\":\"\",\"user_tail\":\"\",\"assistant_head\":\"\","
            "\"assistant_tail\":\"\",\"generation_head\":\"\",\"think_open\":\"\",\"think_close\":\"\",\"end_ids\":[1]}}");
        CHECK(aotx_manifest_line(line, &again) == 0 &&
              same_span(&again.wrap, 0, "\xf0\x9f\x98\x80"), "Unicode surrogate pair differs");
    }
    {
        aotx_wrap full = {0};
        static const char *const names[] = {"system_head", "system_tail", "user_head", "user_tail",
            "assistant_head", "assistant_tail", "generation_head", "think_open", "think_close"};
        size_t used = (size_t)snprintf(line, sizeof(line), "{" BASE ",\"wrap\":{");
        for (unsigned i = 0; i < AOTX_WRAP_SPANS; ++i)
            used += (size_t)snprintf(line + used, sizeof(line) - used, "%s\"%s\":\"%064u\"",
                                    i ? "," : "", names[i], i);
        snprintf(line + used, sizeof(line) - used, ",\"end_ids\":[1]}}");
        CHECK(aotx_manifest_line(line, &again) != 0, "span bytes beyond table capacity read");
        full.end_count = 1u;
        CHECK(aotx_wrap_matches(&full), "explicit spans require a known shape");
    }
}

static void bounds(void)
{
    aotx_manifest_entry entry;
    aotx_wrap w;
    CHECK(aotx_manifest_line("{" BASE "," BLOCK "}", &entry) == 0, "bounds entry fails");
    for (unsigned i = 0; i < AOTX_WRAP_SPANS; ++i) {
        w = entry.wrap; w.offset[i] = UINT16_MAX;
        CHECK(!aotx_wrap_valid(&w), "invalid span offset passes");
        w = entry.wrap; w.length[i] = 65;
        CHECK(!aotx_wrap_valid(&w), "long span passes");
        w = entry.wrap; w.offset[i] = AOTX_WRAP_BYTES;
        CHECK(!aotx_wrap_valid(&w), "span past table passes");
    }
    w = entry.wrap; w.end_count = 9;
    CHECK(!aotx_wrap_valid(&w), "large end list passes");
    w.end_count = 0;
    CHECK(!aotx_wrap_valid(&w), "empty end list passes");
    w = entry.wrap; w.prefix_length = 3;
    CHECK(!aotx_wrap_valid(&w), "prefix past system header passes");
}

static void reducer_cases(void)
{
    static unsigned char vocab[] = "<|im_end|><|endoftext|><|eot_id|><|end_of_text|><|eom_id|>";
    uint64_t offsets[6];
    /* Compute offsets from the vocabulary spellings, not from model ids. */
    const char *words[] = {"<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"};
    aotx_meta meta[3] = {0};
    aotx_modelfile file = {0};
    aotx_wrap wrap;
    aotx_manifest_entry entry;
    const char *templates[] = {wrap_template_0, wrap_template_1, wrap_template_2,
        wrap_template_3, wrap_template_4, call_template_1};
    char altered[sizeof(call_template_1) + 32u];
    offsets[0] = 0;
    for (unsigned i = 0; i < 5u; ++i) offsets[i + 1u] = offsets[i] + strlen(words[i]);
    meta[0].key = "tokenizer.chat_template"; meta[0].type = AOTX_GGUF_STRING;
    meta[1].key = "tokenizer.ggml.tokens"; meta[1].type = AOTX_GGUF_ARRAY;
    meta[1].element_type = AOTX_GGUF_STRING; meta[1].count = 5;
    meta[1].run = vocab; meta[1].offsets = offsets;
    meta[2].key = "tokenizer.ggml.eos_token_id"; meta[2].type = AOTX_GGUF_U32;
    file.meta = meta; file.meta_count = 3;
    snprintf(file.path, sizeof(file.path), "test-model.gguf");
    for (unsigned i = 0; i < sizeof(templates) / sizeof(templates[0]); ++i) {
        int llama = i == 4u;
        meta[0].text = (char *)templates[i]; meta[0].text_bytes = strlen(templates[i]);
        meta[2].u = llama ? 2 : 0;
        CHECK(aotx_wrap_read(&file, NULL, &wrap) == 0, "known template %u refuses", i);
        CHECK(wrap.kind == (llama ? 2u : 1u) && wrap.prefix_length == (llama ? 17u : 0u),
              "known template %u kind or prefix differs", i);
        CHECK(wrap.end_count == (llama ? 3u : 2u), "end token count differs");
        CHECK(same_span(&wrap, AOTX_WRAP_USER_HEAD, llama ?
            "<|start_header_id|>user<|end_header_id|>\n\n" : "<|im_start|>user\n"), "user header differs");
        CHECK(same_span(&wrap, AOTX_WRAP_THINK_OPEN, llama ? "" : "<think>\n\n"), "think span differs");
        CHECK(same_span(&wrap, AOTX_WRAP_GENERATION_HEAD, llama ?
            "<|start_header_id|>assistant<|end_header_id|>\n\n" : "<|im_start|>assistant\n"),
            "generation header repeats a thinking span");
        CHECK(same_span(&wrap, AOTX_WRAP_THINK_CLOSE, llama ? "" : "</think>\n\n"),
              "closing thinking span differs");
        CHECK(aotx_wrap_matches(&wrap), "canonical spans fail the semantic check");
        {
            aotx_wrap changed = wrap;
            uint16_t offset = changed.offset[AOTX_WRAP_USER_HEAD];
            uint8_t length = changed.length[AOTX_WRAP_USER_HEAD];
            changed.offset[AOTX_WRAP_USER_HEAD] = changed.offset[AOTX_WRAP_ASSISTANT_HEAD];
            changed.length[AOTX_WRAP_USER_HEAD] = changed.length[AOTX_WRAP_ASSISTANT_HEAD];
            changed.offset[AOTX_WRAP_ASSISTANT_HEAD] = offset;
            changed.length[AOTX_WRAP_ASSISTANT_HEAD] = length;
            CHECK(aotx_wrap_valid(&changed) && !aotx_wrap_matches(&changed),
                  "swapped role headers pass the semantic check");
            changed = wrap; ++changed.prefix_length;
            CHECK(!aotx_wrap_matches(&changed), "a changed prefix passes the semantic check");
            changed = wrap; changed.end_ids[0] = UINT32_MAX;
            CHECK(aotx_wrap_matches(&changed), "the span check reads end ids");
        }
        strcpy(altered, templates[i]); altered[0] ^= 1;
        meta[0].text = altered;
        CHECK(aotx_wrap_read(&file, NULL, &wrap) != 0, "same-length template change passes");
        strcpy(altered, templates[i]); strcat(altered, "malicious addition");
        meta[0].text_bytes = strlen(altered);
        CHECK(aotx_wrap_read(&file, NULL, &wrap) != 0, "template addition passes");
    }
    meta[0].text = "<|im_start|>system <|im_end|> <|start_header_id|>";
    meta[0].text_bytes = strlen(meta[0].text);
    CHECK(aotx_wrap_read(&file, NULL, &wrap) != 0, "unknown token lookalike template passes");
    CHECK(aotx_manifest_line("{" BASE "," BLOCK "}", &entry) == 0 &&
          aotx_wrap_read(&file, &entry, &wrap) == 0, "unknown template defeats explicit block");
    entry.wrap.length[0] = 65;
    meta[0].text = (char *)wrap_templates[0]; meta[0].text_bytes = strlen(meta[0].text);
    CHECK(aotx_wrap_read(&file, &entry, &wrap) != 0, "invalid explicit block falls back to template");
    meta[2].u = 5;
    CHECK(aotx_wrap_read(&file, NULL, &wrap) != 0, "out-of-vocabulary eos passes");
    file.meta_count = 0;
    CHECK(aotx_wrap_read(&file, NULL, &wrap) != 0, "missing template passes");
}

static void file_cases(void)
{
    char dir[128], path[256];
    aotx_manifest_entry entry;
    FILE *file;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the manifest test directory does not open");
    snprintf(path, sizeof(path), "%s/manifest.jsonl", dir);
    file = fopen(path, "wb");
    CHECK(file != NULL, "the manifest test file does not open");
    if (file != NULL) {
        fputs("{" BASE "}", file); fputc(0, file); fputs("hidden\n", file); fclose(file);
        CHECK(aotx_manifest_read(dir, &entry, 1) == -1, "a literal null hides trailing bytes");
    }
    file = fopen(path, "wb");
    CHECK(file != NULL, "the manifest test file does not reopen");
    if (file != NULL) {
        fputs("\r\n{" BASE "}\r\n", file); fclose(file);
        CHECK(aotx_manifest_read(dir, &entry, 1) == 1, "a CRLF manifest does not read");
    }
    file = fopen(path, "wb");
    CHECK(file != NULL, "the long manifest test file does not open");
    if (file != NULL) {
        for (unsigned i = 0; i < AOTX_MANIFEST_LINE; ++i) fputc(' ', file);
        fclose(file);
        CHECK(aotx_manifest_read(dir, &entry, 1) == -1, "an oversized manifest line reads");
    }
    aotx_remove_tree(dir);
}

int main(void)
{
    for (unsigned n = 1; n <= 64u; n *= 64u) {
        for (unsigned i = 0; i < n; ++i) { manifest_cases(); bounds(); reducer_cases(); file_cases(); }
        printf("wrap_test: batch %u\n", n);
    }
    return aotx_report("wrap_test", 1000);
}
