/* Purpose: Reduce known model templates to bounded turn spans without execution.
 * Owns: No storage; the caller owns the result.
 * Threading: One caller for each model file.
 * Lifetime: One read, check, or print call. */
#include "disk/modelfile/gguf.h"
#include "disk/modelfile/manifest.h"
#include "disk/modelfile/manifest_json.h"
#include <stdio.h>
#include <string.h>

const char *const aotx_wrap_names[AOTX_WRAP_SPANS] = {
    "system_head", "system_tail", "user_head", "user_tail", "assistant_head",
    "assistant_tail", "generation_head", "think_open", "think_close"
};

int aotx_wrap_valid(const aotx_wrap *wrap)
{
    if (wrap == NULL || wrap->end_count == 0u || wrap->end_count > AOTX_WRAP_ENDS ||
        wrap->kind > 2u || wrap->usable > 1u) return 0;
    for (unsigned i = 0; i < AOTX_WRAP_SPANS; ++i) {
        if (wrap->length[i] > AOTX_WRAP_SPAN_BYTES || wrap->offset[i] > AOTX_WRAP_BYTES ||
            wrap->length[i] > AOTX_WRAP_BYTES - wrap->offset[i]) return 0;
    }
    if (wrap->prefix_length > wrap->length[AOTX_WRAP_SYSTEM_HEAD]) return 0;
    for (unsigned i = 0; i < wrap->end_count; ++i)
        for (unsigned k = 0; k < i; ++k)
            if (wrap->end_ids[i] == wrap->end_ids[k]) return 0;
    return 1;
}

static int spans(aotx_wrap *wrap, const char *const *values)
{
    size_t used = 0;
    for (unsigned i = 0; i < AOTX_WRAP_SPANS; ++i) {
        size_t length = strlen(values[i]);
        if (length > AOTX_WRAP_SPAN_BYTES || length > AOTX_WRAP_BYTES - used) return -1;
        wrap->offset[i] = (uint16_t)used;
        wrap->length[i] = (uint8_t)length;
        memcpy(wrap->bytes + used, values[i], length);
        used += length;
    }
    return 0;
}

static int canonical(aotx_wrap *wrap, unsigned kind)
{
    static const char *const qwen[AOTX_WRAP_SPANS] = {
        "<|im_start|>system\n", "<|im_end|>\n", "<|im_start|>user\n", "<|im_end|>\n",
        "<|im_start|>assistant\n", "<|im_end|>\n", "<|im_start|>assistant\n",
        "<think>\n\n", "</think>\n\n"
    };
    static const char *const llama[AOTX_WRAP_SPANS] = {
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>",
        "<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>",
        "<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>",
        "<|start_header_id|>assistant<|end_header_id|>\n\n", "", ""
    };
    wrap->prefix_length = kind == 2u ? 17u : 0u;
    return spans(wrap, kind == 1u ? qwen : llama);
}

int aotx_wrap_matches(const aotx_wrap *wrap)
{
    aotx_wrap expected = {0};
    if (wrap == NULL) return 0;
    if (wrap->kind == 0u) return 1;
    if (wrap->kind > 2u || canonical(&expected, wrap->kind) != 0 ||
        wrap->prefix_length != expected.prefix_length) return 0;
    for (unsigned i = 0; i < AOTX_WRAP_SPANS; ++i) {
        if (wrap->offset[i] > AOTX_WRAP_BYTES ||
            wrap->length[i] > AOTX_WRAP_BYTES - wrap->offset[i] ||
            wrap->length[i] != expected.length[i] ||
            memcmp(wrap->bytes + wrap->offset[i], expected.bytes + expected.offset[i],
                   expected.length[i]) != 0) return 0;
    }
    return 1;
}

static int end_add(aotx_wrap *wrap, uint32_t id, uint64_t count)
{
    if (id >= count) return -1;
    for (unsigned i = 0; i < wrap->end_count; ++i)
        if (wrap->end_ids[i] == id) return 0;
    if (wrap->end_count == AOTX_WRAP_ENDS) return -1;
    wrap->end_ids[wrap->end_count++] = id;
    return 0;
}

static int end_text(aotx_wrap *wrap, const aotx_string_array *vocab, const char *text)
{
    size_t length = strlen(text);
    for (uint64_t i = 0; i < vocab->count; ++i) {
        uint64_t start = vocab->offsets[i], end = vocab->offsets[i + 1u];
        if (end >= start && end - start == length &&
            memcmp(vocab->bytes + start, text, length) == 0)
            return end_add(wrap, (uint32_t)i, vocab->count);
    }
    return -1;
}

static int reduce(const aotx_modelfile *file, aotx_wrap *wrap)
{
    /* Each digest covers all template bytes, including whitespace and tool branches.
     * Only the text-turn spans are emitted; the client owns system preamble text. */
    static const struct { size_t length; const char *digest; unsigned kind; } known[] = {
        {4100u, "57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361", 1u},
        {4168u, "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8", 1u},
        {4761u, "8428c815ac94d82064e35ff1e841dcbe260e7e53a8d0bd3b94afa2eefa9bccab", 1u},
        {4116u, "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5", 1u},
        {3827u, "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4", 2u}
    };
    const char *template;
    size_t length;
    aotx_sha256 hash;
    unsigned char digest[AOTX_SHA256_DIGEST];
    char hex[AOTX_SHA256_HEX];
    aotx_string_array vocab;
    uint32_t id;
    int possible = 0;
    if (aotx_modelfile_string(file, "tokenizer.chat_template", &template, &length) != 0) return -1;
    for (unsigned i = 0; i < sizeof(known) / sizeof(known[0]); ++i)
        if (length == known[i].length) possible = 1;
    if (!possible) return -1;
    aotx_sha256_init(&hash);
    aotx_sha256_update(&hash, template, length);
    aotx_sha256_final(&hash, digest);
    aotx_sha256_text(digest, hex);
    for (unsigned i = 0; i < sizeof(known) / sizeof(known[0]); ++i)
        if (length == known[i].length && strcmp(hex, known[i].digest) == 0) wrap->kind = known[i].kind;
    if (!wrap->kind || canonical(wrap, wrap->kind) != 0) return -1;
    if (aotx_modelfile_strings(file, "tokenizer.ggml.tokens", &vocab) != 0 ||
        !vocab.count || vocab.count > UINT32_MAX ||
        aotx_modelfile_u32(file, "tokenizer.ggml.eos_token_id", &id) != 0 ||
        end_add(wrap, id, vocab.count) != 0) return -1;
    if (aotx_modelfile_u32(file, "tokenizer.ggml.eot_token_id", &id) == 0 &&
        end_add(wrap, id, vocab.count) != 0) return -1;
    if (end_text(wrap, &vocab, wrap->kind == 1u ? "<|im_end|>" : "<|eot_id|>") != 0) return -1;
    if (wrap->kind == 2u && end_text(wrap, &vocab, "<|end_of_text|>") != 0) return -1;
    if (wrap->kind == 1u && end_text(wrap, &vocab, "<|endoftext|>") != 0) return -1;
    if (wrap->kind == 2u) (void)end_text(wrap, &vocab, "<|eom_id|>");
    return aotx_wrap_valid(wrap) ? 0 : -1;
}

int aotx_wrap_read(const aotx_modelfile *file, const aotx_manifest_entry *entry, aotx_wrap *wrap)
{
    const char *path = file != NULL ? file->path : entry != NULL ? entry->path : "model";
    if (wrap == NULL) return -1;
    if (entry != NULL && entry->wrap_present) {
        if (entry->wrap_present == 1u && aotx_wrap_valid(&entry->wrap)) {
            *wrap = entry->wrap;
            wrap->kind = 0;
            wrap->usable = 0;
            wrap->think_open_id = UINT32_MAX;
            wrap->think_close_id = UINT32_MAX;
            return 0;
        }
    } else {
        memset(wrap, 0, sizeof(*wrap));
        wrap->think_open_id = UINT32_MAX;
        wrap->think_close_id = UINT32_MAX;
        if (file != NULL && reduce(file, wrap) == 0) return 0;
    }
    memset(wrap, 0, sizeof(*wrap));
    fprintf(stderr, "aotx_wrap: %s: a valid wrap block is required\n", path);
    return -1;
}

void aotx_wrap_print(const char *name, const aotx_wrap *wrap)
{
    if (!aotx_wrap_valid(wrap)) {
        printf("%s wrap invalid\n", name);
        return;
    }
    printf("%s wrap prefix_length=%u end_ids=[", name, wrap->prefix_length);
    for (unsigned i = 0; i < wrap->end_count; ++i)
        printf("%s%u", i ? "," : "", wrap->end_ids[i]);
    printf("]\n");
    for (unsigned i = 0; i < AOTX_WRAP_SPANS; ++i) {
        printf("  %s=\"", aotx_wrap_names[i]);
        for (unsigned k = 0; k < wrap->length[i]; ++k) {
            unsigned char c = wrap->bytes[wrap->offset[i] + k];
            if (c == '\n') fputs("\\n", stdout);
            else if (c == '\r') fputs("\\r", stdout);
            else if (c == '\t') fputs("\\t", stdout);
            else if (c == '\\' || c == '"') printf("\\%c", c);
            else if (c < 0x20u || c >= 0x7fu) printf("\\x%02x", c);
            else putchar(c);
        }
        printf("\"\n");
    }
}
