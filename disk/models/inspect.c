/* Purpose: Print model header facts and compiled support limits before a full fetch.
 * Owns: One parsed header and one tensor type list.
 * Threading: One command, one source at a time.
 * Lifetime: One command. */
#include "disk/models/inspect.h"
#include "disk/wire/diskwire.h"
#include "cuda/model/kinds_data.h"
#include "cuda/model/names.h"
#include "cuda/text/families.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct aotx_inspect_kind {
    const char *name;
    const aotx_layer_tensor *tensor;
    unsigned int tensors;
} aotx_inspect_kind;

#define AOTX_INSPECT_KIND(name, tensor, state, capture, key, check) \
    { name, tensor, sizeof tensor / sizeof tensor[0] },
static const aotx_inspect_kind kinds[] = { AOTX_LAYER_KIND_TABLE(AOTX_INSPECT_KIND) };
#undef AOTX_INSPECT_KIND
#define AOTX_INSPECT_FAMILY(name, pattern, whole) name,
static const char *families[] = { AOTX_TEXT_FAMILY_TABLE(AOTX_INSPECT_FAMILY) };
#undef AOTX_INSPECT_FAMILY
static const char *whole_names[] = AOTX_DESC_WHOLE_LIST;

/* Escape source bytes so the header cannot write terminal controls or extra lines. */
static void print_text(const char *text, size_t bytes)
{
    size_t limit = bytes < 512u ? bytes : 512u;
    for (size_t i = 0; i < limit; ++i) {
        unsigned char c = (unsigned char)text[i];
        if (c > 32u && c < 127u && c != '\\') putchar(c);
        else printf("\\x%02x", c);
    }
    if (bytes > limit) printf("[cut]");
}

static int text_value(const aotx_modelfile *file, const char *key,
                      const char **text, size_t *bytes, const char *source)
{
    int rc = aotx_modelfile_string(file, key, text, bytes);
    if (rc == 1) { *text = ""; *bytes = 0; return 0; }
    if (rc != 0) {
        fprintf(stderr, "aotx_models: %s: %s is not a string\n", source, key);
        return 2;
    }
    return 0;
}

static int number_value(const aotx_modelfile *file, const char *arch, const char *tail,
                        uint32_t *value, const char *source)
{
    char key[256];
    int n = snprintf(key, sizeof key, "%s.%s", arch, tail);
    if (n < 0 || (size_t)n >= sizeof key) return 2;
    int rc = aotx_modelfile_u32(file, key, value);
    if (rc == 1) { *value = 0; return 0; }
    if (rc != 0) {
        fprintf(stderr, "aotx_models: %s: %s is not an unsigned 32-bit number\n", source, key);
        return 2;
    }
    return 0;
}

static int type_order(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

typedef struct layer_set {
    unsigned int seen[AOTX_LAYER_KIND_COUNT];
    unsigned int extra[AOTX_LAYER_KIND_COUNT];
} layer_set;

/* Each layer must contain exactly one compiled tensor set. Unknown tensors are not ignored. */
static int tensor_set(const aotx_tensor_info *t, uint32_t layers, layer_set *seen,
                      unsigned int *whole)
{
    for (unsigned int i = 0; i < AOTX_DESC_WHOLE; ++i) {
        if (strcmp(t->name, whole_names[i]) == 0) {
            *whole |= 1u << i;
            return 1;
        }
    }
    if (strncmp(t->name, "blk.", 4u) != 0 || t->name[4] < '0' || t->name[4] > '9') return 0;
    char *end;
    errno = 0;
    unsigned long layer = strtoul(t->name + 4, &end, 10);
    if (t->name[4] == '0' && end != t->name + 5) return 0;
    if (errno != 0 || *end != '.' || layer >= layers || layer >= AOTX_MODEL_MAX_LAYERS) return 0;
    unsigned int known = 0;
    for (unsigned int k = 0; k < AOTX_LAYER_KIND_COUNT; ++k) {
        unsigned int matched = 0;
        for (unsigned int i = 0; i < kinds[k].tensors; ++i) {
            const aotx_layer_tensor *slot = &kinds[k].tensor[i];
            size_t n = strlen(slot->name);
            if (strncmp(end + 1, slot->name, n) == 0 && strcmp(end + 1 + n, ".weight") == 0) {
                seen[layer].seen[k] |= 1u << i;
                matched = 1;
                known = 1;
                break;
            }
        }
        if (!matched) ++seen[layer].extra[k];
    }
    return known;
}

static int report(const char *source, aotx_modelfile *file, int remote, uint64_t received)
{
    const char *arch, *pre, *model, *template;
    size_t arch_bytes, pre_bytes, model_bytes, template_bytes;
    if (text_value(file, "general.architecture", &arch, &arch_bytes, source) != 0 ||
        text_value(file, "tokenizer.ggml.pre", &pre, &pre_bytes, source) != 0 ||
        text_value(file, "tokenizer.ggml.model", &model, &model_bytes, source) != 0 ||
        text_value(file, "tokenizer.chat_template", &template, &template_bytes, source) != 0) return 2;
    if (arch_bytes >= 96u || memchr(arch, 0, arch_bytes) != NULL) {
        fprintf(stderr, "aotx_models: %s: the architecture name is too long or holds a zero byte\n", source);
        return 2;
    }
    uint32_t layers = 0, hidden = 0;
    for (size_t i = 0; i < arch_bytes; ++i) {
        unsigned char c = (unsigned char)arch[i];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) {
            fprintf(stderr, "aotx_models: %s: the architecture name has invalid bytes\n", source);
            return 2;
        }
    }
    if (number_value(file, arch, "block_count", &layers, source) != 0 ||
        number_value(file, arch, "embedding_length", &hidden, source) != 0) return 2;
    aotx_string_array tokens = {0};
    int token_rc = aotx_modelfile_strings(file, "tokenizer.ggml.tokens", &tokens);
    if (token_rc == 2) {
        fprintf(stderr, "aotx_models: %s: tokenizer.ggml.tokens is not a string array\n", source);
        return 2;
    }
    int pre_good = 0;
    for (size_t i = 0; i < sizeof families / sizeof families[0]; ++i)
        if (strlen(families[i]) == pre_bytes && memcmp(families[i], pre, pre_bytes) == 0) pre_good = 1;
    int model_good = model_bytes == 4u && memcmp(model, "gpt2", 4u) == 0;
    uint64_t count = aotx_modelfile_tensor_count(file);
    if (count > SIZE_MAX / sizeof(uint32_t)) return 2;
    uint32_t *types = count != 0 ? malloc((size_t)count * sizeof(*types)) : NULL;
    if (count != 0 && types == NULL) {
        fprintf(stderr, "aotx_models: %s: the tensor type list does not allocate\n", source);
        return 1;
    }
    layer_set seen[AOTX_MODEL_MAX_LAYERS] = {0};
    unsigned int whole = 0;
    unsigned int kind_count[AOTX_LAYER_KIND_COUNT] = {0};
    uint64_t unknown = 0, vocab = tokens.count;
    int blocks_good = count != 0, shape_good = hidden != 0 && tokens.count != 0;
    for (uint64_t i = 0; i < count; ++i) {
        aotx_tensor_info t;
        if (aotx_modelfile_tensor(file, i, &t) != 0) { free(types); return 2; }
        types[i] = t.type;
        if (aotx_tensor_type_name(t.type) == NULL) blocks_good = 0;
        if (!tensor_set(&t, layers, seen, &whole)) ++unknown;
        if (strcmp(t.name, "token_embd.weight") == 0) {
            vocab = t.dim_count >= 2u ? t.dims[1] : 0;
            if (t.dim_count != 2u || t.dims[0] != hidden || vocab != tokens.count) shape_good = 0;
        }
    }
    int layer_good = layers != 0 && layers <= AOTX_MODEL_MAX_LAYERS && unknown == 0;
    if (layers <= AOTX_MODEL_MAX_LAYERS) {
        for (uint32_t l = 0; l < layers; ++l) {
            unsigned int matched = 0;
            for (unsigned int k = 0; k < AOTX_LAYER_KIND_COUNT; ++k) {
                unsigned int required = 0, allowed = 0;
                for (unsigned int i = 0; i < kinds[k].tensors; ++i) {
                    const aotx_layer_tensor *t = &kinds[k].tensor[i];
                    allowed |= 1u << i;
                    if (!t->may_be_absent) required |= 1u << i;
                }
                if (seen[l].extra[k] == 0 && (seen[l].seen[k] & required) == required
                    && (seen[l].seen[k] & ~allowed) == 0) {
                    ++kind_count[k]; matched = 1; break;
                }
            }
            if (!matched) layer_good = 0;
        }
    }
    if ((whole & 3u) != 3u) shape_good = 0;
    unsigned char digest[32];
    char hex[65];
    aotx_sha256 hash;
    aotx_sha256_init(&hash);
    aotx_sha256_update(&hash, template, template_bytes);
    aotx_sha256_final(&hash, digest);
    aotx_sha256_text(digest, hex);
    printf("file="); print_text(source, strlen(source)); putchar('\n');
    printf("architecture="); print_text(arch, arch_bytes); putchar('\n');
    printf("pre_tokenizer="); print_text(pre, pre_bytes); printf(" supported=%s\n", pre_good ? "yes" : "no");
    printf("tokenizer_model="); print_text(model, model_bytes); printf(" supported=%s\n", model_good ? "yes" : "no");
    printf("tensors=%" PRIu64 "\n", count);
    if (count != 0) qsort(types, (size_t)count, sizeof(*types), type_order);
    for (uint64_t i = 0; i < count;) {
        uint64_t end = i + 1u;
        while (end < count && types[end] == types[i]) ++end;
        const char *name = aotx_tensor_type_name(types[i]);
        printf("block_type=%s id=%u count=%" PRIu64 " supported=%s\n",
               name != NULL ? name : "unknown", types[i], end - i, name != NULL ? "yes" : "no");
        i = end;
    }
    free(types);
    printf("layers=%u hidden=%u vocabulary=%" PRIu64 "\n", layers, hidden, vocab);
    for (unsigned int k = 0; k < AOTX_LAYER_KIND_COUNT; ++k)
        if (kind_count[k] != 0) printf("layer_type=%s count=%u\n", kinds[k].name, kind_count[k]);
    printf("layer_sets_supported=%s unknown_tensors=%" PRIu64 " layer_limit=%u\n",
           layer_good ? "yes" : "no", unknown, AOTX_MODEL_MAX_LAYERS);
    printf("chat_template_bytes=%zu chat_template_sha256=%s\n", template_bytes, hex);
    printf("file_bytes=%" PRIu64 "\nheader_bytes=%" PRIu64 "\n",
           aotx_modelfile_file_bytes(file), aotx_modelfile_header_bytes(file));
    if (remote) printf("received_bytes=%" PRIu64 "\n", received);
    int good = arch_bytes != 0 && pre_good && model_good && blocks_good && shape_good && layer_good;
    printf("build_support=%s\nrun_verified=no\n", good ? "yes" : "no");
    printf("The support result covers the listed header fields and tensor sets only.\n");
    printf("The header does not prove weight integrity, memory fit, wrap, prefill, or restore.\n");
    if (!good) printf("This build cannot run this file with the listed unsupported or missing fields.\n");
    return 0;
}

int aotx_model_inspect(const char *source)
{
    aotx_modelfile *file = NULL;
    uint64_t received = 0;
    int remote = strncmp(source, "http://", 7u) == 0 || strncmp(source, "https://", 8u) == 0;
    int rc = remote ? aotx_model_inspect_remote(source, &file, &received)
                    : aotx_modelfile_open(source, &file);
    if (rc == 0) rc = report(source, file, remote, received);
    aotx_modelfile_close(file);
    return rc;
}
