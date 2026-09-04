/* Purpose: Check the rule that states the memory a profile needs and the profile table.
 * Owns: The counts of the cases.
 * Launch shape: Host only; the check opens no context and needs no card.
 * Lifetime: One run of the test program. */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "disk/modelfile/modelfile.h"
#include "kvcache/kvcache.cuh"
#include "model/forward.cuh"
#include "model/roles.h"
#include "model/kinds.h"
#include "profile/fit.h"

static unsigned int aotx_profile_test_applied;
static unsigned int aotx_profile_test_failed;

static void aotx_profile_test_check(int ok, const char *what)
{
    aotx_profile_test_applied += 1u;
    if (!ok) {
        aotx_profile_test_failed += 1u;
        printf("profile: FAILED %s\n", what);
    }
}

/* The need of a row, from the figures of that row. */
static unsigned long long aotx_profile_test_need(const aotx_profile_row *row)
{
    return aotx_profile_need(row->weights_bytes, row->ring_slots, row->pages_each);
}

/* The row of the build states the same three figures as the header of the build. */
static void aotx_profile_test_case_row(void)
{
    const aotx_profile_row *row = aotx_profile_row_of(AOTX_PROFILE_NAME);
    aotx_profile_test_check(row != 0, "the table holds a row for the profile of the build");
    if (row == 0) {
        return;
    }
    aotx_profile_test_check(row->weights_bytes == (unsigned long long)AOTX_MEM_WEIGHTS_BYTES,
                            "the row states the weights region of the header");
    aotx_profile_test_check(row->ring_slots == (unsigned long long)AOTX_DEVICE_RING_SLOTS,
                            "the row states the ring slots of the header");
    aotx_profile_test_check(row->pages_each == (unsigned long long)AOTX_KV_PAGES_EACH,
                            "the row states the pages of a slot of the header");
    printf("profile: %s needs %llu MB, arch sm_%d, slots %u\n", AOTX_PROFILE_NAME,
           aotx_profile_test_need(row) >> 20, (int)AOTX_ARCH, (unsigned int)AOTX_SLOTS);
}

/* The four profiles need more memory in the order of the table, and each figure is the
 * figure the rule gives. */
static void aotx_profile_test_case_order(void)
{
    unsigned long long before = 0ull;
    for (unsigned int i = 0u; i < AOTX_PROFILE_ROWS; ++i) {
        unsigned long long need = aotx_profile_test_need(&aotx_profile_table[i]);
        printf("profile: %-4s needs %llu MB\n", aotx_profile_table[i].name, need >> 20);
        aotx_profile_test_check(need > before,
                                "each profile of the table needs more than the one before");
        before = need;
    }
}

/* The rule takes a card whose free memory is over the need and refuses one that is under
 * it by one byte. The check runs at every profile of the table. */
static void aotx_profile_test_case_edges(void)
{
    for (unsigned int i = 0u; i < AOTX_PROFILE_ROWS; ++i) {
        const aotx_profile_row *row = &aotx_profile_table[i];
        unsigned long long need = aotx_profile_test_need(row);
        unsigned long long total = need + (need >> 2);
        const char *fits = aotx_profile_that_fits(need, total);
        const char *under = aotx_profile_that_fits(need - 1ull, total);
        aotx_profile_test_check(fits != 0 && strcmp(fits, row->name) == 0,
                                "a card with the need free holds that profile");
        aotx_profile_test_check(under == 0 || strcmp(under, row->name) != 0,
                                "a card one byte under the need does not hold it");
    }
}

/* The rule of the build refuses a card that has less free memory than the need, and takes
 * one that has the need. A card whose whole memory is under the weights region never holds
 * the profile, whatever its free memory reports. */
static void aotx_profile_test_case_build(void)
{
    unsigned long long need = 0ull;
    unsigned long long large = 1024ull * 1024ull * 1024ull * 1024ull;
    int took = aotx_profile_fits(large, large, &need);
    aotx_profile_test_check(took == 1 && need > 0ull,
                            "a card with a terabyte free holds the profile of the build");
    aotx_profile_test_check(aotx_profile_fits(need, large, 0) == 1,
                            "a card with the need free holds the profile of the build");
    aotx_profile_test_check(aotx_profile_fits(need - 1ull, large, 0) == 0,
                            "a card one byte under the need does not hold it");
    aotx_profile_test_check(
        aotx_profile_fits(large, (unsigned long long)AOTX_MEM_WEIGHTS_BYTES - 1ull, 0) == 0,
        "a card whose memory is under the weights region does not hold the profile");
}

/* Every figure of the profile keeps the shape the kernels need. */
static void aotx_profile_test_case_shape(void)
{
    aotx_profile_test_check(AOTX_SLOTS >= 32u && AOTX_SLOTS <= 256u,
                            "the slot count is from 32 to 256");
    aotx_profile_test_check((AOTX_SLOTS & (AOTX_SLOTS - 1u)) == 0u,
                            "the slot count is a power of two");
    aotx_profile_test_check(
        (AOTX_DEVICE_RING_SLOTS & (AOTX_DEVICE_RING_SLOTS - 1ull)) == 0ull
        && (AOTX_INBOUND_SLOTS & (AOTX_INBOUND_SLOTS - 1ull)) == 0ull,
        "the two ring slot counts are powers of two");
    aotx_profile_test_check(
        (AOTX_HOST_RING_DATA_BYTES & (AOTX_HOST_RING_DATA_BYTES - 1ull)) == 0ull
        && (AOTX_BULK_RING_DATA_BYTES & (AOTX_BULK_RING_DATA_BYTES - 1ull)) == 0ull,
        "the two host ring sizes are powers of two");
    aotx_profile_test_check(AOTX_PROFILE_NAME[0] != '\0'
                            && AOTX_PROFILE_LANGUAGE[0] != '\0',
                            "the profile names itself and its language file");
    aotx_profile_test_check(AOTX_MODULE_SLOTS == AOTX_SLOTS,
                            "the module slots follow the slot count");
    aotx_profile_test_check(AOTX_SKILL_BYTES > 0u && AOTX_CATALOGUE_BYTES > 0ull
                            && AOTX_MODELS_RESIDENT > 0u,
                            "the figures of the later steps carry values");
    aotx_profile_test_check(aotx_role_of(AOTX_PROFILE_LANGUAGE)
                            == (unsigned int)AOTX_PROFILE_LANGUAGE_ROLE,
                            "the language role of the profile is the role of its name");
    printf("profile: the language file is %s, role %u\n", AOTX_PROFILE_LANGUAGE,
           (unsigned int)AOTX_PROFILE_LANGUAGE_ROLE);
}

/* Every state row gives its size rule, growth, paging, restore, and cache manager. */
static void aotx_profile_test_case_state_kinds(void)
{
    for (unsigned int i = 0u; i < AOTX_STATE_KIND_COUNT; ++i) {
        const aotx_state_kind *kind = &aotx_state_kind_table[i];
        aotx_profile_test_check(kind->name[0] != '\0'
                                && kind->bytes_rule < AOTX_STATE_BYTES_RULE_COUNT,
                                "a state kind names its byte rule");
        aotx_profile_test_check(kind->grows_context <= 1u && kind->paged <= 1u,
                                "a state kind states its growth and paging");
        aotx_profile_test_check(kind->restore < AOTX_STATE_RESTORE_COUNT,
                                "a state kind states how restore rebuilds it");
        aotx_profile_test_check(kind->manager < AOTX_STATE_MANAGER_COUNT,
                                "a state kind names its cache manager");
    }
    const aotx_state_kind *kv = aotx_state_kind_of(AOTX_STATE_KIND_KV_PAGES);
    const aotx_state_kind *delta = aotx_state_kind_of(AOTX_STATE_KIND_DELTA_STATE);
    aotx_profile_test_check(
        kv != NULL && strcmp(kv->name, "kv_pages") == 0
        && kv->bytes_rule == AOTX_STATE_BYTES_KV_CONTEXT && kv->grows_context == 1u
        && kv->paged == 1u && kv->restore == AOTX_STATE_RESTORE_REPLAY_PROMPT
        && kv->manager == AOTX_STATE_MANAGER_KV_PAGES,
        "the key value state grows in pages and restore replays the prompt");
    aotx_profile_test_check(
        delta != NULL && strcmp(delta->name, "delta_state") == 0
        && delta->bytes_rule == AOTX_STATE_BYTES_DELTA_HEAD_SQUARE
        && delta->grows_context == 0u && delta->paged == 0u
        && delta->restore == AOTX_STATE_RESTORE_REPLAY_PROMPT
        && delta->manager == AOTX_STATE_MANAGER_NONE,
        "the delta state is fixed and restore replays the prompt");
    char reason[192] = { '\0' };
    aotx_profile_test_check(aotx_kv_state_check(AOTX_STATE_KIND_KV_PAGES,
                                                reason, sizeof reason) == 0,
                            "the cache manager implements key value pages");
    aotx_profile_test_check(
        aotx_kv_state_check(AOTX_STATE_KIND_DELTA_STATE, reason, sizeof reason) != 0
        && strcmp(reason, "the cache manager does not implement state kind delta_state") == 0,
        "an unimplemented state kind is refused by name");
    const unsigned char state[] = {
        AOTX_STATE_KIND_KV_PAGES,
        AOTX_STATE_KIND_DELTA_STATE,
        AOTX_STATE_KIND_KV_PAGES
    };
    aotx_kvl_shape mixed;
    aotx_kvl_shape two;
    aotx_kvl_shape depth;
    aotx_kvl_make_states(&mixed, state, 3u, 8u, 128u);
    aotx_kvl_make(&two, 2u, 8u, 128u);
    aotx_kvl_make(&depth, 3u, 8u, 128u);
    aotx_profile_test_check(mixed.state_layers == 2u
                            && mixed.state_layer[0] == 0u
                            && mixed.state_layer[1] == 0xffu
                            && mixed.state_layer[2] == 1u,
                            "the page layout keeps only key value state layers");
    unsigned int block = (480u / AOTX_KVL_BLOCK) * mixed.state_layers
                       + mixed.state_layer[2];
    aotx_profile_test_check(mixed.block_bytes == 65536u
                            && mixed.blocks_page == 31u && block == 61u
                            && aotx_kvl_page_of(&mixed, 2u, 480u) == 1u,
                            "the mixed state layer has the compact page address");
    aotx_profile_test_check(aotx_kvl_pages(&mixed, 2048u)
                            == aotx_kvl_pages(&two, 2048u)
                            && aotx_kvl_pages(&mixed, 2048u)
                               < aotx_kvl_pages(&depth, 2048u),
                            "sequence pages follow state kinds instead of model depth");
}

typedef struct aotx_profile_fixture {
    unsigned char data[16384];
    size_t used;
    int bad;
} aotx_profile_fixture;

typedef struct aotx_profile_slot {
    const char *name;
    unsigned int slot;
} aotx_profile_slot;

static const aotx_profile_slot aotx_profile_layer_slot[AOTX_LAYER_TENSOR_SLOTS] = {
    { "attn_norm",   offsetof(aotx_model_layer, attn_norm) / sizeof(unsigned long long) },
    { "attn_q",      offsetof(aotx_model_layer, attn_q) / sizeof(unsigned long long) },
    { "attn_k",      offsetof(aotx_model_layer, attn_k) / sizeof(unsigned long long) },
    { "attn_v",      offsetof(aotx_model_layer, attn_v) / sizeof(unsigned long long) },
    { "attn_output", offsetof(aotx_model_layer, attn_o) / sizeof(unsigned long long) },
    { "attn_q_norm", offsetof(aotx_model_layer, attn_q_norm)
                       / sizeof(unsigned long long) },
    { "attn_k_norm", offsetof(aotx_model_layer, attn_k_norm)
                       / sizeof(unsigned long long) },
    { "ffn_norm",    offsetof(aotx_model_layer, ffn_norm) / sizeof(unsigned long long) },
    { "ffn_gate",    offsetof(aotx_model_layer, ffn_gate) / sizeof(unsigned long long) },
    { "ffn_up",      offsetof(aotx_model_layer, ffn_up) / sizeof(unsigned long long) },
    { "ffn_down",    offsetof(aotx_model_layer, ffn_down) / sizeof(unsigned long long) }
};

static void aotx_profile_fixture_raw(aotx_profile_fixture *file,
                                     const void *data, size_t bytes)
{
    if (file->used + bytes > sizeof file->data) {
        file->bad = 1;
        return;
    }
    memcpy(file->data + file->used, data, bytes);
    file->used += bytes;
}

static void aotx_profile_fixture_number(aotx_profile_fixture *file,
                                        uint64_t value, unsigned int bytes)
{
    unsigned char data[8];
    for (unsigned int i = 0u; i < bytes; ++i) {
        data[i] = (unsigned char)(value >> (8u * i));
    }
    aotx_profile_fixture_raw(file, data, bytes);
}

static void aotx_profile_fixture_text(aotx_profile_fixture *file, const char *text)
{
    size_t bytes = strlen(text);
    aotx_profile_fixture_number(file, bytes, 8u);
    aotx_profile_fixture_raw(file, text, bytes);
}

static void aotx_profile_fixture_u32(aotx_profile_fixture *file,
                                     const char *key, uint32_t value)
{
    aotx_profile_fixture_text(file, key);
    aotx_profile_fixture_number(file, AOTX_GGUF_U32, 4u);
    aotx_profile_fixture_number(file, value, 4u);
}

static void aotx_profile_fixture_f32(aotx_profile_fixture *file,
                                     const char *key, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    aotx_profile_fixture_text(file, key);
    aotx_profile_fixture_number(file, AOTX_GGUF_F32, 4u);
    aotx_profile_fixture_number(file, bits, 4u);
}

static unsigned int aotx_profile_fixture_names(
    char name[32][AOTX_DESC_BUFFER], unsigned int kind, int malformed)
{
    unsigned int count = 0u;
    if (malformed == 0) {
        snprintf(name[count++], AOTX_DESC_BUFFER, "token_embd.weight");
    }
    snprintf(name[count++], AOTX_DESC_BUFFER, "output_norm.weight");
    for (unsigned int layer = 0u; layer < 2u; ++layer) {
        unsigned int row_kind = (kind < AOTX_LAYER_KIND_COUNT) ? kind
                              : ((layer == 0u) ? AOTX_LAYER_KIND_ATTENTION
                                               : AOTX_LAYER_KIND_ATTENTION_NO_QK_NORM);
        const aotx_layer_kind *row = &aotx_layer_kind_table[row_kind];
        for (unsigned int i = 0u; i < row->tensors; ++i) {
            if (malformed != 0 && layer == 0u
                && strcmp(row->tensor[i].name, "attn_norm") == 0) {
                continue;
            }
            aotx_layer_name(name[count], AOTX_DESC_BUFFER, layer, &row->tensor[i]);
            count += 1u;
        }
    }
    return count;
}

static int aotx_profile_fixture_open(unsigned int kind, int malformed,
                                     aotx_modelfile **model)
{
    char tensor[32][AOTX_DESC_BUFFER];
    unsigned int tensors = aotx_profile_fixture_names(tensor, kind, malformed);
    aotx_profile_fixture file = {};
    aotx_profile_fixture_raw(&file, "GGUF", 4u);
    aotx_profile_fixture_number(&file, AOTX_GGUF_VERSION, 4u);
    aotx_profile_fixture_number(&file, tensors, 8u);
    aotx_profile_fixture_number(&file, 10u, 8u);
    aotx_profile_fixture_text(&file, "general.architecture");
    aotx_profile_fixture_number(&file, AOTX_GGUF_STRING, 4u);
    aotx_profile_fixture_text(&file, "qwen3");
    aotx_profile_fixture_u32(&file, "qwen3.block_count", 2u);
    aotx_profile_fixture_u32(&file, "qwen3.embedding_length", 32u);
    aotx_profile_fixture_u32(&file, "qwen3.context_length", 128u);
    aotx_profile_fixture_u32(&file, "qwen3.feed_forward_length", 64u);
    aotx_profile_fixture_u32(&file, "qwen3.attention.head_count", 1u);
    aotx_profile_fixture_u32(&file, "qwen3.attention.head_count_kv", 1u);
    aotx_profile_fixture_u32(&file, "qwen3.attention.key_length", 32u);
    aotx_profile_fixture_f32(&file, "qwen3.rope.freq_base", 1000000.0f);
    aotx_profile_fixture_f32(&file, "qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    for (unsigned int i = 0u; i < tensors; ++i) {
        aotx_profile_fixture_text(&file, tensor[i]);
        aotx_profile_fixture_number(&file, 1u, 4u);
        aotx_profile_fixture_number(&file, 32u, 8u);
        aotx_profile_fixture_number(&file, AOTX_TENSOR_F32, 4u);
        aotx_profile_fixture_number(&file, (uint64_t)i * 128u, 8u);
    }
    while ((file.used & 31u) != 0u) {
        aotx_profile_fixture_number(&file, 0u, 1u);
    }
    unsigned char zero[128] = {};
    for (unsigned int i = 0u; i < tensors; ++i) {
        aotx_profile_fixture_raw(&file, zero, sizeof zero);
    }
    char path[] = "/tmp/aotx-profile-model-XXXXXX";
    int fd = mkstemp(path);
    FILE *out = (fd >= 0) ? fdopen(fd, "wb") : NULL;
    int bad = file.bad != 0 || out == NULL;
    if (out != NULL) {
        bad |= fwrite(file.data, 1u, file.used, out) != file.used;
        bad |= fclose(out) != 0;
    } else if (fd >= 0) {
        close(fd);
    }
    if (bad == 0) {
        bad = aotx_modelfile_open(path, model) != 0;
    }
    unlink(path);
    return bad;
}

static void aotx_profile_test_rows(void)
{
    char name[AOTX_DESC_BUFFER];
    for (unsigned int k = 0u; k < AOTX_LAYER_KIND_COUNT; ++k) {
        const aotx_layer_kind *kind = &aotx_layer_kind_table[k];
        aotx_profile_test_check(kind->name[0] != '\0' && kind->tensors != 0u
                                && kind->keys != 0u,
                                "a layer kind row holds its required fields");
        aotx_profile_test_check(kind->state == AOTX_STATE_KIND_NONE
                                || aotx_state_kind_of(kind->state) != NULL,
                                "a present layer state is in the state table");
        for (unsigned int i = 0u; i < kind->tensors; ++i) {
            const aotx_layer_tensor *tensor = &kind->tensor[i];
            aotx_profile_test_check(
                aotx_layer_name(name, sizeof name, AOTX_MODEL_MAX_LAYERS - 1u,
                                tensor) == 0,
                "the name builder builds a tensor name from the kind table");
            aotx_profile_test_check(tensor->slot < AOTX_LAYER_TENSOR_SLOTS,
                                    "a layer tensor slot is in the descriptor row");
            if (tensor->slot < AOTX_LAYER_TENSOR_SLOTS) {
                aotx_profile_test_check(
                    strcmp(tensor->name, aotx_profile_layer_slot[tensor->slot].name) == 0
                    && tensor->slot == aotx_profile_layer_slot[tensor->slot].slot,
                    "a layer tensor slot names its descriptor member");
            }
            for (unsigned int j = 0u; j < i; ++j) {
                aotx_profile_test_check(tensor->slot != kind->tensor[j].slot,
                                        "a layer tensor slot is unique in its row");
            }
        }
        for (unsigned int i = 0u; i < kind->keys; ++i) {
            aotx_profile_test_check(kind->key[i].name[0] != '\0'
                                    && kind->key[i].member < sizeof(aotx_model_desc),
                                    "a layer metadata key names a descriptor member");
        }
    }
}

static int aotx_profile_binding_has(const aotx_model_binding *binding, unsigned int count,
                                    const char *name, unsigned int slot)
{
    for (unsigned int i = 0u; i < count; ++i) {
        if (strcmp(binding[i].name, name) == 0) {
            return binding[i].slot == slot;
        }
    }
    return 0;
}

static void aotx_profile_test_plan(unsigned int kind, int malformed)
{
    aotx_modelfile *file = NULL;
    aotx_profile_test_check(aotx_profile_fixture_open(kind, malformed, &file) == 0,
                            "the described model fixture opens");
    if (file == NULL) {
        return;
    }
    aotx_model_binding binding[AOTX_DESC_WHOLE
                               + AOTX_MODEL_MAX_LAYERS * AOTX_LAYER_TENSOR_SLOTS] = {};
    aotx_model_desc desc;
    char reason[192];
    unsigned int count = 0u;
    int bad = aotx_model_desc_file(file, AOTX_MODEL_LANGUAGE, &desc, binding,
                                   sizeof binding / sizeof binding[0], &count,
                                   reason, sizeof reason);
    if (malformed != 0) {
        aotx_profile_test_check(
            bad != 0 && strcmp(reason, "the model of role 2 has no tensor "
                               "token_embd.weight, and 1 more are absent") == 0,
            "a malformed file names the first whole tensor and missing count");
        aotx_modelfile_close(file);
        return;
    }
    aotx_kvl_shape from_kinds;
    aotx_kvl_shape from_depth;
    aotx_kvl_make_desc(&from_kinds, &desc);
    aotx_kvl_make(&from_depth, desc.layers, desc.kv_heads, desc.head_dim);
    aotx_profile_test_check(
        from_kinds.state_layers
            == aotx_layer_state_count(&desc, AOTX_STATE_KIND_KV_PAGES),
        "the cache shape counts layers that hold key value pages");
    aotx_profile_test_check(aotx_kvl_pages(&from_kinds, desc.context)
                            == aotx_kvl_pages(&from_depth, desc.context),
                            "the present family keeps its page count");
    for (unsigned int layer = 0u; layer < desc.layers; ++layer) {
        aotx_profile_test_check(from_kinds.state_layer[layer] == layer,
                                "the present family keeps its cache layer position");
    }
    for (unsigned int layer = 0u; layer < desc.layers; ++layer) {
        unsigned int expected = (kind < AOTX_LAYER_KIND_COUNT) ? kind
                              : ((layer == 0u) ? AOTX_LAYER_KIND_ATTENTION
                                               : AOTX_LAYER_KIND_ATTENTION_NO_QK_NORM);
        aotx_profile_test_check(desc.kind[layer] == expected,
                                "the loader selects the file's kind for every layer");
    }
    for (unsigned int layer = 0u; layer < desc.layers; ++layer) {
        unsigned int row_kind = (kind < AOTX_LAYER_KIND_COUNT) ? kind
                              : ((layer == 0u) ? AOTX_LAYER_KIND_ATTENTION
                                               : AOTX_LAYER_KIND_ATTENTION_NO_QK_NORM);
        const aotx_layer_kind *row = &aotx_layer_kind_table[row_kind];
        for (unsigned int i = 0u; i < row->tensors; ++i) {
            char name[AOTX_DESC_BUFFER];
            aotx_layer_name(name, sizeof name, layer, &row->tensor[i]);
            unsigned int slot = AOTX_DESC_WHOLE
                              + layer * AOTX_LAYER_TENSOR_SLOTS + row->tensor[i].slot;
            aotx_profile_test_check(aotx_profile_binding_has(binding, count, name, slot) != 0,
                                    "the loader maps a row name to its descriptor slot");
        }
    }
    if (bad != 0) {
        printf("profile: descriptor of kind %u: %s\n", kind, reason);
    }
    aotx_profile_test_check(bad == 0 && reason[0] == '\0',
                            "every described kind has a capture and is runnable");
    aotx_modelfile_close(file);
}

/* The checks use the production file planner, its selected rows, and its binding plan. */
static void aotx_profile_test_case_layer_kinds(void)
{
    aotx_profile_test_rows();
    aotx_profile_test_plan(AOTX_LAYER_KIND_ATTENTION, 0);
    aotx_profile_test_plan(AOTX_LAYER_KIND_ATTENTION_NO_QK_NORM, 0);
    aotx_profile_test_plan(AOTX_LAYER_KIND_COUNT, 0);
    aotx_profile_test_plan(AOTX_LAYER_KIND_ATTENTION, 1);
}

int main(void)
{
    aotx_profile_test_case_row();
    aotx_profile_test_case_order();
    aotx_profile_test_case_edges();
    aotx_profile_test_case_build();
    aotx_profile_test_case_shape();
    aotx_profile_test_case_state_kinds();
    aotx_profile_test_case_layer_kinds();
    printf("profile: %u cases applied, %u passed, %u failed\n", aotx_profile_test_applied,
           aotx_profile_test_applied - aotx_profile_test_failed, aotx_profile_test_failed);
    if (aotx_profile_test_applied == 0u) {
        printf("profile: no case ran\n");
        return 1;
    }
    return (aotx_profile_test_failed == 0u) ? 0 : 1;
}
