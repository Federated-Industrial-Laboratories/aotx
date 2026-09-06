/* Purpose: Check hybrid GGUF bindings, shape refusals, and fixed allocation bytes.
 * Owns: Independent sparse GGUF fixtures for 1 and 64 layers.
 * Launch shape: Host-only descriptor calls; no CUDA context or device allocation.
 * Lifetime: One test process; each sparse file closes after its case. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <string>
#include <vector>

#include "model/forward.cuh"
#include "model/hybrid.cuh"
#include "model/kinds.h"
#include "disk/modelfile/modelfile.h"

struct aotx_hybrid_bytes {
    std::vector<unsigned char> data;
    void integer(uint64_t value, unsigned int bytes)
    {
        for (unsigned int i = 0u; i < bytes; ++i) data.push_back((unsigned char)(value >> (8u * i)));
    }
    void string(const std::string &value)
    {
        integer(value.size(), 8u);
        data.insert(data.end(), value.begin(), value.end());
    }
    void append(const aotx_hybrid_bytes &other)
    {
        data.insert(data.end(), other.data.begin(), other.data.end());
    }
};

struct aotx_hybrid_meta {
    std::string key;
    unsigned int type;
    aotx_hybrid_bytes value;
};

struct aotx_hybrid_tensor_spec {
    const char *suffix;
    unsigned int slot;
    uint64_t first, second;
};

/* Literal file contracts, independent of the runtime kind and tensor tables. */
static const aotx_hybrid_tensor_spec aotx_hybrid_linear[] = {
    {"attn_norm.weight", 0u, 32u, 0u},
    {"attn_qkv.weight", 1u, 32u, 256u},
    {"attn_gate.weight", 2u, 32u, 128u},
    {"ssm_conv1d.weight", 3u, 4u, 256u},
    {"ssm_a", 4u, 4u, 0u},
    {"ssm_dt.bias", 5u, 4u, 0u},
    {"ssm_alpha.weight", 6u, 32u, 4u},
    {"post_attention_norm.weight", 7u, 32u, 0u},
    {"ffn_gate.weight", 8u, 32u, 64u},
    {"ffn_up.weight", 9u, 32u, 64u},
    {"ffn_down.weight", 10u, 64u, 32u},
    {"ssm_beta.weight", 11u, 32u, 4u},
    {"ssm_norm.weight", 12u, 32u, 0u},
    {"ssm_out.weight", 13u, 128u, 32u}
};
static const aotx_hybrid_tensor_spec aotx_hybrid_gated[] = {
    {"attn_norm.weight", 0u, 32u, 0u},
    {"attn_q.weight", 1u, 32u, 1024u},
    {"attn_k.weight", 2u, 32u, 256u},
    {"attn_v.weight", 3u, 32u, 256u},
    {"attn_output.weight", 4u, 512u, 32u},
    {"attn_q_norm.weight", 5u, 256u, 0u},
    {"attn_k_norm.weight", 6u, 256u, 0u},
    {"post_attention_norm.weight", 7u, 32u, 0u},
    {"ffn_gate.weight", 8u, 32u, 64u},
    {"ffn_up.weight", 9u, 32u, 64u},
    {"ffn_down.weight", 10u, 64u, 32u}
};

struct aotx_hybrid_tensor {
    std::string name;
    uint64_t first, second;
    unsigned int type;
};

struct aotx_hybrid_fixture {
    std::vector<aotx_hybrid_meta> metadata;
    std::vector<aotx_hybrid_tensor> tensors;
    std::vector<unsigned char> gated;

    void number(const char *key, uint64_t value, unsigned int type = 4u)
    {
        aotx_hybrid_meta entry = {key, type, {}};
        entry.value.integer(value, type == 10u || type == 11u ? 8u : 4u);
        replace(entry);
    }
    void real(const char *key, float value)
    {
        uint32_t bits;
        memcpy(&bits, &value, sizeof bits);
        number(key, bits, 6u);
    }
    void text(const char *key, const char *value)
    {
        aotx_hybrid_meta entry = {key, 8u, {}};
        entry.value.string(value);
        replace(entry);
    }
    void sections(const std::vector<int32_t> &values)
    {
        aotx_hybrid_meta entry = {"qwen35.rope.dimension_sections", 9u, {}};
        entry.value.integer(5u, 4u);
        entry.value.integer(values.size(), 8u);
        for (int32_t value : values) entry.value.integer((uint32_t)value, 4u);
        replace(entry);
    }
    void replace(const aotx_hybrid_meta &entry)
    {
        for (aotx_hybrid_meta &current : metadata) {
            if (current.key == entry.key) {
                current = entry;
                return;
            }
        }
        metadata.push_back(entry);
    }
    aotx_hybrid_tensor &tensor(const char *name)
    {
        for (aotx_hybrid_tensor &entry : tensors) if (entry.name == name) return entry;
        fprintf(stderr, "hybrid file: fixture has no tensor %s\n", name);
        exit(2);
    }
    void remove(const char *name)
    {
        for (auto i = tensors.begin(); i != tensors.end(); ++i) {
            if (i->name == name) {
                tensors.erase(i);
                return;
            }
        }
        fprintf(stderr, "hybrid file: fixture has no tensor %s\n", name);
        exit(2);
    }
    aotx_hybrid_fixture(unsigned int layers, unsigned int pattern)
    {
        text("general.architecture", "qwen35");
        number("qwen35.block_count", layers);
        number("qwen35.embedding_length", 32u);
        number("qwen35.context_length", 4096u);
        number("qwen35.feed_forward_length", 64u);
        number("qwen35.attention.head_count", 2u);
        number("qwen35.attention.head_count_kv", 1u);
        number("qwen35.attention.key_length", 256u);
        number("qwen35.attention.value_length", 256u);
        number("qwen35.ssm.state_size", 32u);
        number("qwen35.ssm.time_step_rank", 4u);
        number("qwen35.ssm.group_count", 2u);
        number("qwen35.ssm.conv_kernel", 4u);
        number("qwen35.ssm.inner_size", 128u);
        number("qwen35.rope.dimension_count", 64u);
        real("qwen35.rope.freq_base", 1000000.0f);
        real("qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
        sections({11, 11, 10, 0});
        tensors.push_back({"token_embd.weight", 32u, 16u, 0u});
        tensors.push_back({"output_norm.weight", 32u, 0u, 0u});
        tensors.push_back({"output.weight", 32u, 16u, 0u});
        for (unsigned int layer = 0u; layer < layers; ++layer) {
            bool attention = layers == 1u ? pattern != 0u :
                (pattern == 0u ? layer % 4u == 3u : (layer * 13u + 7u) % layers < layers / 4u);
            gated.push_back(attention);
            const aotx_hybrid_tensor_spec *spec = attention ? aotx_hybrid_gated : aotx_hybrid_linear;
            unsigned int count = attention ? 11u : 14u;
            for (unsigned int i = 0u; i < count; ++i) {
                char name[128];
                snprintf(name, sizeof name, "blk.%u.%s", layer, spec[i].suffix);
                tensors.push_back({name, spec[i].first, spec[i].second, 0u});
            }
        }
        /* Tensor-table order does not define either layer kind or binding order. */
        std::reverse(tensors.begin(), tensors.end());
    }
};

static unsigned int aotx_hybrid_checks;
static unsigned int aotx_hybrid_failed;

static void aotx_hybrid_check(bool ok, const char *case_name, const char *contract)
{
    ++aotx_hybrid_checks;
    if (!ok) {
        ++aotx_hybrid_failed;
        printf("hybrid file: FAIL %s: %s\n", case_name, contract);
    }
}

static aotx_modelfile *aotx_hybrid_open(const aotx_hybrid_fixture &fixture)
{
    aotx_hybrid_bytes bytes;
    bytes.integer(0x46554747u, 4u);
    bytes.integer(3u, 4u);
    bytes.integer(fixture.tensors.size(), 8u);
    bytes.integer(fixture.metadata.size(), 8u);
    for (const aotx_hybrid_meta &entry : fixture.metadata) {
        bytes.string(entry.key);
        bytes.integer(entry.type, 4u);
        bytes.append(entry.value);
    }
    uint64_t size = 0u;
    for (const aotx_hybrid_tensor &tensor : fixture.tensors) {
        size = (size + 31u) & ~31ull;
        bytes.string(tensor.name);
        bytes.integer(tensor.second ? 2u : 1u, 4u);
        bytes.integer(tensor.first, 8u);
        if (tensor.second) bytes.integer(tensor.second, 8u);
        bytes.integer(tensor.type, 4u);
        bytes.integer(size, 8u);
        size += tensor.first * (tensor.second ? tensor.second : 1u) * (tensor.type == 1u ? 2u : 4u);
    }
    while (bytes.data.size() % 32u) bytes.data.push_back(0u);
    char path[] = "/tmp/aotx-hybrid-file-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { perror("hybrid file create"); exit(2); }
    FILE *output = fdopen(fd, "wb");
    if (!output) { perror("hybrid file stream"); close(fd); unlink(path); exit(2); }
    if (fwrite(bytes.data.data(), 1u, bytes.data.size(), output) != bytes.data.size()
        || fflush(output) != 0 || ftruncate(fd, (off_t)(bytes.data.size() + size)) != 0) {
        perror("hybrid file write"); fclose(output); unlink(path); exit(2);
    }
    fclose(output);
    aotx_modelfile *file = NULL;
    int status = aotx_modelfile_open(path, &file);
    unlink(path);
    if (status != 0) {
        fprintf(stderr, "hybrid file: sparse fixture reader failed, status=%d\n", status);
        exit(2);
    }
    return file;
}

static unsigned long long aotx_hybrid_expected_bytes(unsigned int recurrent, unsigned int gated,
                                                      unsigned int tokens)
{
    unsigned long long result = 0u;
    std::vector<unsigned long long> spans;
    if (recurrent) {
        spans = {
            (unsigned long long)AOTX_SLOTS * recurrent * 4u * 32u * 32u * 4u,
            (unsigned long long)AOTX_SLOTS * recurrent * 256u * 3u * 4u,
            (unsigned long long)tokens * 256u * 4u,
            (unsigned long long)tokens * 128u * 4u,
            (unsigned long long)tokens * 4u * 4u,
            (unsigned long long)tokens * 4u * 4u,
            (unsigned long long)tokens * 256u * 4u,
            (unsigned long long)tokens * 128u * 4u,
            (unsigned long long)tokens * 128u * 2u
        };
    }
    if (gated) spans.push_back((unsigned long long)tokens * 1024u * 4u);
    for (unsigned long long span : spans) result = ((result + 255u) / 256u) * 256u + span;
    return result;
}

static void aotx_hybrid_case(const aotx_hybrid_fixture &fixture, const char *name,
                               bool valid, const char *reason_part = "")
{
    aotx_modelfile *file = aotx_hybrid_open(fixture);
    aotx_model_desc desc = {};
    std::vector<aotx_model_binding> binding(AOTX_DESC_WHOLE + AOTX_MODEL_MAX_LAYERS * 14u);
    unsigned int count = 0u;
    char reason[512] = {};
    int status = aotx_model_desc_file(file, AOTX_MODEL_LANGUAGE, &desc, binding.data(),
                                      binding.size(), &count, reason, sizeof reason);
    aotx_hybrid_check((status == 0) == valid, name, valid ? "accept valid file" : "refuse invalid file");
    if (status != 0) {
        if (valid) printf("hybrid file: reason: %s\n", reason);
        aotx_hybrid_check(reason[0] != '\0' && strstr(reason, reason_part) != NULL, name, "refusal diagnostic");
    }
    if (status == 0 && valid) {
        unsigned int recurrent = 0u, gated = 0u;
        aotx_kvl_shape shape;
        aotx_kvl_make_desc(&shape, &desc);
        bool kinds = desc.layers == fixture.gated.size(), slots = true, compact = true;
        for (unsigned int layer = 0u; layer < fixture.gated.size(); ++layer) {
            bool attention = fixture.gated[layer] != 0u;
            kinds &= desc.kind[layer] == (attention ? 5u : 4u);
            compact &= shape.state_layer[layer] == (attention ? gated : 0xffu);
            if (attention) ++gated; else ++recurrent;
            const aotx_hybrid_tensor_spec *spec = attention ? aotx_hybrid_gated : aotx_hybrid_linear;
            for (unsigned int i = 0u; i < (attention ? 11u : 14u); ++i) {
                char tensor_name[128];
                snprintf(tensor_name, sizeof tensor_name, "blk.%u.%s", layer, spec[i].suffix);
                unsigned int matches = 0u;
                for (unsigned int b = 0u; b < count; ++b) {
                    if (strcmp(binding[b].name, tensor_name) == 0) {
                        ++matches;
                        slots &= binding[b].needed == 1u && binding[b].slot ==
                            AOTX_DESC_WHOLE + layer * AOTX_LAYER_TENSOR_SLOTS + spec[i].slot;
                    }
                }
                slots &= matches == 1u;
            }
        }
        slots &= count == AOTX_DESC_WHOLE + 14u * recurrent + 11u * gated;
        compact &= shape.state_layers == gated;
        for (unsigned int layer = desc.layers; layer < AOTX_MODEL_MAX_LAYERS; ++layer)
            compact &= shape.state_layer[layer] == 0xffu;
        aotx_hybrid_check(kinds, name, "tensor-selected layer kinds");
        aotx_hybrid_check(slots, name, "exact required tensor binding slots");
        aotx_hybrid_check(compact, name, "compact key/value layer map");
        aotx_hybrid_check(desc.hidden == 32u && desc.ffn == 64u && desc.context == 4096u
            && desc.heads == 2u && desc.kv_heads == 1u && desc.head_dim == 256u
            && desc.delta_dim == 32u && desc.delta_heads == 4u && desc.delta_key_heads == 2u
            && desc.delta_conv == 4u && desc.delta_inner == 128u && desc.rope_dim == 64u
            && desc.rope_theta == 1000000.0f && desc.rms_eps == 1e-6f
            && desc.rope_pairs == AOTX_ROPE_PAIRS_SPLIT, name, "exact descriptor metadata");
        aotx_hybrid_check(aotx_layer_state_count(&desc, AOTX_STATE_KIND_DELTA_STATE) == recurrent
            && aotx_layer_state_count(&desc, AOTX_STATE_KIND_KV_PAGES) == gated, name, "state family counts");
        for (unsigned int tokens : {1u, 64u}) {
            unsigned long long expected = aotx_hybrid_expected_bytes(recurrent, gated, tokens);
            unsigned long long got = aotx_model_hybrid_bytes(&desc, tokens);
            aotx_hybrid_check(got == expected, name, "complete fixed state and scratch bytes");
            aotx_model_desc longer = desc;
            longer.context = 1048576u;
            aotx_hybrid_check(aotx_model_hybrid_bytes(&longer, tokens) == got, name, "context-independent allocation");
        }
    }
    aotx_modelfile_close(file);
}

int main(void)
{
    unsigned int pairs = 91u;
    aotx_hybrid_check(aotx_rope_family_pairs("qwen35", 6u, &pairs) == 1
        && pairs == AOTX_ROPE_PAIRS_SPLIT, "qwen35", "rotary family lookup");
    pairs = 91u;
    aotx_hybrid_check(aotx_rope_family_pairs("qwen35x", 7u, &pairs) == 0 && pairs == 91u,
        "qwen35x", "exact architecture name");
    for (unsigned int layers : {1u, 64u}) {
        for (unsigned int pattern : {0u, 1u}) {
            char name[64];
            snprintf(name, sizeof name, "layers=%u pattern=%u", layers, pattern);
            aotx_hybrid_case(aotx_hybrid_fixture(layers, pattern), name, true);
        }
    }
    const aotx_hybrid_fixture base(4u, 0u);
    {
        aotx_hybrid_fixture f = base;
        f.remove("blk.0.attn_qkv.weight");
        aotx_hybrid_case(f, "missing qkv", false, "attn_qkv.weight");
    }
    {
        aotx_hybrid_fixture f = base;
        f.tensor("blk.3.attn_q.weight").second = 512u;
        aotx_hybrid_case(f, "joint query/gate width", false, "attn_q.weight");
    }
    for (const char *name : {"blk.0.ssm_conv1d.weight", "blk.0.attn_qkv.weight"}) {
        aotx_hybrid_fixture f = base;
        ++f.tensor(name).second;
        aotx_hybrid_case(f, name, false, name);
    }
    {
        aotx_hybrid_fixture f = base;
        f.tensor("blk.0.ssm_conv1d.weight").first = 3u;
        aotx_hybrid_case(f, "history tap width", false, "ssm_conv1d.weight");
    }
    for (const char *name : {"blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_norm.weight",
                              "blk.0.ssm_conv1d.weight", "blk.3.attn_q_norm.weight", "output_norm.weight"}) {
        aotx_hybrid_fixture f = base;
        f.tensor(name).type = 1u;
        aotx_hybrid_case(f, name, false, name);
    }
    const char *zero_keys[] = {
        "qwen35.attention.head_count", "qwen35.attention.head_count_kv",
        "qwen35.attention.key_length", "qwen35.ssm.state_size",
        "qwen35.ssm.time_step_rank", "qwen35.ssm.group_count", "qwen35.rope.dimension_count"
    };
    for (const char *key : zero_keys) {
        aotx_hybrid_fixture f = base;
        f.number(key, 0u);
        aotx_hybrid_case(f, key, false);
    }
    for (unsigned int width : {63u, 258u}) {
        aotx_hybrid_fixture f = base;
        f.number("qwen35.rope.dimension_count", width);
        aotx_hybrid_case(f, "invalid rotary width", false);
    }
    for (const std::vector<int32_t> &sections : {std::vector<int32_t>{11,11,9,0},
             std::vector<int32_t>{11,11,10}, std::vector<int32_t>{-1,11,22,0},
             std::vector<int32_t>{10,10,10,2}, std::vector<int32_t>{0,16,16,0}}) {
        aotx_hybrid_fixture f = base;
        f.sections(sections);
        aotx_hybrid_case(f, "invalid rotary sections", false);
    }
    {
        aotx_hybrid_fixture f = base;
        f.number("qwen35.attention.value_length", 128u);
        aotx_hybrid_case(f, "unequal value width", false);
    }
    for (unsigned int type : {5u, 10u, 8u, 6u}) {
        aotx_hybrid_fixture f = base;
        const char *key = "qwen35.ssm.state_size";
        if (type == 8u) f.text(key, "32");
        else if (type == 6u) f.real(key, 32.0f);
        else f.number(key, type == 10u ? 0x100000020ull : 0xffffffffu, type);
        aotx_hybrid_case(f, "malformed unsigned metadata", false, key);
    }
    for (const char *key : {"qwen35.ssm.conv_kernel", "qwen35.ssm.inner_size",
                             "qwen35.ssm.group_count"}) {
        aotx_hybrid_fixture f = base;
        f.number(key, 17u);
        aotx_hybrid_case(f, key, false);
    }
    printf("hybrid file: %u checks, %u failed\n", aotx_hybrid_checks, aotx_hybrid_failed);
    return aotx_hybrid_failed ? 1 : 0;
}
