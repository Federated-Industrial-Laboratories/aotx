/* Purpose: Read and validate a complete image encoder weight descriptor.
 * Owns: The temporary tensor range table and the result offsets.
 * Threading: One disk reader; no numerical image processing.
 * Lifetime: One model file validation. */
#include "disk/modelfile/vision.h"
#include "disk/modelfile/gguf.h"
#include <stdio.h>
#include <string.h>

static int text_is(const aotx_modelfile *f, const char *key, const char *expected)
{
    const char *text = NULL;
    size_t bytes = 0;
    return aotx_modelfile_string(f, key, &text, &bytes) == 0 &&
        bytes == strlen(expected) && memcmp(text, expected, bytes) == 0;
}

static int metadata(const aotx_modelfile *f)
{
    static const struct { const char *key; uint32_t value; } values[] = {
        {"clip.has_vision_encoder", 1}, {"clip.use_gelu", 1},
        {"clip.vision.projection_dim", 1024}, {"clip.vision.image_size", 768},
        {"clip.vision.patch_size", 16}, {"clip.vision.embedding_length", 768},
        {"clip.vision.feed_forward_length", 3072}, {"clip.vision.block_count", 12},
        {"clip.vision.attention.head_count", 12}, {"clip.vision.spatial_merge_size", 2}
    };
    const uint8_t *deep = NULL;
    const float *mean = NULL, *std = NULL;
    uint64_t n = 0, m = 0, s = 0;
    float epsilon = 0;
    unsigned i;
    if (!f || !text_is(f, "general.architecture", "clip") ||
        !text_is(f, "general.type", "mmproj") ||
        !text_is(f, "clip.projector_type", "qwen3vl_merger")) return 0;
    for (uint64_t a = 0; a < f->meta_count; ++a)
        for (uint64_t b = 0; b < a; ++b)
            if (strcmp(f->meta[a].key, f->meta[b].key) == 0) return 0;
    for (i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        uint32_t value = 0;
        if (aotx_modelfile_u32(f, values[i].key, &value) || value != values[i].value) return 0;
    }
    if (aotx_modelfile_bools(f, "clip.vision.is_deepstack_layers", &deep, &n) || n != 12u ||
        aotx_modelfile_f32s(f, "clip.vision.image_mean", &mean, &m) || m != 3u ||
        aotx_modelfile_f32s(f, "clip.vision.image_std", &std, &s) || s != 3u ||
        aotx_modelfile_f32(f, "clip.vision.attention.layer_norm_epsilon", &epsilon) ||
        epsilon != 1.0e-6f) return 0;
    for (i = 0; i < n; ++i) if (deep[i]) return 0;
    for (i = 0; i < 3u; ++i) if (mean[i] != 0.5f || std[i] != 0.5f) return 0;
    return 1;
}

struct tensor_shape { const char *name; unsigned type, dims; uint64_t size[4]; };
static const struct tensor_shape base[] = {
    {"v.patch_embd.weight", 1, 4, {16,16,3,768}},
    {"v.patch_embd.weight.1", 1, 4, {16,16,3,768}},
    {"v.patch_embd.bias", 0, 1, {768}},
    {"v.position_embd.weight", 0, 2, {768,2304}},
    {"v.post_ln.weight", 0, 1, {768}}, {"v.post_ln.bias", 0, 1, {768}},
    {"mm.0.weight", 1, 2, {3072,3072}}, {"mm.0.bias", 0, 1, {3072}},
    {"mm.2.weight", 1, 2, {3072,1024}}, {"mm.2.bias", 0, 1, {1024}}
};
static const struct tensor_shape layer[] = {
    {"ln1.weight", 0, 1, {768}}, {"ln1.bias", 0, 1, {768}},
    {"ln2.weight", 0, 1, {768}}, {"ln2.bias", 0, 1, {768}},
    {"attn_qkv.weight", 1, 2, {768,2304}}, {"attn_qkv.bias", 0, 1, {2304}},
    {"attn_out.weight", 1, 2, {768,768}}, {"attn_out.bias", 0, 1, {768}},
    {"ffn_up.weight", 1, 2, {768,3072}}, {"ffn_up.bias", 0, 1, {3072}},
    {"ffn_down.weight", 1, 2, {3072,768}}, {"ffn_down.bias", 0, 1, {768}}
};
static int tensor(const aotx_modelfile *f, const char *name, const struct tensor_shape *shape,
                  aotx_tensor_info *ranges, unsigned used, uint64_t *offset)
{
    aotx_tensor_info t;
    if (aotx_modelfile_find(f, name, &t) || t.type != shape->type ||
        t.dim_count != shape->dims || t.offset % 32u) return 0;
    for (unsigned d = 0; d < t.dim_count; ++d) if (t.dims[d] != shape->size[d]) return 0;
    for (unsigned i = 0; i < used; ++i)
        if (t.offset < ranges[i].offset + ranges[i].bytes &&
            ranges[i].offset < t.offset + t.bytes) return 0;
    ranges[used] = t; *offset = t.offset;
    return 1;
}

int aotx_vision_file(const aotx_modelfile *f, aotx_vision_desc *out)
{
    aotx_vision_desc result = {0};
    aotx_tensor_info ranges[154];
    unsigned used = 0;
    if (!out || !metadata(f) || aotx_modelfile_tensor_count(f) != 154u) return 2;
    result.bytes = aotx_modelfile_data_bytes(f);
    for (unsigned i = 0; i < AOTX_VISION_BASE_TENSORS; ++i) {
        if (!tensor(f, base[i].name, &base[i], ranges, used, &result.base[i])) return 2;
        ++used;
    }
    for (unsigned l = 0; l < AOTX_VISION_LAYERS; ++l) {
        for (unsigned i = 0; i < AOTX_VISION_LAYER_TENSORS; ++i) {
            char name[128];
            snprintf(name, sizeof(name), "v.blk.%u.%s", l, layer[i].name);
            if (!tensor(f, name, &layer[i], ranges, used, &result.layer[l][i])) return 2;
            ++used;
        }
    }
    *out = result;
    return 0;
}
