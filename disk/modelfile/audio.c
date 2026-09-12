/* Purpose: Read and validate a complete sound encoder weight descriptor.
 * Owns: The temporary tensor range table and the result offsets.
 * Threading: One disk reader; no numerical sound processing.
 * Lifetime: One model file validation. */
#include "disk/modelfile/audio.h"
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
        {"clip.has_audio_encoder",1}, {"clip.audio.projection_dim",4096},
        {"clip.audio.embedding_length",1280}, {"clip.audio.feed_forward_length",5120},
        {"clip.audio.block_count",32}, {"clip.audio.attention.head_count",20},
        {"clip.audio.num_mel_bins",128}
    };
    float epsilon=0;
    if (!f || !text_is(f,"general.architecture","clip") ||
        !text_is(f,"general.type","mmproj") || !text_is(f,"clip.projector_type","qwen2a")) return 0;
    for (uint64_t a=0;a<f->meta_count;++a)
        for (uint64_t b=0;b<a;++b) if (!strcmp(f->meta[a].key,f->meta[b].key)) return 0;
    for (unsigned i=0;i<sizeof(values)/sizeof(values[0]);++i) {
        uint32_t value=0;
        if (aotx_modelfile_u32(f,values[i].key,&value) || value!=values[i].value) return 0;
    }
    return !aotx_modelfile_f32(f,"clip.audio.attention.layer_norm_epsilon",&epsilon) && epsilon==1.0e-5f;
}

struct tensor_shape { const char *name; unsigned type, dims; uint64_t size[4]; };
static const struct tensor_shape base[] = {
    {"a.conv1d.1.weight", 1, 3, {3,128,1280}},
    {"a.conv1d.1.bias", 0, 2, {1,1280}},
    {"a.conv1d.2.weight", 1, 3, {3,1280,1280}},
    {"a.conv1d.2.bias", 0, 2, {1,1280}},
    {"a.position_embd.weight", 0, 2, {1280,1500}},
    {"a.post_ln.weight", 0, 1, {1280}},
    {"a.post_ln.bias", 0, 1, {1280}},
    {"mm.a.fc.weight", 1, 2, {1280,4096}},
    {"mm.a.fc.bias", 0, 1, {4096}}
};
static const struct tensor_shape layer[] = {
    {"ln1.weight", 0, 1, {1280}},
    {"ln1.bias", 0, 1, {1280}},
    {"ln2.weight", 0, 1, {1280}},
    {"ln2.bias", 0, 1, {1280}},
    {"attn_q.weight", 1, 2, {1280,1280}},
    {"attn_q.bias", 0, 1, {1280}},
    {"attn_k.weight", 1, 2, {1280,1280}},
    {"attn_v.weight", 1, 2, {1280,1280}},
    {"attn_v.bias", 0, 1, {1280}},
    {"attn_out.weight", 1, 2, {1280,1280}},
    {"attn_out.bias", 0, 1, {1280}},
    {"ffn_up.weight", 1, 2, {1280,5120}},
    {"ffn_up.bias", 0, 1, {5120}},
    {"ffn_down.weight", 1, 2, {5120,1280}},
    {"ffn_down.bias", 0, 1, {1280}}
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

int aotx_audio_file(const aotx_modelfile *f, aotx_audio_desc *out)
{
    aotx_audio_desc result = {0};
    aotx_tensor_info ranges[489];
    unsigned used = 0;
    if (!out || !metadata(f) || aotx_modelfile_tensor_count(f) != 489u) return 2;
    result.bytes = aotx_modelfile_data_bytes(f);
    for (unsigned i = 0; i < AOTX_AUDIO_BASE_TENSORS; ++i) {
        if (!tensor(f, base[i].name, &base[i], ranges, used, &result.base[i])) return 2;
        ++used;
    }
    for (unsigned l = 0; l < AOTX_AUDIO_LAYERS; ++l) {
        for (unsigned i = 0; i < AOTX_AUDIO_LAYER_TENSORS; ++i) {
            char name[128];
            snprintf(name, sizeof(name), "a.blk.%u.%s", l, layer[i].name);
            if (!tensor(f, name, &layer[i], ranges, used, &result.layer[l][i])) return 2;
            ++used;
        }
    }
    *out = result;
    return 0;
}
