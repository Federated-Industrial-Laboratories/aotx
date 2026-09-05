/* Purpose: Check the shapes and scalar types required by hybrid layers.
 * Owns: Nothing; the file reader owns the metadata and tensor table.
 * Launch shape: Host glue; one check at model description.
 * Lifetime: One model load. */
#include <limits.h>
#include <math.h>
#include "model/hybrid.cuh"

extern "C" {
#include "disk/modelfile/modelfile.h"
}

static int aotx_hybrid_tensor(const aotx_modelfile *file, const char *name,
                               uint64_t first, uint64_t second, unsigned int scalar,
                               char *reason, size_t size)
{
    aotx_tensor_info tensor;
    unsigned int dims = second ? 2u : 1u;
    if (aotx_modelfile_find(file, name, &tensor) != 0 || tensor.dim_count != dims
        || tensor.dims[0] != first || (second && tensor.dims[1] != second)) {
        snprintf(reason, size, "the hybrid tensor %s has an incompatible shape", name);
        return 1;
    }
    if ((scalar && tensor.type != AOTX_TENSOR_F32) || !aotx_matrix_known(tensor.type)) {
        snprintf(reason, size, "the hybrid tensor %s has an incompatible type", name);
        return 1;
    }
    return 0;
}

static int aotx_hybrid_shape(const aotx_modelfile *file, const aotx_model_desc *desc,
                              char *reason, size_t size)
{
    const char *arch = NULL;
    size_t length = 0u;
    if (aotx_modelfile_string(file, "general.architecture", &arch, &length) != 0
        || length != 6u || memcmp(arch, "qwen35", 6u) != 0) {
        snprintf(reason, size, "the file does not name a supported hybrid architecture");
        return 1;
    }
    uint64_t dim = desc->delta_dim, heads = desc->delta_heads;
    uint64_t channels = 2ull * desc->delta_key_heads * dim + desc->delta_inner;
    uint64_t wide = (uint64_t)desc->heads * desc->head_dim;
    if (desc->hidden == 0u || desc->hidden % 32u || desc->ffn == 0u || desc->ffn % 32u
        || dim == 0u || dim > 128u || dim % 32u || heads == 0u
        || desc->delta_key_heads == 0u || heads % desc->delta_key_heads
        || desc->delta_conv < 2u || desc->delta_conv > 16u
        || desc->delta_inner != heads * dim
        || channels > UINT_MAX / AOTX_MODEL_MAX_TOKENS
        || wide > UINT_MAX / (2u * AOTX_MODEL_MAX_TOKENS)
        || desc->rope_dim == 0u || desc->rope_dim > desc->head_dim || desc->rope_dim % 2u
        || !isfinite(desc->rms_eps) || desc->rms_eps <= 0.0f
        || !isfinite(desc->rope_theta) || desc->rope_theta <= 0.0f) {
        snprintf(reason, size, "the hybrid dimensions or scales are outside the bounds");
        return 1;
    }
    uint32_t value_dim = 0u;
    if (aotx_modelfile_u32(file, "qwen35.attention.value_length", &value_dim) != 0
        || value_dim != desc->head_dim) {
        snprintf(reason, size, "the hybrid key and value widths must be equal");
        return 1;
    }
    const int32_t *sections = NULL;
    uint64_t count = 0u;
    if (aotx_modelfile_i32s(file, "qwen35.rope.dimension_sections", &sections, &count) != 0
        || count != 4u || sections[0] < 0 || sections[1] < 0
        || sections[2] < 0 || sections[3] < 0
        || (uint64_t)sections[0] + sections[1] + sections[2] + sections[3]
           != desc->rope_dim / 2u) {
        snprintf(reason, size, "the rotary sections must cover half the rotary width");
        return 1;
    }
    for (unsigned int pair = 0u; pair < desc->rope_dim / 2u; ++pair) {
        if (pair >= 3ull * (uint64_t)sections[pair % 3u]) {
            snprintf(reason, size, "the text rotary sections select an unsupported fourth coordinate");
            return 1;
        }
    }
    return 0;
}

int aotx_model_check_hybrid(const aotx_modelfile *file, const aotx_model_desc *desc,
                            char *reason, size_t size)
{
    if (aotx_hybrid_shape(file, desc, reason, size)) return 1;
    uint64_t h = desc->hidden, f = desc->ffn, d = desc->delta_dim;
    uint64_t v = desc->delta_inner, heads = desc->delta_heads;
    uint64_t c = 2ull * desc->delta_key_heads * d + v;
    uint64_t q = (uint64_t)desc->heads * desc->head_dim;
    uint64_t k = (uint64_t)desc->kv_heads * desc->head_dim;
    const uint64_t linear[AOTX_DELTA_SPAN][2] = {
        {h,0}, {h,c}, {h,v}, {desc->delta_conv,c}, {heads,0}, {heads,0}, {h,heads},
        {h,0}, {h,f}, {h,f}, {f,h}, {h,heads}, {d,0}, {v,h}
    };
    const uint64_t gated[AOTX_ATTENTION_SPAN][2] = {
        {h,0}, {h,2*q}, {h,k}, {h,k}, {q,h}, {desc->head_dim,0},
        {desc->head_dim,0}, {h,0}, {h,f}, {h,f}, {f,h}
    };
    for (unsigned int layer = 0u; layer < desc->layers; ++layer) {
        unsigned int id = desc->kind[layer];
        if (id != AOTX_LAYER_KIND_LINEAR_DELTA && id != AOTX_LAYER_KIND_ATTENTION_GATED) {
            snprintf(reason, size, "the hybrid layer %u has an incompatible kind", layer);
            return 1;
        }
        const aotx_layer_kind *kind = aotx_layer_kind_of(id);
        const uint64_t (*shape)[2] = id == AOTX_LAYER_KIND_LINEAR_DELTA ? linear : gated;
        for (unsigned int i = 0u; i < kind->tensors; ++i) {
            unsigned int slot = kind->tensor[i].slot;
            char name[AOTX_DESC_BUFFER];
            aotx_layer_name(name, sizeof name, layer, &kind->tensor[i]);
            unsigned int scalar = shape[slot][1] == 0u
                || (id == AOTX_LAYER_KIND_LINEAR_DELTA && slot == AOTX_DELTA_CONV);
            if (aotx_hybrid_tensor(file, name, shape[slot][0], shape[slot][1], scalar,
                                    reason, size)) return 1;
        }
    }
    aotx_tensor_info embedding, output;
    if (aotx_modelfile_find(file, "token_embd.weight", &embedding) != 0
        || embedding.dim_count != 2u || embedding.dims[0] != h
        || embedding.dims[1] == 0u || embedding.dims[1] > UINT_MAX
        || !aotx_matrix_known(embedding.type)) {
        snprintf(reason, size, "the hybrid embedding has an incompatible shape or type");
        return 1;
    }
    if (aotx_hybrid_tensor(file, "output_norm.weight", h, 0u, 1u, reason, size)) return 1;
    if (aotx_modelfile_find(file, "output.weight", &output) == 0
        && aotx_hybrid_tensor(file, "output.weight", h, embedding.dims[1], 0u, reason, size))
        return 1;
    return 0;
}
