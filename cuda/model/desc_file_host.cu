/* Purpose: Read one model file into a descriptor and its tensor binding plan.
 * Owns: Nothing; the caller owns the descriptor and binding storage.
 * Launch shape: Host only; one pass over the layers and kind rows.
 * Lifetime: One model description. */
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "embed/embed.cuh"
#include "kvcache/kvcache.cuh"
#include "model/forward.cuh"
#include "model/kinds.h"
#include "rerank/rerank.cuh"

extern "C" {
#include "disk/modelfile/modelfile.h"
}

#define AOTX_DESC_KEY 96

static const char aotx_desc_whole_name[AOTX_DESC_WHOLE][AOTX_DESC_NAME] =
    AOTX_DESC_WHOLE_LIST;

static int aotx_desc_u32(const aotx_modelfile *file, const char *arch, const char *tail,
                         unsigned int *value, int needed, char *reason, size_t reason_size)
{
    char key[AOTX_DESC_KEY];
    uint32_t got = 0u;
    snprintf(key, sizeof key, "%s.%s", arch, tail);
    if (aotx_modelfile_u32(file, key, &got) != 0) {
        if (needed != 0) {
            snprintf(reason, reason_size, "the model file does not hold %s", key);
            return 1;
        }
        return 0;
    }
    *value = (unsigned int)got;
    return 0;
}

static int aotx_desc_f32(const aotx_modelfile *file, const char *arch, const char *tail,
                         float *value, char *reason, size_t reason_size)
{
    char key[AOTX_DESC_KEY];
    snprintf(key, sizeof key, "%s.%s", arch, tail);
    if (aotx_modelfile_f32(file, key, value) != 0) {
        snprintf(reason, reason_size, "the model file does not hold %s", key);
        return 1;
    }
    return 0;
}

static int aotx_desc_kind_has(const aotx_layer_kind *kind, const char *name)
{
    for (unsigned int i = 0u; i < kind->tensors; ++i) {
        if (strcmp(kind->tensor[i].name, name) == 0) {
            return 1;
        }
    }
    return 0;
}

static int aotx_desc_tensor_seen(unsigned int kind, unsigned int tensor)
{
    const char *name = aotx_layer_kind_table[kind].tensor[tensor].name;
    for (unsigned int k = 0u; k <= kind; ++k) {
        unsigned int limit = (k == kind) ? tensor : aotx_layer_kind_table[k].tensors;
        for (unsigned int i = 0u; i < limit; ++i) {
            if (strcmp(aotx_layer_kind_table[k].tensor[i].name, name) == 0) {
                return 1;
            }
        }
    }
    return 0;
}

static unsigned int aotx_desc_difference(const aotx_modelfile *file, unsigned int layer,
                                         const aotx_layer_kind *candidate)
{
    char name[AOTX_DESC_BUFFER];
    unsigned int count = 0u;
    aotx_tensor_info info;
    for (unsigned int i = 0u; i < candidate->tensors; ++i) {
        const aotx_layer_tensor *tensor = &candidate->tensor[i];
        aotx_layer_name(name, sizeof name, layer, tensor);
        if (tensor->may_be_absent == 0u && aotx_modelfile_find(file, name, &info) != 0) {
            count += 1u;
        }
    }
    for (unsigned int k = 0u; k < AOTX_LAYER_KIND_COUNT; ++k) {
        const aotx_layer_kind *source = &aotx_layer_kind_table[k];
        for (unsigned int i = 0u; i < source->tensors; ++i) {
            const aotx_layer_tensor *tensor = &source->tensor[i];
            if (aotx_desc_kind_has(candidate, tensor->name) != 0
                || aotx_desc_tensor_seen(k, i) != 0) {
                continue;
            }
            aotx_layer_name(name, sizeof name, layer, tensor);
            if (aotx_modelfile_find(file, name, &info) == 0) {
                count += 1u;
            }
        }
    }
    return count;
}

static void aotx_desc_select(const aotx_modelfile *file, aotx_model_desc *desc)
{
    for (unsigned int layer = 0u; layer < desc->layers; ++layer) {
        unsigned int best = AOTX_LAYER_KIND_INVALID;
        unsigned int least = UINT_MAX;
        for (unsigned int kind = 0u; kind < AOTX_LAYER_KIND_COUNT; ++kind) {
            unsigned int difference = aotx_desc_difference(
                file, layer, &aotx_layer_kind_table[kind]);
            if (difference < least) {
                best = kind;
                least = difference;
            }
        }
        desc->kind[layer] = (unsigned char)best;
    }
}

static int aotx_desc_shape(const aotx_modelfile *file, aotx_model_desc *desc,
                           char *reason, size_t reason_size)
{
    const char *arch = NULL;
    size_t length = 0u;
    char name[AOTX_DESC_KEY];
    if (aotx_modelfile_string(file, "general.architecture", &arch, &length) != 0
        || length == 0u || length >= sizeof name) {
        snprintf(reason, reason_size, "the model file does not name an architecture");
        return 1;
    }
    memcpy(name, arch, length);
    name[length] = '\0';
    if (aotx_desc_u32(file, name, "block_count", &desc->layers, 1,
                      reason, reason_size) != 0
        || aotx_desc_u32(file, name, "embedding_length", &desc->hidden, 1,
                         reason, reason_size) != 0
        || aotx_desc_u32(file, name, "context_length", &desc->context, 1,
                         reason, reason_size) != 0
        || aotx_desc_u32(file, name, "pooling_type", &desc->pooling, 0,
                         reason, reason_size) != 0) {
        return 1;
    }
    if (desc->layers > AOTX_MODEL_MAX_LAYERS) {
        snprintf(reason, reason_size, "the layer count is outside the bounds");
        return 1;
    }
    /* The reference turns adjacent pairs for the llama architecture and split pairs for
     * the others of this system. The rule is a fact of the architecture, so the name
     * selects it. */
    desc->rope_pairs = (strcmp(name, "llama") == 0) ? AOTX_ROPE_PAIRS_ADJACENT
                                                    : AOTX_ROPE_PAIRS_SPLIT;
    aotx_desc_select(file, desc);

    unsigned char selected[AOTX_LAYER_KIND_COUNT] = {};
    for (unsigned int layer = 0u; layer < desc->layers; ++layer) {
        selected[desc->kind[layer]] = 1u;
    }
    for (unsigned int kind = 0u; kind < AOTX_LAYER_KIND_COUNT; ++kind) {
        if (selected[kind] == 0u) {
            continue;
        }
        for (unsigned int i = 0u; i < aotx_layer_kind_table[kind].keys; ++i) {
            const aotx_layer_key *key = &aotx_layer_kind_table[kind].key[i];
            void *member = (void *)((char *)desc + key->member);
            int bad = (key->type == AOTX_LAYER_KEY_F32)
                    ? aotx_desc_f32(file, name, key->name, (float *)member,
                                    reason, reason_size)
                    : aotx_desc_u32(file, name, key->name, (unsigned int *)member, 1,
                                    reason, reason_size);
            if (bad != 0) {
                return 1;
            }
        }
    }
    if (desc->heads == 0u || desc->kv_heads == 0u
        || (desc->heads % desc->kv_heads) != 0u) {
        snprintf(reason, reason_size, "the head count is outside the bounds");
        return 1;
    }
    if (desc->head_dim == 0u || desc->head_dim > AOTX_MODEL_HEAD_MAX
        || (desc->head_dim % 32u) != 0u) {
        snprintf(reason, reason_size, "the head width %u is not a multiple of 32 up to %u",
                 desc->head_dim, AOTX_MODEL_HEAD_MAX);
        return 1;
    }
    unsigned int gives = (desc->role == AOTX_MODEL_EMBEDDING) ? AOTX_EMBED_POOL_LAST
                       : ((desc->role == AOTX_MODEL_RERANKER) ? AOTX_RERANK_POOL_RANK : 0u);
    if (desc->pooling != gives) {
        snprintf(reason, reason_size,
                 "the model of role %u asks for the pooling %u and this system gives %u",
                 desc->role, desc->pooling, gives);
        return 1;
    }
    return 0;
}

static int aotx_desc_bindings(const aotx_modelfile *file, const aotx_model_desc *desc,
                              aotx_model_binding *binding, unsigned int capacity,
                              unsigned int *count, char *reason, size_t reason_size)
{
    unsigned int needed = AOTX_DESC_WHOLE;
    for (unsigned int layer = 0u; layer < desc->layers; ++layer) {
        needed += aotx_layer_kind_table[desc->kind[layer]].tensors;
    }
    if (capacity < needed) {
        snprintf(reason, reason_size, "the tensor binding list is too small");
        return 1;
    }
    *count = 0u;
    for (unsigned int i = 0u; i < AOTX_DESC_WHOLE; ++i) {
        snprintf(binding[*count].name, sizeof binding[*count].name, "%s",
                 aotx_desc_whole_name[i]);
        binding[*count].slot = i;
        binding[*count].needed = (i < 2u || (i == 3u && desc->role == AOTX_MODEL_RERANKER));
        *count += 1u;
    }
    for (unsigned int layer = 0u; layer < desc->layers; ++layer) {
        const aotx_layer_kind *kind = &aotx_layer_kind_table[desc->kind[layer]];
        for (unsigned int i = 0u; i < kind->tensors; ++i) {
            const aotx_layer_tensor *tensor = &kind->tensor[i];
            if (aotx_layer_name(binding[*count].name, sizeof binding[*count].name,
                                layer, tensor) != 0) {
                snprintf(reason, reason_size, "a tensor name does not fit the binding list");
                return 1;
            }
            binding[*count].slot = AOTX_DESC_WHOLE
                                 + layer * AOTX_LAYER_TENSOR_SLOTS + tensor->slot;
            binding[*count].needed = (tensor->may_be_absent == 0u);
            *count += 1u;
        }
    }

    unsigned int missing = 0u;
    unsigned int first = 0u;
    aotx_tensor_info info;
    for (unsigned int i = 0u; i < *count; ++i) {
        if (binding[i].needed != 0u
            && aotx_modelfile_find(file, binding[i].name, &info) != 0) {
            if (missing == 0u) {
                first = i;
            }
            missing += 1u;
        }
    }
    if (missing != 0u) {
        snprintf(reason, reason_size,
                 "the model of role %u has no tensor %s, and %u more are absent",
                 desc->role, binding[first].name, missing - 1u);
        return 1;
    }
    for (unsigned int layer = 0u; layer < desc->layers; ++layer) {
        const aotx_layer_kind *kind = &aotx_layer_kind_table[desc->kind[layer]];
        if (aotx_kv_state_check(kind->state, reason, reason_size) != 0) {
            return 1;
        }
        if (kind->capture == NULL) {
            snprintf(reason, reason_size, "the layer kind %s has no capture function",
                     kind->name);
            return 1;
        }
    }
    return 0;
}

int aotx_model_desc_file(const aotx_modelfile *file, unsigned int role,
                         aotx_model_desc *desc, aotx_model_binding *binding,
                         unsigned int capacity, unsigned int *count,
                         char *reason, size_t reason_size)
{
    if (file == NULL || desc == NULL || binding == NULL || count == NULL
        || reason == NULL || reason_size == 0u || role >= AOTX_MODEL_ROLES) {
        return 1;
    }
    reason[0] = '\0';
    *count = 0u;
    memset(desc, 0, sizeof *desc);
    memset(desc->kind, AOTX_LAYER_KIND_INVALID, sizeof desc->kind);
    memset(desc->layer, 0xff, sizeof desc->layer);
    desc->role = role;
    desc->tied_output = 1u;
    desc->token_embd = AOTX_MODEL_ABSENT;
    desc->output_norm = AOTX_MODEL_ABSENT;
    desc->output = AOTX_MODEL_ABSENT;
    desc->cls_output = AOTX_MODEL_ABSENT;
    desc->rope_freqs = AOTX_MODEL_ABSENT;
    if (aotx_desc_shape(file, desc, reason, reason_size) != 0
        || aotx_layer_desc_valid(desc) == 0) {
        if (reason[0] == '\0') {
            snprintf(reason, reason_size, "the descriptor has a layer without a kind");
        }
        return 1;
    }
    return aotx_desc_bindings(file, desc, binding, capacity, count, reason, reason_size);
}
