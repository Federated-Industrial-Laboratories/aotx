/* Purpose: Define the layer kinds, their tensors, state, capture, and metadata keys.
 * Owns: Nothing; each host translation unit holds the constant table.
 * Launch shape: Host only; the table is read during model load and graph capture.
 * Lifetime: The whole run. */
#ifndef AOTX_MODEL_KINDS_H
#define AOTX_MODEL_KINDS_H

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "model/names.h"
#include "model/model.cuh"
#include "model/roles.h"

#define AOTX_LAYER_KIND_ATTENTION           0u
#define AOTX_LAYER_KIND_ATTENTION_NO_QK_NORM 1u
#define AOTX_LAYER_KIND_COUNT               2u
#define AOTX_LAYER_KIND_INVALID             0xffu

#define AOTX_STATE_KIND_KV_PAGES    0u
#define AOTX_STATE_KIND_DELTA_STATE 1u
#define AOTX_STATE_KIND_COUNT       2u
#define AOTX_STATE_KIND_NONE        0xffu

#define AOTX_STATE_BYTES_KV_CONTEXT       0u
#define AOTX_STATE_BYTES_DELTA_HEAD_SQUARE 1u
#define AOTX_STATE_BYTES_RULE_COUNT       2u

#define AOTX_STATE_RESTORE_REPLAY_PROMPT 0u
#define AOTX_STATE_RESTORE_COUNT         1u

#define AOTX_STATE_MANAGER_NONE     0u
#define AOTX_STATE_MANAGER_KV_PAGES 1u
#define AOTX_STATE_MANAGER_COUNT    2u

#define AOTX_LAYER_TENSORS_MAX    11u

#define AOTX_LAYER_KEY_U32        0u
#define AOTX_LAYER_KEY_F32        1u

struct aotx_model_hold;
typedef void (*aotx_layer_capture)(struct aotx_model_hold *hold, unsigned int role,
                                   unsigned int layer);

/* Put the nodes of one attention layer in the graph, with the head norm or without it. */
void aotx_model_capture_attention(struct aotx_model_hold *hold, unsigned int role,
                                  unsigned int layer);
void aotx_model_capture_attention_no_qk_norm(struct aotx_model_hold *hold,
                                             unsigned int role, unsigned int layer);

typedef struct aotx_state_kind {
    const char *name;
    unsigned int bytes_rule;
    unsigned int grows_context;
    unsigned int paged;
    unsigned int restore;
    unsigned int manager;
} aotx_state_kind;

typedef struct aotx_layer_tensor {
    const char *name;
    unsigned int slot;
    unsigned int may_be_absent;
} aotx_layer_tensor;

typedef struct aotx_layer_key {
    const char *name;
    unsigned int type;
    size_t member;
} aotx_layer_key;

typedef struct aotx_layer_kind {
    const char *name;
    const aotx_layer_tensor *tensor;
    unsigned int tensors;
    unsigned int state;
    aotx_layer_capture capture;
    const aotx_layer_key *key;
    unsigned int keys;
} aotx_layer_kind;

static const aotx_state_kind aotx_state_kind_table[AOTX_STATE_KIND_COUNT] = {
    {
        "kv_pages",
        AOTX_STATE_BYTES_KV_CONTEXT,
        1u,
        1u,
        AOTX_STATE_RESTORE_REPLAY_PROMPT,
        AOTX_STATE_MANAGER_KV_PAGES
    },
    {
        "delta_state",
        AOTX_STATE_BYTES_DELTA_HEAD_SQUARE,
        0u,
        0u,
        AOTX_STATE_RESTORE_REPLAY_PROMPT,
        AOTX_STATE_MANAGER_NONE
    }
};

static inline const aotx_state_kind *aotx_state_kind_of(unsigned int state)
{
    return (state < AOTX_STATE_KIND_COUNT) ? &aotx_state_kind_table[state] : NULL;
}

static_assert(sizeof(aotx_model_layer)
              == AOTX_LAYER_TENSOR_SLOTS * sizeof(unsigned long long),
              "a layer must hold only tensor slots");
static_assert(offsetof(aotx_model_desc, layer)
              == offsetof(aotx_model_desc, token_embd)
               + AOTX_DESC_WHOLE * sizeof(unsigned long long),
              "the layer slots must follow the whole-model slots");
static_assert(offsetof(aotx_model_desc, rope_freqs)
              == offsetof(aotx_model_desc, token_embd)
               + (AOTX_DESC_WHOLE - 1u) * sizeof(unsigned long long),
              "the rope factor row is the last whole-model slot");

static const aotx_layer_tensor aotx_layer_attention_tensor[AOTX_LAYER_TENSORS_MAX] = {
    { "attn_norm",   0u, 0u },
    { "attn_q",      1u, 0u },
    { "attn_k",      2u, 0u },
    { "attn_v",      3u, 0u },
    { "attn_output", 4u, 0u },
    { "attn_q_norm", 5u, 0u },
    { "attn_k_norm", 6u, 0u },
    { "ffn_norm",    7u, 0u },
    { "ffn_gate",    8u, 0u },
    { "ffn_up",      9u, 0u },
    { "ffn_down",   10u, 0u }
};

static const aotx_layer_tensor aotx_layer_attention_no_qk_norm_tensor[] = {
    { "attn_norm",   0u, 0u },
    { "attn_q",      1u, 0u },
    { "attn_k",      2u, 0u },
    { "attn_v",      3u, 0u },
    { "attn_output", 4u, 0u },
    { "ffn_norm",    7u, 0u },
    { "ffn_gate",    8u, 0u },
    { "ffn_up",      9u, 0u },
    { "ffn_down",   10u, 0u }
};

static const aotx_layer_key aotx_layer_attention_key[] = {
    { "feed_forward_length",               AOTX_LAYER_KEY_U32,
      offsetof(aotx_model_desc, ffn) },
    { "attention.head_count",               AOTX_LAYER_KEY_U32,
      offsetof(aotx_model_desc, heads) },
    { "attention.head_count_kv",            AOTX_LAYER_KEY_U32,
      offsetof(aotx_model_desc, kv_heads) },
    { "attention.key_length",               AOTX_LAYER_KEY_U32,
      offsetof(aotx_model_desc, head_dim) },
    { "rope.freq_base",                     AOTX_LAYER_KEY_F32,
      offsetof(aotx_model_desc, rope_theta) },
    { "attention.layer_norm_rms_epsilon",   AOTX_LAYER_KEY_F32,
      offsetof(aotx_model_desc, rms_eps) }
};

static const aotx_layer_kind aotx_layer_kind_table[AOTX_LAYER_KIND_COUNT] = {
    {
        "attention",
        aotx_layer_attention_tensor,
        AOTX_LAYER_TENSORS_MAX,
        AOTX_STATE_KIND_KV_PAGES,
        aotx_model_capture_attention,
        aotx_layer_attention_key,
        sizeof aotx_layer_attention_key / sizeof aotx_layer_attention_key[0]
    },
    {
        "attention_no_qk_norm",
        aotx_layer_attention_no_qk_norm_tensor,
        sizeof aotx_layer_attention_no_qk_norm_tensor
            / sizeof aotx_layer_attention_no_qk_norm_tensor[0],
        AOTX_STATE_KIND_KV_PAGES,
        aotx_model_capture_attention_no_qk_norm,
        aotx_layer_attention_key,
        sizeof aotx_layer_attention_key / sizeof aotx_layer_attention_key[0]
    }
};

static inline const aotx_layer_kind *aotx_layer_kind_of(unsigned int kind)
{
    return (kind < AOTX_LAYER_KIND_COUNT) ? &aotx_layer_kind_table[kind] : NULL;
}

/* Count the layers whose kind uses one state kind. */
static inline unsigned int aotx_layer_state_count(const aotx_model_desc *desc,
                                                  unsigned int state)
{
    unsigned int count = 0u;
    for (unsigned int layer = 0u; layer < desc->layers; ++layer) {
        const aotx_layer_kind *kind = aotx_layer_kind_of(desc->kind[layer]);
        count += kind != NULL && kind->state == state;
    }
    return count;
}

static inline int aotx_layer_name(char *out, size_t size, unsigned int layer,
                                  const aotx_layer_tensor *tensor)
{
    int used = snprintf(out, size, "blk.%u.%s.weight", layer, tensor->name);
    return (used < 0 || (size_t)used >= size) ? 1 : 0;
}

static inline int aotx_layer_desc_valid(const aotx_model_desc *desc)
{
    if (desc->layers > AOTX_MODEL_MAX_LAYERS) {
        return 0;
    }
    for (unsigned int layer = 0u; layer < desc->layers; ++layer) {
        const aotx_layer_kind *kind = aotx_layer_kind_of(desc->kind[layer]);
        if (kind == NULL || (kind->state != AOTX_STATE_KIND_NONE
                            && aotx_state_kind_of(kind->state) == NULL)) {
            return 0;
        }
    }
    return 1;
}


/* Print consecutive runs of one descriptor's layer kinds. */
static inline void aotx_layer_print_runs(const aotx_model_desc *one)
{
    for (unsigned int first = 0u; first < one->layers;) {
        unsigned int kind_id = one->kind[first];
        unsigned int last = first + 1u;
        while (last < one->layers && one->kind[last] == kind_id) {
            last += 1u;
        }
        const aotx_layer_kind *kind = aotx_layer_kind_of(kind_id);
        printf(" %u %s", last - first, (kind != NULL) ? kind->name : "unknown");
        first = last;
    }
}

static inline void aotx_layer_print_one(const aotx_model_desc *desc)
{
    printf("layers:");
    aotx_layer_print_runs(desc);
    printf("\n");
}

/* Print consecutive runs of each loaded layer kind. Multiple roles share one line. */
static inline void aotx_layer_print(const aotx_model_desc *desc,
                                    const unsigned int *role, unsigned int roles)
{
    printf("layers:");
    for (unsigned int r = 0u; r < roles; ++r) {
        const aotx_model_desc *one = &desc[role[r]];
        if (roles > 1u) {
            printf("%s%s", (r == 0u) ? " " : "; ", aotx_role_name[role[r]]);
        }
        aotx_layer_print_runs(one);
    }
    printf("\n");
}

#endif
