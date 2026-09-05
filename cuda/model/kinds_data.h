/* Purpose: Share layer tensor sets and state data with the header inspector.
 * Owns: Nothing; the tables contain constant data.
 * Threading: Read only on the disk side and during device setup.
 * Lifetime: The whole program. */
#ifndef AOTX_MODEL_KINDS_DATA_H
#define AOTX_MODEL_KINDS_DATA_H

#include <stddef.h>
#include <limits.h>
#define AOTX_LAYER_KIND_ATTENTION           0u
#define AOTX_LAYER_KIND_ATTENTION_NO_QK_NORM 1u
#define AOTX_LAYER_KIND_FFN_EXPERTS         2u
#define AOTX_LAYER_KIND_ATTENTION_BIAS      3u
#define AOTX_LAYER_KIND_COUNT               4u
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

#define AOTX_LAYER_EXPERTS_MAX    256u

#define AOTX_LAYER_KEY_U32        0u
#define AOTX_LAYER_KEY_F32        1u
#define AOTX_LAYER_KEY_U32_OPTIONAL 2u
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
/* The shared kernels use the same norm and projection slots in each row. */
enum {
    AOTX_SLOT_ATTN_NORM = 0u,
    AOTX_SLOT_ATTN_Q = 1u,
    AOTX_SLOT_ATTN_K = 2u,
    AOTX_SLOT_ATTN_V = 3u,
    AOTX_SLOT_ATTN_O = 4u,
    AOTX_SLOT_FFN_NORM = 7u,
    AOTX_SLOT_FFN_GATE = 8u,
    AOTX_SLOT_FFN_UP = 9u,
    AOTX_SLOT_FFN_DOWN = 10u,
};
enum { AOTX_ATTENTION_Q_NORM = 5u, AOTX_ATTENTION_K_NORM = 6u,
       AOTX_ATTENTION_SPAN = AOTX_SLOT_FFN_DOWN + 1u };
enum { AOTX_ATTENTION_NO_QK_NORM_SPAN = AOTX_SLOT_FFN_DOWN + 1u };
enum { AOTX_EXPERT_ROUTER = 11u, AOTX_EXPERT_SPAN = AOTX_EXPERT_ROUTER + 1u };
enum { AOTX_BIAS_Q = 5u, AOTX_BIAS_K = 6u, AOTX_BIAS_V = 11u,
       AOTX_BIAS_SPAN = AOTX_BIAS_V + 1u };

static const aotx_layer_tensor aotx_layer_attention_tensor[] = {
    { "attn_norm.weight", AOTX_SLOT_ATTN_NORM, 0u },
    { "attn_q.weight", AOTX_SLOT_ATTN_Q, 0u },
    { "attn_k.weight", AOTX_SLOT_ATTN_K, 0u },
    { "attn_v.weight", AOTX_SLOT_ATTN_V, 0u },
    { "attn_output.weight", AOTX_SLOT_ATTN_O, 0u },
    { "attn_q_norm.weight", AOTX_ATTENTION_Q_NORM, 0u },
    { "attn_k_norm.weight", AOTX_ATTENTION_K_NORM, 0u },
    { "ffn_norm.weight", AOTX_SLOT_FFN_NORM, 0u },
    { "ffn_gate.weight", AOTX_SLOT_FFN_GATE, 0u },
    { "ffn_up.weight", AOTX_SLOT_FFN_UP, 0u },
    { "ffn_down.weight", AOTX_SLOT_FFN_DOWN, 0u }
};

static const aotx_layer_tensor aotx_layer_attention_no_qk_norm_tensor[] = {
    { "attn_norm.weight", AOTX_SLOT_ATTN_NORM, 0u },
    { "attn_q.weight", AOTX_SLOT_ATTN_Q, 0u },
    { "attn_k.weight", AOTX_SLOT_ATTN_K, 0u },
    { "attn_v.weight", AOTX_SLOT_ATTN_V, 0u },
    { "attn_output.weight", AOTX_SLOT_ATTN_O, 0u },
    { "ffn_norm.weight", AOTX_SLOT_FFN_NORM, 0u },
    { "ffn_gate.weight", AOTX_SLOT_FFN_GATE, 0u },
    { "ffn_up.weight", AOTX_SLOT_FFN_UP, 0u },
    { "ffn_down.weight", AOTX_SLOT_FFN_DOWN, 0u }
};

static const aotx_layer_tensor aotx_layer_experts_tensor[] = {
    { "attn_norm.weight", AOTX_SLOT_ATTN_NORM, 0u },
    { "attn_q.weight", AOTX_SLOT_ATTN_Q, 0u },
    { "attn_k.weight", AOTX_SLOT_ATTN_K, 0u },
    { "attn_v.weight", AOTX_SLOT_ATTN_V, 0u },
    { "attn_output.weight", AOTX_SLOT_ATTN_O, 0u },
    { "attn_q_norm.weight", AOTX_ATTENTION_Q_NORM, 0u },
    { "attn_k_norm.weight", AOTX_ATTENTION_K_NORM, 0u },
    { "ffn_norm.weight", AOTX_SLOT_FFN_NORM, 0u },
    { "ffn_gate_exps.weight", AOTX_SLOT_FFN_GATE, 0u },
    { "ffn_up_exps.weight", AOTX_SLOT_FFN_UP, 0u },
    { "ffn_down_exps.weight", AOTX_SLOT_FFN_DOWN, 0u },
    { "ffn_gate_inp.weight", AOTX_EXPERT_ROUTER, 0u }
};

static const aotx_layer_tensor aotx_layer_attention_bias_tensor[] = {
    { "attn_norm.weight", AOTX_SLOT_ATTN_NORM, 0u },
    { "attn_q.weight", AOTX_SLOT_ATTN_Q, 0u },
    { "attn_k.weight", AOTX_SLOT_ATTN_K, 0u },
    { "attn_v.weight", AOTX_SLOT_ATTN_V, 0u },
    { "attn_output.weight", AOTX_SLOT_ATTN_O, 0u },
    { "ffn_norm.weight", AOTX_SLOT_FFN_NORM, 0u },
    { "ffn_gate.weight", AOTX_SLOT_FFN_GATE, 0u },
    { "ffn_up.weight", AOTX_SLOT_FFN_UP, 0u },
    { "ffn_down.weight", AOTX_SLOT_FFN_DOWN, 0u },
    { "attn_q.bias", AOTX_BIAS_Q, 0u },
    { "attn_k.bias", AOTX_BIAS_K, 0u },
    { "attn_v.bias", AOTX_BIAS_V, 0u }
};

/* Each row names its tensors, state, capture, keys, file check, and slot span. */
#define AOTX_LAYER_KIND_TABLE(X) \
    X("attention", aotx_layer_attention_tensor, AOTX_STATE_KIND_KV_PAGES, \
      aotx_model_capture_attention, aotx_layer_attention_key, NULL, AOTX_ATTENTION_SPAN) \
    X("attention_no_qk_norm", aotx_layer_attention_no_qk_norm_tensor, AOTX_STATE_KIND_KV_PAGES, \
      aotx_model_capture_attention_no_qk_norm, aotx_layer_attention_key, NULL, AOTX_ATTENTION_NO_QK_NORM_SPAN) \
    X("ffn_experts", aotx_layer_experts_tensor, AOTX_STATE_KIND_KV_PAGES, \
      aotx_model_capture_experts, aotx_layer_experts_key, aotx_model_check_experts, AOTX_EXPERT_SPAN) \
    X("attention_bias", aotx_layer_attention_bias_tensor, AOTX_STATE_KIND_KV_PAGES, \
      aotx_model_capture_attention_bias, aotx_layer_attention_bias_key, aotx_model_check_bias, AOTX_BIAS_SPAN)

#define AOTX_LAYER_STORAGE(name, tensor, state, capture, key, check, span) \
    unsigned char tensor[span];
typedef union aotx_layer_storage {
    AOTX_LAYER_KIND_TABLE(AOTX_LAYER_STORAGE)
} aotx_layer_storage;
#undef AOTX_LAYER_STORAGE
#define AOTX_LAYER_TENSOR_SLOTS ((unsigned int)sizeof(aotx_layer_storage))

#define AOTX_LAYER_MASK_CHECK(name, tensor, state, capture, key, check, span) \
    typedef char tensor##_mask_fits[ \
        sizeof tensor / sizeof tensor[0] <= sizeof(unsigned int) * CHAR_BIT ? 1 : -1];
AOTX_LAYER_KIND_TABLE(AOTX_LAYER_MASK_CHECK)
#undef AOTX_LAYER_MASK_CHECK

#endif
