/* Purpose: Share layer tensor sets and state data with the header inspector.
 * Owns: Nothing; the tables contain constant data.
 * Threading: Read only on the disk side and during device setup.
 * Lifetime: The whole program. */
#ifndef AOTX_MODEL_KINDS_DATA_H
#define AOTX_MODEL_KINDS_DATA_H

#include <stddef.h>
#define AOTX_LAYER_KIND_ATTENTION           0u
#define AOTX_LAYER_KIND_ATTENTION_NO_QK_NORM 1u
#define AOTX_LAYER_KIND_FFN_EXPERTS         2u
#define AOTX_LAYER_KIND_COUNT               3u
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
static const aotx_layer_tensor aotx_layer_attention_tensor[] = {
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

static const aotx_layer_tensor aotx_layer_experts_tensor[] = {
    { "attn_norm",     0u, 0u },
    { "attn_q",        1u, 0u },
    { "attn_k",        2u, 0u },
    { "attn_v",        3u, 0u },
    { "attn_output",   4u, 0u },
    { "attn_q_norm",   5u, 0u },
    { "attn_k_norm",   6u, 0u },
    { "ffn_norm",      7u, 0u },
    { "ffn_gate_exps",  8u, 0u },
    { "ffn_up_exps",    9u, 0u },
    { "ffn_down_exps", 10u, 0u },
    { "ffn_gate_inp",  11u, 0u }
};

/* One row names the tensor set, state, capture, metadata keys, and file check. */
#define AOTX_LAYER_KIND_TABLE(X) \
    X("attention", aotx_layer_attention_tensor, AOTX_STATE_KIND_KV_PAGES, \
      aotx_model_capture_attention, aotx_layer_attention_key, NULL) \
    X("attention_no_qk_norm", aotx_layer_attention_no_qk_norm_tensor, AOTX_STATE_KIND_KV_PAGES, \
      aotx_model_capture_attention_no_qk_norm, aotx_layer_attention_key, NULL) \
    X("ffn_experts", aotx_layer_experts_tensor, AOTX_STATE_KIND_KV_PAGES, \
      aotx_model_capture_experts, aotx_layer_experts_key, aotx_model_check_experts)

#endif
