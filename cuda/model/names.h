/* Purpose: Hold the tensor names of a model, so the device and the host read one list.
 * Owns: Nothing; the two sides each define an array from these lists.
 * Threading: Not applicable; the lists are text.
 * Lifetime: The whole run. */
#ifndef AOTX_MODEL_NAMES_H
#define AOTX_MODEL_NAMES_H

/* Names of the tensors of the whole model, in the order of the first four offsets of the
 * descriptor. The layer list holds the names of one layer, in the order of the layer
 * structure. */
#define AOTX_DESC_WHOLE      4u
#define AOTX_DESC_PER_LAYER  11u
#define AOTX_DESC_NAME       20u

#define AOTX_DESC_WHOLE_LIST { "token_embd.weight", "output_norm.weight", \
                               "output.weight", "cls.output.weight" }

#define AOTX_DESC_LAYER_LIST { "attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", \
                               "attn_q_norm", "attn_k_norm", "ffn_norm", "ffn_gate", \
                               "ffn_up", "ffn_down" }

/* The names of one model: the four of the whole model and eleven for each layer. */
#define AOTX_DESC_NAMES(layers) (AOTX_DESC_WHOLE + (layers) * AOTX_DESC_PER_LAYER)

#endif
