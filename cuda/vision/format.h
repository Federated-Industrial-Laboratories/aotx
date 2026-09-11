/* Purpose: Describe the trained image encoder and its tensor offsets.
 * Owns: No storage; offsets name bytes in a validated weight extent.
 * Threading: The file reader writes the descriptor before device use.
 * Lifetime: From model load to release. */
#ifndef AOTX_VISION_FORMAT_H
#define AOTX_VISION_FORMAT_H
#include <stdint.h>

#define AOTX_VISION_LAYERS 12u
#define AOTX_VISION_WIDTH 768u
#define AOTX_VISION_HIDDEN 3072u
#define AOTX_VISION_OUTPUT 1024u
#define AOTX_VISION_HEADS 12u
#define AOTX_VISION_HEAD 64u
#define AOTX_VISION_MIN_PIXELS 65536u
#define AOTX_VISION_MAX_PIXELS 16777216u

enum aotx_vision_base_tensor {
    AOTX_VISION_PATCH0, AOTX_VISION_PATCH1, AOTX_VISION_PATCH_BIAS,
    AOTX_VISION_POSITION, AOTX_VISION_NORM, AOTX_VISION_NORM_BIAS,
    AOTX_VISION_MERGE0, AOTX_VISION_MERGE0_BIAS,
    AOTX_VISION_MERGE1, AOTX_VISION_MERGE1_BIAS, AOTX_VISION_BASE_TENSORS
};
enum aotx_vision_layer_tensor {
    AOTX_VISION_LN1, AOTX_VISION_LN1_BIAS, AOTX_VISION_LN2, AOTX_VISION_LN2_BIAS,
    AOTX_VISION_QKV, AOTX_VISION_QKV_BIAS, AOTX_VISION_OUT, AOTX_VISION_OUT_BIAS,
    AOTX_VISION_UP, AOTX_VISION_UP_BIAS, AOTX_VISION_DOWN, AOTX_VISION_DOWN_BIAS,
    AOTX_VISION_LAYER_TENSORS
};
typedef struct aotx_vision_desc {
    uint64_t bytes;
    uint64_t base[AOTX_VISION_BASE_TENSORS];
    uint64_t layer[AOTX_VISION_LAYERS][AOTX_VISION_LAYER_TENSORS];
} aotx_vision_desc;
#endif
