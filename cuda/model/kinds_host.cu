/* Purpose: Check each selected layer type before tensor binding.
 * Owns: Nothing; the caller owns the model file and descriptor.
 * Launch shape: Host glue; one check for each selected layer type.
 * Lifetime: One model load. */
#include "kvcache/kvcache.cuh"
#include "model/kinds.h"

int aotx_model_check_layers(const aotx_modelfile *file, const aotx_model_desc *desc,
                            char *reason, size_t reason_size)
{
    unsigned char checked[AOTX_LAYER_KIND_COUNT] = {};
    for (unsigned int layer = 0u; layer < desc->layers; ++layer) {
        unsigned int index = desc->kind[layer];
        const aotx_layer_kind *kind = aotx_layer_kind_of(index);
        if (kind == NULL) {
            snprintf(reason, reason_size, "the layer type is outside the table");
            return 1;
        }
        if (checked[index] != 0u) continue;
        checked[index] = 1u;
        if (aotx_kv_state_check(kind->state, reason, reason_size) != 0) return 1;
        if (kind->capture == NULL) {
            snprintf(reason, reason_size, "the layer kind %s has no capture function", kind->name);
            return 1;
        }
        if (kind->check != NULL && kind->check(file, desc, reason, reason_size) != 0) return 1;
    }
    return 0;
}
