/* Purpose: Hold the names and bounds shared by model tensor binding.
 * Owns: Nothing; the host forms binding names from these definitions.
 * Threading: Not applicable; the definitions are constant.
 * Lifetime: The whole run. */
#ifndef AOTX_MODEL_NAMES_H
#define AOTX_MODEL_NAMES_H

/* Names of the whole model, followed by table-built names for each layer. */
#define AOTX_DESC_WHOLE       4u
#define AOTX_DESC_NAME        20u
#define AOTX_DESC_BUFFER      48u
#define AOTX_LAYER_TENSOR_SLOTS 11u

#define AOTX_DESC_WHOLE_LIST { "token_embd.weight", "output_norm.weight", \
                               "output.weight", "cls.output.weight" }


#endif
