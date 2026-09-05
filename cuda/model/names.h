/* Purpose: Hold the names and bounds shared by model tensor binding.
 * Owns: Nothing; the host forms binding names from these definitions.
 * Threading: Not applicable; the definitions are constant.
 * Lifetime: The whole run. */
#ifndef AOTX_MODEL_NAMES_H
#define AOTX_MODEL_NAMES_H

/* Names of the whole model, followed by table-built names for each layer. The last whole
 * name is the rope frequency factor row, which a file may not hold. */
#define AOTX_DESC_WHOLE       5u
#define AOTX_DESC_NAME        20u
#define AOTX_DESC_BUFFER      48u
#define AOTX_LAYER_TENSOR_SLOTS 11u
#define AOTX_MODEL_MAX_LAYERS  64u

#define AOTX_DESC_WHOLE_LIST { "token_embd.weight", "output_norm.weight", \
                               "output.weight", "cls.output.weight", \
                               "rope_freqs.weight" }


#endif
