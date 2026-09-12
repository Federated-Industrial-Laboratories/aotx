/* Purpose: Read the vocabulary owned by a language sequence.
 * Owns: No storage; saved descriptors refer to immutable token tables.
 * Launch shape: Device callers supply the model role for each row.
 * Lifetime: From vocabulary load through the final reply read. */
#ifndef AOTX_MODEL_VOCAB_CUH
#define AOTX_MODEL_VOCAB_CUH
#include "model/model.cuh"
#include "text/text.cuh"
__device__ __forceinline__ const aotx_text_vocab *aotx_model_vocab(unsigned role)
{
    unsigned at=role==AOTX_MODEL_LANGUAGE_AUDIO?2u:0u;
    return aotx_text_vocab_saved[at].tokens?&aotx_text_vocab_saved[at]:&aotx_text_vocab_table;
}
#endif
