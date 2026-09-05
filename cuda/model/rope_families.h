/* Purpose: Select the rotary pair rule from an architecture name.
 * Owns: The read-only rotary family table.
 * Threading: Disk-side inspection and host model setup only.
 * Lifetime: The whole program. */
#ifndef AOTX_ROPE_FAMILIES_H
#define AOTX_ROPE_FAMILIES_H

#include <stddef.h>
#include <string.h>

/* Split pairs join i and i plus half the head. Adjacent pairs join 2i and 2i plus 1. */
#define AOTX_ROPE_PAIRS_SPLIT    0u
#define AOTX_ROPE_PAIRS_ADJACENT 1u

/* These rows state only the pair rule, not support for the whole architecture.
 * A name outside this table has no default rule and is refused. X(name, pairs). */
#define AOTX_ROPE_FAMILY_TABLE(X) \
    X("llama",    AOTX_ROPE_PAIRS_ADJACENT) \
    X("olmo",     AOTX_ROPE_PAIRS_ADJACENT) \
    X("qwen2",    AOTX_ROPE_PAIRS_SPLIT) \
    X("qwen3",    AOTX_ROPE_PAIRS_SPLIT) \
    X("qwen3moe", AOTX_ROPE_PAIRS_SPLIT) \
    X("olmoe",    AOTX_ROPE_PAIRS_SPLIT)

/* Match the whole name, including its length. Leave the output unchanged on refusal. */
static inline int aotx_rope_family_pairs(const char *name, size_t bytes,
                                        unsigned int *pairs)
{
#define AOTX_ROPE_FAMILY_MATCH(text, rule) \
    if (bytes == sizeof(text) - 1u && memcmp(name, text, sizeof(text) - 1u) == 0) { \
        *pairs = rule; \
        return 1; \
    }
    AOTX_ROPE_FAMILY_TABLE(AOTX_ROPE_FAMILY_MATCH)
#undef AOTX_ROPE_FAMILY_MATCH
    return 0;
}

#endif
