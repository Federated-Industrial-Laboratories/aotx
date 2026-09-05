/* Purpose: Name the compiled tokenizer families and their split patterns.
 * Owns: Nothing; the tables contain constant data.
 * Threading: Read only on the disk side and during device setup.
 * Lifetime: The whole program. */
#ifndef AOTX_TEXT_FAMILIES_H
#define AOTX_TEXT_FAMILIES_H
/* The split patterns of the pre-tokenizer. One row is one pattern and one device function
 * in pretok.cu which matches it. The table gives the pattern numbers and the switch of the
 * state machine. Row 0 is the pattern a zero table selects. X(symbol, function). */
#define AOTX_TEXT_PATTERN_TABLE(X) \
    X(AOTX_TEXT_PATTERN_QWEN2,  aotx_text_match_qwen2) \
    X(AOTX_TEXT_PATTERN_LLAMA3, aotx_text_match_llama3) \
    X(AOTX_TEXT_PATTERN_GPT2,   aotx_text_match_gpt2)

#define AOTX_TEXT_PATTERN_INDEX(symbol, function) symbol,
enum aotx_text_pattern { AOTX_TEXT_PATTERN_TABLE(AOTX_TEXT_PATTERN_INDEX) AOTX_TEXT_PATTERN_COUNT };
#undef AOTX_TEXT_PATTERN_INDEX

/* The tokenizer families. One row is one value of tokenizer.ggml.pre in a model file, the
 * pattern it selects, and the whole piece flag. With the flag set, a whole piece which is a
 * token stands as that token, and the merge step does not run on it. Several names select
 * one pattern, so this table and the pattern table are two tables. Row 0 is the family a
 * zero source selects. X(name, pattern, whole). */
#define AOTX_TEXT_FAMILY_TABLE(X) \
    X("qwen2",            AOTX_TEXT_PATTERN_QWEN2,  0u) \
    X("deepseek-r1-qwen", AOTX_TEXT_PATTERN_QWEN2,  0u) \
    X("kormo",            AOTX_TEXT_PATTERN_QWEN2,  0u) \
    X("f2llmv2",          AOTX_TEXT_PATTERN_QWEN2,  0u) \
    X("llama3",           AOTX_TEXT_PATTERN_LLAMA3, 1u) \
    X("llama-v3",         AOTX_TEXT_PATTERN_LLAMA3, 1u) \
    X("llama-bpe",        AOTX_TEXT_PATTERN_LLAMA3, 1u) \
    X("falcon3",          AOTX_TEXT_PATTERN_LLAMA3, 1u) \
    X("falcon-h1",        AOTX_TEXT_PATTERN_LLAMA3, 1u) \
    X("pixtral",          AOTX_TEXT_PATTERN_LLAMA3, 1u) \
    X("midm-2.0",         AOTX_TEXT_PATTERN_LLAMA3, 1u) \
    X("lfm2",             AOTX_TEXT_PATTERN_LLAMA3, 1u) \
    X("jina-v5-nano",     AOTX_TEXT_PATTERN_LLAMA3, 1u) \
    X("gpt-2",            AOTX_TEXT_PATTERN_GPT2,   0u) \
    X("mpt",              AOTX_TEXT_PATTERN_GPT2,   0u) \
    X("olmo",             AOTX_TEXT_PATTERN_GPT2,   0u) \
    X("jais",             AOTX_TEXT_PATTERN_GPT2,   0u) \
    X("trillion",         AOTX_TEXT_PATTERN_GPT2,   0u) \
    X("granite-docling",  AOTX_TEXT_PATTERN_GPT2,   0u)

#define AOTX_TEXT_FAMILY_ONE(name, pattern, whole) 1u +
#define AOTX_TEXT_FAMILIES  (AOTX_TEXT_FAMILY_TABLE(AOTX_TEXT_FAMILY_ONE) 0u)

#endif
