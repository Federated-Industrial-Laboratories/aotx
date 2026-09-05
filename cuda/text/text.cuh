/* Purpose: Convert between bytes, tokens and text.
 * Owns: The vocabulary tables.
 * Launch shape: One thread for each sequence, then one warp for each chunk.
 * Lifetime: From model load to the end of the run. */
#ifndef AOTX_TEXT_CUH
#define AOTX_TEXT_CUH

#include "text/families.h"

/* The code point that stands for a byte run which is not correct UTF-8. */
#define AOTX_TEXT_REPLACEMENT   0xFFFDu

/* The mark of a byte that continues a character. Only the first byte of a character gives a
 * code point, so the other bytes of that character carry this mark. */
#define AOTX_TEXT_NONE          0xFFFFFFFFu

/* Bytes of the longest chunk that the merge step takes. A pre-token which is longer is cut
 * at a character boundary at this bound.
 *
 * The bound is a real limit and it stays. The merge step gives each warp room for one
 * whole chunk. A pre-token is one run of letters, of numbers, or of marks. A line of
 * ordinary text or of code never comes near 4,096 bytes. A run which is longer than the
 * bound gives more tokens than the reference tokenizer gives, which has no bound. */
#define AOTX_TEXT_CHUNK_BYTES   4096u

/* Bytes of one chunk after the byte to code point map. That map gives one or two bytes for
 * each byte of the chunk. This buffer is the shared memory of a warp. */
#define AOTX_TEXT_SYMBOL_BYTES  (2u * AOTX_TEXT_CHUNK_BYTES)

/* Bytes of device memory that one warp of the merge step holds for the lists of the chunk.
 * Each symbol takes 14 bytes of the lists. They are the place of its text, the symbol
 * after it, the symbol before it, its token, and the rank of the pair it makes. The lists
 * do not go in shared memory, because 4,096 symbols of them would leave room for one warp
 * in a block. The caller gives merge memory of this size for each warp it launches. */
#define AOTX_TEXT_WARP_BYTES    (14u * AOTX_TEXT_CHUNK_BYTES)

/* Warps of one block of the merge step. Each warp holds one chunk. */
#define AOTX_TEXT_WARPS         2u

/* Token types of the model file. A token of the control type or the user type is a whole
 * match. The pre-tokenizer finds it before it looks at the character classes. */
#define AOTX_TEXT_TYPE_UNKNOWN  2
#define AOTX_TEXT_TYPE_CONTROL  3
#define AOTX_TEXT_TYPE_USER     4

/* Special tokens the vocabulary may hold. The Qwen3 files hold 26 and the Llama 3.2 file
 * holds 256. The bound leaves room for a file which adds tokens to that set. */
#define AOTX_TEXT_SPECIAL_MAX   1024u


/* The vocabulary that the device reads. The tokens come from the model file as one byte run
 * with an offset table, and they stay in that form on the device. The two hash tables give
 * a token from a byte run and a rank from a pair of tokens. Both tables use open addressing:
 * a slot that holds AOTX_TEXT_NONE is empty, and a search steps on by one slot. */
typedef struct aotx_text_vocab {
    const unsigned char *token_bytes;    /* every token string, one after the other */
    const unsigned long long *token_at;  /* tokens + 1 offsets into token_bytes */
    unsigned int tokens;                 /* tokens the vocabulary holds */
    unsigned int slots;                  /* slots of the token table; a power of two */
    const unsigned int *slot;            /* token of each slot, or AOTX_TEXT_NONE */
    unsigned int pairs;                  /* slots of the pair table; a power of two */
    const unsigned long long *pair_key;  /* left token and right token of each slot */
    const unsigned int *pair_rank;       /* rank of the pair; a low rank merges first */
    unsigned int pattern;                /* row of the pattern table the pre-tokenizer runs */
    unsigned int whole;                  /* 1 when a whole piece that is a token stands alone */
    unsigned int specials;               /* special tokens the vocabulary holds */
    unsigned int special[AOTX_TEXT_SPECIAL_MAX];  /* token of each special token */
    unsigned long long first[4];         /* bits of the first bytes of the special tokens */
    /* One bit for each token, set when the type of the token is the control type. The
     * detokenizer of a reply reads this bit, because the bytes of a control token are not
     * text of the reply. The words are (tokens + 31) / 32. */
    const unsigned int *control;
    unsigned int control_words;
} aotx_text_vocab;

extern __device__ aotx_text_vocab aotx_text_vocab_table;

/* Report whether a token is of the control type. A table with no control bits gives zero
 * for every token, so a caller that runs before the build sees no control token. */
__device__ __forceinline__ int aotx_text_is_control(const aotx_text_vocab *vocab,
                                                    unsigned int token)
{
    if (vocab->control == 0 || token >= vocab->tokens) {
        return 0;
    }
    unsigned int word = token >> 5;
    if (word >= vocab->control_words) {
        return 0;
    }
    return ((vocab->control[word] >> (token & 31u)) & 1u) != 0u;
}

/* One batch of sequences. A sequence is one line of input or one prompt. The bytes of every
 * sequence are in one run, so the position of a piece in that run names the piece. */
typedef struct aotx_text_batch {
    const unsigned char *bytes;   /* the bytes of every sequence */
    const unsigned int *start;    /* first byte of each sequence in the run */
    const unsigned int *length;   /* bytes of each sequence */
    unsigned int count;           /* sequences of the batch */
} aotx_text_batch;

/* The pieces that the pre-tokenizer gives. A piece is one match of the pattern, or one
 * special token. The lists of a sequence start at the sequence number times the stride. */
typedef struct aotx_text_pieces {
    unsigned int *start;    /* first byte of the piece in the byte run */
    unsigned int *length;   /* bytes of the piece */
    unsigned int *token;    /* token of a special piece, or AOTX_TEXT_NONE */
    unsigned int *count;    /* pieces of each sequence */
    unsigned int *work;     /* piece slots to merge, two entries for each piece */
    unsigned int *works;    /* pieces the work list holds */
    unsigned int stride;    /* piece slots each sequence holds */
} aotx_text_pieces;

/* The tokens that the merge step and the gather step give. */
typedef struct aotx_text_tokens {
    unsigned int *id;       /* tokens of each sequence */
    unsigned int *count;    /* tokens of each sequence */
    unsigned int *chunk;    /* tokens of each piece, by piece slot */
    unsigned int *scratch;  /* tokens of a piece, at the position of the piece in the run */
    unsigned char *merge;   /* AOTX_TEXT_WARP_BYTES for each warp of the merge launch */
    unsigned int warps;     /* warps of the merge launch, which sizes merge */
    unsigned int stride;    /* token slots each sequence holds */
} aotx_text_tokens;

/* What the host glue gives the vocabulary build. The arrays come from the model file and
 * cross to the device without change. The family is a row of the family table, which the
 * host glue finds from the name in the file with aotx_text_family_find. */
typedef struct aotx_text_source {
    const unsigned char *token_bytes;
    const unsigned long long *token_at;
    unsigned long long tokens;
    const unsigned char *merge_bytes;
    const unsigned long long *merge_at;
    unsigned long long merges;
    const int *token_type;
    unsigned int family;
} aotx_text_source;

/* Find the row of the family table which a value of tokenizer.ggml.pre names. The return
 * is zero when a row has that name, and the row goes in row. */
int aotx_text_family_find(const char *name, unsigned long long length, unsigned int *row);

/* What the host glue holds so it can give the memory of the vocabulary back. */
typedef struct aotx_text_store {
    void *block[8];
    unsigned int blocks;
    unsigned long long bytes;
} aotx_text_store;

/* Build the vocabulary on the device from the arrays of a model file. The return is zero
 * when the tables are ready. */
int aotx_text_vocab_build(const aotx_text_source *source, aotx_text_store *store);

/* Give the memory of the vocabulary back. */
void aotx_text_vocab_release(aotx_text_store *store);

/* Compare the tokens of another model file with the tokens of the table that is built. The
 * count of tokens which differ goes in wrong. One table serves every model of a set, so a
 * count above zero means the set does not share one vocabulary. The return is zero when the
 * comparison ran. */
int aotx_text_vocab_prefix(const unsigned char *bytes, const unsigned long long *at,
                           unsigned long long tokens, unsigned int *wrong);

/* The character classes. The tables come from a tool, in unicode_tables.cu. */
__device__ int aotx_text_letter(unsigned int point);
__device__ int aotx_text_number(unsigned int point);
__device__ int aotx_text_space(unsigned int point);

/* The classes of the ASCII range are a small set, so a test of that range needs no table.
 * Text of this system is mostly ASCII, and the fast path keeps the state machine short. */
__device__ __forceinline__ int aotx_text_is_letter(unsigned int point)
{
    if (point < 0x80u) {
        return (point >= 'A' && point <= 'Z') || (point >= 'a' && point <= 'z');
    }
    return aotx_text_letter(point);
}

__device__ __forceinline__ int aotx_text_is_number(unsigned int point)
{
    if (point < 0x80u) {
        return point >= '0' && point <= '9';
    }
    return aotx_text_number(point);
}

__device__ __forceinline__ int aotx_text_is_space(unsigned int point)
{
    if (point < 0x80u) {
        return point == 0x20u || (point >= 0x09u && point <= 0x0Du);
    }
    return aotx_text_space(point);
}

/* Read one character of a byte run and give its code point. The return is the bytes the
 * character holds. A byte run that is not correct UTF-8 gives the replacement code point
 * and one byte, so a decode of any input makes progress. */
__device__ unsigned int aotx_text_decode(const unsigned char *bytes, unsigned int length,
                                         unsigned int at, unsigned int *point);

/* Write one code point as UTF-8. The return is the bytes written, from one to four. */
__device__ unsigned int aotx_text_encode(unsigned int point, unsigned char *out);

/* Digits of the largest unsigned value, which is 20 for 64 bits. */
#define AOTX_TEXT_DIGITS   20u

/* Write an unsigned value as decimal digits. The return is the bytes written.
 *
 * The two writers of whole numbers are in this header and not in a translation unit of
 * their own. A call across translation units is a call of the application binary interface,
 * and the caller keeps a stack frame for it. Every panel and every command line writes
 * numbers, so that frame appeared in each of them. */
__device__ __forceinline__ unsigned int aotx_text_utoa(unsigned long long value, char *out,
                                                       unsigned int max)
{
    char digits[AOTX_TEXT_DIGITS];
    unsigned int count = 0u;
    do {
        digits[count] = (char)('0' + (unsigned int)(value % 10ull));
        value /= 10ull;
        count += 1u;
    } while (value != 0ull && count < AOTX_TEXT_DIGITS);
    unsigned int written = 0u;
    while (count > 0u && written < max) {
        count -= 1u;
        out[written] = digits[count];
        written += 1u;
    }
    return written;
}

/* Write a signed value as decimal digits, with a minus sign for a value below zero. */
__device__ __forceinline__ unsigned int aotx_text_itoa(long long value, char *out,
                                                       unsigned int max)
{
    if (value >= 0ll) {
        return aotx_text_utoa((unsigned long long)value, out, max);
    }
    if (max == 0u) {
        return 0u;
    }
    out[0] = '-';
    return 1u + aotx_text_utoa((unsigned long long)(-value), out + 1, max - 1u);
}

/* Write a value with a point and a given count of digits after the point. */
__device__ unsigned int aotx_text_ftoa(double value, unsigned int after, char *out,
                                       unsigned int max);

/* The byte to code point map of byte level tokens. A byte that a text editor can show keeps
 * its own code point. Every other byte takes a code point above 0xFF, so a token string is
 * correct UTF-8 and holds no space and no control character. */
__device__ __forceinline__ unsigned int aotx_text_byte_point(unsigned int byte)
{
    if ((byte >= 0x21u && byte <= 0x7Eu) || (byte >= 0xA1u && byte <= 0xACu)
        || (byte >= 0xAEu && byte <= 0xFFu)) {
        return byte;
    }
    if (byte <= 0x20u) {
        return 0x100u + byte;
    }
    if (byte <= 0xA0u) {
        return 0x100u + 33u + (byte - 0x7Fu);
    }
    return 0x100u + 67u;
}

/* The other way: a code point of a token string gives the byte it stands for. The return is
 * 0x100 when the code point stands for no byte, and the caller then keeps the code point. */
__device__ __forceinline__ unsigned int aotx_text_point_byte(unsigned int point)
{
    if ((point >= 0x21u && point <= 0x7Eu) || (point >= 0xA1u && point <= 0xACu)
        || (point >= 0xAEu && point <= 0xFFu)) {
        return point;
    }
    if (point < 0x100u || point > 0x143u) {
        return 0x100u;
    }
    unsigned int at = point - 0x100u;
    if (at <= 0x20u) {
        return at;
    }
    if (at <= 66u) {
        return 0x7Fu + (at - 33u);
    }
    return 0xADu;
}

/* Mix the bytes of a token into one value. The mix is FNV-1a, which is short and spreads
 * the low bits of short strings well. */
__device__ __forceinline__ unsigned long long aotx_text_hash(const unsigned char *bytes,
                                                             unsigned int length)
{
    unsigned long long mix = 14695981039346656037ull;
    for (unsigned int i = 0u; i < length; ++i) {
        mix ^= (unsigned long long)bytes[i];
        mix *= 1099511628211ull;
    }
    return mix;
}

/* Mix a pair of tokens into one value for the pair table. */
__device__ __forceinline__ unsigned long long aotx_text_pair_hash(unsigned long long key)
{
    unsigned long long mix = key;
    mix ^= mix >> 33;
    mix *= 0xFF51AFD7ED558CCDull;
    mix ^= mix >> 33;
    mix *= 0xC4CEB9FE1A85EC53ull;
    mix ^= mix >> 33;
    return mix;
}

/* Report whether a token string and a byte run are the same. */
__device__ __forceinline__ int aotx_text_same(const aotx_text_vocab *vocab, unsigned int token,
                                              const unsigned char *bytes, unsigned int length)
{
    unsigned long long first = vocab->token_at[token];
    if (vocab->token_at[token + 1u] - first != (unsigned long long)length) {
        return 0;
    }
    const unsigned char *text = vocab->token_bytes + first;
    for (unsigned int i = 0u; i < length; ++i) {
        if (text[i] != bytes[i]) {
            return 0;
        }
    }
    return 1;
}

/* Find the token of a byte run. The return is AOTX_TEXT_NONE when the vocabulary has no
 * such token. */
__device__ __forceinline__ unsigned int aotx_text_find_token(const aotx_text_vocab *vocab,
                                                             const unsigned char *bytes,
                                                             unsigned int length)
{
    unsigned int mask = vocab->slots - 1u;
    unsigned int at = (unsigned int)aotx_text_hash(bytes, length) & mask;
    for (unsigned int step = 0u; step <= mask; ++step) {
        unsigned int token = vocab->slot[at];
        if (token == AOTX_TEXT_NONE) {
            return AOTX_TEXT_NONE;
        }
        if (aotx_text_same(vocab, token, bytes, length)) {
            return token;
        }
        at = (at + 1u) & mask;
    }
    return AOTX_TEXT_NONE;
}

/* Find the rank of a pair of tokens. The return is AOTX_TEXT_NONE when no merge joins them.
 * A low rank merges before a high rank. */
__device__ __forceinline__ unsigned int aotx_text_find_rank(const aotx_text_vocab *vocab,
                                                            unsigned int left,
                                                            unsigned int right)
{
    unsigned long long key = ((unsigned long long)left << 32) | (unsigned long long)right;
    unsigned int mask = vocab->pairs - 1u;
    unsigned int at = (unsigned int)aotx_text_pair_hash(key) & mask;
    for (unsigned int step = 0u; step <= mask; ++step) {
        unsigned long long got = vocab->pair_key[at];
        if (got == 0xFFFFFFFFFFFFFFFFull) {
            return AOTX_TEXT_NONE;
        }
        if (got == key) {
            return vocab->pair_rank[at];
        }
        at = (at + 1u) & mask;
    }
    return AOTX_TEXT_NONE;
}

/* The kernels that build the vocabulary. The host glue gives the arrays of the model file
 * and then launches these three, in this order. The report holds four counts: tokens put
 * in, merges put in, merges the table refused, and special tokens. */
__global__ void aotx_text_build_tokens(unsigned int *report);
__global__ void aotx_text_build_specials(const int *type, unsigned int *report);
__global__ void aotx_text_build_pairs(const unsigned char *bytes,
                                      const unsigned long long *at_table, unsigned int merges,
                                      unsigned int *report);

/* Compare the tokens of another model file with the tokens of the table, one thread for
 * each token. The report counts the tokens which differ. One table serves every model of
 * the set, so a token of a smaller vocabulary must be the token of the same id here. */
__global__ void aotx_text_check_prefix(const unsigned char *bytes,
                                       const unsigned long long *at_table,
                                       unsigned int tokens, unsigned int *report);

/* The kernels. Every one takes a batch of sequences. */

/* Read the byte runs and give the code points, one thread for each byte. A byte that
 * continues a character gives AOTX_TEXT_NONE. */
__global__ void aotx_text_decode_run(aotx_text_batch batch, unsigned int *point);

/* Take the code points of each sequence in order and drop the marks, one thread for each
 * sequence. The count of each sequence goes in count. */
__global__ void aotx_text_gather_points(aotx_text_batch batch, const unsigned int *point,
                                        unsigned int *out, unsigned int *count,
                                        unsigned int stride);

/* Read each sequence and write it back with every byte run which is not a character as the
 * replacement character. The pre-tokenizer and the merge step take the run this gives. A
 * byte which is not part of a character becomes three bytes which are, as the reference
 * tokenizer does. Sequence i goes at i times the limit, one thread for each sequence. */
__global__ void aotx_text_clean(aotx_text_batch batch, unsigned char *bytes,
                                unsigned int *start, unsigned int *length,
                                unsigned int limit);

/* Write the code points of each sequence back as bytes, one thread for each sequence. */
__global__ void aotx_text_encode_run(const unsigned int *points, const unsigned int *count,
                                     unsigned int sequences, unsigned int stride,
                                     unsigned char *bytes, unsigned int *length,
                                     unsigned int limit);

/* Cut each sequence into pieces, one thread for each sequence. The pattern comes from the
 * pattern row of the vocabulary table. The state machine of each pattern takes the first
 * alternative that matches. The byte run comes from aotx_text_clean, so every byte of it is
 * part of a character. */
__global__ void aotx_text_pretok(aotx_text_batch batch, aotx_text_pieces pieces);

/* Merge the byte pairs of each piece, one warp for each piece. */
__global__ void aotx_text_merge(aotx_text_batch batch, aotx_text_pieces pieces,
                                aotx_text_tokens tokens);

/* Put the tokens of the pieces of a sequence in order, one thread for each sequence. */
__global__ void aotx_text_gather(aotx_text_batch batch, aotx_text_pieces pieces,
                                 aotx_text_tokens tokens);

/* Write the bytes of the tokens of each sequence, one thread for each sequence. */
__global__ void aotx_text_detok(const unsigned int *id, const unsigned int *count,
                                unsigned int sequences, unsigned int stride,
                                unsigned char *bytes, unsigned int *length,
                                unsigned int limit);

#endif
