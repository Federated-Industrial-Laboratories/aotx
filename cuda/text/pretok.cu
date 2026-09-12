/* Purpose: Cut a batch of sequences into the pieces that the merge step takes.
 * Owns: Nothing; the piece lists come from the caller.
 * Launch shape: One thread for each sequence.
 * Lifetime: One launch. */
#include "text/text.cuh"

/* The pre-tokenizer runs one of the patterns of the pattern table, by the pattern row of
 * the vocabulary. Each pattern is one function, which the switch in aotx_text_match calls.
 * The state machine of a pattern takes the first alternative which matches at the position.
 * Each part of an alternative takes as many characters as it can.
 *
 * The pattern of the qwen2 row has seven alternatives:
 *
 * 1 A contraction, in small letters or in capital letters.
 * 2 An optional character which is not a newline, a letter or a number, then letters.
 * 3 One number character.
 *
 * 4 An optional space, then characters which are not space, letter or number, then
 *   newline characters.
 * 5 Space characters which end with newline characters.
 * 6 Space characters which a character that is not a space does not follow.
 * 7 Space characters.
 *
 * The pattern of the llama3 row differs in alternative 3 only. It takes one, two or three
 * number characters. A run of numbers is therefore cut in groups of three from the left.
 *
 * The qwen35 row includes Unicode marks in letter runs and excludes them from punctuation.
 * A leading mark stays in the letter run, including when it is the only character.
 *
 * The pattern of the gpt2 row has six alternatives, and the shape differs:
 *
 * 1 A contraction, in small letters only.
 * 2 An optional space, then letters.
 * 3 An optional space, then numbers.
 * 4 An optional space, then characters which are not space, letter or number.
 * 5 Space characters which a character that is not a space does not follow.
 * 6 Space characters.
 *
 * The lookahead of the run of spaces gives back the last character of the run when a
 * character which is not a space follows it.
 *
 * The functions are inline and the dispatch is a switch, so the state machine kernel holds
 * no indirect call. An indirect call would leave the registers of every function live at
 * the call, and the kernel is one long machine already. */

/* Give the bytes and the code point of the character at a position. */
static __device__ __forceinline__ unsigned int aotx_text_at(const unsigned char *bytes,
                                                            unsigned int end, unsigned int at,
                                                            unsigned int *point)
{
    return aotx_text_decode(bytes, end, at, point);
}

/* Make a capital letter small, for the contractions. */
static __device__ __forceinline__ unsigned int aotx_text_small(unsigned int byte)
{
    return (byte >= 'A' && byte <= 'Z') ? byte + 32u : byte;
}

/* Give the end of the run of space characters which starts at a position, and the start of
 * the last character of that run. */
static __device__ void aotx_text_space_run(const unsigned char *bytes, unsigned int end,
                                           unsigned int at, unsigned int *stop,
                                           unsigned int *last)
{
    unsigned int walk = at;
    unsigned int back = at;
    while (walk < end) {
        unsigned int point = 0u;
        unsigned int span = aotx_text_at(bytes, end, walk, &point);
        if (!aotx_text_is_space(point)) {
            break;
        }
        back = walk;
        walk += span;
    }
    *stop = walk;
    *last = back;
}

/* Give the end of the letter run, with Unicode marks when the pattern includes them. */
static __device__ __forceinline__ unsigned int aotx_text_letter_run(const unsigned char *bytes,
                                                                    unsigned int end,
                                                                    unsigned int at,
                                                                    int marks)
{
    while (at < end) {
        unsigned int point = 0u;
        unsigned int step = aotx_text_at(bytes, end, at, &point);
        if (!aotx_text_is_letter(point) && !(marks && aotx_text_is_mark(point))) {
            break;
        }
        at += step;
    }
    return at;
}

/* Give the end of the run of number characters which starts at a position, and stop after
 * a count of them. A count of zero takes the whole run. */
static __device__ __forceinline__ unsigned int aotx_text_number_run(const unsigned char *bytes,
                                                                    unsigned int end,
                                                                    unsigned int at,
                                                                    unsigned int most)
{
    unsigned int taken = 0u;
    while (at < end && (most == 0u || taken < most)) {
        unsigned int point = 0u;
        unsigned int step = aotx_text_at(bytes, end, at, &point);
        if (!aotx_text_is_number(point)) {
            break;
        }
        at += step;
        taken += 1u;
    }
    return at;
}

/* Give the end of the punctuation run, without Unicode marks when the pattern excludes them. */
static __device__ __forceinline__ unsigned int aotx_text_mark_run(const unsigned char *bytes,
                                                                  unsigned int end,
                                                                  unsigned int at,
                                                                  int marks)
{
    while (at < end) {
        unsigned int point = 0u;
        unsigned int step = aotx_text_at(bytes, end, at, &point);
        if (aotx_text_is_space(point) || aotx_text_is_letter(point)
            || aotx_text_is_number(point) || (marks && aotx_text_is_mark(point))) {
            break;
        }
        at += step;
    }
    return at;
}

/* Match a contraction at a position. The return is its bytes, or zero. A fold of one
 * takes the letters in capital form as well. */
static __device__ __forceinline__ unsigned int aotx_text_contraction(const unsigned char *bytes,
                                                                     unsigned int at,
                                                                     unsigned int end,
                                                                     int fold)
{
    if (bytes[at] != '\'' || at + 1u >= end) {
        return 0u;
    }
    unsigned int one = fold ? aotx_text_small(bytes[at + 1u]) : bytes[at + 1u];
    if (one == 's' || one == 't' || one == 'm' || one == 'd') {
        return 2u;
    }
    if (at + 2u < end) {
        unsigned int two = fold ? aotx_text_small(bytes[at + 2u]) : bytes[at + 2u];
        if ((one == 'r' || one == 'v') && two == 'e') {
            return 3u;
        }
        if (one == 'l' && two == 'l') {
            return 3u;
        }
    }
    return 0u;
}

/* Match the run of space characters at a position, with the three space alternatives. The
 * position holds a space character, because the alternatives before this one take every
 * other character. A newline rule of one takes the alternative which ends with newline
 * characters first. */
static __device__ __forceinline__ unsigned int aotx_text_spaces(const unsigned char *bytes,
                                                                unsigned int at,
                                                                unsigned int end,
                                                                unsigned int span,
                                                                int newline_rule)
{
    unsigned int stop = at;
    unsigned int last = at;
    aotx_text_space_run(bytes, end, at, &stop, &last);
    if (stop == at) {
        return span;
    }

    /* The run ends with newline characters. The first part of the alternative gives back
     * characters until the last part can match. The match therefore ends after the last
     * newline character of the run. */
    if (newline_rule) {
        unsigned int walk = at;
        unsigned int newline = at;
        int found = 0;
        while (walk < stop) {
            unsigned int point = 0u;
            unsigned int step = aotx_text_at(bytes, stop, walk, &point);
            if (point == '\r' || point == '\n') {
                newline = walk;
                found = 1;
            }
            walk += step;
        }
        if (found) {
            return newline + 1u - at;
        }
    }

    /* The run which a character that is not a space does not follow. The run takes as many
     * characters as it can and then gives back the last one. The match therefore ends
     * before the last character of the run when a character that is not a space follows. */
    if (stop == end) {
        return stop - at;
    }
    if (last > at) {
        return last - at;
    }

    /* The run of space characters. */
    return stop - at;
}

/* Match a grouped-number pattern once. The mark flag extends its letter class.
 * The return is the bytes of the match, which is never zero before the end. */
static __device__ __forceinline__ unsigned int aotx_text_match_grouped(const unsigned char *bytes,
                                                                       unsigned int at,
                                                                       unsigned int end,
                                                                       unsigned int numbers,
                                                                       int marks)
{
    unsigned int first = 0u;
    unsigned int span = aotx_text_at(bytes, end, at, &first);

    /* 1: a contraction. The letters after the mark may be capital or small. */
    unsigned int held = aotx_text_contraction(bytes, at, end, 1);
    if (held != 0u) {
        return held;
    }

    /* 2: an optional character, then letters. Keep a leading Unicode mark in the run
     * when marks are letters. This also takes a lone mark after the optional part gives back. */
    {
        unsigned int from = at;
        if (first != '\r' && first != '\n' && !aotx_text_is_letter(first)
            && !aotx_text_is_number(first) && !(marks && aotx_text_is_mark(first))) {
            from = at + span;
        }
        unsigned int walk = aotx_text_letter_run(bytes, end, from, marks);
        if (walk > from) {
            return walk - at;
        }
    }

    /* 3: number characters, up to the count of the pattern. */
    if (aotx_text_is_number(first)) {
        return aotx_text_number_run(bytes, end, at, numbers) - at;
    }

    /* 4: an optional space, then characters which are not space, letter or number, then
     * newline characters. */
    {
        unsigned int from = (first == ' ') ? at + 1u : at;
        unsigned int walk = aotx_text_mark_run(bytes, end, from, marks);
        if (walk > from) {
            while (walk < end && (bytes[walk] == '\r' || bytes[walk] == '\n')) {
                walk += 1u;
            }
            return walk - at;
        }
    }

    /* 5, 6, 7: the run of space characters. */
    return aotx_text_spaces(bytes, at, end, span, 1);
}

static __device__ __forceinline__ unsigned int aotx_text_match_qwen2(const unsigned char *bytes,
                                                                     unsigned int at,
                                                                     unsigned int end)
{
    return aotx_text_match_grouped(bytes, at, end, 1u, 0);
}

static __device__ __forceinline__ unsigned int aotx_text_match_llama3(const unsigned char *bytes,
                                                                      unsigned int at,
                                                                      unsigned int end)
{
    return aotx_text_match_grouped(bytes, at, end, 3u, 0);
}

static __device__ __forceinline__ unsigned int aotx_text_match_qwen35(const unsigned char *bytes,
                                                                      unsigned int at,
                                                                      unsigned int end)
{
    return aotx_text_match_grouped(bytes, at, end, 1u, 1);
}

/* Match the pattern of the gpt2 row once at a position. */
static __device__ __forceinline__ unsigned int aotx_text_match_gpt2(const unsigned char *bytes,
                                                                    unsigned int at,
                                                                    unsigned int end)
{
    unsigned int first = 0u;
    unsigned int span = aotx_text_at(bytes, end, at, &first);

    /* 1: a contraction in small letters. */
    unsigned int held = aotx_text_contraction(bytes, at, end, 0);
    if (held != 0u) {
        return held;
    }

    /* 2, 3, 4: an optional space, then letters, or numbers, or characters which are not
     * space, letter or number. The three runs start at the same place. */
    unsigned int from = (first == ' ') ? at + 1u : at;
    unsigned int walk = aotx_text_letter_run(bytes, end, from, 0);
    if (walk == from) {
        walk = aotx_text_number_run(bytes, end, from, 0u);
    }
    if (walk == from) {
        walk = aotx_text_mark_run(bytes, end, from, 0);
    }
    if (walk > from) {
        return walk - at;
    }

    /* 5, 6: the run of space characters, with no newline alternative. */
    return aotx_text_spaces(bytes, at, end, span, 0);
}

/* Match the pattern of a row once at a position. */
static __device__ __forceinline__ unsigned int aotx_text_match(unsigned int pattern,
                                                               const unsigned char *bytes,
                                                               unsigned int at,
                                                               unsigned int end)
{
    switch (pattern) {
#define AOTX_TEXT_PATTERN_CASE(symbol, function) \
    case symbol: \
        return function(bytes, at, end);
    AOTX_TEXT_PATTERN_TABLE(AOTX_TEXT_PATTERN_CASE)
#undef AOTX_TEXT_PATTERN_CASE
    default:
        return aotx_text_match_qwen2(bytes, at, end);
    }
}

/* Give the bytes of the special token which matches at a position, and its token. The
 * return is zero when no special token matches. The longest match wins. */
static __device__ unsigned int aotx_text_special(const aotx_text_vocab *vocab,
                                                 const unsigned char *bytes, unsigned int at,
                                                 unsigned int end, unsigned int *token)
{
    unsigned int byte = bytes[at];
    if (((vocab->first[byte >> 6] >> (byte & 63u)) & 1ull) == 0ull) {
        return 0u;
    }
    /* The count is bounded by the array, because the build adds to the count before it
     * tests the bound. The host glue refuses a table whose count went past it. */
    unsigned int specials = vocab->specials;
    if (specials > vocab->tokens) {
        specials = vocab->tokens;
    }
    unsigned int best = 0u;
    for (unsigned int i = 0u; i < specials; ++i) {
        unsigned int one = vocab->special[i];
        unsigned long long from = vocab->token_at[one];
        unsigned int length = (unsigned int)(vocab->token_at[one + 1u] - from);
        if (length <= best || at + length > end) {
            continue;
        }
        const unsigned char *text = vocab->token_bytes + from;
        unsigned int i2 = 0u;
        while (i2 < length && text[i2] == bytes[at + i2]) {
            i2 += 1u;
        }
        if (i2 == length) {
            best = length;
            *token = one;
        }
    }
    return best;
}

/* Cut a match which is longer than one chunk at a character boundary. */
static __device__ unsigned int aotx_text_bound(const unsigned char *bytes, unsigned int at,
                                               unsigned int length)
{
    if (length <= AOTX_TEXT_CHUNK_BYTES) {
        return length;
    }
    unsigned int walk = 0u;
    while (walk < length) {
        unsigned int point = 0u;
        unsigned int step = aotx_text_decode(bytes, at + length, at + walk, &point);
        if (walk + step > AOTX_TEXT_CHUNK_BYTES) {
            break;
        }
        walk += step;
    }
    return walk;
}

__global__ void aotx_text_pretok(aotx_text_batch batch, aotx_text_pieces pieces)
{
    unsigned int sequence = blockIdx.x * blockDim.x + threadIdx.x;
    if (sequence >= batch.count) {
        return;
    }
    const aotx_text_vocab *vocab = &aotx_text_vocab_table;
    unsigned int start = batch.start[sequence];
    unsigned int end = start + batch.length[sequence];
    unsigned int slot = sequence * pieces.stride;
    unsigned int count = 0u;
    unsigned int at = start;
    while (at < end && count < pieces.stride) {
        /* A special token is a whole match, and the pattern does not read across it. The
         * search gives the part of the sequence which the pattern reads. */
        unsigned int stop = at;
        unsigned int token = AOTX_TEXT_NONE;
        unsigned int special = 0u;
        while (stop < end) {
            special = aotx_text_special(vocab, batch.bytes, stop, end, &token);
            if (special != 0u) {
                break;
            }
            stop += 1u;
        }
        while (at < stop && count < pieces.stride) {
            unsigned int length = aotx_text_match(vocab->pattern, batch.bytes, at, stop);
            length = aotx_text_bound(batch.bytes, at, length);
            pieces.start[slot + count] = at;
            pieces.length[slot + count] = length;
            pieces.token[slot + count] = AOTX_TEXT_NONE;
            count += 1u;
            at += length;
        }
        if (special != 0u && count < pieces.stride) {
            pieces.start[slot + count] = stop;
            pieces.length[slot + count] = special;
            pieces.token[slot + count] = token;
            count += 1u;
            at = stop + special;
        } else if (special == 0u) {
            at = stop;
        }
    }
    pieces.count[sequence] = count;
    /* The work list holds every piece of the batch, so the merge step gives one warp to
     * each piece and reads no empty slot. */
    unsigned int base = atomicAdd(pieces.works, count);
    for (unsigned int i = 0u; i < count; ++i) {
        pieces.work[base + i] = slot + i;
    }
}
