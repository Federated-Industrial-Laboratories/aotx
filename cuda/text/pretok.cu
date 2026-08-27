/* Purpose: Cut a batch of sequences into the pieces that the merge step takes.
 * Owns: Nothing; the piece lists come from the caller.
 * Launch shape: One thread for each sequence.
 * Lifetime: One launch. */
#include "text/text.cuh"

/* The pattern of this tokenizer family has seven alternatives. The state machine takes the
 * first alternative which matches at the position. Each part of an alternative takes as
 * many characters as it can.
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
 * The lookahead of alternative 6 gives back the last character of the run when a character
 * which is not a space follows it. */

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

/* Match the pattern once at a position. The return is the bytes of the match, which is
 * never zero while the position is before the end. */
static __device__ unsigned int aotx_text_match(const unsigned char *bytes, unsigned int at,
                                               unsigned int end)
{
    unsigned int first = 0u;
    unsigned int span = aotx_text_at(bytes, end, at, &first);

    /* 1: a contraction. The letters after the mark may be capital or small. */
    if (first == '\'' && at + 1u < end) {
        unsigned int one = aotx_text_small(bytes[at + 1u]);
        if (one == 's' || one == 't' || one == 'm' || one == 'd') {
            return 2u;
        }
        if (at + 2u < end) {
            unsigned int two = aotx_text_small(bytes[at + 2u]);
            if ((one == 'r' || one == 'v') && two == 'e') {
                return 3u;
            }
            if (one == 'l' && two == 'l') {
                return 3u;
            }
        }
    }

    /* 2: an optional character, then one letter or more. The optional character is any
     * character which is not a newline, a letter or a number. When no letter follows it,
     * the alternative fails, because the optional character is not a letter either. */
    {
        unsigned int walk = at;
        if (first != '\r' && first != '\n' && !aotx_text_is_letter(first)
            && !aotx_text_is_number(first)) {
            walk = at + span;
        }
        unsigned int from = walk;
        while (walk < end) {
            unsigned int point = 0u;
            unsigned int step = aotx_text_at(bytes, end, walk, &point);
            if (!aotx_text_is_letter(point)) {
                break;
            }
            walk += step;
        }
        if (walk > from) {
            return walk - at;
        }
    }

    /* 3: one number character. */
    if (aotx_text_is_number(first)) {
        return span;
    }

    /* 4: an optional space, then characters which are not space, letter or number, then
     * newline characters. */
    {
        unsigned int walk = (first == ' ') ? at + 1u : at;
        unsigned int from = walk;
        while (walk < end) {
            unsigned int point = 0u;
            unsigned int step = aotx_text_at(bytes, end, walk, &point);
            if (aotx_text_is_space(point) || aotx_text_is_letter(point)
                || aotx_text_is_number(point)) {
                break;
            }
            walk += step;
        }
        if (walk > from) {
            while (walk < end && (bytes[walk] == '\r' || bytes[walk] == '\n')) {
                walk += 1u;
            }
            return walk - at;
        }
    }

    /* The position holds a space character now, because the alternatives above take every
     * other character. The last three alternatives read the run of space characters. */
    unsigned int stop = at;
    unsigned int last = at;
    aotx_text_space_run(bytes, end, at, &stop, &last);
    if (stop == at) {
        return span;
    }

    /* 5: the run ends with newline characters. The first part of the alternative gives
     * back characters until the last part can match. The match therefore ends after the
     * last newline character of the run. */
    {
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

    /* 6: the run which a character that is not a space does not follow. The run takes as
     * many characters as it can and then gives back the last one. The match therefore ends
     * before the last character of the run when a character that is not a space follows. */
    if (stop == end) {
        return stop - at;
    }
    if (last > at) {
        return last - at;
    }

    /* 7: the run of space characters. */
    return stop - at;
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
    unsigned int best = 0u;
    for (unsigned int i = 0u; i < vocab->specials; ++i) {
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
            unsigned int length = aotx_text_match(batch.bytes, at, stop);
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
