/* Purpose: Join the byte pairs of each piece into tokens, by rank order.
 * Owns: Nothing; the token lists and the merge memory come from the caller.
 * Launch shape: One warp for each piece, and one thread for each sequence in the gather.
 * Lifetime: One launch. */
#include "text/text.cuh"

/* The merge step is byte level. The piece becomes one symbol for each byte, and the byte
 * to code point map gives the text of each symbol. The pair with the lowest rank joins
 * first. A pair which the table does not name has no rank and never joins.
 *
 * The symbols of a piece are one run of bytes. A join makes the left symbol longer and
 * drops the right one, so no text moves.
 *
 * The lanes of a warp read the pairs of one piece together. The warp then takes the lowest
 * rank of them. Two pairs with the same rank take the one on the left. That order is the
 * order a single thread with a queue of pairs would take.
 *
 * The text of a chunk is the shared memory of the warp. The lists of the chunk are device
 * memory, one block of AOTX_TEXT_WARP_BYTES for each warp. Shared memory holds 4,096
 * symbols of lists for one warp of a block only.
 *
 * The rank of each pair is held and not read again. A join changes two pairs only. They
 * are the pair the left symbol now makes and the pair the symbol before it makes. The step
 * which looks for the lowest rank reads the held ranks. It makes two searches of the table
 * for each join, and not one search for each pair of the chunk. */

/* The mark of a symbol which has no symbol after it, and the mark of a symbol which a join
 * dropped. Both are above the largest symbol number, which is AOTX_TEXT_CHUNK_BYTES. */
#define AOTX_TEXT_LAST   0xFFFFu
#define AOTX_TEXT_DEAD   0xFFFEu

/* Give the bytes of the text of a symbol. The symbols hold one run of bytes in order, so
 * the text of a symbol ends where the text of the symbol after it starts. */
static __device__ __forceinline__ unsigned int aotx_text_span(const unsigned short *at,
                                                              const unsigned short *next,
                                                              unsigned int symbol,
                                                              unsigned int total)
{
    unsigned int right = next[symbol];
    unsigned int stop = (right == AOTX_TEXT_LAST) ? total : (unsigned int)at[right];
    return stop - (unsigned int)at[symbol];
}

__global__ void aotx_text_merge(aotx_text_batch batch, aotx_text_pieces pieces,
                                aotx_text_tokens tokens)
{
    __shared__ unsigned char text[AOTX_TEXT_WARPS][AOTX_TEXT_SYMBOL_BYTES];

    const aotx_text_vocab *vocab = &aotx_text_vocab_table;
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int warps = (blockDim.x >> 5) * gridDim.x;
    unsigned int first = (blockIdx.x * (blockDim.x >> 5)) + warp;
    unsigned int works = *pieces.works;
    /* The merge memory holds one block for each warp the caller counted. A launch with
     * more warps than that would leave the pieces of the warps above the count undone. The
     * walk over the pieces steps by the count and not by the warps of the launch. */
    if (warps > tokens.warps) {
        warps = tokens.warps;
    }
    if (first >= warps) {
        return;
    }
    unsigned char *own = tokens.merge + (unsigned long long)first * AOTX_TEXT_WARP_BYTES;
    unsigned short *at = (unsigned short *)own;
    unsigned short *next = (unsigned short *)(own + 2u * AOTX_TEXT_CHUNK_BYTES);
    unsigned short *prev = (unsigned short *)(own + 4u * AOTX_TEXT_CHUNK_BYTES);
    unsigned int *hold = (unsigned int *)(own + 6u * AOTX_TEXT_CHUNK_BYTES);
    unsigned int *rank = (unsigned int *)(own + 10u * AOTX_TEXT_CHUNK_BYTES);

    for (unsigned int work = first; work < works; work += warps) {
        unsigned int slot = pieces.work[work];
        unsigned int from = pieces.start[slot];
        unsigned int length = pieces.length[slot];
        unsigned int special = pieces.token[slot];
        if (special != AOTX_TEXT_NONE) {
            /* A special token is one whole token, and the merge step does not read it. */
            if (lane == 0u) {
                tokens.scratch[from] = special;
                tokens.chunk[slot] = 1u;
            }
            continue;
        }

        /* The byte to code point map gives one or two bytes for each byte of the piece.
         * The scan of the warp gives each byte the place of its text. */
        unsigned int total = 0u;
        for (unsigned int off = 0u; off < length; off += 32u) {
            unsigned int i = off + lane;
            unsigned int point = 0u;
            unsigned int bytes = 0u;
            if (i < length) {
                point = aotx_text_byte_point(batch.bytes[from + i]);
                bytes = (point < 0x80u) ? 1u : 2u;
            }
            unsigned int scan = bytes;
            for (unsigned int step = 1u; step < 32u; step <<= 1) {
                unsigned int got = __shfl_up_sync(0xFFFFFFFFu, scan, step);
                if (lane >= step) {
                    scan += got;
                }
            }
            unsigned int place = total + scan - bytes;
            if (i < length) {
                at[i] = (unsigned short)place;
                next[i] = (i + 1u < length) ? (unsigned short)(i + 1u) : AOTX_TEXT_LAST;
                prev[i] = (i > 0u) ? (unsigned short)(i - 1u) : AOTX_TEXT_LAST;
                aotx_text_encode(point, &text[warp][place]);
            }
            total += __shfl_sync(0xFFFFFFFFu, scan, 31);
        }
        __syncwarp();
        /* With the whole piece flag, a piece which is itself a token stands as that token,
         * and the merge step does not run on it. The lookup reads the mapped text of the
         * piece, which is the text the symbol lookups read. Every lane reads the same slot,
         * so every lane takes the same branch. */
        if (vocab->whole != 0u) {
            unsigned int token = aotx_text_find_token(vocab, &text[warp][0], total);
            if (token != AOTX_TEXT_NONE) {
                if (lane == 0u) {
                    tokens.scratch[from] = token;
                    tokens.chunk[slot] = 1u;
                }
                __syncwarp();
                continue;
            }
        }
        for (unsigned int i = lane; i < length; i += 32u) {
            hold[i] = aotx_text_find_token(vocab, &text[warp][at[i]],
                                           aotx_text_span(at, next, i, total));
        }
        __syncwarp();
        for (unsigned int i = lane; i < length; i += 32u) {
            rank[i] = (next[i] == AOTX_TEXT_LAST)
                    ? AOTX_TEXT_NONE
                    : aotx_text_find_rank(vocab, hold[i], hold[next[i]]);
        }
        __syncwarp();

        /* Join the pair with the lowest rank until no pair of the piece has a rank. */
        for (;;) {
            unsigned long long best = 0xFFFFFFFFFFFFFFFFull;
            for (unsigned int i = lane; i < length; i += 32u) {
                unsigned int held = rank[i];
                if (held == AOTX_TEXT_NONE) {
                    continue;
                }
                unsigned long long key = ((unsigned long long)held << 32)
                                       | (unsigned long long)i;
                if (key < best) {
                    best = key;
                }
            }
            for (unsigned int step = 16u; step > 0u; step >>= 1) {
                unsigned long long got = __shfl_down_sync(0xFFFFFFFFu, best, step);
                if (got < best) {
                    best = got;
                }
            }
            best = __shfl_sync(0xFFFFFFFFu, best, 0);
            if (best == 0xFFFFFFFFFFFFFFFFull) {
                break;
            }
            if (lane == 0u) {
                unsigned int left = (unsigned int)(best & 0xFFFFFFFFull);
                unsigned int right = next[left];
                unsigned int after = next[right];
                next[left] = (unsigned short)after;
                next[right] = AOTX_TEXT_DEAD;
                rank[right] = AOTX_TEXT_NONE;
                if (after != AOTX_TEXT_LAST) {
                    prev[after] = (unsigned short)left;
                }
                hold[left] = aotx_text_find_token(vocab, &text[warp][at[left]],
                                                  aotx_text_span(at, next, left, total));
                /* A join changes the pair the left symbol makes and the pair the symbol
                 * before it makes. No other pair of the chunk changes. */
                rank[left] = (after == AOTX_TEXT_LAST)
                           ? AOTX_TEXT_NONE
                           : aotx_text_find_rank(vocab, hold[left], hold[after]);
                unsigned int before = prev[left];
                if (before != AOTX_TEXT_LAST) {
                    rank[before] = aotx_text_find_rank(vocab, hold[before], hold[left]);
                }
            }
            __syncwarp();
        }

        /* The first symbol is always the left side of a join, so the walk starts there. */
        if (lane == 0u) {
            unsigned int count = 0u;
            unsigned int walk = 0u;
            while (walk != AOTX_TEXT_LAST && length != 0u) {
                tokens.scratch[from + count] = hold[walk];
                count += 1u;
                walk = next[walk];
            }
            tokens.chunk[slot] = count;
        }
        __syncwarp();
    }
}

__global__ void aotx_text_gather(aotx_text_batch batch, aotx_text_pieces pieces,
                                 aotx_text_tokens tokens)
{
    unsigned int sequence = blockIdx.x * blockDim.x + threadIdx.x;
    if (sequence >= batch.count) {
        return;
    }
    unsigned int slot = sequence * pieces.stride;
    unsigned int out = sequence * tokens.stride;
    unsigned int count = 0u;
    for (unsigned int piece = 0u; piece < pieces.count[sequence]; ++piece) {
        unsigned int from = pieces.start[slot + piece];
        unsigned int held = tokens.chunk[slot + piece];
        for (unsigned int i = 0u; i < held && count < tokens.stride; ++i) {
            tokens.id[out + count] = tokens.scratch[from + i];
            count += 1u;
        }
    }
    tokens.count[sequence] = count;
}
