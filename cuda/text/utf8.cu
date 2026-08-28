/* Purpose: Read and write UTF-8 characters of a batch of byte runs.
 * Owns: Nothing; every buffer comes from the caller.
 * Launch shape: One block row for each sequence, then one thread for each byte.
 * Lifetime: One launch. */
#include "text/text.cuh"

/* Report whether a byte continues a character. */
static __device__ __forceinline__ int aotx_text_tail(unsigned int byte)
{
    return (byte & 0xC0u) == 0x80u;
}

__device__ unsigned int aotx_text_decode(const unsigned char *bytes, unsigned int length,
                                         unsigned int at, unsigned int *point)
{
    unsigned int first = bytes[at];
    unsigned int left = length - at;
    if (first < 0x80u) {
        *point = first;
        return 1u;
    }
    /* The lead byte gives the length. A byte from 0x80 to 0xBF continues a character and is
     * not a lead byte, so it gives the replacement.
     *
     * This reader takes a form which is longer than the code point needs, and the code
     * points of the surrogate range. The reference tokenizer takes them, and the token ids
     * of the same bytes must be the same. A byte run of C0 AF therefore gives the code
     * point 002F, which is what the reference gives.
     *
     * A four byte form with a code point above 10FFFF is the one refusal which the
     * reference does not share. The reference stops the whole run there, because it cannot
     * write that code point back as bytes. The device gives the replacement instead, so one
     * bad byte run does not stop a batch of sequences. */
    if (first < 0xC0u) {
        *point = AOTX_TEXT_REPLACEMENT;
        return 1u;
    }
    if (first < 0xE0u && left >= 2u && aotx_text_tail(bytes[at + 1u])) {
        *point = ((first & 0x1Fu) << 6) | (bytes[at + 1u] & 0x3Fu);
        return 2u;
    }
    if (first >= 0xE0u && first < 0xF0u && left >= 3u && aotx_text_tail(bytes[at + 1u])
        && aotx_text_tail(bytes[at + 2u])) {
        *point = ((first & 0x0Fu) << 12) | ((bytes[at + 1u] & 0x3Fu) << 6)
               | (bytes[at + 2u] & 0x3Fu);
        return 3u;
    }
    if (first >= 0xF0u && first < 0xF8u && left >= 4u && aotx_text_tail(bytes[at + 1u])
        && aotx_text_tail(bytes[at + 2u]) && aotx_text_tail(bytes[at + 3u])) {
        unsigned int got = ((first & 0x07u) << 18) | ((bytes[at + 1u] & 0x3Fu) << 12)
                         | ((bytes[at + 2u] & 0x3Fu) << 6) | (bytes[at + 3u] & 0x3Fu);
        if (got <= 0x10FFFFu) {
            *point = got;
            return 4u;
        }
    }
    *point = AOTX_TEXT_REPLACEMENT;
    return 1u;
}

__device__ unsigned int aotx_text_encode(unsigned int point, unsigned char *out)
{
    if (point < 0x80u) {
        out[0] = (unsigned char)point;
        return 1u;
    }
    if (point < 0x800u) {
        out[0] = (unsigned char)(0xC0u | (point >> 6));
        out[1] = (unsigned char)(0x80u | (point & 0x3Fu));
        return 2u;
    }
    if (point < 0x10000u) {
        out[0] = (unsigned char)(0xE0u | (point >> 12));
        out[1] = (unsigned char)(0x80u | ((point >> 6) & 0x3Fu));
        out[2] = (unsigned char)(0x80u | (point & 0x3Fu));
        return 3u;
    }
    out[0] = (unsigned char)(0xF0u | (point >> 18));
    out[1] = (unsigned char)(0x80u | ((point >> 12) & 0x3Fu));
    out[2] = (unsigned char)(0x80u | ((point >> 6) & 0x3Fu));
    out[3] = (unsigned char)(0x80u | (point & 0x3Fu));
    return 4u;
}

/* Report whether a character which starts before this byte holds this byte. A character
 * holds four bytes at most, so the look back stops at three. This test lets one thread
 * read each byte without a walk from the start of the sequence. */
static __device__ int aotx_text_held(const unsigned char *bytes, unsigned int start,
                                     unsigned int end, unsigned int at)
{
    for (unsigned int back = 1u; back <= 3u && at >= start + back; ++back) {
        unsigned int point = 0u;
        unsigned int span = aotx_text_decode(bytes, end, at - back, &point);
        if (span > back) {
            return 1;
        }
    }
    return 0;
}

__global__ void aotx_text_decode_run(aotx_text_batch batch, unsigned int *point)
{
    unsigned int sequence = blockIdx.y;
    if (sequence >= batch.count) {
        return;
    }
    unsigned int start = batch.start[sequence];
    unsigned int end = start + batch.length[sequence];
    unsigned int step = blockDim.x * gridDim.x;
    for (unsigned int at = start + blockIdx.x * blockDim.x + threadIdx.x; at < end;
         at += step) {
        if (aotx_text_held(batch.bytes, start, end, at)) {
            point[at] = AOTX_TEXT_NONE;
        } else {
            unsigned int got = 0u;
            aotx_text_decode(batch.bytes, end, at, &got);
            point[at] = got;
        }
    }
}

__global__ void aotx_text_gather_points(aotx_text_batch batch, const unsigned int *point,
                                        unsigned int *out, unsigned int *count,
                                        unsigned int stride)
{
    unsigned int sequence = blockIdx.x * blockDim.x + threadIdx.x;
    if (sequence >= batch.count) {
        return;
    }
    unsigned int start = batch.start[sequence];
    unsigned int end = start + batch.length[sequence];
    unsigned int at = 0u;
    for (unsigned int byte = start; byte < end; ++byte) {
        unsigned int got = point[byte];
        if (got != AOTX_TEXT_NONE && at < stride) {
            out[sequence * stride + at] = got;
            at += 1u;
        }
    }
    count[sequence] = at;
}

__global__ void aotx_text_encode_run(const unsigned int *points, const unsigned int *count,
                                     unsigned int sequences, unsigned int stride,
                                     unsigned char *bytes, unsigned int *length,
                                     unsigned int limit)
{
    unsigned int sequence = blockIdx.x * blockDim.x + threadIdx.x;
    if (sequence >= sequences) {
        return;
    }
    unsigned char *out = bytes + (unsigned long long)sequence * limit;
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < count[sequence]; ++i) {
        if (at + 4u > limit) {
            break;
        }
        at += aotx_text_encode(points[sequence * stride + i], out + at);
    }
    length[sequence] = at;
}

__global__ void aotx_text_clean(aotx_text_batch batch, unsigned char *bytes,
                                unsigned int *start, unsigned int *length, unsigned int limit)
{
    unsigned int sequence = blockIdx.x * blockDim.x + threadIdx.x;
    if (sequence >= batch.count) {
        return;
    }
    unsigned int from = batch.start[sequence];
    unsigned int end = from + batch.length[sequence];
    unsigned char *out = bytes + (unsigned long long)sequence * limit;
    unsigned int at = 0u;
    for (unsigned int walk = from; walk < end && at + 4u <= limit; ) {
        unsigned int point = 0u;
        walk += aotx_text_decode(batch.bytes, end, walk, &point);
        at += aotx_text_encode(point, out + at);
    }
    start[sequence] = sequence * limit;
    length[sequence] = at;
}
