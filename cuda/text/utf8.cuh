/* Purpose: Read one UTF-8 code point with the shared tokenizer byte rules.
 * Owns: Scalar character decoding; no buffer or allocation.
 * Launch shape: Called within each byte or vocabulary thread.
 * Lifetime: One code point read. */
#ifndef AOTX_TEXT_UTF8_CUH
#define AOTX_TEXT_UTF8_CUH
#include "text/text.cuh"

/* Report whether a byte continues a character. */
static __device__ __forceinline__ int aotx_text_tail(unsigned int byte)
{
    return (byte & 0xC0u) == 0x80u;
}

__device__ __forceinline__ unsigned int aotx_text_decode_point(const unsigned char *bytes, unsigned int length,
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

#endif
