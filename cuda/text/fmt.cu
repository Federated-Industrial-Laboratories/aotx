/* Purpose: Write numbers as text on the device.
 * Owns: Nothing; the caller gives the buffer.
 * Launch shape: One thread for each value; no kernels of its own.
 * Lifetime: One call. */
#include "text/text.cuh"

/* The device has no library for numbers. The seam rules refuse a print call in a device
 * file. Every panel and every record line therefore comes through the three writers. */

/* Digits after the point that a value may ask for. */
#define AOTX_TEXT_AFTER    9u

/* The two writers of whole numbers are in text.cuh, where every caller takes them inline.
 * The writer of values with a point stays here. It is longer, and no hot path calls it
 * more than one time for a line. */

__device__ unsigned int aotx_text_ftoa(double value, unsigned int after, char *out,
                                       unsigned int max)
{
    unsigned int at = 0u;
    if (after > AOTX_TEXT_AFTER) {
        after = AOTX_TEXT_AFTER;
    }
    if (value < 0.0) {
        if (max == 0u) {
            return 0u;
        }
        out[0] = '-';
        at = 1u;
        value = -value;
    }
    /* The round comes before the split, so a value such as 9.99 with one digit after the
     * point gives 10.0 and not 9.9. */
    double step = 1.0;
    for (unsigned int i = 0u; i < after; ++i) {
        step *= 10.0;
    }
    unsigned long long whole = (unsigned long long)value;
    unsigned long long part = (unsigned long long)((value - (double)whole) * step + 0.5);
    if (part >= (unsigned long long)step) {
        whole += 1ull;
        part = 0ull;
    }
    at += aotx_text_utoa(whole, out + at, max - at);
    if (after == 0u || at >= max) {
        return at;
    }
    out[at] = '.';
    at += 1u;
    /* The digits after the point keep the zeros in front of them, which the digit writer
     * would drop. */
    for (unsigned int i = after; i > 0u && at < max; --i) {
        unsigned long long scale = 1ull;
        for (unsigned int k = 1u; k < i; ++k) {
            scale *= 10ull;
        }
        out[at] = (char)('0' + (unsigned int)((part / scale) % 10ull));
        at += 1u;
    }
    return at;
}
