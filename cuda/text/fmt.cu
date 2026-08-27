/* Purpose: Write numbers as text on the device.
 * Owns: Nothing; the caller gives the buffer.
 * Launch shape: One thread for each value; no kernels of its own.
 * Lifetime: One call. */
#include "text/text.cuh"

/* The device has no library for numbers. The seam rules refuse a print call in a device
 * file. Every panel and every record line therefore comes through these three functions. */

/* Digits of the largest unsigned value, which is 20 for 64 bits. */
#define AOTX_TEXT_DIGITS   20u

/* Digits after the point that a value may ask for. */
#define AOTX_TEXT_AFTER    9u

__device__ unsigned int aotx_text_utoa(unsigned long long value, char *out, unsigned int max)
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

__device__ unsigned int aotx_text_itoa(long long value, char *out, unsigned int max)
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
