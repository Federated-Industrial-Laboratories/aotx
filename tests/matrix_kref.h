/* Purpose: Give a host reading of packed weight blocks in double precision.
 * Owns: Nothing; the caller holds the bytes.
 * Launch shape: Host only; one weight for each call.
 * Lifetime: One matrix test case.
 *
 * This reading is written from the block definitions of the model file format. It is not
 * written from the device readers, so a fault of one is not a fault of both. It includes
 * no header of the device side. The super block holds 256 weights as 8 sub blocks of 32.
 *
 * Q4_K, 144 bytes: half d at 0, half dmin at 2, 12 scale bytes at 4, 128 nibble bytes at
 * 16. Nibble byte c times 32 plus l holds weight l of sub block 2c in its low nibble. The
 * high nibble of that byte holds weight l of sub block 2c plus 1. The weight is d times s
 * times the nibble, less dmin times m.
 *
 * Sub block j takes a 6 bit scale s and a 6 bit offset m from the 12 scale bytes. For j
 * under 4, s is the low 6 bits of byte j and m is the low 6 bits of byte j plus 4. For j
 * from 4, s is the low nibble of byte j plus 4, under the high 2 bits of byte j minus 4.
 * Then m is the high nibble of byte j plus 4, under the high 2 bits of byte j.
 *
 * Q5_K, 176 bytes: half d at 0, half dmin at 2, 12 scale bytes at 4, 32 high bit bytes at
 * 16. The 128 nibble bytes are at 48. The nibbles and the scales lie as in Q4_K. High bit
 * byte l holds bit j for weight l of sub block j, and that bit adds 16 to the nibble.
 *
 * Q6_K, 210 bytes: 128 nibble bytes at 0, 64 high bit bytes at 128, 16 signed scale bytes
 * at 192, half d at 208. The super block is two halves of 128 weights. In half h, nibble
 * byte h times 64 plus l holds weight l of sub block 4h in its low nibble. Its high nibble
 * holds weight l of sub block 4h plus 2. Nibble byte h times 64 plus 32 plus l holds sub
 * blocks 4h plus 1 and 4h plus 3 the same way.
 *
 * High bit byte h times 32 plus l holds 2 bits for weight l of each sub block of half h.
 * Sub block 4h takes the low pair, and sub block 4h plus 3 takes the high pair. Each run
 * of 16 weights takes one signed scale byte, in order. The weight is d times the scale
 * times the 6 bit code less 32. */

/* Q4_1 has d, m, and 16 nibble bytes in 20 bytes. Q5_0 has d, four high-bit bytes,
 * and 16 nibble bytes in 22 bytes. Q5_1 adds m after d, for 24 bytes.
 * Each high bit belongs to its weight index. Q5_0 subtracts 16 from the code.
 *
 * Q2_K has 16 scale/minimum bytes, 64 code bytes, d, and dmin in 84 bytes.
 * Q3_K has 32 sign-mask bytes, 64 code bytes, 12 scale bytes, and d in 110 bytes.
 * Both low-bit K types apply one scale to each consecutive run of 16 weights. */
#ifndef AOTX_MATRIX_KREF_H
#define AOTX_MATRIX_KREF_H

#include <math.h>
#include <stdint.h>

#define AOTX_KREF_SUPER      256u
#define AOTX_KREF_Q4K_BYTES  144u
#define AOTX_KREF_Q5K_BYTES  176u
#define AOTX_KREF_Q6K_BYTES  210u
#define AOTX_KREF_Q4K        12u
#define AOTX_KREF_Q5K        13u
#define AOTX_KREF_Q6K        14u
#define AOTX_KREF_Q41        3u
#define AOTX_KREF_Q50        6u
#define AOTX_KREF_Q51        7u
#define AOTX_KREF_Q2K        10u
#define AOTX_KREF_Q3K        11u

/* A half value from its 16 bits, by the IEEE 754 binary16 definition. */
static double aotx_kref_half(const unsigned char *at)
{
    unsigned int bits = (unsigned int)at[0] | ((unsigned int)at[1] << 8);
    unsigned int sign = bits >> 15;
    unsigned int exponent = (bits >> 10) & 0x1Fu;
    unsigned int fraction = bits & 0x3FFu;
    double value;
    if (exponent == 0u) {
        value = ldexp((double)fraction, -24);
    } else if (exponent == 31u) {
        value = (fraction == 0u) ? INFINITY : NAN;
    } else {
        value = ldexp((double)(fraction | 0x400u), (int)exponent - 25);
    }
    return sign ? -value : value;
}

/* The 6 bit scale and offset of sub block j from the 12 scale bytes of Q4_K and Q5_K. */
static void aotx_kref_scale(const unsigned char *scales, unsigned int j, unsigned int *s,
                            unsigned int *m)
{
    if (j < 4u) {
        *s = scales[j] & 63u;
        *m = scales[j + 4u] & 63u;
    } else {
        *s = (scales[j + 4u] & 15u) | ((scales[j - 4u] >> 6) << 4);
        *m = (scales[j + 4u] >> 4) | ((scales[j] >> 6) << 4);
    }
}

/* The bytes of one row of k weights of a packed type. */
static uint64_t aotx_kref_row_bytes(unsigned int type, unsigned int k)
{
    switch (type) {
    case AOTX_KREF_Q41: return (uint64_t)(k / 32u) * 20u;
    case AOTX_KREF_Q50: return (uint64_t)(k / 32u) * 22u;
    case AOTX_KREF_Q51: return (uint64_t)(k / 32u) * 24u;
    case AOTX_KREF_Q2K: return (uint64_t)(k / 256u) * 84u;
    case AOTX_KREF_Q3K: return (uint64_t)(k / 256u) * 110u;
    case AOTX_KREF_Q4K: case AOTX_KREF_Q5K: case AOTX_KREF_Q6K: break;
    default: return 0u;
    }
    uint64_t per = (type == AOTX_KREF_Q4K) ? AOTX_KREF_Q4K_BYTES
                 : ((type == AOTX_KREF_Q5K) ? AOTX_KREF_Q5K_BYTES : AOTX_KREF_Q6K_BYTES);
    return (uint64_t)(k / AOTX_KREF_SUPER) * per;
}

/* Weight j of one row, in double. */
static double aotx_kref_weight(const unsigned char *row, unsigned int type, unsigned int j)
{
    if (type == AOTX_KREF_Q41 || type == AOTX_KREF_Q50 || type == AOTX_KREF_Q51) {
        const unsigned char *one = row + (j / 32u) * aotx_kref_row_bytes(type, 32u);
        unsigned int position = j % 32u;
        unsigned int start = type == AOTX_KREF_Q41 ? 4u : (type == AOTX_KREF_Q50 ? 6u : 8u);
        unsigned int pair = one[start + position % 16u];
        int code = position < 16u ? (int)(pair % 16u) : (int)(pair / 16u);
        if (type != AOTX_KREF_Q41) {
            unsigned int mask_at = type == AOTX_KREF_Q50 ? 2u : 4u;
            unsigned int bit = (one[mask_at + position / 8u] / (1u << (position % 8u))) % 2u;
            code += (int)bit * 16;
        }
        if (type == AOTX_KREF_Q50) {
            return aotx_kref_half(one) * (double)(code - 16);
        }
        return aotx_kref_half(one) * (double)code + aotx_kref_half(one + 2u);
    }
    if (type == AOTX_KREF_Q2K || type == AOTX_KREF_Q3K) {
        const unsigned char *one = row + (j / 256u) * aotx_kref_row_bytes(type, 256u);
        unsigned int position = j % 256u;
        unsigned int scale_index = position / 16u;
        unsigned int plane = (position % 128u) / 32u;
        unsigned int lane = position % 32u;
        unsigned int start = type == AOTX_KREF_Q2K ? 16u : 32u;
        int code = (one[start + (position / 128u) * 32u + lane] / (1u << (2u * plane))) % 4u;
        if (type == AOTX_KREF_Q2K) {
            unsigned int packed = one[scale_index];
            return aotx_kref_half(one + 80u) * (double)(packed % 16u) * (double)code
                 - aotx_kref_half(one + 82u) * (double)(packed / 16u);
        }
        /* Eight low nibbles, eight high nibbles, then four interleaved high pairs. */
        unsigned int low_byte = one[96u + scale_index % 8u];
        unsigned int low_scale = scale_index < 8u ? low_byte % 16u : low_byte / 16u;
        unsigned int high_scale = (one[104u + scale_index % 4u]
            / (1u << (2u * (scale_index / 4u)))) % 4u;
        int scale = (int)(low_scale + 16u * high_scale) - 32;
        if ((one[lane] / (1u << (position / 32u))) % 2u == 0u) {
            code -= 4;
        }
        return aotx_kref_half(one + 108u) * (double)scale * (double)code;
    }
    if (type != AOTX_KREF_Q4K && type != AOTX_KREF_Q5K && type != AOTX_KREF_Q6K) {
        return NAN;
    }
    unsigned int in = j % AOTX_KREF_SUPER;
    unsigned int sub = in / 32u;
    unsigned int l = in % 32u;
    const unsigned char *one = row + (size_t)(j / AOTX_KREF_SUPER) * aotx_kref_row_bytes(type, AOTX_KREF_SUPER);
    if (type == AOTX_KREF_Q6K) {
        unsigned int h = sub / 4u;
        unsigned int quarter = sub % 4u;
        unsigned int nibble_byte = one[h * 64u + (quarter % 2u) * 32u + l];
        unsigned int low = (quarter < 2u) ? (nibble_byte & 15u) : (nibble_byte >> 4);
        unsigned int high = (one[128u + h * 32u + l] >> (quarter * 2u)) & 3u;
        int code = (int)(low | (high << 4)) - 32;
        int scale = (int)(signed char)one[192u + h * 8u + quarter * 2u + l / 16u];
        return aotx_kref_half(one + 208u) * (double)scale * (double)code;
    }
    unsigned int s;
    unsigned int m;
    aotx_kref_scale(one + 4u, sub, &s, &m);
    double d = aotx_kref_half(one) * (double)s;
    double dmin = aotx_kref_half(one + 2u) * (double)m;
    unsigned int nibble_at = (type == AOTX_KREF_Q4K) ? 16u : 48u;
    unsigned int nibble_byte = one[nibble_at + (sub / 2u) * 32u + l];
    unsigned int q = (sub % 2u == 0u) ? (nibble_byte & 15u) : (nibble_byte >> 4);
    if (type == AOTX_KREF_Q5K) {
        q += ((one[16u + l] >> sub) & 1u) ? 16u : 0u;
    }
    return d * (double)q - dmin;
}

#endif
