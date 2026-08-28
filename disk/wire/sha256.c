/* Purpose: Compute the SHA-256 digest of a byte run or of a file, one piece at a time.
 * Owns: The state of one digest: the eight words, the byte count, and the part block.
 * Threading: One thread for each state; the state holds no lock.
 * Lifetime: From the init call to the final call. */
#include "disk/wire/diskwire.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

/* FIPS 180-4 section 4.2.2 gives the 64 constants. Each one is the first 32 bits of the
 * fraction part of the cube root of one of the first 64 prime numbers. */
static const uint32_t aotx_sha256_k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static uint32_t rotate_right(uint32_t x, unsigned n)
{
    return (x >> n) | (x << (32u - n));
}

/* One compression of a 64-byte block, from FIPS 180-4 section 6.2.2. */
static void compress(uint32_t *h, const unsigned char *block)
{
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, hh;
    int i;
    /* The message schedule reads the block as 16 big-endian words. */
    for (i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) | (uint32_t)block[i * 4 + 3];
    }
    for (i = 16; i < 64; i++) {
        uint32_t s0 = rotate_right(w[i - 15], 7) ^ rotate_right(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotate_right(w[i - 2], 17) ^ rotate_right(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    a = h[0];
    b = h[1];
    c = h[2];
    d = h[3];
    e = h[4];
    f = h[5];
    g = h[6];
    hh = h[7];
    for (i = 0; i < 64; i++) {
        uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = hh + s1 + ch + aotx_sha256_k[i] + w[i];
        uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

void aotx_sha256_init(aotx_sha256 *s)
{
    /* FIPS 180-4 section 5.3.3: the fraction parts of the square roots of the first eight
     * prime numbers. */
    s->h[0] = 0x6a09e667u;
    s->h[1] = 0xbb67ae85u;
    s->h[2] = 0x3c6ef372u;
    s->h[3] = 0xa54ff53au;
    s->h[4] = 0x510e527fu;
    s->h[5] = 0x9b05688cu;
    s->h[6] = 0x1f83d9abu;
    s->h[7] = 0x5be0cd19u;
    s->bytes = 0;
    s->fill = 0;
    memset(s->block, 0, sizeof(s->block));
}

void aotx_sha256_update(aotx_sha256 *s, const void *data, size_t bytes)
{
    const unsigned char *p = (const unsigned char *)data;
    size_t left = bytes;
    s->bytes += (uint64_t)bytes;
    /* A part block from an earlier call comes first, so a run of small calls gives the
     * same digest as one large call. */
    if (s->fill > 0) {
        size_t need = 64u - s->fill;
        size_t take = (left < need) ? left : need;
        memcpy(s->block + s->fill, p, take);
        s->fill += take;
        p += take;
        left -= take;
        if (s->fill < 64u) {
            return;
        }
        compress(s->h, s->block);
        s->fill = 0;
    }
    while (left >= 64u) {
        compress(s->h, p);
        p += 64;
        left -= 64;
    }
    if (left > 0) {
        memcpy(s->block, p, left);
        s->fill = left;
    }
}

void aotx_sha256_final(aotx_sha256 *s, unsigned char digest[AOTX_SHA256_DIGEST])
{
    uint64_t bits = s->bytes * 8u;
    unsigned char tail[72];
    size_t pad;
    int i;
    /* FIPS 180-4 section 5.1.1 gives the padding. One set bit comes first, then zero
     * bits, then the length in bits as eight bytes with the high byte first. The result
     * is a multiple of 64 bytes. */
    memset(tail, 0, sizeof(tail));
    tail[0] = 0x80u;
    pad = (s->fill < 56u) ? (56u - s->fill) : (120u - s->fill);
    for (i = 0; i < 8; i++) {
        tail[pad + (size_t)i] = (unsigned char)(bits >> (56 - i * 8));
    }
    aotx_sha256_update(s, tail, pad + 8u);
    for (i = 0; i < 8; i++) {
        digest[i * 4] = (unsigned char)(s->h[i] >> 24);
        digest[i * 4 + 1] = (unsigned char)(s->h[i] >> 16);
        digest[i * 4 + 2] = (unsigned char)(s->h[i] >> 8);
        digest[i * 4 + 3] = (unsigned char)s->h[i];
    }
}

void aotx_sha256_text(const unsigned char digest[AOTX_SHA256_DIGEST], char *out)
{
    static const char hex[] = "0123456789abcdef";
    int i;
    for (i = 0; i < AOTX_SHA256_DIGEST; i++) {
        out[i * 2] = hex[(digest[i] >> 4) & 0xfu];
        out[i * 2 + 1] = hex[digest[i] & 0xfu];
    }
    out[AOTX_SHA256_DIGEST * 2] = '\0';
}

int aotx_sha256_read(int fd, uint64_t offset, uint64_t bytes, void *buffer, size_t buffer_bytes,
                     aotx_sha256 *state)
{
    unsigned char *at = (unsigned char *)buffer;
    uint64_t left = bytes;
    if (buffer_bytes == 0) {
        return 1;
    }
    while (left > 0) {
        uint64_t want = (left < (uint64_t)buffer_bytes) ? left : (uint64_t)buffer_bytes;
        ssize_t got = pread(fd, at, (size_t)want, (off_t)offset);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            return 1;
        }
        if (got == 0) {
            /* The file is shorter than the caller asked for. */
            return 1;
        }
        aotx_sha256_update(state, at, (size_t)got);
        offset += (uint64_t)got;
        left -= (uint64_t)got;
    }
    return 0;
}

int aotx_sha256_file(const char *path, char *text, uint64_t *bytes, void *buffer,
                     size_t buffer_bytes)
{
    aotx_sha256 state;
    unsigned char digest[AOTX_SHA256_DIGEST];
    unsigned char *at = (unsigned char *)buffer;
    uint64_t total = 0;
    int fd;
    if (buffer_bytes == 0) {
        return 1;
    }
    fd = open(path, O_RDONLY);
    if (fd < 0) {
        return 1;
    }
    aotx_sha256_init(&state);
    for (;;) {
        ssize_t got = read(fd, at, buffer_bytes);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            close(fd);
            return 1;
        }
        if (got == 0) {
            break;
        }
        aotx_sha256_update(&state, at, (size_t)got);
        total += (uint64_t)got;
    }
    close(fd);
    aotx_sha256_final(&state, digest);
    aotx_sha256_text(digest, text);
    if (bytes != NULL) {
        *bytes = total;
    }
    return 0;
}
