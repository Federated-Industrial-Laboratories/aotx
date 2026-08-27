/* Purpose: Give each agent a random number stream.
 * Owns: The Philox state for each agent.
 * Launch shape: One thread for each agent.
 * Lifetime: From agent creation to agent release. */
#ifndef AOTX_RNG_CUH
#define AOTX_RNG_CUH

/* Philox4x32-10, a counter based generator. A stream has no state to carry. A thread makes
 * its numbers from a key and a counter. A replay of the same key and counter gives the same
 * numbers. The key holds the seed and the stream identity. The counter holds the position
 * in the stream. */

#define AOTX_RNG_ROUNDS   10u
#define AOTX_RNG_MULT_0   0xD2511F53u  /* first multiplier of the round function */
#define AOTX_RNG_MULT_1   0xCD9E8D57u  /* second multiplier of the round function */
#define AOTX_RNG_BUMP_0   0x9E3779B9u  /* key increment for the low word */
#define AOTX_RNG_BUMP_1   0xBB67AE85u  /* key increment for the high word */

typedef struct aotx_rng_key {
    unsigned int k[2];
} aotx_rng_key;

typedef struct aotx_rng_counter {
    unsigned int c[4];
} aotx_rng_counter;

/* One round mixes the counter with the key. The two products give a high half and a low half;
 * the high halves cross over to the other side of the counter. */
__device__ __forceinline__ void aotx_rng_round(aotx_rng_counter *ctr, const aotx_rng_key *key)
{
    unsigned int lo0 = AOTX_RNG_MULT_0 * ctr->c[0];
    unsigned int hi0 = __umulhi(AOTX_RNG_MULT_0, ctr->c[0]);
    unsigned int lo1 = AOTX_RNG_MULT_1 * ctr->c[2];
    unsigned int hi1 = __umulhi(AOTX_RNG_MULT_1, ctr->c[2]);
    unsigned int n0 = hi1 ^ ctr->c[1] ^ key->k[0];
    unsigned int n1 = lo1;
    unsigned int n2 = hi0 ^ ctr->c[3] ^ key->k[1];
    unsigned int n3 = lo0;
    ctr->c[0] = n0;
    ctr->c[1] = n1;
    ctr->c[2] = n2;
    ctr->c[3] = n3;
}

/* The key advances by a fixed amount between rounds, which keeps the rounds distinct. */
__device__ __forceinline__ void aotx_rng_bump(aotx_rng_key *key)
{
    key->k[0] += AOTX_RNG_BUMP_0;
    key->k[1] += AOTX_RNG_BUMP_1;
}

/* Ten rounds of the counter and the key give four random words. The first round takes the
 * key as given; each later round takes the key after one increment. */
__device__ __forceinline__ uint4 aotx_rng_philox(aotx_rng_counter ctr, aotx_rng_key key)
{
    #pragma unroll
    for (unsigned int r = 0; r < AOTX_RNG_ROUNDS; ++r) {
        if (r != 0) {
            aotx_rng_bump(&key);
        }
        aotx_rng_round(&ctr, &key);
    }
    uint4 out;
    out.x = ctr.c[0];
    out.y = ctr.c[1];
    out.z = ctr.c[2];
    out.w = ctr.c[3];
    return out;
}

/* The stream of one lane. The seed and the stream go in the key; the lane and the position
 * go in the counter. Two lanes with different identities never share numbers. */
__device__ __forceinline__ uint4 aotx_rng_lane(unsigned long long seed,
                                               unsigned int stream,
                                               unsigned int lane,
                                               unsigned long long position)
{
    aotx_rng_key key;
    aotx_rng_counter ctr;
    key.k[0] = (unsigned int)(seed & 0xFFFFFFFFull);
    key.k[1] = (unsigned int)(seed >> 32);
    ctr.c[0] = (unsigned int)(position & 0xFFFFFFFFull);
    ctr.c[1] = (unsigned int)(position >> 32);
    ctr.c[2] = lane;
    ctr.c[3] = stream;
    return aotx_rng_philox(ctr, key);
}

/* A float in the half open range from zero to one. The top 24 bits give the mantissa, which
 * is the width of a float, so every value in the range is reachable. */
__device__ __forceinline__ float aotx_rng_unit(unsigned int word)
{
    return (float)(word >> 8) * (1.0f / 16777216.0f);
}

#endif
