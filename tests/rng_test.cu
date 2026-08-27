/* Purpose: Check the Philox generator against known answers, and check its spread.
 * Owns: The test fixtures and the counts of the cases.
 * Launch shape: One thread for each lane of the batch.
 * Lifetime: One run of the test program. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "rng/rng.cuh"

#define AOTX_TEST_BUCKETS 16u

/* The batch: one counter and one key for each lane, four words out of each lane. */
__global__ void aotx_rng_test_fill(const aotx_rng_counter *counters,
                                   const aotx_rng_key *keys,
                                   uint4 *out, unsigned int lanes,
                                   unsigned int draws)
{
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) {
        return;
    }
    for (unsigned int draw = 0u; draw < draws; ++draw) {
        aotx_rng_counter counter = counters[lane];
        counter.c[0] += draw;
        out[(size_t)draw * lanes + lane] = aotx_rng_philox(counter, keys[lane]);
    }
}

/* The same generator on the host, so a device answer is checked against a second
 * implementation and not only against a stored vector. */
static void aotx_rng_host(const unsigned int *ctr_in, const unsigned int *key_in,
                          unsigned int *out)
{
    unsigned int c[4];
    unsigned int k[2];
    memcpy(c, ctr_in, sizeof c);
    memcpy(k, key_in, sizeof k);
    for (unsigned int r = 0u; r < 10u; ++r) {
        if (r != 0u) {
            k[0] += 0x9E3779B9u;
            k[1] += 0xBB67AE85u;
        }
        unsigned long long p0 = (unsigned long long)0xD2511F53u * c[0];
        unsigned long long p1 = (unsigned long long)0xCD9E8D57u * c[2];
        unsigned int n0 = (unsigned int)(p1 >> 32) ^ c[1] ^ k[0];
        unsigned int n1 = (unsigned int)p1;
        unsigned int n2 = (unsigned int)(p0 >> 32) ^ c[3] ^ k[1];
        unsigned int n3 = (unsigned int)p0;
        c[0] = n0;
        c[1] = n1;
        c[2] = n2;
        c[3] = n3;
    }
    memcpy(out, c, 4u * sizeof(unsigned int));
}

/* The known answers of Philox4x32-10, from the Random123 reference vectors. Each one is
 * confirmed here by the host implementation above. */
static const unsigned int aotx_rng_known_ctr[3][4] = {
    { 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u },
    { 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu },
    { 0x243F6A88u, 0x85A308D3u, 0x13198A2Eu, 0x03707344u }
};
static const unsigned int aotx_rng_known_key[3][2] = {
    { 0x00000000u, 0x00000000u },
    { 0xFFFFFFFFu, 0xFFFFFFFFu },
    { 0xA4093822u, 0x299F31D0u }
};
static const unsigned int aotx_rng_known_out[3][4] = {
    { 0x6627E8D5u, 0xE169C58Du, 0xBC57AC4Cu, 0x9B00DBD8u },
    { 0x408F276Du, 0x41C83B0Eu, 0xA20BC7C6u, 0x6D5451FDu },
    { 0xD16CFE09u, 0x94FDCCEBu, 0x5001E420u, 0x24126EA1u }
};

static unsigned int aotx_rng_run(unsigned int lanes, unsigned int draws,
                                 const aotx_rng_counter *counters,
                                 const aotx_rng_key *keys, uint4 *out)
{
    aotx_rng_counter *device_counters = NULL;
    aotx_rng_key *device_keys = NULL;
    uint4 *device_out = NULL;
    size_t words = (size_t)lanes * draws;
    aotx_check_runtime(cudaMalloc(&device_counters, lanes * sizeof *counters), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&device_keys, lanes * sizeof *keys), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&device_out, words * sizeof *out), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device_counters, counters, lanes * sizeof *counters,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(device_keys, keys, lanes * sizeof *keys,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    unsigned int threads = 128u;
    unsigned int blocks = (lanes + threads - 1u) / threads;
    aotx_rng_test_fill<<<blocks, threads>>>(device_counters, device_keys, device_out,
                                            lanes, draws);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(out, device_out, words * sizeof *out,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    cudaFree(device_counters);
    cudaFree(device_keys);
    cudaFree(device_out);
    return lanes * draws;
}

int main(void)
{
    unsigned int applied = 0u;
    unsigned int failed = 0u;

    /* Case set 1: the known answers, at a batch of one and inside a batch of three. */
    for (unsigned int single = 0u; single < 3u; ++single) {
        aotx_rng_counter counter;
        aotx_rng_key key;
        uint4 got;
        memcpy(counter.c, aotx_rng_known_ctr[single], sizeof counter.c);
        memcpy(key.k, aotx_rng_known_key[single], sizeof key.k);
        aotx_rng_run(1u, 1u, &counter, &key, &got);
        unsigned int want[4];
        aotx_rng_host(aotx_rng_known_ctr[single], aotx_rng_known_key[single], want);
        applied += 2u;
        if (memcmp(want, aotx_rng_known_out[single], sizeof want) != 0) {
            printf("rng: host answer %u differs from the reference vector\n", single);
            failed += 1u;
        }
        if (got.x != want[0] || got.y != want[1] || got.z != want[2] || got.w != want[3]) {
            printf("rng: device answer %u differs from the reference vector\n", single);
            failed += 1u;
        }
    }

    /* Case set 2: a batch of 64 lanes, every lane with a distinct counter and key. */
    const unsigned int lanes = 64u;
    const unsigned int draws = 4096u;
    aotx_rng_counter *counters = (aotx_rng_counter *)malloc(lanes * sizeof *counters);
    aotx_rng_key *keys = (aotx_rng_key *)malloc(lanes * sizeof *keys);
    uint4 *out = (uint4 *)malloc((size_t)lanes * draws * sizeof *out);
    for (unsigned int lane = 0u; lane < lanes; ++lane) {
        counters[lane].c[0] = 0u;
        counters[lane].c[1] = lane * 7u + 1u;
        counters[lane].c[2] = lane;
        counters[lane].c[3] = 0x5A5A0000u + lane;
        keys[lane].k[0] = 0x1234u + lane * 31u;
        keys[lane].k[1] = 0xABCD0000u + lane;
    }
    aotx_rng_run(lanes, draws, counters, keys, out);

    /* Every lane agrees with the host, so no lane reads another lane's counter. */
    for (unsigned int lane = 0u; lane < lanes; ++lane) {
        for (unsigned int draw = 0u; draw < 4u; ++draw) {
            unsigned int ctr[4];
            unsigned int want[4];
            memcpy(ctr, counters[lane].c, sizeof ctr);
            ctr[0] += draw;
            aotx_rng_host(ctr, keys[lane].k, want);
            uint4 got = out[(size_t)draw * lanes + lane];
            applied += 1u;
            if (got.x != want[0] || got.y != want[1] || got.z != want[2] || got.w != want[3]) {
                printf("rng: lane %u draw %u differs from the host\n", lane, draw);
                failed += 1u;
            }
        }
    }

    /* The spread: the words fall in even buckets, and no two draws of a lane repeat. */
    size_t total = (size_t)lanes * draws * 4u;
    unsigned long long bucket[AOTX_TEST_BUCKETS];
    memset(bucket, 0, sizeof bucket);
    double sum = 0.0;
    for (size_t i = 0; i < (size_t)lanes * draws; ++i) {
        const unsigned int *word = (const unsigned int *)&out[i];
        for (unsigned int w = 0u; w < 4u; ++w) {
            bucket[word[w] >> 28] += 1ull;
            sum += (double)(word[w] >> 8) * (1.0 / 16777216.0);
        }
    }
    double mean = sum / (double)total;
    applied += 1u;
    if (mean < 0.49 || mean > 0.51) {
        printf("rng: the mean is %f, which is outside the band\n", mean);
        failed += 1u;
    }
    double share = (double)total / (double)AOTX_TEST_BUCKETS;
    for (unsigned int b = 0u; b < AOTX_TEST_BUCKETS; ++b) {
        applied += 1u;
        double ratio = (double)bucket[b] / share;
        if (ratio < 0.95 || ratio > 1.05) {
            printf("rng: bucket %u holds %f of its share\n", b, ratio);
            failed += 1u;
        }
    }

    free(counters);
    free(keys);
    free(out);
    printf("rng: %u cases applied, %u failed\n", applied, failed);
    return failed == 0u ? 0 : 1;
}
