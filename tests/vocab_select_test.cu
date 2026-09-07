/* Purpose: Check distinct token IDs through both vocabulary selections.
 * Owns: Two vocabulary stores and a batch of distinct text keys.
 * Launch shape: One thread for each text key at one slot and at all slots.
 * Lifetime: One test process. */
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "boot/check.h"
#include "profile/profile.cuh"
#include "text/text.cuh"

__global__ void aotx_vocab_lookup_test(unsigned int *out, unsigned int count)
{
    unsigned int slot = threadIdx.x;
    if (slot >= count) return;
    unsigned char key[4] = {'n', (unsigned char)('0' + slot / 100u),
                           (unsigned char)('0' + (slot / 10u) % 10u),
                           (unsigned char)('0' + slot % 10u)};
    out[slot] = aotx_text_find_token(&aotx_text_vocab_table, key, 4u);
}

static int build(aotx_text_store *store, unsigned int reverse)
{
    static_assert(AOTX_SLOTS <= 1000u, "the text keys need more digits");
    unsigned char text[AOTX_SLOTS * 4u + 32u];
    unsigned long long offset[AOTX_SLOTS + 5u];
    int types[AOTX_SLOTS + 4u];
    unsigned int used = 0u;
    for (unsigned int i = 0u; i < AOTX_SLOTS; ++i) {
        offset[i] = used;
        unsigned int key = reverse ? AOTX_SLOTS - 1u - i : i;
        int bytes = snprintf((char *)text + used, sizeof text - used, "n%03u", key);
        if (bytes != 4 || (unsigned int)bytes >= sizeof text - used) return 1;
        used += (unsigned int)bytes;
        types[i] = 1;
    }
    const char *extra[] = {"a", "b", "ab", "<|endoftext|>"};
    for (unsigned int i = 0u; i < 4u; ++i) {
        offset[AOTX_SLOTS + i] = used;
        size_t bytes = strlen(extra[i]);
        memcpy(text + used, extra[i], bytes);
        used += (unsigned int)bytes;
        types[AOTX_SLOTS + i] = i == 3u ? AOTX_TEXT_TYPE_CONTROL : 1;
    }
    offset[AOTX_SLOTS + 4u] = used;
    const unsigned char merge[] = "a b";
    const unsigned long long merge_at[] = {0ull, 3ull};
    aotx_text_source source = {};
    source.token_bytes = text;
    source.token_at = offset;
    source.tokens = AOTX_SLOTS + 4u;
    source.merge_bytes = merge;
    source.merge_at = merge_at;
    source.merges = 1u;
    source.token_type = types;
    if (aotx_text_family_find("qwen2", 5u, &source.family) != 0) return 1;
    if (aotx_text_vocab_build(&source, store) != 0) return 1;
    aotx_text_vocab table;
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_text_vocab_table, sizeof table),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_text_vocab_saved, &table, sizeof table,
                                         reverse * sizeof table), "cudaMemcpyToSymbol");
    return 0;
}
#include "vocab_order.h"


int main(void)
{
    aotx_text_store store[2] = {};
    if (build(&store[0], 0u) || build(&store[1], 1u)) return 1;
    unsigned int *device = NULL;
    aotx_check_runtime(cudaMalloc(&device, AOTX_SLOTS * sizeof(unsigned int)), "cudaMalloc");
    unsigned int out[AOTX_SLOTS];
    unsigned int failures = 0u;
    unsigned int checks = 0u;
    const unsigned int sizes[] = {1u, AOTX_SLOTS};
    const unsigned int choices[] = {0u, 1u, 2u, 0u};
    for (unsigned int count : sizes) {
        for (unsigned int choice : choices) {
            aotx_text_vocab_select<<<1, 128>>>(choice);
            aotx_vocab_lookup_test<<<1, AOTX_SLOTS>>>(device, count);
            aotx_check_runtime(cudaMemcpy(out, device, count * sizeof(unsigned int),
                                          cudaMemcpyDeviceToHost), "cudaMemcpy");
            unsigned int wrong = 0u;
            for (unsigned int i = 0u; i < count; ++i) {
                unsigned int expected = choice == 0u ? i : AOTX_SLOTS - 1u - i;
                wrong += out[i] != expected;
            }
            ++checks;
            failures += wrong != 0u;
            printf("vocabulary selection %u N=%u: %u wrong IDs\n", choice, count, wrong);
        }
    }
    cudaFree(device);
    aotx_text_vocab_release(&store[0]);
    aotx_text_vocab_release(&store[1]);
    failures += aotx_vocab_order_cases(&checks);
    printf("vocabulary: %u checks, %u failed\n", checks, failures);
    return failures != 0u;
}
