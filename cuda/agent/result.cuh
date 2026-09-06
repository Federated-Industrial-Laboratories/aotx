/* Purpose: Measure result bytes in the selected text encoding.
 * Owns: No storage; source bytes can occupy a circular buffer.
 * Launch shape: One thread for each result in a batch.
 * Lifetime: One result or prompt build. */
#ifndef AOTX_AGENT_RESULT_CUH
#define AOTX_AGENT_RESULT_CUH

#define AOTX_RESULT_CUT_TEXT " ... the result is cut to the room of this prompt"

/* Invalid spans return no prefix. The budget counts encoded bytes, not source bytes. */
__device__ __forceinline__ unsigned int aotx_result_prefix(
    const unsigned char *source, unsigned int base, unsigned int length,
    unsigned int capacity, int json, unsigned int budget)
{
    if (source == 0 || capacity == 0u || base >= capacity || length > capacity) return 0u;
    if (!json) return length < budget ? length : budget;
    unsigned int i = 0u;
    unsigned int at = base;
    for (; i < length; ++i) {
        unsigned char byte = source[at];
        unsigned int need = json && byte < 0x20u ? 6u
                          : json && (byte == '"' || byte == '\\') ? 2u : 1u;
        if (need > budget) break;
        budget -= need;
        if (++at == capacity) at = 0u;
    }
    return i;
}

/* Invalid spans or an unrepresentable encoded length return the maximum unsigned value. */
__device__ __forceinline__ unsigned int aotx_result_bytes(
    const unsigned char *source, unsigned int base, unsigned int length,
    unsigned int capacity, int json)
{
    if (source == 0 || capacity == 0u || base >= capacity || length > capacity) return ~0u;
    if (!json) return length;
    unsigned int bytes = 0u;
    unsigned int at = base;
    for (unsigned int i = 0u; i < length; ++i) {
        unsigned char byte = source[at];
        unsigned int need = json && byte < 0x20u ? 6u
                          : json && (byte == '"' || byte == '\\') ? 2u : 1u;
        if (need > ~0u - bytes) return ~0u;
        bytes += need;
        if (++at == capacity) at = 0u;
    }
    return bytes;
}

#endif
