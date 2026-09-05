/* Purpose: Check turn bytes, token order and the first reply token at model load.
 * Owns: The per-role wrap table.
 * Launch shape: One thread for each text row; one block for an argmax.
 * Lifetime: The wrap table lasts for the loaded model. */
#include <math.h>
#include "model/wrap_check.cuh"
#include "model/forward.cuh"
#include "kvcache/kvcache.cuh"

__device__ aotx_wrap aotx_model_wrap[AOTX_MODEL_ROLES];

static __device__ unsigned int aotx_wrap_check_text(unsigned char *out, unsigned int at,
                                                   const char *text)
{
    unsigned int length = 0u;
    while (text[length] != '\0') ++length;
    return aotx_wrap_run(out, at, AOTX_WRAP_CHECK_BYTES,
                         (const unsigned char *)text, length);
}

static __device__ unsigned int aotx_wrap_check_render(unsigned char *out,
                                                      const aotx_wrap *wrap)
{
    unsigned int at = aotx_wrap_put(out, 0u, AOTX_WRAP_CHECK_BYTES, wrap,
                                     AOTX_WRAP_SYSTEM_HEAD);
    at = aotx_wrap_check_text(out, at, "You answer briefly.");
    at = aotx_wrap_put(out, at, AOTX_WRAP_CHECK_BYTES, wrap, AOTX_WRAP_SYSTEM_TAIL);
    at = aotx_wrap_put(out, at, AOTX_WRAP_CHECK_BYTES, wrap, AOTX_WRAP_USER_HEAD);
    at = aotx_wrap_check_text(out, at, "Name one color.");
    at = aotx_wrap_put(out, at, AOTX_WRAP_CHECK_BYTES, wrap, AOTX_WRAP_USER_TAIL);
    at = aotx_wrap_put(out, at, AOTX_WRAP_CHECK_BYTES, wrap, AOTX_WRAP_ASSISTANT_HEAD);
    at = aotx_wrap_check_text(out, at, "Blue.");
    at = aotx_wrap_put(out, at, AOTX_WRAP_CHECK_BYTES, wrap, AOTX_WRAP_ASSISTANT_TAIL);
    at = aotx_wrap_put(out, at, AOTX_WRAP_CHECK_BYTES, wrap, AOTX_WRAP_USER_HEAD);
    at = aotx_wrap_check_text(out, at, "Name one animal.");
    at = aotx_wrap_put(out, at, AOTX_WRAP_CHECK_BYTES, wrap, AOTX_WRAP_USER_TAIL);
    return aotx_wrap_generation(out, at, AOTX_WRAP_CHECK_BYTES, wrap);
}

__global__ void aotx_wrap_check_build(unsigned int role, aotx_wrap_check_work *work)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= AOTX_WRAP_CHECK_ROWS) return;
    work->start[row] = row * AOTX_WRAP_CHECK_BYTES;
    if (row == 0u) work->argmax = ~0u;
    const aotx_wrap *wrap = row == 0u ? &aotx_model_wrap[role] : &work->expected;
    for (unsigned int s = 0u; s < AOTX_WRAP_SPANS; ++s) {
        if (wrap->length[s] > AOTX_WRAP_SPAN_BYTES
            || wrap->offset[s] + wrap->length[s] > AOTX_WRAP_BYTES) return;
    }
    if (row < 2u) {
        work->length[row] = aotx_wrap_check_render(work->raw[row], wrap);
    } else if (row < 2u + AOTX_WRAP_SPANS) {
        unsigned int s = row - 2u;
        work->length[row] = aotx_wrap_put(work->raw[row], 0u, AOTX_WRAP_CHECK_BYTES, wrap, s);
    } else {
        unsigned int i = row - 2u - AOTX_WRAP_SPANS;
        if (i >= wrap->end_count) return;
        unsigned int id = wrap->end_ids[i];
        const aotx_text_vocab *vocab = &aotx_text_vocab_table;
        if (id >= vocab->tokens || id >= aotx_model[role].vocab) return;
        unsigned long long first = vocab->token_at[id];
        unsigned long long length = vocab->token_at[id + 1u] - first;
        if (length == 0ull || length > AOTX_WRAP_CHECK_BYTES) return;
        work->length[row] = aotx_wrap_run(work->raw[row], 0u, AOTX_WRAP_CHECK_BYTES,
                                          vocab->token_bytes + first, (unsigned int)length);
    }
}

__global__ void aotx_wrap_check_tokens(unsigned int role, aotx_wrap_check_work *work)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    unsigned int total = work->count[0];
    unsigned int good = total != 0u && total <= AOTX_MODEL_MAX_TOKENS
                      && total == work->count[1] && work->length[0] == work->length[1];
    for (unsigned int i = 0u; good && i < total; ++i)
        good &= work->id[0][i] == work->id[1][i];
    for (unsigned int i = 0u; good && i < work->length[0]; ++i)
        good &= work->raw[0][i] == work->raw[1][i];
    const unsigned int order[] = {
        AOTX_WRAP_SYSTEM_HEAD, AOTX_WRAP_SYSTEM_TAIL, AOTX_WRAP_USER_HEAD,
        AOTX_WRAP_USER_TAIL, AOTX_WRAP_ASSISTANT_HEAD, AOTX_WRAP_ASSISTANT_TAIL,
        AOTX_WRAP_USER_HEAD, AOTX_WRAP_USER_TAIL, AOTX_WRAP_GENERATION_HEAD,
        AOTX_WRAP_THINK_OPEN, AOTX_WRAP_THINK_CLOSE
    };
    unsigned int at = 0u;
    for (unsigned int s = 0u; good && s < sizeof(order) / sizeof(order[0]); ++s) {
        unsigned int row = 2u + order[s], n = work->count[row];
        if (n == 0u) {
            good &= work->expected.length[order[s]] == 0u;
            continue;
        }
        unsigned int found = 0u;
        for (; at + n <= total; ++at) {
            unsigned int same = 1u;
            for (unsigned int j = 0u; j < n; ++j)
                same &= work->id[0][at + j] == work->id[row][j];
            if (same) { found = 1u; at += n; break; }
        }
        good &= found;
    }
    work->order_ok = good && aotx_model_wrap[role].prefix_length == work->expected.prefix_length;
    work->ends_ok = work->expected.end_count > 0u
                  && work->expected.end_count <= AOTX_WRAP_ENDS
                  && aotx_model_wrap[role].end_count == work->expected.end_count;
    for (unsigned int i = 0u; i < work->expected.end_count && i < AOTX_WRAP_ENDS; ++i) {
        unsigned int row = 2u + AOTX_WRAP_SPANS + i;
        work->ends_ok &= work->count[row] == 1u
                      && work->id[row][0] == work->expected.end_ids[i]
                      && aotx_model_wrap[role].end_ids[i] == work->expected.end_ids[i];
    }
    aotx_wrap *wrap = &aotx_model_wrap[role];
    wrap->usable = 0u;
    wrap->think_open_id = work->count[2u + AOTX_WRAP_THINK_OPEN] != 0u
                       ? work->id[2u + AOTX_WRAP_THINK_OPEN][0] : ~0u;
    wrap->think_close_id = work->count[2u + AOTX_WRAP_THINK_CLOSE] != 0u
                        ? work->id[2u + AOTX_WRAP_THINK_CLOSE][0] : ~0u;
    work->offset[0] = 0u;
    work->offset[1] = total;
    work->agent = 0u;
}

__global__ void aotx_wrap_check_argmax(unsigned int role, const float *logits,
                                      aotx_wrap_check_work *work)
{
    __shared__ float value[256];
    __shared__ unsigned int id[256], invalid;
    unsigned int lane = threadIdx.x;
    if (lane == 0u) invalid = 0u;
    __syncthreads();
    float best = -INFINITY;
    unsigned int token = ~0u;
    for (unsigned int i = lane; i < aotx_model[role].vocab; i += blockDim.x) {
        float next = logits[i];
        if (!isfinite(next)) atomicExch(&invalid, 1u);
        if (next > best || (next == best && i < token)) { best = next; token = i; }
    }
    value[lane] = best; id[lane] = token;
    __syncthreads();
    for (unsigned int n = 128u; n != 0u; n >>= 1u) {
        if (lane < n && (value[lane + n] > value[lane]
            || (value[lane + n] == value[lane] && id[lane + n] < id[lane]))) {
            value[lane] = value[lane + n]; id[lane] = id[lane + n];
        }
        __syncthreads();
    }
    if (lane == 0u) {
        work->argmax = id[0];
        work->prefill_ok = invalid == 0u && aotx_model_faults == 0u
                         && id[0] < aotx_model[role].vocab && !aotx_wrap_end(role, id[0]);
        aotx_model_wrap[role].usable = work->order_ok && work->ends_ok && work->prefill_ok;
    }
}

/* These private pages replace no live content. The host restores the saved table after
 * the check and releases only this allocation. No page request or record is made. */
__global__ void aotx_wrap_check_cache(unsigned long long pages, unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < AOTX_KV_PAGES_EACH)
        aotx_kv.page[0][i] = i < count ? pages + (unsigned long long)i * AOTX_KV_PAGE_BYTES : 0ull;
    if (i == 0u) { aotx_kv.count[0] = count; aotx_model_seen[0] = 0u; aotx_model_faults = 0u; }
}
