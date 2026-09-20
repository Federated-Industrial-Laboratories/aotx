/* Purpose: Publish validated object batches and export exact state checkpoints.
 * Owns: Staged state changes and recorded tail application.
 * Launch shape: One 64-thread block; the caller serializes state operations.
 * Lifetime: Caller-owned live, stage, image and result buffers. */
#include "cognitive/validate.cuh"

static __device__ void aotx_cog_copy(unsigned char *out, const unsigned char *in, uint64_t bytes) {
    for (uint64_t j = threadIdx.x; j < bytes; j += blockDim.x) out[j] = in[j];
}
static __device__ void aotx_cog_error(uint32_t *error, uint32_t status) {
    if (status) atomicMin(error, status);
}
static __device__ void aotx_cog_result(aotx_cognitive_result *r, uint32_t error,
                                      uint32_t applied, uint64_t bytes, uint64_t sequence) {
    if (!threadIdx.x) {
        r->status = error == UINT32_MAX ? AOTX_COG_OK : error;
        r->applied = r->status ? 0 : applied;
        r->bytes = r->status ? 0 : bytes;
        r->sequence = sequence;
    }
}
static __device__ void aotx_cog_publish(aotx_cognitive_store *live, aotx_cognitive_store *stage,
                                        uint32_t *error) {
    if (!threadIdx.x) {
        uint64_t bytes = 0, covered = 0;
        for (uint32_t j = 0; j < stage->count; ++j) {
            uint64_t length = aotx_cog_resident_bytes(stage->objects[j]);
            if (length > stage->bytes) { aotx_cog_error(error, AOTX_COG_FORMAT); break; }
            bytes += length;
            if (aotx_cog_u64(stage->objects[j] + AOTX_CO_UPDATED) > stage->retry_floor) ++covered;
        }
        if (bytes != stage->bytes) aotx_cog_error(error, AOTX_COG_FORMAT);
        if (stage->pressure_percent && covered != stage->sequence - stage->retry_floor)
            aotx_cog_error(error, AOTX_COG_SEQUENCE);
    }
    for (uint32_t j = threadIdx.x; j < stage->count; j += blockDim.x)
        aotx_cog_error(error, aotx_cog_validate(stage, j));
    __syncthreads();
    if (*error == UINT32_MAX)
        aotx_cog_copy((unsigned char *)live, (const unsigned char *)stage, sizeof(*live));
    __syncthreads();
}

static __device__ void aotx_cog_tail_layout(const unsigned char *tail, uint32_t count,
                                            uint64_t payload, uint32_t *error) {
    uint64_t first = aotx_cog_u64(tail + 32);
    for (uint32_t j = threadIdx.x; j < count; j += blockDim.x) {
        const unsigned char *r = tail + AOTX_COG_HEADER + j * AOTX_COG_OBJECT;
        if (aotx_cog_cold(r)) { aotx_cog_error(error, AOTX_COG_FORMAT); continue; }
        uint64_t offset = aotx_cog_u64(r + AOTX_CO_OFFSET), length = aotx_cog_u64(r + AOTX_CO_BYTES);
        if (aotx_cog_u64(r + AOTX_CO_UPDATED) != first + j) aotx_cog_error(error, AOTX_COG_SEQUENCE);
        if (offset > payload || length > payload - offset || (!length && offset)) {
            aotx_cog_error(error, AOTX_COG_FORMAT); continue;
        }
        for (uint32_t k = 0; k < j; ++k) {
            const unsigned char *p = tail + AOTX_COG_HEADER + k * AOTX_COG_OBJECT;
            uint64_t start = aotx_cog_u64(p + AOTX_CO_OFFSET), bytes = aotx_cog_u64(p + AOTX_CO_BYTES);
            if (start > payload || bytes > payload - start) {
                aotx_cog_error(error, AOTX_COG_FORMAT); continue;
            }
            if (length && bytes && offset < start + bytes && start < offset + length)
                aotx_cog_error(error, AOTX_COG_FORMAT);
        }
    }
    if (!threadIdx.x) {
        uint64_t bytes = 0;
        for (uint32_t j = 0; j < count; ++j) {
            uint64_t length = aotx_cog_u64(tail + AOTX_COG_HEADER + j * AOTX_COG_OBJECT + AOTX_CO_BYTES);
            if (length > payload) { aotx_cog_error(error, AOTX_COG_FORMAT); return; }
            bytes += length;
        }
        if (bytes != payload) aotx_cog_error(error, AOTX_COG_FORMAT);
    }
}

__device__ void aotx_cognitive_restore_block(aotx_cognitive_store *live, aotx_cognitive_store *stage,
    const unsigned char *image, uint64_t bytes, aotx_cognitive_result *result) {
    if (blockIdx.x) return;
    __shared__ uint32_t error;
    if (!threadIdx.x) {
        error = UINT32_MAX;
        aotx_cog_error(&error, live == stage ? AOTX_COG_FORMAT : aotx_cog_header(image, bytes, false));
    }
    __syncthreads();
    if (error != UINT32_MAX) { aotx_cog_result(result, error, 0, 0, live->sequence); return; }
    for (uint64_t j = threadIdx.x; j < sizeof(*stage); j += blockDim.x) ((unsigned char *)stage)[j] = 0;
    __syncthreads();
    if (!threadIdx.x) {
        stage->count = aotx_cog_u32(image + 20);
        stage->bytes = (uint32_t)aotx_cog_u64(image + 24);
        stage->sequence = aotx_cog_u64(image + 32);
        stage->tick = aotx_cog_u64(image + 40);
        aotx_cog_policy_read(stage, image);
    }
    aotx_cog_copy(stage->lineage, image + 48, 16);
    __syncthreads();
    aotx_cog_copy(&stage->objects[0][0], image + AOTX_COG_HEADER, (uint64_t)stage->count * AOTX_COG_OBJECT);
    aotx_cog_copy(stage->payload, image + aotx_cog_u64(image + 72), stage->bytes);
    __syncthreads();
    aotx_cog_publish(live, stage, &error);
    aotx_cog_result(result, error, stage->count, bytes, live->sequence);
}

__device__ void aotx_cognitive_apply_block(aotx_cognitive_store *live, aotx_cognitive_store *stage,
    const unsigned char *tail, uint64_t bytes, aotx_cognitive_result *result) {
    if (blockIdx.x) return;
    __shared__ uint32_t error, skip, count, extra;
    if (!threadIdx.x) {
        error = UINT32_MAX; skip = 0; count = 0; extra = 0;
        aotx_cog_error(&error, live == stage ? AOTX_COG_FORMAT : aotx_cog_header(tail, bytes, true));
        if (error == UINT32_MAX) {
            count = aotx_cog_u32(tail + 20);
            uint64_t first = aotx_cog_u64(tail + 32);
            if (!aotx_cog_equal(live->lineage, tail + 48)) aotx_cog_error(&error, AOTX_COG_SOURCE);
            if (first > live->sequence && first - live->sequence != 1) aotx_cog_error(&error, AOTX_COG_SEQUENCE);
            if (first <= live->sequence) {
                uint64_t covered = live->sequence - first;
                skip = covered >= count - 1 ? count : (uint32_t)covered + 1;
            }
            if (live->pressure_percent && first <= live->retry_floor) aotx_cog_error(&error, AOTX_COG_STALE);
            if (count > skip && (aotx_cog_u32(tail + 8) != (live->pressure_percent ? 2u : 1u) ||
                (live->pressure_percent && (aotx_cog_u64(tail + 96) != live->root_sequence ||
                 aotx_cog_u64(tail + 104) != live->retry_floor || aotx_cog_u32(tail + 112) != live->keep_recent ||
                 aotx_cog_u32(tail + 116) != live->max_age || aotx_cog_u32(tail + 120) != live->maintenance ||
                 aotx_cog_u32(tail + 124) != live->pressure_percent)))) aotx_cog_error(&error, AOTX_COG_STALE);
            if (count - skip > AOTX_COG_OBJECTS - live->count) aotx_cog_error(&error, AOTX_COG_CAPACITY);
            if (count > skip && aotx_cog_u64(tail + 40) < live->tick) aotx_cog_error(&error, AOTX_COG_SEQUENCE);
        }
    }
    __syncthreads();
    if (error != UINT32_MAX) { aotx_cog_result(result, error, 0, 0, live->sequence); return; }
    uint64_t payload = aotx_cog_u64(tail + 24), payload_start = aotx_cog_u64(tail + 72);
    aotx_cog_tail_layout(tail, count, payload, &error);
    __syncthreads();
    if (error != UINT32_MAX) { aotx_cog_result(result, error, 0, 0, live->sequence); return; }
    /* Covered records must match the admitted bytes. A conflicting retry is a refusal. */
    for (uint32_t j = threadIdx.x; j < skip; j += blockDim.x) {
        const unsigned char *r = tail + AOTX_COG_HEADER + j * AOTX_COG_OBJECT;
        int found = aotx_cog_find(live, r + AOTX_CO_ID, aotx_cog_u64(r + AOTX_CO_VERSION));
        if (found < 0) { aotx_cog_error(&error, AOTX_COG_REFERENCE); continue; }
        const unsigned char *old = live->objects[found];
        if (aotx_cog_cold(old)) { aotx_cog_error(&error, AOTX_COG_UNAVAILABLE); continue; }
        if (!aotx_cog_equal(old, r, AOTX_CO_OFFSET) ||
            !aotx_cog_equal(old + AOTX_CO_BYTES, r + AOTX_CO_BYTES, AOTX_COG_OBJECT - AOTX_CO_BYTES)) {
            aotx_cog_error(&error, AOTX_COG_VERSION); continue;
        }
        uint64_t length = aotx_cog_u64(r + AOTX_CO_BYTES);
        if (!aotx_cog_equal(live->payload + aotx_cog_u64(old + AOTX_CO_OFFSET),
                            tail + payload_start + aotx_cog_u64(r + AOTX_CO_OFFSET), (unsigned)length))
            aotx_cog_error(&error, AOTX_COG_VERSION);
    }
    __syncthreads();
    if (!threadIdx.x) {
        uint64_t total = 0;
        for (uint32_t j = skip; j < count; ++j)
            total += aotx_cog_u64(tail + AOTX_COG_HEADER + j * AOTX_COG_OBJECT + AOTX_CO_BYTES);
        if (total > AOTX_COG_PAYLOAD - live->bytes) aotx_cog_error(&error, AOTX_COG_CAPACITY);
        else extra = (uint32_t)total;
    }
    __syncthreads();
    if (error != UINT32_MAX || count == skip) {
        aotx_cog_result(result, error, 0, 0, live->sequence); return;
    }
    aotx_cog_copy((unsigned char *)stage, (const unsigned char *)live, sizeof(*live));
    __syncthreads();
    for (uint32_t j = skip; j < count; ++j) {
        const unsigned char *r = tail + AOTX_COG_HEADER + j * AOTX_COG_OBJECT;
        unsigned char *out = stage->objects[live->count + j - skip];
        uint64_t offset = live->bytes;
        for (uint32_t k = skip; k < j; ++k)
            offset += aotx_cog_u64(tail + AOTX_COG_HEADER + k * AOTX_COG_OBJECT + AOTX_CO_BYTES);
        uint64_t length = aotx_cog_u64(r + AOTX_CO_BYTES);
        aotx_cog_copy(out, r, AOTX_COG_OBJECT);
        aotx_cog_copy(stage->payload + offset, tail + payload_start + aotx_cog_u64(r + AOTX_CO_OFFSET), length);
        __syncthreads();
        if (!threadIdx.x) aotx_cog_put(out + AOTX_CO_OFFSET, length ? offset : 0, 8);
    }
    __syncthreads();
    if (!threadIdx.x) {
        stage->count += count - skip; stage->bytes += extra;
        stage->sequence = aotx_cog_u64(tail + 32) + count - 1;
        stage->tick = aotx_cog_u64(tail + 40);
    }
    __syncthreads();
    aotx_cog_publish(live, stage, &error);
    aotx_cog_result(result, error, count - skip, 0, live->sequence);
}

__device__ void aotx_cognitive_checkpoint_header_block(const aotx_cognitive_store *live,
    unsigned char *image, uint64_t capacity, aotx_cognitive_result *result) {
    if (blockIdx.x) return;
    uint64_t payload = AOTX_COG_HEADER + (uint64_t)live->count * AOTX_COG_OBJECT;
    uint64_t bytes = payload + live->bytes;
    if (capacity < bytes) { aotx_cog_result(result, AOTX_COG_CAPACITY, 0, 0, live->sequence); return; }
    for (uint32_t j = threadIdx.x; j < AOTX_COG_HEADER; j += blockDim.x) image[j] = 0;
    __syncthreads();
    if (!threadIdx.x) {
        const char *magic = "AOTXOBJ1";
        for (unsigned j = 0; j < 8; ++j) image[j] = magic[j];
        aotx_cog_put(image + 8, 1, 4); aotx_cog_put(image + 12, AOTX_COG_HEADER, 4);
        aotx_cog_put(image + 16, AOTX_COG_OBJECT, 4); aotx_cog_put(image + 20, live->count, 4);
        aotx_cog_put(image + 24, live->bytes, 8); aotx_cog_put(image + 32, live->sequence, 8);
        aotx_cog_put(image + 40, live->tick, 8); aotx_cog_put(image + 64, AOTX_COG_HEADER, 8);
        aotx_cog_put(image + 72, payload, 8); aotx_cog_put(image + 80, bytes, 8);
        aotx_cog_put(image + 88, 1, 4);
        aotx_cog_policy_write(image, live);
    }
    aotx_cog_copy(image + 48, live->lineage, 16);
    aotx_cog_result(result, UINT32_MAX, live->count, bytes, live->sequence);
}

__device__ void aotx_cognitive_checkpoint_block(const aotx_cognitive_store *live,
    unsigned char *image, uint64_t capacity, aotx_cognitive_result *result) {
    aotx_cognitive_checkpoint_header_block(live, image, capacity, result);
    __syncthreads();
    if (result->status) return;
    uint64_t payload = AOTX_COG_HEADER + (uint64_t)live->count * AOTX_COG_OBJECT;
    aotx_cog_copy(image + AOTX_COG_HEADER, &live->objects[0][0], (uint64_t)live->count * AOTX_COG_OBJECT);
    aotx_cog_copy(image + payload, live->payload, live->bytes);
}

__global__ void aotx_cognitive_checkpoint(const aotx_cognitive_store *live,
    unsigned char *image, uint64_t capacity, aotx_cognitive_result *result) {
    aotx_cognitive_checkpoint_block(live, image, capacity, result);
}

__global__ void aotx_cognitive_restore(aotx_cognitive_store *live, aotx_cognitive_store *stage,
    const unsigned char *image, uint64_t bytes, aotx_cognitive_result *result) {
    aotx_cognitive_restore_block(live, stage, image, bytes, result);
}

__global__ void aotx_cognitive_apply(aotx_cognitive_store *live, aotx_cognitive_store *stage,
    const unsigned char *tail, uint64_t bytes, aotx_cognitive_result *result) {
    aotx_cognitive_apply_block(live, stage, tail, bytes, result);
}
