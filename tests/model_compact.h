/* Purpose: Check a mixed model layer address on two mapped key and value pages.
 * Owns: Nothing; the model check owns the page map and device buffers.
 * Launch shape: One thread addresses one row; attention uses one token and one head.
 * Lifetime: One run of the model check. */
#ifndef AOTX_TEST_MODEL_COMPACT_H
#define AOTX_TEST_MODEL_COMPACT_H

/* Request or release the two pages of the mixed state fixture. */
__global__ void aotx_test_compact_pages(unsigned int pages)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    if (pages == 0u) {
        aotx_kv_release(0u);
    } else {
        aotx_kv_request(0u, pages);
    }
}

/* Read and then write the last row of the mixed state fixture through the layout. */
__global__ void aotx_test_compact_rows(aotx_kvl_shape shape, float *read)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    half *key = aotx_kvl_key(&shape, 0u, 2u, 0u, 480u);
    half *value = aotx_kvl_value(&shape, 0u, 2u, 0u, 480u);
    if (key == 0 || value == 0) {
        read[0] = -1.0f;
        read[1] = -1.0f;
        return;
    }
    read[0] = __half2float(*key);
    read[1] = __half2float(*value);
    *key = __float2half(7.5f);
    *value = __float2half(-8.25f);
}

/* Check the mixed physical-layer to compact-state address on two mapped pages. */
static void aotx_test_compact(aotx_kv_map *map)
{
    const unsigned char state[] = {
        AOTX_STATE_KIND_KV_PAGES,
        AOTX_STATE_KIND_DELTA_STATE,
        AOTX_STATE_KIND_KV_PAGES
    };
    aotx_kvl_shape shape;
    aotx_kvl_make_states(&shape, state, 3u, 8u, 128u);
    aotx_test_compact_pages<<<1, 1>>>(2u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    int served = aotx_kv_serve(map, 0);
    aotx_kv_table table;
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_kv, sizeof table),
                       "cudaMemcpyFromSymbol");
    aotx_test_note("mixed fixture maps two pages",
                   served == 1 && table.count[0] == 2u && table.mapped_pages == 2u,
                   "mapped", (double)table.mapped_pages, 2.0);

    unsigned int block = (480u / AOTX_KVL_BLOCK) * shape.state_layers
                       + shape.state_layer[2];
    unsigned long long at = (unsigned long long)(block % shape.blocks_page)
                          * shape.block_bytes;
    half *key_at = (half *)(table.page[0][1] + AOTX_KVL_HEADER + at);
    half *value_at = key_at
                   + (unsigned long long)shape.kv_heads * AOTX_KVL_BLOCK * shape.head_dim;
    half key_seed = __float2half(3.25f);
    half value_seed = __float2half(-4.5f);
    aotx_check_runtime(cudaMemcpy(key_at, &key_seed, sizeof key_seed, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(value_at, &value_seed, sizeof value_seed,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    float *read = (float *)aotx_test_take(2u * sizeof(float));
    aotx_test_compact_rows<<<1, 1>>>(shape, read);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    float seen[2];
    half key_written;
    half value_written;
    aotx_check_runtime(cudaMemcpy(seen, read, sizeof seen, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(&key_written, key_at, sizeof key_written,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(&value_written, value_at, sizeof value_written,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_test_note("mixed layer rows use compact block",
                   seen[0] == 3.25f && seen[1] == -4.5f
                   && __half2float(key_written) == 7.5f
                   && __half2float(value_written) == -8.25f,
                   "block", (double)block, 61.0);
    cudaFree(read);

    for (unsigned int page = 0u; page < 2u; ++page) {
        aotx_check_runtime(cudaMemset((void *)(table.page[0][page] + AOTX_KVL_HEADER), 0,
                                      (size_t)(AOTX_KV_PAGE_BYTES - AOTX_KVL_HEADER)),
                           "cudaMemset");
    }
    unsigned int host_offset[2] = { 0u, 1u };
    unsigned int host_agent = 0u;
    unsigned int host_base = 480u;
    unsigned int *offset = (unsigned int *)aotx_test_take(sizeof host_offset);
    unsigned int *agent = (unsigned int *)aotx_test_take(sizeof host_agent);
    unsigned int *base = (unsigned int *)aotx_test_take(sizeof host_base);
    half *query = (half *)aotx_test_take(128u * sizeof(half));
    half *attention = (half *)aotx_test_take(128u * sizeof(half));
    aotx_check_runtime(cudaMemcpy(offset, host_offset, sizeof host_offset,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(agent, &host_agent, sizeof host_agent,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(base, &host_base, sizeof host_base,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");

    aotx_model_desc desc = {};
    desc.role = AOTX_MODEL_LANGUAGE;
    desc.layers = 3u;
    desc.heads = 8u;
    desc.kv_heads = 8u;
    desc.head_dim = 128u;
    aotx_model_work work = {};
    work.qh = query;
    work.att = attention;
    work.base = base;
    work.shape = shape;
    aotx_model_run run = {};
    run.offset = offset;
    run.agent = agent;
    run.seqs = 1u;
    run.tokens = 1u;
    run.telemetry = 1u;
    static float none[AOTX_SLOTS * AOTX_KV_PAGES_EACH];
    static float mass[AOTX_SLOTS * AOTX_KV_PAGES_EACH];
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &desc, sizeof desc,
                                          (size_t)AOTX_MODEL_LANGUAGE * sizeof desc),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_space, &work, sizeof work,
                                          (size_t)AOTX_MODEL_LANGUAGE * sizeof work),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &run, sizeof run,
                                          (size_t)AOTX_MODEL_LANGUAGE * sizeof run),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_page_mass, none, sizeof none),
                       "cudaMemcpyToSymbol");
    aotx_model_attend<<<dim3(1u, 1u), AOTX_MODEL_ATTN_THREADS>>>(AOTX_MODEL_LANGUAGE, 2u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(mass, aotx_page_mass, sizeof mass),
                       "cudaMemcpyFromSymbol");
    float page_one = 241.0f / 481.0f;
    aotx_test_note("attention mass uses compact page",
                   fabsf(mass[0] - 240.0f / 481.0f) < 2.0e-4f
                   && fabsf(mass[1] - page_one) < 2.0e-4f && mass[2] == 0.0f,
                   "mass", (double)mass[1], (double)page_one);
    unsigned int no_fault = 0u;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_faults, &no_fault, sizeof no_fault),
                       "cudaMemcpyToSymbol");

    cudaFree(attention);
    cudaFree(query);
    cudaFree(base);
    cudaFree(agent);
    cudaFree(offset);
    aotx_test_compact_pages<<<1, 1>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_kv_serve(map, 0);
}

#endif
