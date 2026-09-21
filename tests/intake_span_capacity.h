/* Purpose: Verify exact compact output reservations before either decoder call.
 * Owns: Independent byte arithmetic, escaped quotes and source span limits.
 * Launch shape: N=1 and N=64 bounded source rows.
 * Lifetime: One capacity fixture without model weights. */
#ifndef AOTX_TEST_INTAKE_SPAN_CAPACITY_H
#define AOTX_TEST_INTAKE_SPAN_CAPACITY_H
__global__ void aotx_span_capacity_run(unsigned n, unsigned phase, unsigned *out) {
    unsigned i = threadIdx.x; if (i >= n) return;
    auto *r = aotx_intake.rows + i; r->phase = phase;
    const auto *q = aotx_live.requests + 64 + i * AOTX_RECALL_QUERY;
    bool okay = aotx_source_split(q + 4640, aotx_cog_u32(q + 148), aotx_intake_index_rows[i].sizes,
        r->statements, &r->source_count, true);
    r->first_count = r->source_count;
    out[i] = okay ? aotx_intake_span_capacity(i) : AOTX_COG_SOURCE;
    out[n + i] = r->source_count;
}
static void aotx_stage_span_capacity(unsigned n) {
    aotx_intake_device d(n); aotx_fixture empty;
    d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
    unsigned *out; AOTX_CUDA(cudaMallocManaged(&out, 2 * n * sizeof(*out)));
    for (unsigned phase : {1u, 2u}) for (bool escape : {false, true}) for (unsigned mode = 0; mode < 3; ++mode) {
        unsigned count = phase == 1 ? 215 : 341, extra = phase == 1 ? 9 + mode : 2 + mode;
        auto query = aotx_stage_query(n);
        for (unsigned i = 0; i < n; ++i) {
            std::string source(extra - (escape ? 1 : 0), 'A');
            if (escape) source += '\\';
            for (unsigned j = 0; j < count; ++j) source += j ? " A!" : "A!";
            auto q = query.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
            memset(q + 4640, 0, 2048); memcpy(q + 4640, source.data(), source.size()); aotx_put(q + 148, source.size(), 4);
        }
        for (unsigned i = 0; i < n; ++i)
            AOTX_CUDA(cudaMemcpyToSymbol(aotx_live, query.data() + 128 + i * AOTX_LIVE_QUERY_ROW, AOTX_RECALL_QUERY,
                offsetof(aotx_live_state, requests) + 64 + i * AOTX_RECALL_QUERY));
        aotx_span_capacity_run<<<1,64>>>(n, phase, out); AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) {
            unsigned reserved = 1 + 2 * count + extra + (escape ? 1 : 0) + (phase == 1 ? 17 : 10) * count;
            aotx_check(out[n + i] == count, "capacity case retains every complete source span");
            aotx_check(out[i] == (reserved <= 4096 ? AOTX_COG_OK : AOTX_COG_CAPACITY),
                "compact output reserve accepts its exact bound and refuses one additional byte");
        }
    }
    cudaFree(out);
}
#endif
