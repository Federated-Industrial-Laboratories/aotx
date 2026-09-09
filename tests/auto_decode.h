/* Purpose: Check language token publication after an embedding pass uses a free slot.
 * Owns: Distinct shared cache cursors and active language token fixtures.
 * Launch shape: One and 64 slots through the actual decode plan and commit.
 * Lifetime: One automatic memory test process. */
#ifndef AOTX_TEST_AUTO_DECODE_H
#define AOTX_TEST_AUTO_DECODE_H

__global__ void aotx_auto_decode_seed(unsigned n, bool active) {
    unsigned i = threadIdx.x;
    if (!i) { aotx_decode.role = AOTX_MODEL_LANGUAGE; aotx_decode.ready = 1; }
    if (i >= n) return;
    auto s = aotx_seqs.slot + i;
    s->state = active ? AOTX_SEQ_STATE_PREFILL : AOTX_SEQ_STATE_FREE;
    s->role = AOTX_MODEL_LANGUAGE; s->prompt = active ? 32 : 0;
    s->limit = 1; s->page_limit = 16; s->seed = 7 + i;
    aotx_model_seen[i] = active ? 2 + i % 7 : 13 + i;
    aotx_decode.first[i] = 0; aotx_decode.rows[i] = active ? 2 + i % 7 : 0;
    for (unsigned j = 0; j < 32; ++j) aotx_seqs.tokens[i][j] = 1000 + i * 32 + j;
}
static void aotx_auto_decode(unsigned n) {
    for (unsigned active = 0; active < 2; ++active) {
        aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_seqs); AOTX_LIVE_CLEAR(aotx_seq_kept); AOTX_LIVE_CLEAR(aotx_decode);
        aotx_auto_decode_seed<<<1,64>>>(n, active);
        if (!active) aotx_decode_plan<<<1,AOTX_SLOTS>>>(1);
        AOTX_CUDA(cudaDeviceSynchronize()); auto before = d.seam();
        aotx_decode_commit<<<1,AOTX_SLOTS>>>(1); AOTX_CUDA(cudaDeviceSynchronize()); auto after = d.seam();
        unsigned expected = 0;
        for (unsigned i = 0; i < n; ++i) {
            unsigned kept = 0; AOTX_CUDA(cudaMemcpyFromSymbol(&kept, aotx_seq_kept, sizeof(kept), i * sizeof(kept)));
            unsigned count = active ? 2 + i % 7 : 0; expected += count;
            aotx_check(kept == count, "only active language rows advance the token journal cursor");
        }
        aotx_check(after.dev.tail - before.dev.tail == expected, "free cache cursors publish no language token records");
        if (!active) aotx_check(after.apply.state_hash == before.apply.state_hash, "free language slots leave the state hash unchanged");
        else {
            aotx_live_records records(expected);
            AOTX_CUDA(cudaMemcpy(records.data(), d.out + before.dev.tail * AOTX_SLOT_BYTES,
                expected * AOTX_SLOT_BYTES, cudaMemcpyDeviceToHost));
            for (unsigned i = 0, at = 0; i < n; ++i) for (unsigned j = 0; j < 2 + i % 7; ++j, ++at) {
                auto h = (const aotx_record_header *)records[at].data();
                auto b = (const aotx_token_body *)(records[at].data() + AOTX_HEADER_BYTES);
                aotx_check(h->type == AOTX_REC_TOKEN && b->slot == i && b->position == j &&
                    b->token == 1000 + i * 32 + j && b->role == AOTX_MODEL_LANGUAGE && b->flags == AOTX_TOKEN_PROMPT,
                    "active language rows still publish their exact distinct token records");
            }
        }
    }
}
#endif
