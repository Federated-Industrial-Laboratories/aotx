/* Purpose: Check publication and replay hashes for ordinary and service token batches.
 * Owns: Distinct sequence fixtures, a device record ring and an independent host hash.
 * Launch shape: The real commit kernel at N=1 and N=64.
 * Lifetime: Each batch releases its isolated record ring. */
#include "model/decode_state.cuh"
#include "service/service.cuh"
#include "cognitive/intake.cuh"
#include "sched/sched.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static unsigned checks, failures;
static void check(bool good, const char *label)
{ ++checks; if (!good) { ++failures; fprintf(stderr, "FAIL %s\n", label); } }
static void cu(cudaError_t status)
{ if (status != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(status)); exit(1); } }
__global__ void aotx_service_journal_seed(unsigned n, unsigned mode)
{
    unsigned i = threadIdx.x;
    aotx_seqs.slot[i] = {};
    aotx_seq_kept[i] = 0; aotx_intake.row[i] = 0;
    aotx_service.slot[i] = mode == 2 || (mode == 1 && i % 2) ? i + 1 : 0;
    if (!i) {
        aotx_service.enabled = mode != 0; aotx_sched.held = 0;
        aotx_decode.role = AOTX_MODEL_LANGUAGE; aotx_time_tick = 17;
        aotx_model_wrap[AOTX_MODEL_LANGUAGE] = {};
    }
    if (i >= n) return;
    aotx_seq &s = aotx_seqs.slot[i];
    s.state = AOTX_SEQ_STATE_PREFILL; s.role = AOTX_MODEL_LANGUAGE;
    s.prompt = 2; s.limit = 1; s.stop = ~0u; s.seed = 700 + i;
    s.opened = 16;
    aotx_seqs.tokens[i][0] = 100 + i; aotx_seqs.tokens[i][1] = 200 + i;
    aotx_decode.rows[i] = 2; aotx_decode.first[i] = 0; aotx_decode.place[i] = i;
    aotx_decode.token[i] = 300 + i; aotx_decode.draw[i] = 400 + i;
    aotx_model_seen[i] = 2;
}
static void run(unsigned n, unsigned mode)
{
    unsigned char *ring = nullptr;
    cu(cudaMalloc(&ring, 1024 * AOTX_SLOT_BYTES));
    cu(cudaMemset(ring, 0, 1024 * AOTX_SLOT_BYTES));
    aotx_seam_state seam = {};
    seam.dev.base = ring; seam.dev.slot_count = 1024; seam.dev.mask = 1023;
    seam.apply.state_hash = 14695981039346656037ull;
    cu(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof seam));
    aotx_service_journal_seed<<<1,AOTX_SLOTS>>>(n, mode);
    aotx_decode_commit<<<1,AOTX_SLOTS>>>(17); cu(cudaDeviceSynchronize());
    cu(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam));
    check(seam.dev.tail == 4 * n, "each slot publishes three tokens and one completion");
    std::vector<unsigned char> records((size_t)seam.dev.tail * AOTX_SLOT_BYTES);
    cu(cudaMemcpy(records.data(), ring, records.size(), cudaMemcpyDeviceToHost));
    unsigned applied = 0, events = 0;
    unsigned long long hash = 14695981039346656037ull;
    for (unsigned r = 0; r < seam.dev.tail; ++r) {
        const auto *h = (const aotx_record_header *)(records.data() + r * AOTX_SLOT_BYTES);
        check(h->seq == r + 1, "hash reads preserve every record publication marker");
        if (h->type == AOTX_REC_SEQUENCE) { ++events; continue; }
        const auto *body = (const aotx_token_body *)((const unsigned char *)h + AOTX_HEADER_BYTES);
        unsigned slot = r / 3, position = r % 3;
        bool service = mode == 2 || (mode == 1 && slot % 2);
        check(h->type == (service ? AOTX_REC_SERVICE_TOKEN : AOTX_REC_TOKEN) &&
            h->cls == (service ? AOTX_CLASS_B : AOTX_CLASS_A), "each token has its owner's replay class");
        check(body->slot == slot && body->position == position && body->token == 100 * (position + 1) + slot &&
            body->seed == 700 + slot, "token identity survives the mixed batch");
        if (!service) {
            const unsigned char *bytes = (const unsigned char *)body;
            for (unsigned j = 0; j < sizeof *body; ++j) { hash ^= bytes[j]; hash *= 1099511628211ull; }
            ++applied;
        }
    }
    check(events == n, "all sequence completions remain published");
    check(seam.apply.applied_count == applied && seam.apply.state_hash == hash,
        "only ordinary tokens enter the independent replay hash");
    cu(cudaFree(ring));
}
int main(void)
{
    for (unsigned n : {1u, AOTX_SLOTS}) for (unsigned mode = 0; mode < 3; ++mode) run(n, mode);
    printf("service journal: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
