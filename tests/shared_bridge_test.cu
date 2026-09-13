/* Purpose: Check memory stage stop reasons and interrupted result recovery.
 * Owns: Distinct receipt and slot fixtures at N=1 and N=64.
 * Launch shape: Batches of the 63 available shared execution slots.
 * Lifetime: Each callback preserves its recorded memory effect and prior stop reason. */
#include "shared/bridge.cuh"
#include "shared/internal.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
static unsigned checks, failures;
static void cu(cudaError_t status)
{ if (status != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); std::exit(1); } }
__global__ void aotx_shared_bridge_gap_test(unsigned count, unsigned *out)
{
    unsigned i = threadIdx.x;
    if (i < count) {
        aotx_shared_receipt &r = aotx_shared.receipts[i];
        r = {}; r.phase = AOTX_SHARED_QUEUED; r.slot = AOTX_SLOTS;
        r.conversation = AOTX_SHARED_NONE; r.actor[0] = i + 1; r.sequence = i + 1;
        unsigned char p[64] = {};
        aotx_service_put(p, i, 4); aotx_service_put(p + 4, 598, 4);
        aotx_service_put(p + 8, i + 17, 4); aotx_service_put(p + 12, i + 3, 4);
        aotx_service_bytes(p + 32, r.actor, 16); aotx_service_put(p + 48, r.sequence, 8);
        out[i] = aotx_shared_apply(AOTX_SHARED_COMPLETE_RECORD, p, sizeof(p), i + 500, true);
    }
    __syncthreads();
    if (!i) aotx_shared_restore_end();
    __syncthreads();
    if (i < count) {
        const aotx_shared_receipt &r = aotx_shared.receipts[i];
        out[i] &= r.phase == AOTX_SHARED_INTERRUPTED && r.status == 598 && r.gap &&
            r.saved_admission && r.saved_terminal && r.prompt == i + 17 && r.sampled == i + 3;
    }
}
__global__ void aotx_shared_bridge_callback_test(unsigned first, unsigned count, unsigned prior,
                                                unsigned memory_status, unsigned *out)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned index = first + i, slot = i + 1;
    aotx_shared_receipt &r = aotx_shared.receipts[index];
    r = {}; r.phase = AOTX_SHARED_RUNNING; r.slot = slot; r.actor[0] = index + 1;
    aotx_shared.slot[slot] = index + 1;
    aotx_shared_execution &x = aotx_shared_execution_slots[slot];
    x = {}; x.stage = AOTX_SHARED_MEMORY; x.status = prior; x.opened = index + 17;
    aotx_shared_memory_choice(slot, memory_status);
    unsigned wanted = prior ? prior : memory_status ? 503 : 0;
    out[index] = (x.status == wanted ? 1u : 0u) |
        (x.stage == (wanted ? AOTX_SHARED_END : AOTX_SHARED_PROMPT) ? 2u : 0u) |
        (r.input_committed == !memory_status ? 4u : 0u) |
        (x.opened == index + 17 && r.actor[0] == index + 1 ? 8u : 0u);
    aotx_shared.slot[slot] = 0;
}
static void run(unsigned count)
{
    aotx_shared_state state = {}; state.enabled = 1; state.receipt_capacity = count;
    cu(cudaMalloc(&state.receipts, count * sizeof(*state.receipts)));
    cu(cudaMemcpyToSymbol(aotx_shared, &state, sizeof(state)));
    unsigned *result; cu(cudaMalloc(&result, count * sizeof(*result)));
    std::vector<unsigned> values(count);
    for (unsigned prior : {0u, 403u, 409u, 504u}) {
        for (unsigned memory_status : {0u, 1u}) {
            for (unsigned first = 0; first < count; first += AOTX_SLOTS - 1) {
                unsigned n = count - first;
                if (n > AOTX_SLOTS - 1) n = AOTX_SLOTS - 1;
                aotx_shared_bridge_callback_test<<<1,64>>>(first, n, prior, memory_status, result);
            }
            cu(cudaDeviceSynchronize());
            cu(cudaMemcpy(values.data(), result, count * sizeof(*result), cudaMemcpyDeviceToHost));
            for (unsigned i = 0; i < count; ++i) {
                ++checks;
                if (values[i] != 15) {
                    ++failures; std::fprintf(stderr, "FAIL callback N=%u row=%u prior=%u memory=%u flags=%u\n",
                        count, i, prior, memory_status, values[i]);
                }
            }
        }
    }
    aotx_shared_bridge_gap_test<<<1,64>>>(count, result); cu(cudaDeviceSynchronize());
    cu(cudaMemcpy(values.data(), result, count * sizeof(*result), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < count; ++i) {
        ++checks;
        if (!values[i]) { ++failures; std::fprintf(stderr, "FAIL interrupted gap N=%u row=%u\n", count, i); }
    }
    cudaFree(result); cudaFree(state.receipts);
    std::printf("shared-bridge N=%u checks=%u failures=%u\n", count, checks, failures);
}
int main()
{
    run(1); run(64);
    std::printf("shared-bridge total checks=%u failures=%u\n", checks, failures);
    return failures ? 1 : 0;
}
