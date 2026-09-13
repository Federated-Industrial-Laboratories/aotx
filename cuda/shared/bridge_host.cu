/* Purpose: Allocate the required shared tables before replay and capture their graph nodes.
 * Owns: The device table batch and its allocation handles.
 * Launch shape: Host glue only; ordered device nodes perform all state operations.
 * Lifetime: The shared runtime; allocations do not depend on network grants. */
#include "shared/host.h"
#include "shared/bridge.cuh"
#include "disk/runtime/runtime.h"
#include <cuda_runtime.h>
#include <stdint.h>

static aotx_shared_state aotx_shared_allocations;

void aotx_shared_close(void)
{
    aotx_shared_state empty = {};
    cudaMemcpyToSymbol(aotx_shared, &empty, sizeof empty);
    cudaFree(aotx_shared_allocations.participants); cudaFree(aotx_shared_allocations.spaces);
    cudaFree(aotx_shared_allocations.members); cudaFree(aotx_shared_allocations.conversations);
    cudaFree(aotx_shared_allocations.receipts);
    aotx_shared_allocations = {};
}
static int aotx_shared_table(void **table, size_t count, size_t row_bytes)
{
    if (!count || count > SIZE_MAX / row_bytes) return 1;
    size_t bytes = count * row_bytes;
    return cudaMalloc(table, bytes) != cudaSuccess || cudaMemset(*table, 0, bytes) != cudaSuccess;
}
int aotx_shared_open(const aotx_runtime_shared_profile *p)
{
    if (!p || !aotx_runtime_shared_fits(p) || aotx_shared_allocations.enabled) return 1;
    aotx_shared_state *s = &aotx_shared_allocations;
    s->participant_capacity = p->participants; s->space_capacity = p->spaces;
    s->member_capacity = p->members; s->conversation_capacity = p->conversations;
    s->receipt_capacity = p->receipts;
    if (aotx_shared_table((void **)&s->participants, p->participants, sizeof(aotx_shared_participant)) ||
        aotx_shared_table((void **)&s->spaces, p->spaces, sizeof(aotx_shared_space)) ||
        aotx_shared_table((void **)&s->members, p->members, sizeof(aotx_shared_member)) ||
        aotx_shared_table((void **)&s->conversations, p->conversations, sizeof(aotx_shared_conversation)) ||
        aotx_shared_table((void **)&s->receipts, p->receipts, sizeof(aotx_shared_receipt))) {
        aotx_shared_close(); return 1;
    }
    void *execution = NULL;
    s->enabled = 1;
    if (cudaGetSymbolAddress(&execution, aotx_shared_execution_slots) != cudaSuccess ||
        cudaMemset(execution, 0, sizeof(aotx_shared_execution) * AOTX_SLOTS) != cudaSuccess ||
        cudaMemcpyToSymbol(aotx_shared, s, sizeof *s) != cudaSuccess) {
        aotx_shared_close(); return 1;
    }
    return 0;
}
void aotx_shared_capture(void *stream)
{
    if (aotx_shared_allocations.enabled)
        aotx_shared_work<<<1, 1, 0, (cudaStream_t)stream>>>();
}
void aotx_shared_output_capture(void *stream)
{
    if (!aotx_shared_allocations.enabled) return;
    aotx_shared_results<<<1, 1, 0, (cudaStream_t)stream>>>();
    aotx_shared_emit<<<1, 1, 0, (cudaStream_t)stream>>>();
}
