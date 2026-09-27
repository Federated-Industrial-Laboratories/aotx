/* Purpose: Commit execution leases, exact output bytes and final result metadata.
 * Owns: Complete shared execution transitions and their validation.
 * Launch shape: Bounded lease rows and one ordered record consumer.
 * Lifetime: One persistent operation receipt. */
#include "shared/internal.cuh"
#include "shared/affect.cuh"
#include "cognitive/intake_capability.cuh"
static __device__ unsigned aotx_shared_apply_requests[AOTX_SLOTS], aotx_shared_apply_slots[AOTX_SLOTS];
static __device__ unsigned aotx_shared_apply_retention[AOTX_SLOTS];
static __device__ unsigned aotx_shared_apply_affect[AOTX_SLOTS];
static __device__ unsigned char aotx_shared_complete_affect[64];
static __device__ bool aotx_shared_match(const unsigned char *p, unsigned index, unsigned actor_at, unsigned sequence_at)
{
    return index < aotx_shared.receipt_capacity && aotx_shared.receipts[index].phase &&
        aotx_shared_id(aotx_shared.receipts[index].actor, p + actor_at) &&
        aotx_shared.receipts[index].sequence == aotx_shared_u64(p + sequence_at);
}
__device__ bool aotx_shared_lease(const unsigned *requests, const unsigned *slots, unsigned count)
{
    if (!count || count > AOTX_SLOTS) return false;
    unsigned revision = 4;
    for (unsigned i = 0; i < count; ++i) {
        if (requests[i] >= aotx_shared.receipt_capacity || slots[i] >= AOTX_SLOTS || aotx_shared.slot[slots[i]]) return false;
        const aotx_shared_receipt &r = aotx_shared.receipts[requests[i]];
        const aotx_service_grant *g = aotx_service_granted(r.actor);
        if (r.phase != AOTX_SHARED_QUEUED || !r.saved_admission || r.cancel ||
            !g || g->revision != r.revision || !aotx_shared_authorized(&r, 2)) return false;
#ifdef AOTX_AFFECT
        if (aotx_shared_affect_managed(&r)) revision = 5;
#endif
        for (unsigned j = 0; j < i; ++j) {
            if (requests[i] == requests[j] || slots[i] == slots[j]) return false;
#ifdef AOTX_AFFECT
            if (aotx_shared_affect_conflict(&r, &aotx_shared.receipts[requests[j]])) return false;
#endif
        }
    }
    if (!aotx_shared_begin(AOTX_SHARED_LEASE_RECORD, 8 + count * 40)) return false;
    aotx_service_put(aotx_shared.transfer, count, 4);
    aotx_service_put(aotx_shared.transfer + 4, revision, 4);
    for (unsigned i = 0; i < count; ++i) {
        unsigned char *p = aotx_shared.transfer + 8 + i * 40;
        aotx_service_put(p, requests[i], 4); aotx_service_put(p + 4, slots[i], 4);
        aotx_service_put(p + 8, aotx_shared.receipts[requests[i]].sequence, 8);
        aotx_service_bytes(p + 16, aotx_shared.receipts[requests[i]].actor, 16);
        aotx_service_put(p + 32, aotx_intake_qualified(aotx_shared.receipts[requests[i]].role) ? 2 : 1, 4);
#ifdef AOTX_AFFECT
        aotx_service_put(p + 36, aotx_shared_affect_managed(&aotx_shared.receipts[requests[i]]), 4);
#endif
    }
    return true;
}
__device__ bool aotx_shared_output(unsigned request, const unsigned char *bytes, unsigned count)
{
    if (request >= aotx_shared.receipt_capacity || !count || count > AOTX_SHARED_TRANSFER - 48) return false;
    const aotx_shared_receipt &r = aotx_shared.receipts[request];
    const aotx_service_grant *g = aotx_service_granted(r.actor);
    if (r.phase != AOTX_SHARED_RUNNING || r.output > AOTX_SHARED_RESULT_BYTES ||
        count > AOTX_SHARED_RESULT_BYTES - r.output || r.cancel || !g || g->revision != r.revision ||
        !aotx_shared_authorized(&r, 2) || !aotx_shared_begin(AOTX_SHARED_OUTPUT_RECORD, 48 + count)) return false;
    unsigned char *p = aotx_shared.transfer;
    aotx_service_put(p, request, 4); aotx_service_put(p + 4, r.output, 4); aotx_service_put(p + 8, count, 4);
    aotx_service_bytes(p + 16, r.actor, 16); aotx_service_put(p + 32, r.sequence, 8);
    aotx_service_bytes(p + 48, bytes, count); return true;
}
__device__ bool aotx_shared_complete(unsigned request, unsigned status, unsigned prompt, unsigned sampled, unsigned finish)
{
    if (request >= aotx_shared.receipt_capacity || status < 200 || status > 599) return false;
    const aotx_shared_receipt &r = aotx_shared.receipts[request];
    unsigned affect = 0;
#ifdef AOTX_AFFECT
    affect = aotx_shared_affect_encode(&r, status, finish, aotx_shared_complete_affect);
#endif
    if (!r.phase || r.terminal_source || !aotx_shared_begin(AOTX_SHARED_COMPLETE_RECORD, 64 + affect)) return false;
    unsigned char *p = aotx_shared.transfer;
    aotx_service_put(p, request, 4); aotx_service_put(p + 4, status, 4);
    aotx_service_put(p + 8, prompt, 4); aotx_service_put(p + 12, sampled, 4); aotx_service_put(p + 16, finish, 4);
    aotx_service_put(p + 20, r.cancel, 4); aotx_service_put(p + 24, r.output, 4);
    aotx_service_bytes(p + 32, r.actor, 16); aotx_service_put(p + 48, r.sequence, 8);
    if (affect) {
        aotx_service_put(p + 28, 1, 4);
        aotx_service_bytes(p + 64, aotx_shared_complete_affect, affect);
    }
    return true;
}
static __device__ __noinline__ bool aotx_shared_execution_apply(unsigned kind, const unsigned char *p, unsigned n,
                                  unsigned long long source, bool replay)
{
    if (kind == AOTX_SHARED_LEASE_RECORD) {
        unsigned count = n >= 8 ? aotx_shared_u32(p) : 0;
        unsigned *requests = aotx_shared_apply_requests, *slots = aotx_shared_apply_slots;
        unsigned revision = n >= 8 ? aotx_shared_u32(p + 4) : ~0u, stride = revision >= 2 ? 40 : 32;
        unsigned *retention = aotx_shared_apply_retention, *affect = aotx_shared_apply_affect;
        if (!count || count > AOTX_SLOTS || revision > 5 || n != 8 + count * stride) return false;
        for (unsigned i = 0; i < count; ++i) {
            const unsigned char *row = p + 8 + i * stride;
            requests[i] = aotx_shared_u32(row); slots[i] = aotx_shared_u32(row + 4);
            retention[i] = revision >= 2 ? aotx_shared_u32(row + 32) : 2;
            affect[i] = revision >= 2 ? aotx_shared_u32(row + 36) : 0;
            if (revision >= 2 && ((retention[i] != 1 && retention[i] != 2) ||
                affect[i] > (revision == 5 ? 1u : 0u))) return false;
            if (!aotx_shared_match(row, requests[i], 16, 8) || slots[i] >= AOTX_SLOTS || aotx_shared.slot[slots[i]]) return false;
            const aotx_shared_receipt &r = aotx_shared.receipts[requests[i]];
            if (r.phase != AOTX_SHARED_QUEUED || r.slot != AOTX_SLOTS) return false;
            for (unsigned j = 0; j < i; ++j) if (requests[i] == requests[j] || slots[i] == slots[j]) return false;
        }
        if (!aotx_shared_bridge_lease(requests, slots, count, replay, revision, retention, affect)) return false;
        for (unsigned i = 0; i < count; ++i) {
            aotx_shared_receipt &r = aotx_shared.receipts[requests[i]];
            r.phase = AOTX_SHARED_RUNNING; r.slot = slots[i]; aotx_shared.slot[slots[i]] = requests[i] + 1;
        }
        return true;
    }
    unsigned request = n >= 4 ? aotx_shared_u32(p) : AOTX_SHARED_NONE;
    if (kind == AOTX_SHARED_OUTPUT_RECORD) {
        if (n <= 48 || !aotx_shared_match(p, request, 16, 32) || aotx_shared_u32(p + 12) || aotx_shared_u64(p + 40)) return false;
        aotx_shared_receipt &r = aotx_shared.receipts[request];
        unsigned count = aotx_shared_u32(p + 8);
        if (r.phase != AOTX_SHARED_RUNNING || count != n - 48 || aotx_shared_u32(p + 4) != r.output ||
            count > AOTX_SHARED_RESULT_BYTES - r.output) return false;
        aotx_service_bytes(r.result + r.output, p + 48, count); r.output += count; return true;
    }
    if (kind != AOTX_SHARED_COMPLETE_RECORD || (n != 64 && n != 128) ||
        !aotx_shared_match(p, request, 32, 48) ||
        aotx_shared_u32(p + 28) != (n == 128 ? 1u : 0u) || aotx_shared_u64(p + 56)) return false;
    aotx_shared_receipt &r = aotx_shared.receipts[request];
    unsigned status = aotx_shared_u32(p + 4);
    if (r.terminal_source || status < 200 || status > 599 || aotx_shared_u32(p + 24) != r.output ||
        aotx_shared_u32(p + 20) != r.cancel) return false;
    if (r.sample.affect && status != 598 && n == 64 &&
        (aotx_shared_u32(p + 8) || aotx_shared_u32(p + 12))) return false;
    if (n == 128 && (!aotx_shared_u32(p + 8) || status == 598)) return false;
    if (n == 128) {
#ifdef AOTX_AFFECT
        if (!aotx_shared_affect_apply(&r, p + 64, 64)) return false;
#else
        return false;
#endif
    }
    r.status = status; r.prompt = aotx_shared_u32(p + 8); r.sampled = aotx_shared_u32(p + 12);
    r.finish = aotx_shared_u32(p + 16); r.terminal_source = source;
    r.phase = status == 598 ? AOTX_SHARED_INTERRUPTED : r.cancel ? AOTX_SHARED_CANCELLED :
        status < 400 ? AOTX_SHARED_DONE : AOTX_SHARED_FAILED;
    if (status == 598) r.gap = 1;
    if (r.slot < AOTX_SLOTS) {
        unsigned slot = r.slot;
        aotx_shared_bridge_release(request, replay); aotx_shared.slot[slot] = 0;
    }
    r.slot = AOTX_SLOTS;
    if (r.conversation < aotx_shared.conversation_capacity &&
        aotx_shared.conversations[r.conversation].request == request + 1) aotx_shared.conversations[r.conversation].request = 0;
    return true;
}

__device__ bool aotx_shared_apply(unsigned kind, const unsigned char *p, unsigned n,
    unsigned long long source, bool replay)
{
    if (kind == AOTX_SHARED_ADMIT_RECORD) return aotx_shared_admission_apply(p, n, source, replay);
    return aotx_shared_execution_apply(kind, p, n, source, replay);
}
