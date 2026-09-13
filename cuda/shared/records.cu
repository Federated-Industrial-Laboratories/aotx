/* Purpose: Record bounded shared transitions and restore complete transfers.
 * Owns: Ordered class A parts, complete transfer publication and source watermarks.
 * Launch shape: One emitter thread writes at most sixteen records per tick.
 * Lifetime: Admission through complete runtime replay. */
#include "shared/internal.cuh"
static __device__ unsigned char aotx_shared_record_body[192];
__device__ bool aotx_shared_begin(unsigned kind, unsigned bytes)
{
    if (!aotx_shared.enabled || aotx_shared.fatal || aotx_shared.kind || aotx_shared.received ||
        aotx_seam.replaying || !bytes || bytes > AOTX_SHARED_TRANSFER || aotx_shared.serial >= ~0ull / 64 - 1) return false;
    aotx_shared.kind = kind; aotx_shared.total = bytes; aotx_shared.written = 0;
    aotx_shared.transfer_serial = ++aotx_shared.serial;
    aotx_shared_zero(aotx_shared.transfer, bytes);
    return true;
}
__global__ void aotx_shared_emit(void)
{
    if (threadIdx.x || blockIdx.x || !aotx_shared.enabled || aotx_shared.fatal ||
        aotx_sched.held || aotx_seam.replaying || !aotx_shared.kind) return;
    unsigned char *body = aotx_shared_record_body;
    for (unsigned part = 0; part < AOTX_SHARED_EMIT && aotx_shared.written < aotx_shared.total; ++part) {
        unsigned bytes = min(aotx_shared.total - aotx_shared.written, AOTX_SHARED_RECORD_DATA);
        aotx_shared_zero(body, 32);
        aotx_service_put(body, 1, 4); aotx_service_put(body + 4, aotx_shared.kind, 4);
        aotx_service_put(body + 8, aotx_shared.transfer_serial, 8);
        aotx_service_put(body + 24, aotx_shared.total, 4); aotx_service_put(body + 28, aotx_shared.written, 4);
        aotx_service_bytes(body + 32, aotx_shared.transfer + aotx_shared.written, bytes);
        aotx_shared.source = aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_A, AOTX_SHARED_RECORD, 0, body, 32 + bytes);
        aotx_seam.apply.state_hash = aotx_seam_fnv1a(aotx_seam.apply.state_hash, body, 32 + bytes);
        ++aotx_seam.apply.applied_count; aotx_shared.written += bytes;
    }
    if (aotx_shared.written != aotx_shared.total) return;
    unsigned kind = aotx_shared.kind;
    if (!aotx_shared_apply(kind, aotx_shared.transfer, aotx_shared.total, aotx_shared.source, false)) aotx_shared.fatal = 1;
    aotx_shared.kind = aotx_shared.total = aotx_shared.written = 0;
}
__device__ void aotx_shared_part(const unsigned char *body, unsigned bytes, unsigned long long source)
{
    if (!aotx_shared.enabled || aotx_shared.fatal || !aotx_seam.replaying || bytes <= 32 || bytes > 192) {
        aotx_shared.fatal = 1; return;
    }
    unsigned kind = aotx_shared_u32(body + 4), total = aotx_shared_u32(body + 24), offset = aotx_shared_u32(body + 28);
    unsigned long long serial = aotx_shared_u64(body + 8);
    if (aotx_shared_u32(body) != 1 || !kind || kind > AOTX_SHARED_COMPLETE_RECORD ||
        aotx_shared_u64(body + 16) || !total || total > AOTX_SHARED_TRANSFER ||
        offset > total || bytes - 32 != min(total - offset, AOTX_SHARED_RECORD_DATA) || !source) {
        aotx_shared.fatal = 1; return;
    }
    if (!offset) {
        if (aotx_shared.kind || aotx_shared.received || serial != aotx_shared.serial + 1 || serial >= ~0ull / 64) {
            aotx_shared.fatal = 1; return;
        }
        aotx_shared.kind = kind; aotx_shared.total = total; aotx_shared.transfer_serial = serial;
    }
    if (aotx_shared.kind != kind || aotx_shared.total != total || aotx_shared.transfer_serial != serial ||
        aotx_shared.received != offset) { aotx_shared.fatal = 1; return; }
    aotx_service_bytes(aotx_shared.transfer + offset, body + 32, bytes - 32);
    aotx_shared.received += bytes - 32; aotx_shared.source = source;
    if (aotx_shared.received != total) return;
    aotx_shared.serial = serial;
    if (!aotx_shared_apply(kind, aotx_shared.transfer, total, source, true)) aotx_shared.fatal = 1;
    aotx_shared.kind = aotx_shared.total = aotx_shared.received = 0;
}
__device__ unsigned aotx_shared_window(unsigned long long base, unsigned count)
{
    if (aotx_shared.enabled && !aotx_seam.replaying && aotx_shared.kind == AOTX_SHARED_ADMIT_RECORD &&
        aotx_shared_candidate.operation == AOTX_SHARED_PUBLISH) return 0;
    if (!aotx_shared.enabled || !aotx_seam.replaying) return count;
    for (unsigned i = 0; i < count; ++i) {
        const unsigned char *p = aotx_seam.in.slots + ((base + i) & aotx_seam.in.mask) * AOTX_SLOT_BYTES;
        const aotx_record_header *h = (const aotx_record_header *)p;
        if (h->type != AOTX_SHARED_RECORD || h->body_len < 32 || h->body_len > 192) continue;
        const unsigned char *b = p + AOTX_HEADER_BYTES;
        unsigned kind = aotx_shared_u32(b + 4);
        if ((kind == AOTX_SHARED_LEASE_RECORD || kind == AOTX_SHARED_COMPLETE_RECORD) &&
            aotx_shared_u32(b + 28) + h->body_len - 32 == aotx_shared_u32(b + 24)) return i + 1;
    }
    return count;
}
__device__ bool aotx_shared_restore_end(void)
{
    if (!aotx_shared.enabled) return true;
    if (aotx_shared.fatal || aotx_shared.kind || aotx_shared.received) return false;
    for (unsigned i = 0; i < aotx_shared.receipt_capacity; ++i) {
        aotx_shared_receipt &r = aotx_shared.receipts[i];
        if (!r.phase) continue;
        r.saved_admission = 1; r.saved_terminal = r.terminal_source != 0;
        if (r.phase < AOTX_SHARED_DONE) {
            if (r.slot < AOTX_SLOTS) aotx_shared_bridge_release(i, true);
            if (r.conversation < aotx_shared.conversation_capacity) aotx_shared.conversations[r.conversation].request = 0;
            r.slot = AOTX_SLOTS; r.phase = AOTX_SHARED_INTERRUPTED; r.status = 598; r.gap = 1;
        }
    }
    for (unsigned i = 0; i < AOTX_SLOTS; ++i) aotx_shared.slot[i] = 0;
    aotx_shared.replaying = 0; return true;
}
