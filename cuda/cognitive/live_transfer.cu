/* Purpose: Stage complete typed transfers and enforce apply-window boundaries.
 * Owns: The transfer buffer, sequence and bounded output diagnostics.
 * Launch shape: The inbound serial thread calls these functions.
 * Lifetime: One runtime and its journal replay. */
#include "cognitive/intake.cuh"
#include "cognitive/codec.cuh"
#include "cli/cli.cuh"
#include "seam/seam.cuh"

__device__ aotx_live_state aotx_live;
__device__ aotx_cognitive_store aotx_live_store;
__device__ aotx_live_binding aotx_live_bindings[AOTX_SLOTS];

static __device__ aotx_cli_out aotx_live_line;
__device__ void aotx_live_note(uint32_t op, uint32_t status, uint32_t count) {
    aotx_cli_out &out = aotx_live_line;
    aotx_cli_clear(&out);
    aotx_cli_say(&out, "memory: operation "); aotx_cli_num(&out, op);
    aotx_cli_say(&out, " status "); aotx_cli_num(&out, status);
    aotx_cli_say(&out, " rows "); aotx_cli_num(&out, count);
    aotx_console_write(out.text, out.at);
}

/* A complete transfer must reach its graph nodes before the next transfer is applied. */
__device__ unsigned int aotx_live_window(uint64_t base, unsigned int count) {
    for (uint32_t i = 0; i < count; ++i) {
        const volatile aotx_record_header *h = (const volatile aotx_record_header *)
            (aotx_seam.in.slots + ((base + i) & aotx_seam.in.mask) * AOTX_SLOT_BYTES);
        if (h->type != AOTX_LIVE_RECORD || h->cls != AOTX_CLASS_A) continue;
        if (aotx_live.phase == AOTX_LIVE_WRITE || (aotx_live.phase == AOTX_LIVE_ENCODING || aotx_live.phase == AOTX_INTAKE_RUN)) return i;
        if (h->body_len <= AOTX_LIVE_PART || h->body_len > AOTX_BODY_BYTES) return i + 1;
        const volatile unsigned char *p = (const volatile unsigned char *)h + AOTX_HEADER_BYTES;
        unsigned char head[AOTX_LIVE_PART];
        for (uint32_t j = 0; j < AOTX_LIVE_PART; ++j) head[j] = p[j];
        uint32_t total = aotx_cog_u32(head + 24), offset = aotx_cog_u32(head + 28);
        if (offset >= total || h->body_len - AOTX_LIVE_PART >= total - offset) return i + 1;
    }
    return count;
}

__device__ void aotx_live_part(const unsigned char *p, uint32_t bytes, uint64_t seq, uint32_t flags) {
    uint32_t status = AOTX_COG_FORMAT;
    if (bytes <= AOTX_LIVE_PART || bytes > AOTX_BODY_BYTES) goto failed;
    {
        uint32_t op = aotx_cog_u32(p + 4), total = aotx_cog_u32(p + 24), offset = aotx_cog_u32(p + 28);
        if (op == AOTX_LIVE_ADMISSION) {
            if (!aotx_live_admission_take(p, bytes)) goto failed;
            return;
        }
        uint32_t data = bytes - AOTX_LIVE_PART;
        bool choice = op == AOTX_LIVE_CHOICE || op == AOTX_LIVE_TEXT_CHOICE || op == AOTX_LIVE_RETAINED || op == AOTX_LIVE_AUTO_CHOICE || op == AOTX_INTAKE_CHOICE;
        if (aotx_cog_u32(p) != AOTX_LIVE_SCHEMA || op < AOTX_LIVE_LOAD || op > AOTX_INTAKE_CHOICE ||
            aotx_cog_zero(p + 8, 16) || !total || total > AOTX_LIVE_BYTES || offset >= total ||
            data != (total - offset < AOTX_LIVE_DATA ? total - offset : AOTX_LIVE_DATA) ||
            (choice && (!aotx_seam.replaying || op != aotx_live_result_op()))) goto failed;
        if (!offset) {
            if (aotx_live.received || aotx_live.phase !=
                (choice ? AOTX_LIVE_WAIT : AOTX_LIVE_IDLE)) goto failed;
            aotx_live.op = op; aotx_live.total = total;
            aotx_live.admission = aotx_live_direct(op) && (!aotx_seam.replaying || (flags & AOTX_FLAG_ADMISSION)) ? 1 : 0;
            aotx_live.pressure = 0;
            for (uint32_t j = 0; j < 16; ++j) aotx_live.transfer_id[j] = p[8 + j];
        }
        if (op != aotx_live.op || total != aotx_live.total || offset != aotx_live.received ||
            !aotx_cog_equal(p + 8, aotx_live.transfer_id)) goto failed;
        if (aotx_seam.replaying && aotx_live_direct(op) &&
            !!(flags & AOTX_FLAG_ADMISSION) != (aotx_live.admission == 1)) goto failed;
        for (uint32_t j = 0; j < data; ++j) aotx_live.input[offset + j] = p[AOTX_LIVE_PART + j];
        aotx_live.received += data;
        if (aotx_live.received == total) {
            aotx_live.source_seq = seq;
            aotx_live.phase = choice ? AOTX_LIVE_REPLAY : AOTX_LIVE_READY;
        }
        return;
    }
failed:
    if ((aotx_live.phase == AOTX_LIVE_ENCODING || aotx_live.phase == AOTX_INTAKE_RUN)) { aotx_live.status = status; return; }
    aotx_live_note(aotx_live.op, status, 0);
    ++aotx_live.refused;
    if (aotx_seam.replaying) aotx_live.fatal = 1;
    aotx_live.received = 0; aotx_live.phase = AOTX_LIVE_IDLE;
}
__device__ bool aotx_live_restore_end(void) {
    if (aotx_live.received || aotx_live.phase != AOTX_LIVE_IDLE) {
        aotx_live.fatal = 1;
        aotx_live_note(aotx_live.op, AOTX_COG_MISSING, 0);
    }
    return aotx_live.fatal == 0;
}
