/* Purpose: Process bounded scoped service commands in the finite tick graph.
 * Owns: Mailbox validation and request status routing.
 * Launch shape: Parallel mailbox copy, then one ordered admission batch.
 * Lifetime: One mapped service channel. */
#include "service/internal.cuh"
#include "shared/state.cuh"
#include "model/decode.cuh"
__device__ aotx_service_state aotx_service;

__global__ void aotx_service_copy(void)
{
    unsigned channel = blockIdx.x;
    if (!aotx_service.enabled || channel >= AOTX_SERVICE_CHANNELS || aotx_seam.replaying) return;
    aotx_service_mailbox *m = aotx_service.mailbox + channel;
    __shared__ unsigned pending;
    __shared__ unsigned long long length;
    if (!threadIdx.x) {
        pending = aotx_seam_acquire_sys((const unsigned long long *)&m->state) == 1;
        length = pending ? m->length : 0;
    }
    __syncthreads();
    if (!pending) return;
    unsigned char *f = aotx_service.frames + (unsigned long long)channel * AOTX_SERVICE_FRAME;
    bool valid = length >= AOTX_SERVICE_HEAD && length <= AOTX_SERVICE_FRAME;
    unsigned n = valid ? (unsigned)length : AOTX_SERVICE_HEAD;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) f[i] = valid ? m->bytes[i] : 0;
    if (!threadIdx.x) aotx_service.ready[channel] = valid ? (unsigned)length : 1u;
}
static __device__ bool aotx_service_header(const unsigned char *f, unsigned length)
{
    if (length < AOTX_SERVICE_HEAD || aotx_service_u32(f + 88) != length - AOTX_SERVICE_HEAD) return false;
    for (unsigned i = 0; i < 8; ++i) if (f[i] != AOTX_SERVICE_MAGIC[i]) return false;
    for (unsigned i = 92; i < AOTX_SERVICE_HEAD; ++i) if (f[i]) return false;
    return aotx_service_u32(f + 12) == 0;
}
static __device__ void aotx_service_read(unsigned channel, const aotx_service_grant *g, bool cancel)
{
    unsigned char *f = aotx_service.frames + (unsigned long long)channel * AOTX_SERVICE_FRAME;
    if (aotx_service_get(f + 40, 8) != aotx_service.epoch) {
        aotx_service_answer(channel, 410, 0); return;
    }
    aotx_service_job *j = aotx_service_request(f + 48, f + 16);
    if (!j || j->revision != g->revision || !(g->models & (1u << j->role))) {
        aotx_service_answer(channel, 404, 0); return;
    }
    unsigned long long cursor = aotx_service_get(f + 64, 8);
    if (cursor > j->output || (cancel && cursor)) { aotx_service_answer(channel, 409, j->phase); return; }
    if (cancel && j->phase < AOTX_SERVICE_DONE) {
        if (j->slot < AOTX_SLOTS) aotx_seq_stop(j->slot);
        else j->phase = AOTX_SERVICE_CANCELLED;
        j->cancel = 1; j->status = 409; j->changed = aotx_service.clock;
    }
    unsigned take = min(j->output - (unsigned)cursor, AOTX_SERVICE_DATA);
    aotx_service_bytes(f + AOTX_SERVICE_HEAD, j->result + cursor, take);
    aotx_service_put(f + 72, j->role, 4);
    aotx_service_put(f + 76, j->output, 4); aotx_service_put(f + 80, j->prompt, 4);
    aotx_service_put(f + 84, j->sampled, 4); aotx_service_put(f + 92, j->finish, 4);
    aotx_service_put(f + 96, j->status, 4);
    aotx_service_put(f + 100, j->cancel, 4);
    aotx_service_answer(channel, 200, j->phase, take);
}
__global__ void aotx_service_admit(void)
{
    if (!aotx_service.enabled || aotx_seam.replaying) return;
    aotx_service.clock = aotx_sched.start_ns;
    if (!aotx_sched.held) aotx_service_media_expire();
    unsigned taken = 0, start = aotx_service.cursor;
    for (unsigned pass = 0; pass < AOTX_SERVICE_CHANNELS; ++pass) {
        unsigned channel = !pass ? 0 : 1 + (start + pass - 1) % (AOTX_SERVICE_CHANNELS - 1);
        unsigned length = aotx_service.ready[channel];
        if (!length) continue;
        unsigned char *f = aotx_service.frames + (unsigned long long)channel * AOTX_SERVICE_FRAME;
        if (!aotx_service_header(f, length)) { aotx_service_answer(channel, 400, 0); continue; }
        unsigned op = aotx_service_u32(f + 8);
        if (!channel) {
            unsigned status = op == AOTX_SERVICE_GRANTS ? aotx_service_install(f, length) : 403;
            aotx_service_answer(channel, status, 0); continue;
        }
        if (taken++ == AOTX_SLOTS) break;
        aotx_service.cursor = channel % (AOTX_SERVICE_CHANNELS - 1);
        aotx_service_grant *g = aotx_service_granted(f + 16);
        if (!g || g->revision != aotx_service_get(f + 32, 8)) {
            aotx_service_answer(channel, 403, 0); continue;
        }
        bool shape = true;
        if (op != AOTX_SERVICE_SUBMIT)
            for (unsigned i = 72; i < 88; ++i) shape &= f[i] == 0;
        if (op == AOTX_SERVICE_INFO || op == AOTX_SERVICE_METRICS)
            for (unsigned i = 40; i < 72; ++i) shape &= f[i] == 0;
        if (op == AOTX_SERVICE_MEDIA || op == AOTX_SERVICE_MEDIA_READ)
            shape &= !aotx_service_get(f + 40, 8) && !aotx_service_get(f + 64, 8);
        if (op == AOTX_SERVICE_MEDIA_LIST)
            shape &= !aotx_service_get(f + 40, 8) && !aotx_service_nonzero(f + 48, 16) && !aotx_service_u32(f + 88);
        if (!shape) { aotx_service_answer(channel, 400, 0); continue; }
        /* A journal hold permits scoped reads and deployment grants, with no new recorded work. */
        if (aotx_sched.held && (op == AOTX_SERVICE_SUBMIT || op == AOTX_SERVICE_CANCEL ||
            op == AOTX_SERVICE_MEDIA || op == 10)) {
            unsigned action = op == AOTX_SERVICE_MEDIA ? AOTX_SERVICE_UPLOAD :
                op == 10 ? AOTX_SHARED_WRITE_ACTION : AOTX_SERVICE_INFER;
            aotx_service_answer(channel, g->actions & action ? 429 : 403, 0); continue;
        }
        if (op == AOTX_SERVICE_INFO || op == AOTX_SERVICE_METRICS) {
            aotx_service_information(channel, g, op == AOTX_SERVICE_METRICS); continue;
        }
        if (op == AOTX_SERVICE_MEDIA || op == AOTX_SERVICE_MEDIA_READ) {
            aotx_service_media(channel, g, op == AOTX_SERVICE_MEDIA_READ); continue;
        }
        if (op == AOTX_SERVICE_MEDIA_LIST) { aotx_service_media_list(channel, g); continue; }
        if (op == 10 || op == 11) { aotx_shared_handle(channel, g, f); continue; }
        if (!(g->actions & AOTX_SERVICE_INFER)) { aotx_service_answer(channel, 403, 0); continue; }
        if (op == AOTX_SERVICE_READ || op == AOTX_SERVICE_CANCEL) {
            if (length != AOTX_SERVICE_HEAD) aotx_service_answer(channel, 400, 0);
            else aotx_service_read(channel, g, op == AOTX_SERVICE_CANCEL);
        } else if (op == AOTX_SERVICE_SUBMIT) {
            unsigned status = aotx_service_submit(g, f);
            aotx_service_answer(channel, status, status == 202 ? AOTX_SERVICE_QUEUED : 0);
        } else aotx_service_answer(channel, 400, 0);
    }
}
