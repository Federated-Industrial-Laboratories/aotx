/* Purpose: Read and change bounded runtime affect settings through operator grants.
 * Owns: Revision checks and exact setting record publication.
 * Launch shape: One ordered service admission batch.
 * Lifetime: Settings and revisions follow the complete journal history. */
#include "service/internal.cuh"
#include "settings/settings.cuh"
#include "cognitive/checkpoint.cuh"
static __device__ aotx_setting_body aotx_affect_setting_body;
static __device__ unsigned aotx_affect_pending(void)
{
    unsigned count = 0;
    for (unsigned i = 0; i < aotx_setting_table.pending_count; ++i) {
        const auto &p = aotx_setting_table.pending[i]; unsigned index;
        if (aotx_setting_find(p.key, p.key_len, &index) && aotx_setting_affect(index)) ++count;
    }
    return count;
}
__device__ void aotx_service_affect_settings(unsigned channel, const aotx_service_grant *g)
{
    unsigned char *f = aotx_service.frames + (unsigned long long)channel * AOTX_SERVICE_FRAME;
    unsigned bytes = aotx_service_u32(f + 88); bool write = bytes != 0;
    unsigned permitted = write ? AOTX_SERVICE_AFFECT_MANAGE : AOTX_SERVICE_AFFECT_MANAGE | AOTX_SERVICE_TELEMETRY;
    if (!(g->actions & permitted)) { aotx_service_answer(channel, 403, 0); return; }
    if (aotx_service_nonzero(f + 48, 24) || (!write && aotx_service_get(f + 40, 8)) || (write && bytes != 96)) {
        aotx_service_answer(channel, 400, 0); return;
    }
#ifndef AOTX_AFFECT
    aotx_service_answer(channel, 501, 0); return;
#else
    unsigned pending = aotx_affect_pending();
    if (write) {
        const unsigned char *p = f + AOTX_SERVICE_HEAD;
        unsigned length = aotx_service_u32(p + 4), index = 0;
        long long value = (long long)aotx_service_get(p + 16, 8);
        unsigned scale = aotx_service_u32(p + 24);
        if (aotx_service_u32(p) != 1 || !length || length > 63 || aotx_service_u32(p + 28) ||
            aotx_service_nonzero(p + 32 + length, 64 - length) ||
            !aotx_setting_find((const char *)p + 32, length, &index) || !aotx_setting_affect(index) ||
            aotx_settings_judge((const char *)p + 32, length, value, scale, &index) != AOTX_SETTING_TOOK) {
            aotx_service_answer(channel, 400, 0); return;
        }
        if (aotx_service_get(f + 40, 8) != aotx_service.epoch) { aotx_service_answer(channel, 410, 0); return; }
        if (aotx_service_get(p + 8, 8) != aotx_setting_table.affect_revision || aotx_setting_table.affect_revision == ~0ull) {
            aotx_service_answer(channel, 409, 0); return;
        }
        if (pending || aotx_sched.held || aotx_checkpoint_pressure()) { aotx_service_answer(channel, 429, 0); return; }
        auto &body = aotx_affect_setting_body; body = {};
        body.value = value; body.scale = scale; body.key_len = length;
        aotx_service_bytes((unsigned char *)body.key, p + 32, length);
        aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_A, AOTX_REC_SETTING, 0, &body, sizeof(body));
        aotx_seam.apply.state_hash = aotx_seam_fnv1a(aotx_seam.apply.state_hash, (const unsigned char *)&body, sizeof(body));
        ++aotx_seam.apply.applied_count;
        aotx_settings_apply(&body, aotx_time_tick);
    }
    unsigned char *out = f + AOTX_SERVICE_HEAD;
    for (unsigned i = 0; i < 32; ++i) out[i] = 0;
    aotx_service_put(out, 1, 4); aotx_service_put(out + 4, 13, 4);
    aotx_service_put(out + 8, aotx_setting_table.affect_revision, 8);
    aotx_service_put(out + 16, !!(g->actions & AOTX_SERVICE_AFFECT_MANAGE), 4);
    aotx_service_put(out + 20, pending, 4);
    unsigned count = 0;
    for (unsigned i = 0; i < AOTX_SETTING_NUMBER_COUNT; ++i) {
        if (!aotx_setting_affect(i)) continue;
        unsigned char *p = out + 32 + count++ * 64;
        for (unsigned j = 0; j < 64; ++j) p[j] = 0;
        const char *name = aotx_setting_name(i);
        for (unsigned j = 0; name[j] && j < 31; ++j) p[j] = name[j];
        aotx_service_put(p + 32, (unsigned long long)aotx_setting_value(i), 8);
        aotx_service_put(p + 40, (unsigned long long)aotx_setting_least(i), 8);
        aotx_service_put(p + 48, (unsigned long long)aotx_setting_most(i), 8);
        aotx_service_put(p + 56, aotx_setting_scale(i), 4);
    }
    aotx_service_answer(channel, 200, 0, 32 + count * 64);
#endif
}
