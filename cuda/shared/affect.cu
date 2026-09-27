/* Purpose: Record absolute affect successors within their authorized scope.
 * Owns: Scoped affect state; physical slots are temporary working copies.
 * Launch shape: One ordered transition batch on the device.
 * Lifetime: Complete shared journal replay restores exact values without inference. */
#include "shared/affect.cuh"
#include "shared/bridge.cuh"
#include "shared/internal.cuh"
#ifdef AOTX_AFFECT
/* The ordered completion writer owns this candidate until encoding ends. */
static __device__ aotx_affect_agent_state aotx_shared_affect_next;
static __device__ unsigned aotx_shared_affect_reason;
static __device__ aotx_affect_agent_state aotx_shared_affect_neutral(void)
{
    aotx_affect_agent_state value = {};
    value.scale = AOTX_AFFECT_SCALE_ONE; value.axes = AOTX_AFFECT_DATA_AXES;
    return value;
}
__device__ aotx_shared_affect_state *aotx_shared_affect_scope(const aotx_shared_receipt *r)
{
    if (!r || r->space >= aotx_shared.space_capacity || r->conversation >= aotx_shared.conversation_capacity)
        return 0;
    aotx_shared_space &s = aotx_shared.spaces[r->space];
    aotx_shared_conversation &c = aotx_shared.conversations[r->conversation];
    if (!s.active || !c.active || c.space != r->space) return 0;
    return s.scope ? &s.affect : &c.affect;
}
__device__ unsigned aotx_shared_affect_managed(const aotx_shared_receipt *r)
{
    const aotx_shared_affect_state *s = aotx_shared_affect_scope(r);
    return s && (s->enabled || aotx_setting_count(AOTX_SET_AFFECT_ON)) ? 1 : 0;
}
__device__ bool aotx_shared_affect_conflict(const aotx_shared_receipt *a, const aotx_shared_receipt *b)
{
    const aotx_shared_affect_state *s = aotx_shared_affect_scope(a);
    return s && s == aotx_shared_affect_scope(b) &&
        (aotx_shared_affect_managed(a) || a->sample.affect || b->sample.affect);
}
__device__ void aotx_shared_affect_clear(unsigned slot)
{
    if (slot >= AOTX_SLOTS) return;
    aotx_affect_state[slot] = aotx_shared_affect_neutral();
    aotx_affect_acc[slot] = {}; aotx_affect_laws[slot] = {};
}
__device__ void aotx_shared_affect_lease(aotx_shared_receipt *r, unsigned slot, unsigned managed)
{
    aotx_shared_affect_clear(slot); r->sample.affect = managed;
    const aotx_shared_affect_state *s = aotx_shared_affect_scope(r);
    if (managed && s && s->enabled) aotx_affect_state[slot] = s->value;
}
__device__ unsigned aotx_shared_affect_encode(const aotx_shared_receipt *r, unsigned status,
    unsigned finish, unsigned char *out)
{
    if (!r->sample.affect || r->slot >= AOTX_SLOTS ||
        !aotx_shared_execution_slots[r->slot].model_opened) return 0;
    unsigned events = r->cancel ? 1u << AOTX_AFFECT_EVENT_OPERATOR_STOP :
        status == 504 ? 1u << AOTX_AFFECT_EVENT_DEADLINE :
        finish == 1 ? 1u << AOTX_AFFECT_EVENT_STOP :
        finish == 2 ? 1u << AOTX_AFFECT_EVENT_LIMIT : 1u << AOTX_AFFECT_EVENT_TASK_FAILED;
    unsigned &reason = aotx_shared_affect_reason;
    reason = 0;
    unsigned available = 0, enabled = aotx_affect_acc[r->slot].flag;
    aotx_affect_agent_state &next = aotx_shared_affect_next;
    next = aotx_shared_affect_neutral();
    if (enabled && !aotx_affect_predict(r->slot, events, &next, &reason)) return 0;
    if (enabled && aotx_control_matches(&aotx_affect_rows.identity, r->role))
        for (unsigned i = 0; i < aotx_affect_rows.count; ++i)
            if (!aotx_affect_rows.row[i].monitor && aotx_affect_rows.row[i].axis < AOTX_AFFECT_DATA_AXES)
                available |= 1u << aotx_affect_rows.row[i].axis;
    const aotx_shared_affect_state *s = aotx_shared_affect_scope(r);
    if (!s || s->revision == ~0ull) return 0;
    aotx_shared_zero(out, 64);
    aotx_service_put(out, s->revision, 8); aotx_service_put(out + 8, s->revision + 1, 8);
    for (unsigned i = 0; i < 4; ++i) {
        aotx_service_put(out + 16 + i * 2, (unsigned short)next.fast[i], 2);
        aotx_service_put(out + 24 + i * 2, (unsigned short)next.slow[i], 2);
    }
    aotx_service_put(out + 32, next.scale, 2); aotx_service_put(out + 34, next.axes, 2);
    aotx_service_put(out + 36, next.actuator_flags, 4);
    aotx_service_put(out + 40, __float_as_uint(next.budget_spent), 4);
    aotx_service_put(out + 44, reason, 4); aotx_service_put(out + 48, available, 4);
    aotx_service_put(out + 52, enabled, 4); return 64;
}
__device__ bool aotx_shared_affect_apply(const aotx_shared_receipt *r, const unsigned char *p, unsigned n)
{
    aotx_shared_affect_state *s = aotx_shared_affect_scope(r);
    if (!s || !r->sample.affect || r->phase != AOTX_SHARED_RUNNING || r->slot >= AOTX_SLOTS || n != 64 ||
        aotx_shared_u64(p) != s->revision || s->revision == ~0ull ||
        aotx_shared_u64(p + 8) != s->revision + 1 || aotx_shared_u64(p + 56) ||
        (aotx_shared_u32(p + 44) & ~AOTX_AFFECT_EVENT_MASK) || (aotx_shared_u32(p + 48) & ~3u) ||
        aotx_shared_u32(p + 52) > 1) return false;
    aotx_affect_agent_state next = {};
    for (unsigned i = 0; i < 4; ++i) {
        next.fast[i] = (short)aotx_service_get(p + 16 + i * 2, 2);
        next.slow[i] = (short)aotx_service_get(p + 24 + i * 2, 2);
    }
    next.scale = (unsigned short)aotx_service_get(p + 32, 2);
    next.axes = (unsigned short)aotx_service_get(p + 34, 2);
    next.actuator_flags = aotx_shared_u32(p + 36);
    next.budget_spent = __uint_as_float(aotx_shared_u32(p + 40));
    if (next.axes != AOTX_AFFECT_DATA_AXES || (next.actuator_flags & ~15u) ||
        next.fast[2] || next.fast[3] || next.slow[2] || next.slow[3] ||
        !isfinite(next.budget_spent) || next.budget_spent < 0) return false;
    unsigned enabled = aotx_shared_u32(p + 52);
    if (!enabled && (next.fast[0] || next.fast[1] || next.slow[0] || next.slow[1] ||
        next.scale != AOTX_AFFECT_SCALE_ONE || next.actuator_flags || next.budget_spent ||
        aotx_shared_u32(p + 44) || aotx_shared_u32(p + 48))) return false;
    s->value = next; s->enabled = enabled; ++s->revision;
    s->reason = aotx_shared_u32(p + 44); s->available = aotx_shared_u32(p + 48); s->role = r->role;
    aotx_service_bytes(s->model_digest, r->model_digest, 32); return true;
}
__device__ void aotx_shared_affect_read(unsigned conversation, unsigned char *out)
{
    const aotx_shared_conversation &c = aotx_shared.conversations[conversation];
    const aotx_shared_space &space = aotx_shared.spaces[c.space];
    const aotx_shared_affect_state &s = space.scope ? space.affect : c.affect;
    aotx_affect_agent_state value = s.revision ? s.value : aotx_shared_affect_neutral();
    aotx_shared_zero(out, 96);
    aotx_service_put(out, 1, 4); aotx_service_put(out + 4, s.enabled, 4);
    aotx_service_put(out + 8, s.revision, 8);
    for (unsigned i = 0; i < 4; ++i) {
        aotx_service_put(out + 16 + i * 2, (unsigned short)value.fast[i], 2);
        aotx_service_put(out + 24 + i * 2, (unsigned short)value.slow[i], 2);
    }
    aotx_service_put(out + 32, value.scale, 2); aotx_service_put(out + 34, value.axes, 2);
    aotx_service_put(out + 36, value.actuator_flags, 4);
    aotx_service_put(out + 40, __float_as_uint(value.budget_spent), 4);
    aotx_service_put(out + 44, s.reason, 4); aotx_service_put(out + 48, s.available, 4);
    aotx_service_put(out + 52, s.role, 4); aotx_service_bytes(out + 56, s.model_digest, 32);
}
#endif
