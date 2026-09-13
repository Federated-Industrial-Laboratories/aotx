/* Purpose: Find persistent shared resources and intersect their membership rights.
 * Owns: The shared device state and scoped table lookup.
 * Launch shape: One ordered device batch and slot-indexed execution readers.
 * Lifetime: The complete runtime lineage. */
#include "shared/internal.cuh"
__device__ aotx_shared_state aotx_shared;
__device__ unsigned aotx_shared_participant_find(const unsigned char *id)
{
    for (unsigned i = 0; i < aotx_shared.participant_capacity; ++i)
        if (aotx_shared.participants[i].active && aotx_shared_id(id, aotx_shared.participants[i].id)) return i;
    return AOTX_SHARED_NONE;
}
__device__ unsigned aotx_shared_space_find(const unsigned char *id)
{
    for (unsigned i = 0; i < aotx_shared.space_capacity; ++i)
        if (aotx_shared.spaces[i].active && aotx_shared_id(id, aotx_shared.spaces[i].id)) return i;
    return AOTX_SHARED_NONE;
}
__device__ unsigned aotx_shared_conversation_find(const unsigned char *id)
{
    for (unsigned i = 0; i < aotx_shared.conversation_capacity; ++i)
        if (aotx_shared.conversations[i].active && aotx_shared_id(id, aotx_shared.conversations[i].id)) return i;
    return AOTX_SHARED_NONE;
}
__device__ unsigned aotx_shared_receipt_find(unsigned participant, unsigned long long sequence)
{
    for (unsigned i = 0; i < aotx_shared.receipt_capacity; ++i) {
        const aotx_shared_receipt &r = aotx_shared.receipts[i];
        if (r.phase && r.participant == participant && r.sequence == sequence) return i;
    }
    return AOTX_SHARED_NONE;
}
__device__ unsigned aotx_shared_id_find(const unsigned char *id)
{
    for (unsigned i = 0; i < aotx_shared.receipt_capacity; ++i)
        if (aotx_shared.receipts[i].phase && aotx_shared_id(id, aotx_shared.receipts[i].id)) return i;
    return AOTX_SHARED_NONE;
}
__device__ unsigned aotx_shared_key_find(unsigned participant, const unsigned char *key)
{
    for (unsigned i = 0; i < aotx_shared.receipt_capacity; ++i) {
        const aotx_shared_receipt &r = aotx_shared.receipts[i];
        if (r.phase && r.participant == participant && aotx_shared_id(key, r.key)) return i;
    }
    return AOTX_SHARED_NONE;
}
__device__ unsigned aotx_shared_rights(unsigned participant, unsigned space)
{
    if (participant >= aotx_shared.participant_capacity || !aotx_shared.participants[participant].active ||
        space >= aotx_shared.space_capacity || !aotx_shared.spaces[space].active) return 0;
    const aotx_shared_space &s = aotx_shared.spaces[space];
    if (aotx_shared_id(s.owner, aotx_shared.participants[participant].id)) return 7;
    for (unsigned i = 0; i < aotx_shared.member_capacity; ++i) {
        const aotx_shared_member &m = aotx_shared.members[i];
        if (m.active && m.participant == participant && m.space == space) return m.permissions;
    }
    return s.scope == 2 ? 3 : 0;
}
__device__ bool aotx_shared_visible(unsigned participant, unsigned space, unsigned rights)
{ return (aotx_shared_rights(participant, space) & rights) == rights; }
__device__ bool aotx_shared_authorized(const aotx_shared_receipt *r, unsigned rights)
{
    const aotx_service_grant *g = aotx_service_granted(r->actor);
    unsigned actions = (rights & 1 ? AOTX_SHARED_READ_ACTION : 0) |
        (rights & 2 ? AOTX_SHARED_WRITE_ACTION : 0) | (rights & 4 ? AOTX_SHARED_MANAGE_ACTION : 0);
    return g && (g->actions & actions) == actions &&
        (r->space == AOTX_SHARED_NONE || aotx_shared_visible(r->participant, r->space, rights)) &&
        (r->operation != AOTX_SHARED_INPUT || (r->role < 32 && (g->models & (1u << r->role))));
}
__device__ bool aotx_shared_owns(unsigned slot)
{ return aotx_shared.enabled && slot < AOTX_SLOTS && aotx_shared.slot[slot] != 0; }
__device__ aotx_shared_receipt *aotx_shared_request(unsigned slot)
{
    if (!aotx_shared_owns(slot) || aotx_shared.slot[slot] > aotx_shared.receipt_capacity) return 0;
    return aotx_shared.receipts + aotx_shared.slot[slot] - 1;
}
__device__ const unsigned char *aotx_shared_actor(unsigned slot)
{ const aotx_shared_receipt *r = aotx_shared_request(slot); return r ? r->actor : 0; }
__device__ bool aotx_shared_quiet(void)
{
    if (!aotx_shared.enabled) return true;
    if (aotx_shared.kind || aotx_shared.received || aotx_shared.fatal) return false;
    for (unsigned i = 0; i < AOTX_SLOTS; ++i) if (aotx_shared.slot[i]) return false;
    return true;
}
__device__ void aotx_shared_ack(unsigned long long source, unsigned long long generation,
                               const unsigned char *incarnation, unsigned long long boot,
                               const unsigned char *commit_digest)
{
    if (boot == aotx_shared.saved_boot && aotx_shared_id(incarnation, aotx_shared.saved_incarnation) &&
        (source < aotx_shared.saved_source || generation < aotx_shared.saved_generation)) return;
    aotx_shared.saved_source = source; aotx_shared.saved_generation = generation;
    aotx_shared.saved_boot = boot;
    aotx_service_bytes(aotx_shared.saved_commit_digest, commit_digest, 32);
    aotx_service_bytes(aotx_shared.saved_incarnation, incarnation, 16);
    for (unsigned i = 0; i < aotx_shared.receipt_capacity; ++i) {
        aotx_shared_receipt &r = aotx_shared.receipts[i];
        if (!r.phase) continue;
        if (r.admission_source && r.admission_source <= source) r.saved_admission = 1;
        if (r.terminal_source && r.terminal_source <= source) r.saved_terminal = 1;
    }
}
__device__ void aotx_shared_handle(unsigned channel, const aotx_service_grant *grant, unsigned char *frame)
{
    if (!aotx_shared.enabled || aotx_shared.fatal) { aotx_service_answer(channel, 503, 0); return; }
    unsigned n = aotx_shared_u32(frame + 88), op = aotx_shared_u32(frame + 8);
    if (op == 11) {
        unsigned char read[AOTX_SHARED_READ_HEAD];
        if (n != sizeof(read)) { aotx_service_answer(channel, 400, 0); return; }
        aotx_service_bytes(read, frame + AOTX_SERVICE_HEAD, n);
        aotx_shared_read(channel, grant, read, n); return;
    }
    if (op != 10) { aotx_service_answer(channel, 400, 0); return; }
    unsigned receipt = AOTX_SHARED_NONE;
    unsigned status = aotx_shared_admit(grant, frame + AOTX_SERVICE_HEAD, n, &receipt);
    if (status < 400 && receipt != AOTX_SHARED_NONE)
        aotx_shared_receipt_reply(channel, AOTX_SHARED_OPERATION_READ, receipt, 0, status);
    else aotx_service_answer(channel, status, 0);
}
