/* Purpose: Project permitted shared resources and exact byte cursors.
 * Owns: Bounded mailbox replies with scoped counters and save status.
 * Launch shape: One ordered read batch.
 * Lifetime: One authenticated read against current membership and grants. */
#include "shared/internal.cuh"
#include "shared/affect.cuh"
__device__ unsigned char *aotx_shared_reply(unsigned channel, unsigned kind)
{
    unsigned char *out = aotx_service.frames + (unsigned long long)channel * AOTX_SERVICE_FRAME + AOTX_SERVICE_HEAD;
    aotx_shared_zero(out, AOTX_SHARED_REPLY_HEAD);
    aotx_service_bytes(out, (const unsigned char *)AOTX_SHARED_MAGIC, 8);
    aotx_service_put(out + 8, kind, 4); aotx_service_bytes(out + 16, aotx_live_store.lineage, 16);
    aotx_service_put(out + 136, aotx_shared.saved_source, 8);
    aotx_service_put(out + 144, aotx_shared.saved_generation, 8);
    aotx_service_bytes(out + 152, aotx_shared.saved_incarnation, 16);
    aotx_service_put(out + 216, aotx_shared.pending_bytes, 8); aotx_service_put(out + 224, aotx_shared.disk_error, 4);
    aotx_service_put(out + 256, aotx_shared.saved_boot, 8);
    aotx_service_bytes(out + 264, aotx_shared.saved_commit_digest, 32); return out;
}
__device__ void aotx_shared_receipt_reply(unsigned channel, unsigned kind, unsigned index,
                                         unsigned long long cursor, unsigned http_status)
{
    const aotx_shared_receipt &r = index == aotx_shared.receipt_capacity ? aotx_shared_candidate : aotx_shared.receipts[index];
    if (cursor > r.output) { aotx_service_answer(channel, 409, r.phase); return; }
    unsigned char *out = aotx_shared_reply(channel, kind);
    aotx_service_put(out + 12, r.phase, 4); aotx_service_bytes(out + 32, r.command + 56, 16);
    if (r.space < aotx_shared.space_capacity) {
        aotx_service_bytes(out + 48, aotx_shared.spaces[r.space].id, 16);
        aotx_service_put(out + 228, aotx_shared.spaces[r.space].scope, 4);
    }
    aotx_service_bytes(out + 64, r.actor, 16);
    bool own = aotx_shared_id(aotx_service.frames + (unsigned long long)channel * AOTX_SERVICE_FRAME + 16, r.actor);
    if (own) aotx_service_put(out + 80, r.sequence, 8);
    if (r.participant < aotx_shared.participant_capacity &&
        aotx_shared_id(aotx_service.frames + (unsigned long long)channel * AOTX_SERVICE_FRAME + 16, r.actor)) {
        const aotx_shared_participant &p = aotx_shared.participants[r.participant];
        aotx_service_put(out + 88, p.active ? p.next : 1, 8); aotx_service_put(out + 96, p.active ? p.floor : 1, 8);
    }
    aotx_service_put(out + 104, r.order, 8);
    if (r.conversation < aotx_shared.conversation_capacity)
        aotx_service_put(out + 112, aotx_shared.conversations[r.conversation].event_floor, 8);
    aotx_service_put(out + 120, r.admission_source, 8); aotx_service_put(out + 128, r.terminal_source, 8);
    unsigned flags = (r.phase != AOTX_SHARED_ACCEPTED ? 1u : 0u) | (r.saved_admission ? 2u : 0u) |
        (r.saved_terminal ? 4u : 0u) | (r.gap ? 8u : 0u);
    aotx_service_put(out + 168, r.status, 4); aotx_service_put(out + 172, flags, 4);
    aotx_service_put(out + 176, r.output, 4); aotx_service_put(out + 180, r.prompt, 4);
    aotx_service_put(out + 184, r.sampled, 4); aotx_service_put(out + 188, r.finish, 4);
    aotx_service_put(out + 208, cursor, 8); aotx_service_put(out + 236, r.operation, 4);
    aotx_service_bytes(out + 240, r.key, 16); aotx_service_bytes(out + 296, r.id, 16);
    unsigned take = min(r.output - (unsigned)cursor, AOTX_SERVICE_DATA - AOTX_SHARED_REPLY_HEAD);
    aotx_service_bytes(out + AOTX_SHARED_REPLY_HEAD, r.result + cursor, take);
    aotx_service_answer(channel, http_status, r.phase, AOTX_SHARED_REPLY_HEAD + take);
}
static __device__ unsigned aotx_shared_operation_visible(const aotx_service_grant *g, unsigned participant,
                                                         const unsigned char *read)
{
    if (aotx_service_nonzero(read + 48, 16)) return AOTX_SHARED_NONE;
    if (aotx_shared.kind == AOTX_SHARED_ADMIT_RECORD && aotx_shared_id(read + 32, aotx_shared_candidate.id) &&
        aotx_shared_id(g->principal, aotx_shared_candidate.actor)) return aotx_shared.receipt_capacity;
    unsigned index = aotx_shared_id_find(read + 32);
    if (index == AOTX_SHARED_NONE) return index;
    const aotx_shared_receipt &r = aotx_shared.receipts[index];
    if (r.participant != participant && r.operation != AOTX_SHARED_INPUT) return AOTX_SHARED_NONE;
    if (r.space != AOTX_SHARED_NONE && !aotx_shared_visible(participant, r.space, 1)) return AOTX_SHARED_NONE;
    if (r.operation == AOTX_SHARED_INPUT && (r.role >= 32 || !(g->models & (1u << r.role)))) return AOTX_SHARED_NONE;
    return index;
}
static __device__ unsigned aotx_shared_list(unsigned kind, unsigned participant, unsigned space,
                                            unsigned long long cursor, unsigned limit, unsigned char *out)
{
    unsigned row = kind == AOTX_SHARED_MEMBERS_READ ? 32 : 64;
    unsigned capacity = kind == AOTX_SHARED_SPACES_READ ? aotx_shared.space_capacity :
        kind == AOTX_SHARED_MEMBERS_READ ? aotx_shared.member_capacity + 1 : aotx_shared.conversation_capacity;
    unsigned count = 0; unsigned long long visible = 0; bool more = false;
    for (unsigned i = 0; i < capacity; ++i) {
        bool allowed = false;
        if (kind == AOTX_SHARED_SPACES_READ) allowed = aotx_shared_visible(participant, i, 1);
        else if (kind == AOTX_SHARED_CONVERSATIONS_READ) allowed = aotx_shared.conversations[i].active && aotx_shared.conversations[i].space == space;
        else allowed = !i || (aotx_shared.members[i - 1].active && aotx_shared.members[i - 1].space == space);
        if (!allowed) continue;
        if (visible++ < cursor) continue;
        if (count == limit) { --visible; more = true; break; }
        unsigned char *p = out + AOTX_SHARED_REPLY_HEAD + count * row;
        aotx_shared_zero(p, row);
        if (kind == AOTX_SHARED_SPACES_READ) {
            const aotx_shared_space &s = aotx_shared.spaces[i];
            aotx_service_bytes(p, s.id, 16); aotx_service_bytes(p + 16, s.owner, 16);
            aotx_service_put(p + 32, s.scope, 4); aotx_service_put(p + 36, aotx_shared_rights(participant, i), 4);
        } else if (kind == AOTX_SHARED_CONVERSATIONS_READ) {
            const aotx_shared_conversation &c = aotx_shared.conversations[i];
            aotx_service_bytes(p, c.id, 16); aotx_service_bytes(p + 16, aotx_shared.spaces[space].id, 16);
            aotx_service_put(p + 32, c.next_order, 8); aotx_service_put(p + 40, c.event_floor, 8);
            aotx_service_put(p + 48, c.request != 0, 4);
        } else {
            const unsigned char *id = !i ? aotx_shared.spaces[space].owner :
                aotx_shared.participants[aotx_shared.members[i - 1].participant].id;
            aotx_service_bytes(p, id, 16); aotx_service_put(p + 16, !i ? 7 : aotx_shared.members[i - 1].permissions, 4);
        }
        ++count;
    }
    aotx_service_put(out + 192, count, 4); aotx_service_put(out + 196, row, 4);
    if (cursor > visible) return AOTX_SHARED_NONE;
    aotx_service_put(out + 200, more ? cursor + count : 0, 8); return count * row;
}
static __device__ unsigned aotx_shared_events(const aotx_service_grant *g, unsigned conversation,
                                              unsigned long long cursor, unsigned limit, unsigned char *out)
{
    const aotx_shared_conversation &c = aotx_shared.conversations[conversation];
    unsigned count = 0;
    unsigned long long order = max(cursor, c.event_floor);
    if (cursor && cursor < c.event_floor) aotx_service_put(out + 172, 8, 4);
    for (; order < c.next_order && count < limit; ++order) {
        for (unsigned i = 0; i < aotx_shared.receipt_capacity; ++i) {
            const aotx_shared_receipt &r = aotx_shared.receipts[i];
            if (!r.phase || r.operation != AOTX_SHARED_INPUT || r.conversation != conversation || r.order != order) continue;
            if (r.role >= 32 || !(g->models & (1u << r.role))) break;
            unsigned char *p = out + AOTX_SHARED_REPLY_HEAD + count++ * 96;
            aotx_shared_zero(p, 96); aotx_service_bytes(p, r.id, 16); aotx_service_bytes(p + 16, r.actor, 16);
            if (aotx_shared_id(g->principal, r.actor)) aotx_service_put(p + 32, r.sequence, 8);
            aotx_service_put(p + 40, r.order, 8);
            aotx_service_put(p + 48, r.phase, 4); aotx_service_put(p + 52, r.status, 4);
            aotx_service_put(p + 56, r.output, 4); aotx_service_put(p + 60, r.sampled, 4);
            aotx_service_put(p + 64, r.finish, 4); aotx_service_put(p + 68, r.saved_admission | (r.saved_terminal << 1), 4);
            aotx_service_put(p + 72, r.admission_source, 8); aotx_service_put(p + 80, r.terminal_source, 8);
            aotx_service_put(p + 88, r.gap, 4); break;
        }
    }
    aotx_service_put(out + 192, count, 4); aotx_service_put(out + 196, 96, 4);
    aotx_service_put(out + 200, order, 8); return count * 96;
}
__device__ void aotx_shared_read(unsigned channel, const aotx_service_grant *g, const unsigned char *read, unsigned bytes)
{
    if (!(g->actions & AOTX_SHARED_READ_ACTION)) { aotx_service_answer(channel, 403, 0); return; }
    unsigned kind = aotx_shared_u32(read + 8), limit = aotx_shared_u32(read + 80);
    bool known = kind >= AOTX_SHARED_CAPABILITIES && kind <= AOTX_SHARED_PROMPT_READ;
    bool discovery = kind == AOTX_SHARED_CAPABILITIES || kind == AOTX_SHARED_PARTICIPANT;
    if (bytes != AOTX_SHARED_READ_HEAD || !known || !limit || limit > 256 ||
        !aotx_service_equal(read, (const unsigned char *)AOTX_SHARED_MAGIC, 8) || aotx_shared_u32(read + 12) ||
        aotx_service_nonzero(read + 84, 12) ||
        (!aotx_shared_id(read + 16, aotx_live_store.lineage) && !(discovery && !aotx_service_nonzero(read + 16, 16)))) {
        aotx_service_answer(channel, 400, 0); return;
    }
    bool has_target = kind == AOTX_SHARED_SPACE_READ || kind == AOTX_SHARED_MEMBERS_READ ||
        kind == AOTX_SHARED_CONVERSATIONS_READ || kind == AOTX_SHARED_CONVERSATION_READ ||
        kind == AOTX_SHARED_OPERATION_READ || kind == AOTX_SHARED_EVENTS_READ || kind == AOTX_SHARED_MEMORY_READ ||
        kind == AOTX_SHARED_AFFECT_READ || kind == AOTX_SHARED_PROMPT_READ;
    bool list = kind == AOTX_SHARED_SPACES_READ || kind == AOTX_SHARED_MEMBERS_READ ||
        kind == AOTX_SHARED_CONVERSATIONS_READ || kind == AOTX_SHARED_EVENTS_READ || kind == AOTX_SHARED_MEMORY_READ;
    if (aotx_service_nonzero(read + 32, 16) != has_target ||
        (kind != AOTX_SHARED_MEMORY_READ && aotx_service_nonzero(read + 48, 16)) ||
        (!list && aotx_shared_u64(read + 64)) ||
        (kind != AOTX_SHARED_OPERATION_READ && kind != AOTX_SHARED_MEMORY_READ && aotx_shared_u64(read + 72))) {
        aotx_service_answer(channel, 400, 0); return;
    }
    unsigned participant = aotx_shared_participant_find(g->principal);
    unsigned long long cursor = aotx_shared_u64(read + 64), byte = aotx_shared_u64(read + 72);
    if (kind == AOTX_SHARED_OPERATION_READ) {
        unsigned index = aotx_shared_operation_visible(g, participant, read);
        if (index == AOTX_SHARED_NONE) { aotx_service_answer(channel, 404, 0); return; }
        aotx_shared_receipt_reply(channel, kind, index, byte, 200); return;
    }
    if (participant == AOTX_SHARED_NONE && !discovery) { aotx_service_answer(channel, 404, 0); return; }
    unsigned space = AOTX_SHARED_NONE, conversation = AOTX_SHARED_NONE;
    bool conv = kind == AOTX_SHARED_CONVERSATION_READ || kind == AOTX_SHARED_EVENTS_READ || kind == AOTX_SHARED_AFFECT_READ || kind == AOTX_SHARED_PROMPT_READ;
    bool scoped = conv || kind == AOTX_SHARED_SPACE_READ || kind == AOTX_SHARED_MEMBERS_READ ||
        kind == AOTX_SHARED_CONVERSATIONS_READ || kind == AOTX_SHARED_MEMORY_READ;
    if (conv) {
        conversation = aotx_shared_conversation_find(read + 32);
        if (conversation != AOTX_SHARED_NONE) space = aotx_shared.conversations[conversation].space;
    } else if (scoped) space = aotx_shared_space_find(read + 32);
    if (scoped && !aotx_shared_visible(participant, space, 1)) { aotx_service_answer(channel, 404, 0); return; }
    if (kind == AOTX_SHARED_MEMORY_READ) { aotx_shared_memory_read(channel, g, read, space); return; }
    unsigned char *out = aotx_shared_reply(channel, kind); unsigned tail = 0;
    aotx_service_bytes(out + 64, g->principal, 16);
    if (discovery) {
        aotx_service_put(out + 88, participant == AOTX_SHARED_NONE ? 1 : aotx_shared.participants[participant].next, 8);
        aotx_service_put(out + 96, participant == AOTX_SHARED_NONE ? 1 : aotx_shared.participants[participant].floor, 8);
        aotx_service_put(out + 172, participant == AOTX_SHARED_NONE ? 0 : 1, 4);
    }
    if (kind == AOTX_SHARED_CAPABILITIES) {
        unsigned values[12] = {aotx_shared.participant_capacity, aotx_shared.space_capacity, aotx_shared.conversation_capacity,
            aotx_shared.member_capacity, aotx_shared.receipt_capacity, AOTX_SHARED_COMMAND_BYTES, AOTX_SHARED_RESULT_BYTES,
            2048, AOTX_SHARED_MEDIA_REFS, AOTX_SHARED_EMIT, 1, AOTX_SHARED_PROMPT_BYTES};
        for (unsigned i = 0; i < 12; ++i) aotx_service_put(out + AOTX_SHARED_REPLY_HEAD + i * 4, values[i], 4);
        aotx_service_put(out + 192, 1, 4); aotx_service_put(out + 196, 48, 4); tail = 48;
    }
    if (scoped) {
        aotx_service_bytes(out + 32, read + 32, 16); aotx_service_bytes(out + 48, aotx_shared.spaces[space].id, 16);
        aotx_service_put(out + 228, aotx_shared.spaces[space].scope, 4);
        aotx_service_put(out + 232, aotx_shared_rights(participant, space), 4);
    }
    if (conv) {
        const aotx_shared_conversation &c = aotx_shared.conversations[conversation];
        aotx_service_put(out + 104, c.next_order, 8); aotx_service_put(out + 112, c.event_floor, 8);
        aotx_service_put(out + 12, c.request ? aotx_shared.receipts[c.request - 1].phase : AOTX_SHARED_DONE, 4);
    }
    if (kind == AOTX_SHARED_PROMPT_READ) {
        const aotx_shared_conversation &c = aotx_shared.conversations[conversation];
        unsigned char *p = out + AOTX_SHARED_REPLY_HEAD;
        aotx_service_put(p, 1, 4); aotx_service_put(p + 4, c.prompt_mode, 4);
        aotx_service_put(p + 8, c.prompt_length, 4); aotx_service_put(p + 12, 0, 4);
        aotx_service_bytes(p + 16, c.prompt, c.prompt_length);
        tail = 16 + c.prompt_length;
        aotx_service_put(out + 192, 1, 4); aotx_service_put(out + 196, tail, 4);
    }
    if (kind == AOTX_SHARED_AFFECT_READ) {
#ifdef AOTX_AFFECT
        aotx_shared_affect_read(conversation, out + AOTX_SHARED_REPLY_HEAD);
        aotx_service_put(out + 192, 1, 4); aotx_service_put(out + 196, 96, 4); tail = 96;
#else
        aotx_service_answer(channel, 501, 0); return;
#endif
    }
    if (kind == AOTX_SHARED_SPACES_READ || kind == AOTX_SHARED_CONVERSATIONS_READ || kind == AOTX_SHARED_MEMBERS_READ)
        tail = aotx_shared_list(kind, participant, space, cursor, limit, out);
    if (kind == AOTX_SHARED_EVENTS_READ) tail = aotx_shared_events(g, conversation, cursor, limit, out);
    if (tail == AOTX_SHARED_NONE) { aotx_service_answer(channel, 409, 0); return; }
    aotx_service_answer(channel, 200, 0, AOTX_SHARED_REPLY_HEAD + tail);
}
