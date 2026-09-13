/* Purpose: Apply complete shared admissions without repeating current grant decisions.
 * Owns: Persistent participant, space, member, conversation and retry transitions.
 * Launch shape: One ordered record consumer.
 * Lifetime: One complete runtime lineage. */
#include "shared/internal.cuh"
#include <stddef.h>
static __device__ void aotx_shared_retire(unsigned participant, unsigned long long floor)
{
    for (unsigned i = 0; i < aotx_shared.receipt_capacity; ++i) {
        aotx_shared_receipt &r = aotx_shared.receipts[i];
        if (!r.phase || r.participant != participant || r.sequence >= floor) continue;
        if (r.operation == AOTX_SHARED_INPUT && r.conversation < aotx_shared.conversation_capacity) {
            aotx_shared_conversation &c = aotx_shared.conversations[r.conversation];
            c.event_floor = max(c.event_floor, r.order + 1);
        }
        r.phase = AOTX_SHARED_FREE;
    }
    aotx_shared.participants[participant].floor = floor;
}
__device__ bool aotx_shared_admission_apply(const unsigned char *p, unsigned n,
                                            unsigned long long source, bool replay)
{
    if (n < AOTX_SHARED_ADMIT_HEAD + AOTX_SHARED_COMMAND_HEAD || n > AOTX_SHARED_ADMIT_HEAD + AOTX_SHARED_COMMAND_BYTES) return false;
    const unsigned char *cmd = p + AOTX_SHARED_ADMIT_HEAD;
    if (aotx_shared_command_check(cmd, n - AOTX_SHARED_ADMIT_HEAD) != 200 ||
        aotx_shared_u32(cmd + 140) != aotx_shared_u32(p + 80)) return false;
    unsigned index = aotx_shared_u32(p + 24), participant = aotx_shared_u32(p + 28);
    unsigned space = aotx_shared_u32(p + 32), conversation = aotx_shared_u32(p + 36);
    unsigned op = aotx_shared_u32(cmd + 8), media = aotx_shared_u32(p + 80);
    unsigned long long sequence = aotx_shared_u64(cmd + 16);
    if (index >= aotx_shared.receipt_capacity || participant >= aotx_shared.participant_capacity ||
        !op || op > AOTX_SHARED_SAVE || media > AOTX_SHARED_MEDIA_REFS || !sequence || sequence == ~0ull ||
        !aotx_service_equal(cmd, (const unsigned char *)AOTX_SHARED_MAGIC, 8) ||
        !aotx_shared_id(cmd + 40, aotx_live_store.lineage) || !aotx_service_nonzero(p, 16) ||
        !aotx_service_nonzero(cmd + 24, 16) || !aotx_service_nonzero(p + 96, 16) ||
        aotx_shared_id_find(p + 96) != AOTX_SHARED_NONE || aotx_shared_u32(p + 84) || aotx_shared_u32(p + 92)) return false;
    aotx_shared_participant &person = aotx_shared.participants[participant];
    if (person.active ? (!aotx_shared_id(person.id, p) || person.next != sequence || op == AOTX_SHARED_REGISTER) :
        (sequence != 1 || op != AOTX_SHARED_REGISTER)) return false;
    if (op == AOTX_SHARED_RETIRE) {
        unsigned long long floor = aotx_shared_u64(cmd + 128);
        if (floor <= person.floor || floor > person.next) return false;
        for (unsigned i = 0; i < aotx_shared.receipt_capacity; ++i) {
            const aotx_shared_receipt &old = aotx_shared.receipts[i];
            if (old.phase && old.participant == participant && old.sequence < floor && old.phase < AOTX_SHARED_DONE) return false;
        }
        aotx_shared_retire(participant, floor);
    }
    if (aotx_shared.receipts[index].phase) return false;
    if ((op == AOTX_SHARED_SPACE || op == AOTX_SHARED_MEMBER || op == AOTX_SHARED_CONVERSATION ||
         op == AOTX_SHARED_INPUT || op == AOTX_SHARED_PUBLISH) && space >= aotx_shared.space_capacity) return false;
    if ((op == AOTX_SHARED_CONVERSATION || op == AOTX_SHARED_INPUT) && conversation >= aotx_shared.conversation_capacity) return false;
    aotx_shared_receipt &r = aotx_shared.receipts[index];
    aotx_shared_zero(&r, (unsigned)offsetof(aotx_shared_receipt, command));
    aotx_service_bytes(r.actor, p, 16); aotx_service_bytes(r.key, cmd + 24, 16);
    aotx_service_bytes(r.model_digest, p + 48, 32); aotx_service_bytes(r.id, p + 96, 16);
    r.sequence = sequence; r.revision = aotx_shared_u64(p + 16); r.order = aotx_shared_u64(p + 40);
    r.operation = op; r.participant = participant; r.space = space; r.conversation = conversation;
    r.length = n - AOTX_SHARED_ADMIT_HEAD; r.admission_source = source;
    r.slot = AOTX_SLOTS; r.role = aotx_shared_u32(cmd + 104); r.limit = aotx_shared_u32(cmd + 108);
    r.pages = aotx_shared_u32(p + 88); r.media_count = media;
    r.sample.temperature = __uint_as_float(aotx_shared_u32(cmd + 120));
    r.sample.top_p = __uint_as_float(aotx_shared_u32(cmd + 124));
    r.sample.repeat_penalty = 1; r.sample.voice = ~0u;
    for (unsigned i = 0; i < AOTX_MODEL_STEERS; ++i) r.sample.steer[i] = ~0u;
    for (unsigned i = 0; i < media; ++i) {
        if (aotx_shared_u32(p + 116 + i * 16)) return false;
        r.media[i] = {aotx_shared_u32(p + 112 + i * 16), aotx_shared_u64(p + 120 + i * 16)};
    }
    aotx_service_bytes(r.command, cmd, r.length);
    r.phase = op == AOTX_SHARED_INPUT || op == AOTX_SHARED_PUBLISH ? AOTX_SHARED_QUEUED : AOTX_SHARED_DONE;
    r.status = r.phase == AOTX_SHARED_DONE ? 200 : 0;
    if (r.phase == AOTX_SHARED_DONE) r.terminal_source = source;
    if (op == AOTX_SHARED_REGISTER) {
        aotx_service_bytes(person.id, p, 16); person.active = 1; person.floor = 1;
    } else if (op == AOTX_SHARED_SPACE) {
        aotx_shared_space &s = aotx_shared.spaces[space];
        if (s.active || aotx_shared_space_find(cmd + 56) != AOTX_SHARED_NONE) return false;
        aotx_service_bytes(s.id, cmd + 56, 16); aotx_service_bytes(s.owner, p, 16);
        s.active = 1; s.scope = aotx_shared_u32(cmd + 12);
    } else if (op == AOTX_SHARED_MEMBER) {
        unsigned member = aotx_shared_participant_find(cmd + 88), at = AOTX_SHARED_NONE;
        if (member == AOTX_SHARED_NONE) return false;
        for (unsigned i = 0; i < aotx_shared.member_capacity; ++i) {
            const aotx_shared_member &m = aotx_shared.members[i];
            if (m.active && m.space == space && m.participant == member) { at = i; break; }
            if (!m.active && at == AOTX_SHARED_NONE) at = i;
        }
        if (at == AOTX_SHARED_NONE) return false;
        aotx_shared.members[at] = {member, space, aotx_shared_u32(cmd + 116), 1};
    } else if (op == AOTX_SHARED_CONVERSATION) {
        aotx_shared_conversation &c = aotx_shared.conversations[conversation];
        if (c.active || aotx_shared_conversation_find(cmd + 56) != AOTX_SHARED_NONE) return false;
        aotx_shared_zero(&c, sizeof(c)); aotx_service_bytes(c.id, cmd + 56, 16);
        c.active = 1; c.space = space; c.next_order = c.event_floor = 1;
        c.binding.active = 1; c.binding.scope = aotx_shared.spaces[space].scope; c.binding.auto_retain = 1;
        aotx_service_bytes(c.binding.principal, aotx_shared.spaces[space].id, 16);
        aotx_service_bytes(c.binding.room, aotx_shared.spaces[space].id, 16);
        aotx_service_bytes(c.binding.conversation, c.id, 16);
    } else if (op == AOTX_SHARED_INPUT) {
        aotx_shared_conversation &c = aotx_shared.conversations[conversation];
        if (!c.active || c.space != space || c.request || r.order != c.next_order ||
            !aotx_shared_id(c.id, cmd + 56)) return false;
        ++c.next_order; c.request = index + 1;
    } else if (op == AOTX_SHARED_CANCEL) {
        unsigned target = aotx_shared_id_find(cmd + 56);
        if (target == AOTX_SHARED_NONE || aotx_shared.receipts[target].participant != participant ||
            aotx_shared.receipts[target].sequence != aotx_shared_u64(cmd + 128)) return false;
        aotx_shared.receipts[target].cancel = 1;
    } else if (op == AOTX_SHARED_PUBLISH) {
        if (!aotx_shared_publish_apply(index, replay)) return false;
    }
    person.next = sequence + 1;
    return true;
}
