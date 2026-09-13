/* Purpose: Validate canonical shared commands before their ordered record commit.
 * Owns: Retry comparison, resource reservations and accepted receipt bytes.
 * Launch shape: The service admission node processes a bounded mailbox batch.
 * Lifetime: Admission through exact receipt retirement. */
#include "shared/internal.cuh"
#include <stddef.h>
#include "rng/rng.cuh"
__device__ aotx_shared_receipt aotx_shared_candidate;
static __device__ bool aotx_shared_text(const unsigned char *p, unsigned n)
{
    for (unsigned i = 0; i < n;) {
        unsigned c = p[i++], follow = 0, point = c, least = 0;
        if (!c) return false;
        if (c >= 0xc2 && c <= 0xdf) { follow = 1; point = c & 31; least = 0x80; }
        else if (c >= 0xe0 && c <= 0xef) { follow = 2; point = c & 15; least = 0x800; }
        else if (c >= 0xf0 && c <= 0xf4) { follow = 3; point = c & 7; least = 0x10000; }
        else if (c >= 0x80) return false;
        if (follow > n - i) return false;
        for (unsigned j = 0; j < follow; ++j) {
            c = p[i++]; if ((c & 0xc0) != 0x80) return false;
            point = (point << 6) | (c & 63);
        }
        if (point < least || point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff)) return false;
    }
    return true;
}
__device__ unsigned aotx_shared_command_check(const unsigned char *p, unsigned n)
{
    if (n < AOTX_SHARED_COMMAND_HEAD || n > AOTX_SHARED_COMMAND_BYTES ||
        !aotx_service_equal(p, (const unsigned char *)AOTX_SHARED_MAGIC, 8)) return 400;
    unsigned op = aotx_shared_u32(p + 8), text = aotx_shared_u32(p + 136), media = aotx_shared_u32(p + 140);
    if (!op || op > AOTX_SHARED_SAVE || !aotx_shared_u64(p + 16) ||
        !aotx_service_nonzero(p + 24, 16) || !aotx_shared_id(p + 40, aotx_live_store.lineage) ||
        aotx_service_nonzero(p + 144, 48) || text > 2048 || media > AOTX_SHARED_MEDIA_REFS ||
        n != AOTX_SHARED_COMMAND_HEAD + text + media * AOTX_SHARED_MEDIA_ROW) return 400;
    bool target = op == AOTX_SHARED_SPACE || op == AOTX_SHARED_MEMBER || op == AOTX_SHARED_CONVERSATION ||
        op == AOTX_SHARED_INPUT || op == AOTX_SHARED_CANCEL || op == AOTX_SHARED_PUBLISH;
    if (aotx_service_nonzero(p + 56, 16) != target) return 400;
    if (op != AOTX_SHARED_CONVERSATION && op != AOTX_SHARED_PUBLISH && aotx_service_nonzero(p + 72, 16)) return 400;
    if ((op == AOTX_SHARED_CONVERSATION || op == AOTX_SHARED_PUBLISH) && !aotx_service_nonzero(p + 72, 16)) return 400;
    if ((op == AOTX_SHARED_MEMBER) != aotx_service_nonzero(p + 88, 16)) return 400;
    if ((op != AOTX_SHARED_SPACE && aotx_shared_u32(p + 12)) || aotx_shared_u32(p + 12) > 2) return 400;
    if ((op != AOTX_SHARED_MEMBER && aotx_shared_u32(p + 116)) || aotx_shared_u32(p + 116) > 7) return 400;
    if (op != AOTX_SHARED_INPUT && (aotx_service_nonzero(p + 104, 12) ||
        aotx_service_nonzero(p + 120, 8) || text || media)) return 400;
    if (op != AOTX_SHARED_RETIRE && op != AOTX_SHARED_CANCEL && op != AOTX_SHARED_PUBLISH && aotx_shared_u64(p + 128)) return 400;
    if (op == AOTX_SHARED_INPUT && ((!text && !media) || !aotx_shared_text(p + AOTX_SHARED_COMMAND_HEAD, text))) return 400;
    for (unsigned i = 0; i < media; ++i) {
        const unsigned char *m = p + AOTX_SHARED_COMMAND_HEAD + text + i * AOTX_SHARED_MEDIA_ROW;
        if ((aotx_shared_u32(m) != 1 && aotx_shared_u32(m) != 2) || aotx_shared_u32(m + 4) ||
            !aotx_service_nonzero(m + 8, 32)) return 400;
    }
    return 200;
}
static __device__ unsigned aotx_shared_resources(aotx_shared_receipt *r, const aotx_service_grant *g)
{
    const unsigned char *p = r->command;
    unsigned op = r->operation, participant = r->participant;
    if (op == AOTX_SHARED_REGISTER) return aotx_shared.participants[participant].active ? 409 : 200;
    if (!aotx_shared.participants[participant].active) return 404;
    if (op == AOTX_SHARED_SPACE) {
        if (aotx_shared_space_find(p + 56) != AOTX_SHARED_NONE) return 409;
        if (aotx_shared_u32(p + 12) && !(g->actions & AOTX_SHARED_MANAGE_ACTION)) return 403;
        for (unsigned i = 0; i < aotx_shared.space_capacity; ++i)
            if (!aotx_shared.spaces[i].active) { r->space = i; return 200; }
        return 429;
    }
    if (op == AOTX_SHARED_RETIRE) {
        unsigned long long floor = aotx_shared_u64(p + 128);
        const aotx_shared_participant &person = aotx_shared.participants[participant];
        if (floor <= person.floor || floor > person.next) return 409;
        for (unsigned i = 0; i < aotx_shared.receipt_capacity; ++i) {
            const aotx_shared_receipt &old = aotx_shared.receipts[i];
            if (!old.phase || old.participant != participant || old.sequence >= floor) continue;
            if (old.phase < AOTX_SHARED_DONE || !old.saved_terminal) return 409;
        }
        return 200;
    }
    if (op == AOTX_SHARED_SAVE) return 200;
    if (op == AOTX_SHARED_CONVERSATION) {
        r->space = aotx_shared_space_find(p + 72);
        if (!aotx_shared_visible(participant, r->space, 2)) return 404;
        if (aotx_shared_conversation_find(p + 56) != AOTX_SHARED_NONE) return 409;
        for (unsigned i = 0; i < aotx_shared.conversation_capacity; ++i)
            if (!aotx_shared.conversations[i].active) { r->conversation = i; return 200; }
        return 429;
    }
    if (op == AOTX_SHARED_MEMBER) {
        r->space = aotx_shared_space_find(p + 56);
        if (!(g->actions & AOTX_SHARED_MANAGE_ACTION) || !aotx_shared_visible(participant, r->space, 4)) return 404;
        unsigned member = aotx_shared_participant_find(p + 88);
        if (member == AOTX_SHARED_NONE || aotx_shared_id(p + 88, aotx_shared.spaces[r->space].owner)) return 409;
        for (unsigned i = 0; i < aotx_shared.member_capacity; ++i)
            if (!aotx_shared.members[i].active ||
                (aotx_shared.members[i].space == r->space && aotx_shared.members[i].participant == member)) return 200;
        return 429;
    }
    if (op == AOTX_SHARED_PUBLISH) {
        r->space = aotx_shared_space_find(p + 72);
        if (!(g->actions & AOTX_SHARED_MANAGE_ACTION) || !aotx_shared_visible(participant, r->space, 6)) return 404;
        return aotx_shared_publish_check(r, g);
    }
    if (op == AOTX_SHARED_CANCEL) {
        unsigned target = aotx_shared_id_find(p + 56);
        if (target == AOTX_SHARED_NONE || aotx_shared.receipts[target].participant != participant ||
            aotx_shared.receipts[target].sequence != aotx_shared_u64(p + 128)) return 404;
        r->space = aotx_shared.receipts[target].space;
        r->conversation = aotx_shared.receipts[target].conversation;
        return aotx_shared_authorized(aotx_shared.receipts + target, 2) ? 200 : 404;
    }
    r->conversation = aotx_shared_conversation_find(p + 56);
    if (r->conversation == AOTX_SHARED_NONE) return 404;
    aotx_shared_conversation &c = aotx_shared.conversations[r->conversation];
    r->space = c.space;
    if (!aotx_shared_visible(participant, r->space, 2)) return 404;
    if (c.request || c.next_order == ~0ull) return 409;
    unsigned active = 0;
    for (unsigned i = 0; i < aotx_shared.receipt_capacity; ++i) {
        const aotx_shared_receipt &old = aotx_shared.receipts[i];
        if (old.phase && old.phase < AOTX_SHARED_DONE && old.participant == participant) ++active;
    }
    if (active >= g->requests) return 429;
    r->order = c.next_order;
    return aotx_shared_input_check(r, g);
}
__device__ unsigned aotx_shared_admit(const aotx_service_grant *g, const unsigned char *p,
                                      unsigned n, unsigned *receipt)
{
    if (!(g->actions & AOTX_SHARED_WRITE_ACTION)) return 403;
    unsigned status = aotx_shared_command_check(p, n);
    if (status != 200) return status;
    unsigned participant = aotx_shared_participant_find(g->principal);
    unsigned long long sequence = aotx_shared_u64(p + 16);
    if (participant != AOTX_SHARED_NONE && sequence < aotx_shared.participants[participant].floor) return 410;
    if (aotx_shared.kind == AOTX_SHARED_ADMIT_RECORD && aotx_shared_id(aotx_shared_candidate.actor, g->principal)) {
        const aotx_shared_receipt &old = aotx_shared_candidate;
        if (old.sequence == sequence) {
            if (old.length != n || !aotx_service_equal(old.command, p, n)) return 409;
            *receipt = aotx_shared.receipt_capacity; return 202;
        }
        if (aotx_shared_id(old.key, p + 24)) return 409;
    }
    for (unsigned i = 0; i < aotx_shared.receipt_capacity; ++i) {
        const aotx_shared_receipt &old = aotx_shared.receipts[i];
        if (!old.phase || !aotx_shared_id(old.actor, g->principal)) continue;
        if (old.sequence == sequence) {
            if (old.length != n || !aotx_service_equal(p, old.command, n)) return 409;
            if (old.space != AOTX_SHARED_NONE && old.operation != AOTX_SHARED_SPACE &&
                !aotx_shared_visible(participant, old.space, 1)) return 404;
            *receipt = i; return old.phase == AOTX_SHARED_ACCEPTED ? 202 : 200;
        }
        if (aotx_shared_id(old.key, p + 24)) return 409;
    }
    unsigned long long next = participant == AOTX_SHARED_NONE ? 1 : aotx_shared.participants[participant].next;
    unsigned long long floor = participant == AOTX_SHARED_NONE ? 1 : aotx_shared.participants[participant].floor;
    if (sequence < floor) return 410;
    if (sequence != next || next == ~0ull) return 409;
    if (aotx_shared.kind || aotx_shared.received || aotx_shared.pressure || aotx_shared.disk_error) return 429;
    if (participant == AOTX_SHARED_NONE) {
        if (aotx_shared_u32(p + 8) != AOTX_SHARED_REGISTER) return 404;
        for (unsigned i = 0; i < aotx_shared.participant_capacity; ++i)
            if (!aotx_shared.participants[i].active) { participant = i; break; }
        if (participant == AOTX_SHARED_NONE) return 429;
    }
    aotx_shared_receipt *r = &aotx_shared_candidate;
    aotx_shared_zero(r, (unsigned)offsetof(aotx_shared_receipt, command));
    aotx_service_bytes(r->actor, g->principal, 16); aotx_service_bytes(r->key, p + 24, 16);
    aotx_service_bytes(r->command, p, n); r->length = n; r->sequence = sequence; r->revision = g->revision;
    r->operation = aotx_shared_u32(p + 8); r->participant = participant; r->space = r->conversation = AOTX_SHARED_NONE;
    r->slot = AOTX_SLOTS; r->phase = AOTX_SHARED_ACCEPTED;
    r->role = aotx_shared_u32(p + 104); r->limit = aotx_shared_u32(p + 108); r->pages = aotx_shared_u32(p + 112);
    status = aotx_shared_resources(r, g);
    if (status != 200) return status;
    unsigned index = AOTX_SHARED_NONE;
    for (unsigned i = 0; i < aotx_shared.receipt_capacity; ++i) {
        const aotx_shared_receipt &old = aotx_shared.receipts[i];
        if (!old.phase || (r->operation == AOTX_SHARED_RETIRE && old.participant == participant &&
            old.sequence < aotx_shared_u64(p + 128))) { index = i; break; }
    }
    unsigned long long identity_serial = aotx_shared.serial + 1;
    uint4 identity = aotx_rng_lane(aotx_seam.boot_id, 0x53485231u, 0, identity_serial);
    aotx_service_put(r->id, identity.x, 4); aotx_service_put(r->id + 4, identity.y, 4);
    aotx_service_put(r->id + 8, identity.z, 4); aotx_service_put(r->id + 12, identity.w, 4);
    if (!aotx_service_nonzero(r->id, 16) || aotx_shared_id_find(r->id) != AOTX_SHARED_NONE) return 409;
    if (index == AOTX_SHARED_NONE || !aotx_shared_begin(AOTX_SHARED_ADMIT_RECORD, AOTX_SHARED_ADMIT_HEAD + n)) return 429;
    unsigned char *out = aotx_shared.transfer;
    aotx_service_bytes(out, r->actor, 16); aotx_service_put(out + 16, r->revision, 8);
    aotx_service_put(out + 24, index, 4); aotx_service_put(out + 28, participant, 4);
    aotx_service_put(out + 32, r->space, 4); aotx_service_put(out + 36, r->conversation, 4);
    aotx_service_put(out + 40, r->order, 8); aotx_service_bytes(out + 48, r->model_digest, 32);
    aotx_service_put(out + 80, r->media_count, 4); aotx_service_put(out + 88, r->pages, 4);
    aotx_service_bytes(out + 96, r->id, 16);
    for (unsigned i = 0; i < r->media_count; ++i) {
        aotx_service_put(out + 112 + i * 16, r->media[i].object, 4);
        aotx_service_put(out + 120 + i * 16, r->media[i].generation, 8);
    }
    aotx_service_bytes(out + AOTX_SHARED_ADMIT_HEAD, p, n);
    aotx_shared.pending_receipt = index; *receipt = aotx_shared.receipt_capacity;
    return 202;
}
