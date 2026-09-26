/* Purpose: Keep shared membership, conversations and exact operation results on the device.
 * Owns: Persistent tables and bounded journal transfer state.
 * Launch shape: One ordered admission batch and one finite record emitter.
 * Lifetime: The complete runtime lineage; execution slots are temporary. */
#ifndef AOTX_SHARED_STATE_CUH
#define AOTX_SHARED_STATE_CUH
#include "shared/profile.h"
#include "shared/wire.h"
#include "service/service.cuh"
#include "cognitive/live.cuh"
#ifdef AOTX_AFFECT
#include "affect/affect.cuh"
struct aotx_shared_affect_state {
    aotx_affect_agent_state value;
    unsigned enabled;
    unsigned long long revision;
    unsigned reason, available, role;
    unsigned char model_digest[32];
};
#endif
struct aotx_shared_participant {
    unsigned char id[16];
    unsigned long long next, floor;
    unsigned active;
};
struct aotx_shared_space {
    unsigned char id[16], owner[16];
    unsigned active, scope;
#ifdef AOTX_AFFECT
    aotx_shared_affect_state affect;
#endif
};
struct aotx_shared_member {
    unsigned participant, space, permissions, active;
};
struct aotx_shared_conversation {
    unsigned char id[16];
    unsigned active, space, request, reserved;
    unsigned long long next_order, event_floor;
    aotx_live_binding binding;
#ifdef AOTX_AFFECT
    aotx_shared_affect_state affect;
#endif
};
struct aotx_shared_receipt {
    unsigned char actor[16], key[16], id[16], model_digest[32];
    unsigned long long sequence, revision, order, admission_source, terminal_source;
    unsigned phase, status, operation, participant, space, conversation, slot;
    unsigned length, output, prompt, sampled, finish, cancel, gap, input_committed;
    unsigned saved_admission, saved_terminal;
    unsigned role, limit, pages, media_count;
    aotx_model_how sample;
    aotx_media_reference media[AOTX_SHARED_MEDIA_REFS];
    unsigned char command[AOTX_SHARED_COMMAND_BYTES];
    unsigned char result[AOTX_SHARED_RESULT_BYTES];
};
struct aotx_shared_state {
    aotx_shared_participant *participants;
    aotx_shared_space *spaces;
    aotx_shared_member *members;
    aotx_shared_conversation *conversations;
    aotx_shared_receipt *receipts;
    unsigned participant_capacity, space_capacity, member_capacity, conversation_capacity, receipt_capacity;
    unsigned enabled, replaying, fatal, slot[AOTX_SLOTS];
    unsigned kind, total, written, received, pending_receipt;
    unsigned long long serial, transfer_serial, source, saved_source, saved_generation;
    unsigned char saved_incarnation[16], saved_commit_digest[32];
    unsigned long long saved_boot;
    unsigned long long pending_bytes;
    unsigned disk_error, pressure;
    unsigned char transfer[AOTX_SHARED_TRANSFER];
};
extern __device__ aotx_shared_state aotx_shared;
__device__ bool aotx_shared_owns(unsigned slot);
__device__ const unsigned char *aotx_shared_actor(unsigned slot);
__device__ aotx_shared_receipt *aotx_shared_request(unsigned slot);
__device__ bool aotx_shared_authorized(const aotx_shared_receipt *receipt, unsigned rights);
__device__ void aotx_shared_handle(unsigned channel, const aotx_service_grant *grant, unsigned char *frame);
__global__ void aotx_shared_emit(void);
__device__ void aotx_shared_part(const unsigned char *body, unsigned bytes, unsigned long long source);
__device__ bool aotx_shared_restore_end(void);
__device__ unsigned aotx_shared_window(unsigned long long base, unsigned count);
__device__ bool aotx_shared_lease(const unsigned *requests, const unsigned *slots, unsigned count);
__device__ bool aotx_shared_output(unsigned request, const unsigned char *bytes, unsigned count);
__device__ bool aotx_shared_complete(unsigned request, unsigned status, unsigned prompt,
                                    unsigned sampled, unsigned finish);
__device__ bool aotx_shared_quiet(void);
__device__ void aotx_shared_ack(unsigned long long source, unsigned long long generation,
                               const unsigned char *incarnation, unsigned long long boot,
                               const unsigned char *commit_digest);
/* The bridge validates new input, then records its actual model identity.
 * Replay reads this identity without consulting current deployment grants. */
__device__ unsigned aotx_shared_input_check(aotx_shared_receipt *receipt, const aotx_service_grant *grant);
__device__ bool aotx_shared_bridge_lease(const unsigned *requests, const unsigned *slots,
    unsigned count, bool replay, unsigned recall_revision = 0, const unsigned *retention = 0, const unsigned *affect = 0);
__device__ void aotx_shared_bridge_release(unsigned request, bool replay);
__device__ void aotx_shared_memory_read(unsigned channel, const aotx_service_grant *grant,
                                       const unsigned char *read, unsigned space);
__device__ unsigned aotx_shared_publish_check(const aotx_shared_receipt *receipt,
                                             const aotx_service_grant *grant);
__device__ bool aotx_shared_publish_apply(unsigned request, bool replay);
#endif
