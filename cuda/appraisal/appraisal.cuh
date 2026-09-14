/* Purpose: Coordinate source appraisal through the resident model and memory store.
 * Owns: Batched candidate metadata and bounded work progress on the device.
 * Launch shape: One source row per leased sequence, up to AOTX_RECALL_BATCH.
 * Lifetime: One recorded work batch; durable state resides in typed objects. */
#ifndef AOTX_APPRAISAL_APPRAISAL_CUH
#define AOTX_APPRAISAL_APPRAISAL_CUH
#include "appraisal/format.h"
#include "cognitive/intake.cuh"
typedef struct aotx_appraisal_prefix {
    uint32_t stage, field, digits, number, gap, evidence, task, trust;
    aotx_intake_prefix quote;
} aotx_appraisal_prefix;
typedef struct aotx_appraisal_row {
    uint32_t queue, source, slot, status, task_source;
    uint32_t quote_start, quote_length, task_start, task_length;
    uint32_t commitment_start, commitment_length, correction;
    uint32_t values[AOTX_APPRAISAL_VALUES];
    uint32_t prior_count, prior[AOTX_RECALL_LIMIT];
    unsigned char task[16], model[32];
    unsigned char ids[2][16];
    aotx_appraisal_prefix prefix;
} aotx_appraisal_row;
typedef struct aotx_appraisal_state {
    uint32_t active, background, count, status, config;
    uint32_t pages, tokens, ticks, pending, last_status;
    uint32_t enqueue_count, enqueue_first, enqueue_offset;
    uint32_t observed_count, observed_bytes, write_flags;
    uint32_t control_pending, explicit_pending, blocked_count, blocked_bytes, recovery;
    uint64_t blocked_sequence, blocked_root;
    unsigned char control[AOTX_APPRAISAL_CONFIG_BYTES];
    uint64_t observed_maintenance;
    unsigned char enqueue_ids[AOTX_RECALL_BATCH][16];
    uint32_t enqueue_rows[AOTX_RECALL_BATCH];
    uint64_t observed, observed_root, revision, calls, completed, refused, interrupted;
    aotx_appraisal_row rows[AOTX_RECALL_BATCH];
} aotx_appraisal_state;
extern __device__ aotx_appraisal_state aotx_appraisal;
extern __device__ const unsigned char aotx_appraisal_processor[32];
__device__ uint32_t aotx_appraisal_pending(void);
__device__ uint64_t aotx_appraisal_revision(void);
__device__ bool aotx_appraisal_enabled(void);
__device__ bool aotx_appraisal_interrupted(void);
__device__ bool aotx_appraisal_foreground(void);
__device__ uint32_t aotx_appraisal_parse(uint32_t row);
__device__ uint32_t aotx_appraisal_prompt(uint32_t row, uint32_t slot);
__device__ bool aotx_appraisal_advance(uint32_t row, const unsigned char *, uint32_t);
__device__ void aotx_appraisal_refresh(void);
__device__ uint32_t aotx_appraisal_resolve(const aotx_cognitive_query *);
__device__ void aotx_appraisal_auto_request(void);
__device__ void aotx_appraisal_begin(void);
__device__ void aotx_appraisal_decide(void);
__device__ void aotx_appraisal_control(void);
__device__ void aotx_appraisal_control_request(void);
__device__ uint32_t aotx_appraisal_request(bool);
__device__ void aotx_appraisal_publish(void);
__device__ uint32_t aotx_appraisal_queue_prepare(uint32_t *, uint32_t *);
__device__ void aotx_appraisal_queue_encode(unsigned char *, uint32_t, uint32_t);
struct aotx_cli_out;
__device__ void aotx_appraisal_command(const unsigned char *, uint32_t, struct aotx_cli_out *);
#endif
