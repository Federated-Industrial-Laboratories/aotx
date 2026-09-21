/* Purpose: Coordinate source interpretation through leased language sequences.
 * Owns: Per-source output, parsed spans and temporary sequence ownership.
 * Launch shape: One thread per request with serial whole-batch admission.
 * Lifetime: One combined memory decision; accepted bytes are journaled. */
#ifndef AOTX_COGNITIVE_INTAKE_CUH
#define AOTX_COGNITIVE_INTAKE_CUH
#include "cognitive/live.cuh"
#include "cognitive/intake.h"
#include "cognitive/source_spans.cuh"
#include "model/forward.cuh"
#include "disk/modelfile/wrap.h"
#define AOTX_INTAKE_RUN 9u
#define AOTX_INTAKE_DONE 10u
typedef struct aotx_intake_item {
    uint32_t kind, start, length, target;
    unsigned char id[16];
} aotx_intake_item;
typedef aotx_source_span aotx_intake_span;
typedef struct aotx_intake_prefix {
    uint32_t stage, kind, node, length, start, number, digits, gap, escape, code, high;
    uint32_t utf8_left, utf8_value, utf8_min, items, used, filtered, source_end, spaces;
} aotx_intake_prefix;
typedef struct aotx_intake_row {
    uint32_t state, status, bytes, count, ticks, tokens, prompt, limit;
    unsigned char model[32], reply[AOTX_INTAKE_REPLY], quote[AOTX_RECALL_TEXT];
    aotx_intake_item items[AOTX_INTAKE_ITEMS];
    aotx_intake_prefix prefix;
    uint32_t target_count, target_bytes, target_role, target_capacity, target_index[AOTX_RECALL_LIMIT];
    unsigned char targets[AOTX_INTAKE_TARGETS];
    uint32_t phase, first_bytes, first_count, second_call, source_count;
    unsigned char first_reply[AOTX_INTAKE_REPLY];
    aotx_intake_span statements[AOTX_INTAKE_ITEMS];
    aotx_wrap wrapper;
    uint32_t target_groups[AOTX_RECALL_LIMIT], target_previous[AOTX_RECALL_LIMIT];
    uint32_t target_need[AOTX_COG_WORDS], target_done[AOTX_COG_WORDS];
} aotx_intake_row;
typedef struct aotx_intake_state {
    uint32_t row[AOTX_SLOTS], objects, payload;
    unsigned long long calls;
    aotx_model_how sample;
    aotx_intake_row rows[AOTX_RECALL_BATCH];
} aotx_intake_state;
extern __device__ aotx_intake_state aotx_intake;
extern __device__ const unsigned char aotx_intake_processor[32];
extern __device__ const unsigned char aotx_intake_source_processor[32];
extern __device__ const unsigned char aotx_intake_statement_processor[32];
__device__ __forceinline__ bool aotx_intake_owns(uint32_t slot) {
    return slot < AOTX_SLOTS && aotx_intake.row[slot] != 0;
}
__device__ uint32_t aotx_intake_spans(uint32_t row);
__device__ uint32_t aotx_intake_span_capacity(uint32_t row);
__device__ uint32_t aotx_intake_target_capacity(uint32_t row, uint32_t role);
__device__ void aotx_intake_open(uint32_t slot);
__device__ void aotx_intake_begin(void);
__device__ uint32_t aotx_intake_classify(uint32_t row, uint32_t slot);
__global__ void aotx_intake_step(void);
__device__ bool aotx_intake_advance(uint32_t row, const unsigned char *bytes, uint32_t length);
#endif
