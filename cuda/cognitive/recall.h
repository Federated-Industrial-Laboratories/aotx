/* Purpose: Define prepared memory payloads and bounded recall results.
 * Owns: Shared byte sizes and result transport; no device address is stored.
 * Launch shape: Batches of 1 to 64 requests.
 * Lifetime: One explicit context preparation or recorded replay. */
#ifndef AOTX_COGNITIVE_RECALL_H
#define AOTX_COGNITIVE_RECALL_H
#include "cognitive/format.h"
#define AOTX_RECALL_BATCH 64u
#define AOTX_RECALL_WIDTH 1024u
#define AOTX_RECALL_LIMIT 16u
#define AOTX_RECALL_PINS 8u
#define AOTX_RECALL_HEADER 64u
#define AOTX_RECALL_QUERY 8192u
#define AOTX_RECALL_REQUESTS (AOTX_RECALL_HEADER + AOTX_RECALL_BATCH * AOTX_RECALL_QUERY)
#define AOTX_RECALL_TEXT 2048u
#define AOTX_RECALL_BUDGET 4096u
#define AOTX_RECALL_CONTEXT 8192u
#define AOTX_RECALL_SELECTION (16u + AOTX_RECALL_LIMIT * 32u)
#define AOTX_RECALL_EXTENSION 6688u
#define AOTX_RECALL_SUBJECTS 64u
#define AOTX_RECALL_ACTOR (AOTX_RECALL_EXTENSION + 1072u)
#define AOTX_RECALL_TASKS 1u
#define AOTX_RECALL_APPRAISE 2u
#define AOTX_RECALL_OBLIGATION 4u
#define AOTX_RECALL_ASSESSMENT 5u
#define AOTX_RECALL_SIGNIFICANT 6u
#define AOTX_RECALL_REQUIRED 1u
#define AOTX_RECALL_FOCUS 2u
#define AOTX_RECALL_SEMANTIC 3u
typedef struct aotx_recall_result {
    uint32_t status, count, context_bytes, searches;
    uint64_t cut;
    unsigned char request_id[16], selection_id[16];
    uint32_t index[AOTX_RECALL_LIMIT], reason[AOTX_RECALL_LIMIT];
    unsigned char selection[AOTX_RECALL_SELECTION];
    unsigned char context[AOTX_RECALL_CONTEXT];
} aotx_recall_result;
#endif
