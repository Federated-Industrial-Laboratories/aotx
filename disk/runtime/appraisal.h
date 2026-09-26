/* Purpose: Admit the built-in appraisal processor and its saved model dependencies.
 * Owns: Required runtime metadata and bounded disk identity checks.
 * Threading: One reader checks the complete memory and asset extent batch.
 * Lifetime: Packing, activation and durable runtime generations. */
#ifndef AOTX_RUNTIME_APPRAISAL_H
#define AOTX_RUNTIME_APPRAISAL_H
#include "disk/runtime/runtime.h"
#include "disk/ccir/internal.h"
#include "cuda/appraisal/format.h"
#include "cuda/seam/wire.h"
#include "cognitive/cold.h"
#include "reflection/format.h"
#include <string.h>
typedef struct aotx_runtime_appraisal_models {
    uint32_t count;
    unsigned char selected[32], digest[8][32];
} aotx_runtime_appraisal_models;
static inline uint32_t aotx_runtime_appraisal_contract(const unsigned char *p) {
    static const unsigned char current[32] = AOTX_APPRAISAL_PROCESSOR_BYTES;
    static const unsigned char legacy[32] = AOTX_APPRAISAL_LEGACY_PROCESSOR_BYTES;
    return !memcmp(p, current, 32) ? 2 : !memcmp(p, legacy, 32) ? 1 : 0;
}
static inline int aotx_runtime_appraisal_profile(const unsigned char *h) {
    if (!(aotx_ccir_u32(h + 20) & AOTX_RUNTIME_APPRAISAL))
        return aotx_ccir_zero(h + 188, 68) ? 0 : AOTX_CCIR_INVALID;
    if (aotx_ccir_u32(h + 188) != 1 || !aotx_runtime_appraisal_contract(h + 192)) return AOTX_CCIR_UNSUPPORTED;
    return aotx_ccir_zero(h + 224, 32) ? AOTX_CCIR_INVALID : 0;
}
static inline uint32_t aotx_runtime_appraisal_record(const aotx_record_header *r) {
    if (r->cls != AOTX_CLASS_A) return 0;
    if (r->type == AOTX_REC_POLICY_CONTROL) return AOTX_RUNTIME_REVIEW;
    if (r->type != 33 || r->body_len < 8) return 0;
    const unsigned char *body = (const unsigned char *)r + AOTX_HEADER_BYTES;
    uint32_t operation = aotx_ccir_u32(body + 4);
    if (operation == AOTX_REVIEW_REQUEST || operation == AOTX_REVIEW_RESULT) return AOTX_RUNTIME_REVIEW;
    if (operation == AOTX_COLD_CONTROL || operation == AOTX_COLD_RESULT) return AOTX_RUNTIME_COLD;
    return operation >= AOTX_APPRAISAL_CONTROL && operation <= AOTX_APPRAISAL_RESULT ? AOTX_RUNTIME_APPRAISAL : 0;
}
int aotx_runtime_appraisal_model_read(int fd, uint64_t offset, uint64_t bytes,
    const char *roles, aotx_runtime_appraisal_models *models);
int aotx_runtime_appraisal_model_view(const aotx_ccir_view *view, const aotx_runtime_index *index,
    aotx_runtime_appraisal_models *models);
int aotx_runtime_appraisal_scan(const unsigned char *memory, int fd, uint64_t offset, uint64_t bytes,
    const aotx_runtime_appraisal_models *models, uint32_t *required);
void aotx_runtime_appraisal_require(unsigned char *header, const aotx_runtime_appraisal_models *models);
int aotx_runtime_appraisal_dependencies(const aotx_ccir_view *view, const aotx_runtime_index *index);
int aotx_runtime_appraisal_checkpoint(const aotx_ccir_view *view, aotx_runtime_index *index,
    const unsigned char *memory, uint64_t bytes, uint32_t replay_features);
int aotx_runtime_appraisal_replay_check(const aotx_ccir_view *replay, const aotx_ccir_view *assets,
    const aotx_runtime_index *index);
struct aotx_runtime_pack;
int aotx_runtime_pack_appraisal(struct aotx_runtime_pack *pack);
#endif
