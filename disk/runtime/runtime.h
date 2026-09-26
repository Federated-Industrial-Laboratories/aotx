/* Purpose: Define the required text runtime index and its asset references.
 * Owns: Portable byte fields and disk metadata declarations.
 * Threading: One leased file reader or writer processes the complete asset batch.
 * Lifetime: Runtime index schemas 1 through 7. */
#ifndef AOTX_RUNTIME_H
#define AOTX_RUNTIME_H
#include "disk/ccir/ccir.h"
#define AOTX_RUNTIME_HEADER 256u
#define AOTX_RUNTIME_ROW 384u
#define AOTX_RUNTIME_NAME 256u
#define AOTX_RUNTIME_AFFECT 1u
#define AOTX_RUNTIME_VISION 2u
#define AOTX_RUNTIME_AUDIO 4u
#define AOTX_RUNTIME_SHARED 8u
#define AOTX_RUNTIME_POLICY 16u
#define AOTX_RUNTIME_APPRAISAL 32u
#define AOTX_RUNTIME_POLICY_HISTORY 64u
#define AOTX_RUNTIME_COLD 128u
#define AOTX_RUNTIME_REVIEW 256u
#define AOTX_RUNTIME_ABI 1u
#define AOTX_RUNTIME_SHARED_SCHEMA 1u
#define AOTX_RUNTIME_SHARED_BYTES 48u
/* Header: magic AOTXRT01, schema/row/count/features at 8/12/16/20.
 * Wire layout/slots/object capacity/architecture at 24/28/32/36.
 * Payload capacity at 40, runtime ABI at 48, zero at 52..63.
 * Initial model roles at 64..127; replay ID at 128..143.
 *
 * Shared profile: AOTXSH01 at 144, schema/bytes at 152/156.
 * Participant/space/conversation/member/receipt capacities at 160/164/168/172/176.
 * Command/result byte capacities at 180/184; zero at 188..255 unless appraisal is required.
 * Without the shared feature, bytes 144..187 are zero.
 *
 * Appraisal profiles use section schema 4: profile schema 1 at 188, processor digest at 192, model digest at 224.
 * Without appraisal, bytes 188..255 are zero.
 * Without appraisal, policy profiles require section schema 3, with or without shared tables.
 * Without either feature, shared profiles require section schema 2; base profiles require section schema 1.
 *
 * Row: section ID at 0, kind/flags at 16/20, bytes at 24, digest at 32.
 * Name at 64..319; zero at 320..383. Text has a zero terminator and zero padding.
 *
 * Kind 1 is a model asset. Kind 2 is a data module asset. All rows are required.
 * Kind 3 is the unique policy.bin asset and requires the policy feature.
 * Kind 4 is policy-history.bin. It requires the policy history feature and section schema 5. */
typedef struct aotx_runtime_index {
    unsigned char header[AOTX_RUNTIME_HEADER];
    unsigned char rows[AOTX_CCIR_SECTIONS][AOTX_RUNTIME_ROW];
    uint32_t count;
} aotx_runtime_index;
typedef struct aotx_runtime_shared_profile {
    uint32_t participants, spaces, conversations, members, receipts;
    uint32_t command_bytes, result_bytes;
} aotx_runtime_shared_profile;
#ifdef __cplusplus
extern "C" {
#endif
int aotx_runtime_name(const char *name);
int aotx_runtime_index_read(int fd, const aotx_ccir_view *view, aotx_runtime_index *index);
int aotx_runtime_profile(int fd, const aotx_ccir_view *view, const unsigned char id[16]);
int aotx_runtime_section(const aotx_ccir_view *view, const unsigned char id[16]);
int aotx_runtime_dependencies(const aotx_ccir_view *view);
uint16_t aotx_runtime_schema(uint32_t features);
void aotx_runtime_revision(const aotx_ccir_view *view, unsigned char digest[32]);
int aotx_runtime_shared_read(const unsigned char header[AOTX_RUNTIME_HEADER],
    aotx_runtime_shared_profile *profile);
void aotx_runtime_shared_write(unsigned char header[AOTX_RUNTIME_HEADER],
    const aotx_runtime_shared_profile *profile);
int aotx_runtime_shared_fits(const aotx_runtime_shared_profile *profile);
void aotx_runtime_shared_current(aotx_runtime_shared_profile *profile);
#ifdef __cplusplus
}
#endif
#endif
