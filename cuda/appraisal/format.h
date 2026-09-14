/* Purpose: Define scoped appraisal work and relationship evidence formats.
 * Owns: Portable payload fields and configuration limits.
 * Launch shape: Source batches share the configured memory allocation.
 * Lifetime: Typed memory, ordinary journal and complete runtime files. */
#ifndef AOTX_APPRAISAL_FORMAT_H
#define AOTX_APPRAISAL_FORMAT_H
#define AOTX_APPRAISAL_PROCESSOR_BYTES {0xc8, 0x3f, 0xd9, 0xf8, 0xca, 0x7e, 0x8f, 0xc4, 0xf2, 0x18, 0x39, 0x4a, 0xc1, 0xc4, 0x48, 0x0f, 0x85, 0xcf, 0x29, 0x06, 0x12, 0x00, 0xa4, 0x41, 0xdd, 0xd2, 0xa7, 0x9d, 0x5f, 0x3f, 0x37, 0x80}
#define AOTX_APPRAISAL_RESULT_ROW 4160u
#define AOTX_APPRAISAL_CONFIG_BYTES 96u
#define AOTX_APPRAISAL_QUEUE_BYTES 160u
#define AOTX_APPRAISAL_RELATION_BYTES 192u
#define AOTX_APPRAISAL_ASSESS_BYTES 128u
#define AOTX_APPRAISAL_VALUES 9u
#define AOTX_APPRAISAL_WRITE 1u
#define AOTX_APPRAISAL_RECALL 2u
#define AOTX_APPRAISAL_BACKGROUND 4u
#define AOTX_APPRAISAL_PENDING 0u
#define AOTX_APPRAISAL_COMPLETE 1u
#define AOTX_APPRAISAL_REFUSED 2u
#define AOTX_APPRAISAL_INTERRUPTED 3u
#define AOTX_APPRAISAL_CONTROL 15u
#define AOTX_APPRAISAL_REQUEST 16u
#define AOTX_APPRAISAL_RESULT 17u
/* Configuration: AOTXAPC1, schema/flags at 8/12, pages/tokens/ticks at 16/20/24.
 * Recall floor/priority are at 28/32, work rows at 36, processor SHA-256 at 40.
 * Bytes 72..95 are zero. Its object version is the configuration revision. */

/* Queue: AOTXAPQ1, schema/status at 8/12, config ID/version at 16/32.
 * Task ID is at 40, result status at 56, zero at 60, processor/model SHA-256 at 64/96.
 * Optional task descriptor ID/version are at 128/144; bytes 152..159 are zero.
 * Object source and subject bind the admitted event. */

/* Assessment: existing 32-byte appraisal prefix with schema 2.
 * Processor/model SHA-256 are at 32/64, queue ID/version at 96/112, source quote start/length at 120/124. */

/* Relationship: AOTXREL1, schema at 8, exposure count at 12, task ID at 32.
 * Regard gain/loss and task trust gain/loss are at 16/20/24/28.
 * Source, task and commitment quote pairs are at 48/52, 56/60 and 64/68.
 * Processor/model SHA-256 are at 72/104, queue ID/version at 136/152; bytes 160..191 are zero.
 * Quote pairs are start/length.
 *
 * Each completed external source contributes one exposure. Repeated recall does not.
 * Gain and loss values use AOTX_COG_SCALE with explicit unknown values. */
#endif
