/* Purpose: Define bounded memory transfers for live conversations and journal replay.
 * Owns: Portable byte layouts only; no device pointer or credential is stored.
 * Launch shape: Batches of 1 to 64 conversation rows.
 * Lifetime: Schema 1 transfers in class A record type 33. */
#ifndef AOTX_COGNITIVE_LIVE_H
#define AOTX_COGNITIVE_LIVE_H
#include "cognitive/recall.h"
#include "cognitive/checkpoint.h"
#include "cognitive/intake.h"
#define AOTX_LIVE_SCHEMA 1u
#define AOTX_LIVE_RECORD 33u
#define AOTX_LIVE_LOAD 1u
#define AOTX_LIVE_UPDATE 2u
#define AOTX_LIVE_BIND 3u
#define AOTX_LIVE_QUERY 4u
#define AOTX_LIVE_CHOICE 5u
#define AOTX_LIVE_TEXT 6u
#define AOTX_LIVE_TEXT_CHOICE 7u
#define AOTX_LIVE_RETAIN 8u
#define AOTX_LIVE_RETAINED 9u
#define AOTX_LIVE_AUTO_CHOICE 10u
#define AOTX_LIVE_ADMISSION 12u
#define AOTX_LIVE_MAINTAIN 13u
#define AOTX_LIVE_RETAIN_ROW 160u
#define AOTX_LIVE_RETAINED_ROW 384u
#define AOTX_LIVE_RETAINED_BYTES (64u + 64u * AOTX_LIVE_RETAINED_ROW + AOTX_COG_IMAGE)
#define AOTX_LIVE_TEXT_BYTES AOTX_RECALL_TEXT
#define AOTX_LIVE_TEXT_TICKS 128u
#define AOTX_LIVE_PART 32u
#define AOTX_LIVE_DATA 160u
#define AOTX_LIVE_EMIT 64u
#define AOTX_LIVE_TRANSFER_BYTES ((16u + 2u * AOTX_COG_IMAGE) > AOTX_LIVE_RESULTS ? \
    (16u + 2u * AOTX_COG_IMAGE) : AOTX_LIVE_RESULTS)
#define AOTX_LIVE_BYTES (AOTX_LIVE_TRANSFER_BYTES > AOTX_CP_BYTES ? AOTX_LIVE_TRANSFER_BYTES : AOTX_CP_BYTES)
#define AOTX_LIVE_HEADER 64u
#define AOTX_LIVE_BIND_ROW 64u
#define AOTX_LIVE_QUERY_ROW (64u + AOTX_RECALL_QUERY)
#define AOTX_LIVE_CHOICE_ROW (64u + AOTX_RECALL_SELECTION)
#define AOTX_LIVE_TEXT_CHOICE_ROW (AOTX_LIVE_QUERY_ROW + AOTX_RECALL_SELECTION)
#define AOTX_LIVE_TEXT_CHOICES (AOTX_LIVE_HEADER + AOTX_RECALL_BATCH * AOTX_LIVE_TEXT_CHOICE_ROW)
#define AOTX_LIVE_CHOICES (AOTX_LIVE_HEADER + AOTX_RECALL_BATCH * AOTX_LIVE_CHOICE_ROW)
#define AOTX_LIVE_AUTO_ROW (AOTX_LIVE_TEXT_CHOICE_ROW + AOTX_LIVE_RETAINED_ROW)
#define AOTX_LIVE_AUTO_BYTES (64u + 64u * AOTX_LIVE_AUTO_ROW + AOTX_COG_IMAGE)
/* Control batches fit even when the stored object allocation is small. */
#define AOTX_LIVE_INTAKE_ROW (AOTX_LIVE_AUTO_ROW + AOTX_INTAKE_EXTRA)
#define AOTX_LIVE_RESULTS (AOTX_LIVE_AUTO_BYTES + AOTX_RECALL_BATCH * AOTX_INTAKE_EXTRA)
/* Part: schema/op uint32 at 0/4, transfer ID at 8, total/offset uint32 at 24/28.
 * Data follows at 32. Each part has 160 data bytes except the last part.
 * Load: checkpoint/tail lengths uint64 at 0/8, then those exact image bytes.
 * Update: one canonical typed tail image.
 *
 * Bind/query/choice: magic AOTXBND1/AOTXLIV1/AOTXCHO1, count/schema at 8/12,
 * lineage at 16, store sequence uint64 at 32, row size at 40, zero 44..63.
 * Bind row: slot/scope uint32 at 0/4, principal/room/conversation IDs at 8/24/40,
 * page cap uint32 at 56, automatic retention (0 or 1) at 60. Binding requires a fresh idle agent slot.
 *
 * Query row: slot at 0, working focus flag (0 or 1) at 4, zero 8..15.
 * Conversation ID at 16, ordinal uint64 at 32, zero 40..63, then one prepared query row. Ordinals start at 1 per binding.
 *
 * Choice row: exact query row prefix, then the schema-1 ordered selection bytes.
 * Choice status uint32 at 44 is zero for success; a refusal has no rows.
 * Query and choice transfer IDs match. Only the device writes live choice records.
 *
 * Automatic choice: magic AOTXACH1, row size 9168, status at 44, uint64 tail bytes at 48,
 * zero 56..63. Row: prefix 64, prepared query 8192, selection 528, retained result 384.
 * Explicit-mode rows have a zero retained result. A canonical tail follows all rows.
 * Refusal: no rows, no tail, and a 64-byte header. */
#endif
