/* Purpose: Define live checkpoint images and their bounded disk transport.
 * Owns: Byte offsets and mapped ring fields; no process address is stored in a file.
 * Launch shape: Batched bindings and a configured ring of complete images.
 * Lifetime: Checkpoint schema 1 and transport layout 2. */
#ifndef AOTX_COGNITIVE_CHECKPOINT_H
#define AOTX_COGNITIVE_CHECKPOINT_H
#include "cognitive/recall.h"
#include "profile/profile.cuh"
#ifndef AOTX_MEMORY_SNAPSHOTS
#define AOTX_MEMORY_SNAPSHOTS 2u
#endif
#define AOTX_CP_MAGIC 0x50435841u
#define AOTX_CP_LAYOUT 2u
#define AOTX_CP_HEADER 128u
#define AOTX_CP_RESULT (184u + AOTX_RECALL_SELECTION + AOTX_RECALL_CONTEXT)
#define AOTX_CP_ROW (128u + AOTX_RECALL_QUERY + AOTX_CP_RESULT + AOTX_RECALL_PINS * 24u)
#define AOTX_CP_BINDINGS (AOTX_CP_HEADER + AOTX_SLOTS * AOTX_CP_ROW)
#define AOTX_CP_BYTES (AOTX_CP_BINDINGS + AOTX_COG_IMAGE)
#define AOTX_CP_COPY 262144u
#define AOTX_CP_RESUME 11u
#define AOTX_CP_SLOT_HEADER 64u
#define AOTX_CP_SLOT_BYTES (AOTX_CP_SLOT_HEADER + (uint64_t)AOTX_CP_BYTES)
#define AOTX_CP_RING_BYTES (sizeof(aotx_checkpoint_ring) + AOTX_MEMORY_SNAPSHOTS * AOTX_CP_SLOT_BYTES)
#if AOTX_MEMORY_SNAPSHOTS < 1 || AOTX_MEMORY_SNAPSHOTS > 65536
#error "checkpoint ring count is outside the transport range"
#endif
/* Image: AOTXLCP1 at 0; schema/row/count at 8/12/16, zero at 20.
 * Object image bytes at 24; lineage at 32; object sequence/tick at 48/56.
 * Accepted operations at 64; source runtime tick at 72; zero at 80..127.
 * Rows start at 128. The complete object image follows the rows.
 *
 * Row: slot/pages/scope/context bytes at 0/4/8/12; ordinal at 16.
 * Principal/room/conversation at 24/40/56; focus count/auto retain at 72/76.
 * Completed turn/open count at 80/84; zero at 88..127.
 * Query, encoded result and focus follow the 128-byte row header.
 *
 * Result: status/count/context bytes/searches at 0/4/8/12; cut at 16;
 * request/selection IDs at 24/40; 16 indices at 56; 16 reasons at 120;
 * selection then context at 184. All integers are little endian.
 * Slot: boot/serial/image bytes at 0/8/16; runtime source sequence at 24;
 * zero at 32..63; image at 64. The runtime source sequence is zero in memory mode.
 * Ring reserved[0] selects complete runtime state; reserved[1] is its durable source sequence.
 * In runtime mode, the first 32 pad_head bytes hold SHA-256 of the initial prologue and commit digests.
 *
 * An odd ack_serial means the disk writer is changing the acknowledgment.
 * A complete acknowledgment has ack_serial equal to twice consumed.
 * Incarnation and commit_digest contain the selected file identity in byte order. */
typedef struct aotx_checkpoint_ring {
    uint32_t magic, layout;
    uint64_t boot, slots, slot_bytes;
    uint64_t reserved[4];
    uint64_t head;
    uint64_t pad_head[7];
    uint64_t consumed;
    uint64_t durable_sequence, durable_revision, generation, error, ack_boot;
    uint64_t ack_serial, pad_ack;
    uint64_t incarnation[2], commit_digest[4];
    uint64_t pad_identity[2];
} aotx_checkpoint_ring;
typedef char aotx_checkpoint_ring_size[(sizeof(aotx_checkpoint_ring) == 256) ? 1 : -1];
#endif
