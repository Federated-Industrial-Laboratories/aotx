/* Purpose: Define canonical shared commands, reads and journal parts.
 * Owns: Portable little-endian fields and bounded record kinds.
 * Threading: Complete mailbox commands enter one ordered device batch.
 * Lifetime: Persistent IDs belong to the memory lineage. */
#ifndef AOTX_SHARED_WIRE_H
#define AOTX_SHARED_WIRE_H
#define AOTX_SHARED_MAGIC "AOTXSHR1"
#define AOTX_SHARED_COMMAND_HEAD 192u
#define AOTX_SHARED_READ_HEAD 96u
#define AOTX_SHARED_REPLY_HEAD 320u
#define AOTX_SHARED_RECORD 37u
#define AOTX_SHARED_RECORD_DATA 160u
#define AOTX_SHARED_MEDIA_ROW 40u
#define AOTX_SHARED_MEDIA_REFS 8u
#define AOTX_SHARED_READ_ACTION 16u
#define AOTX_SHARED_WRITE_ACTION 32u
#define AOTX_SHARED_MANAGE_ACTION 64u
enum aotx_shared_operation {
    AOTX_SHARED_REGISTER = 1, AOTX_SHARED_SPACE, AOTX_SHARED_MEMBER,
    AOTX_SHARED_CONVERSATION, AOTX_SHARED_INPUT, AOTX_SHARED_CANCEL,
    AOTX_SHARED_RETIRE, AOTX_SHARED_PUBLISH, AOTX_SHARED_SAVE
};
enum aotx_shared_read_kind {
    AOTX_SHARED_CAPABILITIES = 1, AOTX_SHARED_PARTICIPANT,
    AOTX_SHARED_SPACES_READ, AOTX_SHARED_SPACE_READ, AOTX_SHARED_MEMBERS_READ,
    AOTX_SHARED_CONVERSATIONS_READ, AOTX_SHARED_CONVERSATION_READ,
    AOTX_SHARED_OPERATION_READ, AOTX_SHARED_EVENTS_READ,
    AOTX_SHARED_MEMORY_READ, AOTX_SHARED_SAVE_READ, AOTX_SHARED_AFFECT_READ
};
enum aotx_shared_phase {
    AOTX_SHARED_FREE, AOTX_SHARED_ACCEPTED, AOTX_SHARED_QUEUED,
    AOTX_SHARED_RUNNING, AOTX_SHARED_DONE, AOTX_SHARED_FAILED,
    AOTX_SHARED_CANCELLED, AOTX_SHARED_INTERRUPTED
};
enum aotx_shared_record_kind {
    AOTX_SHARED_ADMIT_RECORD = 1, AOTX_SHARED_LEASE_RECORD,
    AOTX_SHARED_OUTPUT_RECORD, AOTX_SHARED_COMPLETE_RECORD
};
/* Command offsets: magic 0, operation 8, scope 12, sequence 16, key 24, lineage 40.
 * Target, space and member IDs are at 56, 72 and 88.
 * Model, token limit, pages and member rights are at 104, 108, 112 and 116. */

/* Temperature and top_p are at 120 and 124. Target sequence is at 128.
 * Text length and media count are at 136 and 140. Bytes 144..191 are zero.
 * Text follows the header. Media rows contain kind at 0, zero at 4 and digest at 8. */

/* Read offsets: magic 0, kind 8, zero 12, lineage 16, target 32, parent 48.
 * Order cursor is at 64. Byte cursor is at 72. Limit is at 80.
 * Bytes 84..95 are zero. */

/* Record offsets: schema 0, kind 4, serial 8, zero 16, total 24, offset 28, data 32. */

/* Reply offsets: magic 0, kind 8, state 12, lineage 16, target 32, space 48, actor 64.
 * Sequence, next sequence and retry floor are at 80, 88 and 96.
 * Input order and event floor are at 104 and 112. */

/* Admission source, terminal source and saved source are at 120, 128 and 136.
 * Saved generation and incarnation are at 144 and 152. Status and flags are at 168 and 172.
 * Output bytes and input tokens are at 176 and 180. Output tokens and finish reason are at 184 and 188. */

/* Row count, row bytes and next cursor are at 192, 196 and 200.
 * Byte cursor, pending bytes and disk error are at 208, 216 and 224.
 * Scope, rights, command operation and key are at 228, 232, 236 and 240. */

/* Saved boot is at 256. Saved commit digest is at 264. Receipt ID is at 296.
 * Bytes 312..319 are zero. Rows or exact output follow at 320.
 * Flags: device commit 1, saved admission 2, saved terminal 4, output gap 8. */
#endif
