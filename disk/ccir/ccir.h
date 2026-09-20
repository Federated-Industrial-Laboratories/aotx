/* Purpose: Store and read bounded CCIR section batches.
 * Owns: File leases and validated section directories.
 * Threading: One caller for each open view; writers take an exclusive lease.
 * Lifetime: A view retains its lease until close. */
#ifndef AOTX_CCIR_H
#define AOTX_CCIR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AOTX_CCIR_SECTIONS 256u
#define AOTX_CCIR_PAGE 4096u
#define AOTX_CCIR_ROOT_A 4096u
#define AOTX_CCIR_ROOT_B 8192u
#define AOTX_CCIR_DATA 12288u
#define AOTX_CCIR_ROW 128u
#define AOTX_CCIR_COMMIT 256u
#define AOTX_CCIR_MANIFEST_BYTES 96u
#define AOTX_CCIR_REQUIRED 1u
#define AOTX_CCIR_MANIFEST 1u
#define AOTX_CCIR_CHECKPOINT 2u
#define AOTX_CCIR_TAIL 3u
#define AOTX_CCIR_LIVE 4u
#define AOTX_CCIR_RUNTIME 5u
#define AOTX_CCIR_ASSET 6u
#define AOTX_CCIR_REPLAY 7u
#define AOTX_CCIR_COLD 8u
#define AOTX_CCIR_MEMORY 0u
#define AOTX_CCIR_FILE 1u
#define AOTX_CCIR_REUSE 2u

/* Prologue offsets: version 8, geometry 12, features 16, lineage 24,
 * incarnation 40, root geometry 56 and digest 4064.
 * Root offsets: generation 16, commit offset 24, commit bytes 32, end 40,
 * commit digest 48, prologue digest 80 and root digest 4064.
 * Commit offsets: generation 8, previous digest 16, checkpoint sequence 48,
 * durable sequence 56 and tick 64. Directory offset, count and row bytes start
 * at 72, 80 and 84. Directory bytes, digest and end start at 88, 96 and 128.
 * Section IDs start at 136, 152 and 168. */

enum aotx_ccir_status {
    AOTX_CCIR_OK = 0, AOTX_CCIR_IO, AOTX_CCIR_INVALID,
    AOTX_CCIR_UNSUPPORTED, AOTX_CCIR_LIMIT, AOTX_CCIR_BUSY,
    AOTX_CCIR_EXISTS, AOTX_CCIR_CHANGED
};

typedef struct aotx_ccir_limits {
    uint64_t file_bytes;
    uint64_t section_bytes;
    uint32_t sections;
} aotx_ccir_limits;

typedef struct aotx_ccir_meta {
    uint64_t checkpoint_sequence;
    uint64_t durable_sequence;
    uint64_t source_tick;
} aotx_ccir_meta;

typedef struct aotx_ccir_section {
    uint32_t type;
    uint16_t schema;
    uint16_t flags;
    unsigned char id[16];
    uint64_t offset;
    uint64_t bytes;
    uint32_t alignment;
    unsigned char digest[32];
} aotx_ccir_section;

typedef struct aotx_ccir_input {
    aotx_ccir_section section;
    uint32_t source;
    const void *data;
    int fd;
    uint64_t source_offset;
} aotx_ccir_input;

typedef struct aotx_ccir_view {
    int fd;
    uint32_t count;
    uint32_t root_slot;
    uint32_t fallback;
    uint64_t generation;
    uint64_t end;
    uint64_t trailing_bytes;
    uint64_t directory_offset;
    uint64_t commit_offset;
    unsigned char lineage[16];
    unsigned char incarnation[16];
    unsigned char prologue_digest[32];
    unsigned char commit_digest[32];
    aotx_ccir_meta meta;
    aotx_ccir_section sections[AOTX_CCIR_SECTIONS];
} aotx_ccir_view;

typedef struct aotx_ccir_read {
    uint32_t section;
    uint64_t offset;
    size_t bytes;
    void *data;
} aotx_ccir_read;

typedef struct aotx_ccir_revision {
    unsigned char prologue_digest[32];
    unsigned char commit_digest[32];
} aotx_ccir_revision;

/* Null limits use the configured file cap for both file and section bytes, with 256 sections. */
void aotx_ccir_default_limits(aotx_ccir_limits *limits);
const char *aotx_ccir_status_text(int status);

/* Manifest magic starts at 0. Schema, object ABI, record bytes and representation
 * start at 8, 12, 16 and 20. Checkpoint ID starts at 24; tail ID starts at 40.
 * Bytes 56 through 95 are zero.
 * All integers are little endian. A null tail ID means no tail. */
void aotx_ccir_manifest(unsigned char out[AOTX_CCIR_MANIFEST_BYTES],
                        const unsigned char checkpoint[16],
                        const unsigned char tail[16]);
void aotx_ccir_live_manifest(unsigned char out[AOTX_CCIR_MANIFEST_BYTES],
    const unsigned char checkpoint[16], const unsigned char live[16]);

/* Open verifies every selected extent and retains a shared, nonblocking lease.
 * Unsupported required sections refuse the newest intact generation.
 * A failed open leaves fd equal to -1. Close only a successfully opened view. */
int aotx_ccir_open(const char *path, const aotx_ccir_limits *limits,
                   aotx_ccir_view *view);
void aotx_ccir_close(aotx_ccir_view *view);
/* The writer retains an exclusive lease through all append operations. */
int aotx_ccir_writer_open(const char *path, const aotx_ccir_limits *limits, aotx_ccir_view *view);
int aotx_ccir_writer_append(aotx_ccir_view *view, const aotx_ccir_input *inputs,
    uint32_t count, const aotx_ccir_meta *meta, const aotx_ccir_limits *limits);
int aotx_ccir_writer_replace(aotx_ccir_view *view, const char *path,
    const aotx_ccir_input *inputs, uint32_t count, const aotx_ccir_meta *meta,
    const aotx_ccir_limits *limits);
int aotx_ccir_writer_sync(aotx_ccir_view *view, const char *path);
int aotx_ccir_read_batch(const aotx_ccir_view *view,
                         const aotx_ccir_read *reads, uint32_t count);

/* Create uses exclusive creation. Append takes a complete directory batch.
 * REUSE names an unchanged section ID from the selected generation.
 * An IO error can follow publication; reopen to find the complete generation.
 * Callers keep input buffers and source files stable until the call returns. */
int aotx_ccir_create(const char *path, const unsigned char lineage[16],
                     const aotx_ccir_input *inputs, uint32_t count,
                     const aotx_ccir_meta *meta, const aotx_ccir_limits *limits);
int aotx_ccir_append(const char *path, const aotx_ccir_input *inputs,
                     uint32_t count, const aotx_ccir_meta *meta,
                     const aotx_ccir_limits *limits);
/* Copy both digests from the source view before close. Conditional append checks
 * that identity under the exclusive lease. A mismatch returns CHANGED without writes. */
int aotx_ccir_append_if(const char *path, const aotx_ccir_revision *expected,
                        const aotx_ccir_input *inputs, uint32_t count,
                        const aotx_ccir_meta *meta, const aotx_ccir_limits *limits);
int aotx_ccir_compact(const char *source, const char *destination,
                      const aotx_ccir_limits *limits);

#ifdef __cplusplus
}
#endif
#endif
