/* Purpose: Frame complete journal blocks for portable runtime recovery.
 * Owns: Replay header fields and bounded disk transfer declarations.
 * Threading: One drain or restore process handles a complete block batch.
 * Lifetime: One checkpoint generation or activation. */
#ifndef AOTX_RUNTIME_REPLAY_H
#define AOTX_RUNTIME_REPLAY_H
#include "disk/runtime/runtime.h"
#include "disk/restore/scan.h"
#include <stdio.h>
#define AOTX_RUNTIME_REPLAY_HEADER 128u
/* AOTXRPL1 at 0; schema/mode at 8/12. Mode 1 creates a new runtime; 2 replays.
 * Boot/tick/hash/record count/block count/payload bytes at 16/24/32/40/48/56.
 * Memory operation revision/runtime source sequence at 64/72; zero at 80..127.
 * Each payload frame is a u64 byte count followed by a complete journal block. */
#ifdef __cplusplus
extern "C" {
#endif
int aotx_runtime_replay_header(int fd, const aotx_ccir_section *section, unsigned char out[128]);
int aotx_runtime_replay_collect(const char *journal, uint64_t boot, uint64_t tick,
    uint64_t memory_revision, uint64_t runtime_sequence, uint64_t limit, FILE **file, uint64_t *bytes);
int aotx_runtime_replay_walk(const aotx_ccir_view *view, unsigned char *buffer, uint32_t bytes,
    aotx_block_fn fn, void *context, aotx_journal_scan *scan);
struct aotx_checkpoint_disk;
int aotx_runtime_checkpoint_write(struct aotx_checkpoint_disk *disk,
    const unsigned char *image, uint64_t bytes, uint32_t base, int same);
#ifdef __cplusplus
}
#endif
#endif
