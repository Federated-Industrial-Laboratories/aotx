/* Purpose: Declare checkpoint file framing and ordered drain operations.
 * Owns: File leases and mapped transport; no cognitive decisions.
 * Threading: One disk consumer takes a batch of complete ring entries.
 * Lifetime: The optional live memory mirror. */
#ifndef AOTX_CHECKPOINT_IO_H
#define AOTX_CHECKPOINT_IO_H
#include "cognitive/checkpoint.h"
#include "disk/ccir/ccir.h"
#include "disk/wire/diskwire.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct aotx_checkpoint_disk {
    aotx_map map;
    aotx_checkpoint_ring *ring;
    aotx_ccir_view view;
    const char *path;
    uint64_t next_retry;
    const char *journal;
    uint64_t runtime_sequence;
    unsigned runtime, runtime_verified;
    unsigned char runtime_revision[32];
    struct aotx_cold_worker *cold_worker;
} aotx_checkpoint_disk;
uint64_t aotx_cp_get(const unsigned char *p, uint32_t bytes);
int aotx_checkpoint_framing(const unsigned char *image, uint64_t bytes, uint32_t *base);
int aotx_checkpoint_file_read(const char *path, unsigned char **image, uint32_t *bytes);
int aotx_checkpoint_file_write(aotx_checkpoint_disk *disk, const unsigned char *image, uint64_t bytes);
int aotx_checkpoint_disk_open(aotx_checkpoint_disk *disk, int fd, const char *path);
int aotx_checkpoint_disk_pass(aotx_checkpoint_disk *disk);
void aotx_checkpoint_disk_close(aotx_checkpoint_disk *disk);
#ifdef __cplusplus
}
#endif
#endif
