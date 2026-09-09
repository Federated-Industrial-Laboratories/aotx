/* Purpose: Share bounded CCIR file operations.
 * Owns: Framing constants and internal function declarations.
 * Threading: The caller holds the required file lease.
 * Lifetime: One read or write transaction. */
#ifndef AOTX_CCIR_INTERNAL_H
#define AOTX_CCIR_INTERNAL_H
#include "disk/ccir/ccir.h"
#include "disk/wire/diskwire.h"

#define AOTX_CCIR_HASH_OFFSET 4064u
#define AOTX_CCIR_CHUNK 65536u

uint16_t aotx_ccir_u16(const unsigned char *p);
uint32_t aotx_ccir_u32(const unsigned char *p);
uint64_t aotx_ccir_u64(const unsigned char *p);
void aotx_ccir_put(unsigned char *p, uint64_t n, unsigned int bytes);
int aotx_ccir_zero(const unsigned char *p, size_t bytes);
void aotx_ccir_hash(const void *data, size_t bytes, unsigned char out[32]);
int aotx_ccir_pread(int fd, void *data, size_t bytes, uint64_t offset);
int aotx_ccir_pwrite(int fd, const void *data, size_t bytes, uint64_t offset);
int aotx_ccir_hash_fd(int fd, uint64_t offset, uint64_t bytes, unsigned char out[32]);
int aotx_ccir_lock(const char *path, int write, int create, int *fd);
int aotx_ccir_parent_sync(const char *path);
int aotx_ccir_limits_get(const aotx_ccir_limits *in, aotx_ccir_limits *out);
int aotx_ccir_load(int fd, const aotx_ccir_limits *limits, aotx_ccir_view *view);
int aotx_ccir_profile(int fd, const aotx_ccir_view *view,
                      const unsigned char commit[AOTX_CCIR_COMMIT]);
void aotx_ccir_encode_row(const aotx_ccir_section *section, unsigned char row[128]);
int aotx_ccir_decode_row(const unsigned char row[128], aotx_ccir_section *section,
                         const aotx_ccir_limits *limits, uint64_t end);
int aotx_ccir_write_generation(int fd, const aotx_ccir_view *old,
                               const aotx_ccir_input *inputs, uint32_t count,
                               const aotx_ccir_meta *meta,
                               const aotx_ccir_limits *limits);
#endif
