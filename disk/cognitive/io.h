/* Purpose: Read and write bounded typed state sections for the device loader.
 * Owns: File leases, host buffers and command-line output.
 * Threading: One caller; all section reads and writes use the CCIR batch API.
 * Lifetime: Open through write or explicit close. */
#ifndef AOTX_COGNITIVE_IO_H
#define AOTX_COGNITIVE_IO_H
#include "ccir/ccir.h"
#include "cognitive/format.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct aotx_cognitive_file {
    aotx_ccir_view view;
    unsigned char *checkpoint, *tail;
    uint64_t checkpoint_bytes, tail_bytes;
    uint32_t checkpoint_index, manifest_index, tail_index;
    unsigned char manifest[AOTX_CCIR_MANIFEST_BYTES];
} aotx_cognitive_file;

int aotx_cognitive_file_open(const char *path, aotx_cognitive_file *file);
int aotx_cognitive_file_write(aotx_cognitive_file *file, const char *path,
                              const unsigned char *checkpoint, uint64_t bytes);
void aotx_cognitive_file_close(aotx_cognitive_file *file);
int aotx_cognitive_file_options(int argc, char **argv);
void aotx_cognitive_file_report(int status, uint64_t sequence, uint64_t bytes, uint32_t fallback);
#ifdef __cplusplus
}
#endif
#endif
