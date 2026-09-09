/* Purpose: Read recall requests and write recorded context results.
 * Owns: Bounded disk buffers, file leases and JSON output.
 * Threading: One caller; each operation takes a complete request batch.
 * Lifetime: Open through output creation and result output. */
#ifndef AOTX_COGNITIVE_RECALL_IO_H
#define AOTX_COGNITIVE_RECALL_IO_H
#include "cognitive/io.h"
#include "cognitive/recall.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct aotx_recall_file {
    aotx_cognitive_file source;
    unsigned char *requests, *checkpoint;
    aotx_recall_result *rows;
    uint64_t request_bytes;
    uint32_t count;
} aotx_recall_file;

/* Options return 0 for select, 1 for replay, 2 for help, or -1 for an error. */
int aotx_recall_file_options(int argc, char **argv);
int aotx_recall_file_open(const char *path, const char *requests, aotx_recall_file *file);
int aotx_recall_file_write(aotx_recall_file *file, const char *path, uint64_t bytes);
int aotx_recall_file_rows(const aotx_recall_file *file);
void aotx_recall_file_report(int status);
void aotx_recall_file_close(aotx_recall_file *file);
#ifdef __cplusplus
}
#endif
#endif
