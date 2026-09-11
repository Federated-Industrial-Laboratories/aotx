/* Purpose: Define the required text runtime index and its asset references.
 * Owns: Portable byte fields and disk metadata declarations.
 * Threading: One leased file reader or writer processes the complete asset batch.
 * Lifetime: Runtime index schema 1. */
#ifndef AOTX_RUNTIME_H
#define AOTX_RUNTIME_H
#include "disk/ccir/ccir.h"
#define AOTX_RUNTIME_HEADER 256u
#define AOTX_RUNTIME_ROW 384u
#define AOTX_RUNTIME_NAME 256u
#define AOTX_RUNTIME_AFFECT 1u
#define AOTX_RUNTIME_ABI 1u
/* Header: magic AOTXRT01, schema/row/count/features at 8/12/16/20.
 * Wire layout/slots/object capacity/architecture at 24/28/32/36.
 * Payload capacity at 40, runtime ABI at 48, zero at 52..63.
 * Initial model roles at 64..127; replay ID at 128..143; zero at 144..255.
 *
 * Row: section ID at 0, kind/flags at 16/20, bytes at 24, digest at 32.
 * Name at 64..319; zero at 320..383. Text has a zero terminator and zero padding.
 * Kind 1 is a model asset. Kind 2 is a data module asset. All rows are required. */
typedef struct aotx_runtime_index {
    unsigned char header[AOTX_RUNTIME_HEADER];
    unsigned char rows[AOTX_CCIR_SECTIONS][AOTX_RUNTIME_ROW];
    uint32_t count;
} aotx_runtime_index;
#ifdef __cplusplus
extern "C" {
#endif
int aotx_runtime_name(const char *name);
int aotx_runtime_index_read(int fd, const aotx_ccir_view *view, aotx_runtime_index *index);
int aotx_runtime_profile(int fd, const aotx_ccir_view *view, const unsigned char id[16]);
int aotx_runtime_section(const aotx_ccir_view *view, const unsigned char id[16]);
int aotx_runtime_dependencies(const aotx_ccir_view *view);
void aotx_runtime_revision(const aotx_ccir_view *view, unsigned char digest[32]);
#ifdef __cplusplus
}
#endif
#endif
