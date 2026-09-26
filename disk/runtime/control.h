/* Purpose: Bind fitted control assets to exact model and turn-format bytes.
 * Owns: The versioned asset binding and its disk checks.
 * Threading: One reader or writer for each asset in a bounded batch.
 * Lifetime: The binding stays beside its asset through packing and recovery. */
#ifndef AOTX_RUNTIME_CONTROL_H
#define AOTX_RUNTIME_CONTROL_H
#include "disk/modelfile/wrap.h"
#include <stdio.h>
#define AOTX_CONTROL_BYTES 608u
#define AOTX_CONTROL_VECTOR 1u
#define AOTX_CONTROL_PROBE 2u
#define AOTX_CONTROL_CALIBRATION 3u
/* Hook 1 reads or adds after the complete layer residual update. */
#define AOTX_CONTROL_HOOK 1u
#define AOTX_CONTROL_ALL 0u
#define AOTX_CONTROL_RESPONSE 1u
typedef struct aotx_control_identity {
    unsigned char model[32];
    aotx_wrap wrap;
} aotx_control_identity;
#ifdef __cplusplus
extern "C" {
#endif
int aotx_control_read(const char *store, const char *name, unsigned kind,
    const aotx_control_identity *expected, FILE *asset, unsigned *positions);
int aotx_control_digest(const char *store, const char *name, char text[65]);
int aotx_control_pair(const char *line, unsigned char digest[2][32]);
int aotx_control_write(const char *path, unsigned kind, const aotx_control_identity *identity, unsigned positions);
int aotx_control_decode(const unsigned char bytes[AOTX_CONTROL_BYTES], unsigned kind,
    aotx_control_identity *identity, unsigned char digest[32], unsigned *positions);
struct aotx_ccir_view;
struct aotx_runtime_index;
int aotx_control_reference(const struct aotx_ccir_view *view,
    const struct aotx_runtime_index *index, const char *name, unsigned kind, unsigned response);
#ifdef __cplusplus
}
#endif
#endif
