/* Purpose: Inspect model headers without a model store or device.
 * Owns: Nothing; each call closes its source and model metadata.
 * Threading: One command, one source at a time.
 * Lifetime: One command. */
#ifndef AOTX_MODELS_INSPECT_H
#define AOTX_MODELS_INSPECT_H

#include "disk/modelfile/modelfile.h"

/* Return 0 for a complete report, 1 for a read error, or 2 for a bad header. */
int aotx_model_inspect(const char *source);

/* Read HTTP ranges into the shared header reader. No tensor data handle is kept.
 * received gives the body bytes read, including bounded read-ahead. */
int aotx_model_inspect_remote(const char *source, aotx_modelfile **file,
                              uint64_t *received);

#endif
