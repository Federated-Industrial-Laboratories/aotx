/* Purpose: Rebuild the resident tensor layout for a one-resident language replacement. */
#ifndef AOTX_MODEL_LAYOUT_HOST_H
#define AOTX_MODEL_LAYOUT_HOST_H

#include "disk/modelfile/modelfile.h"
#include "model/load.cuh"

int aotx_model_layout_replace(const char *dir, const aotx_manifest_entry *entry,
                              unsigned int entries, const aotx_model_load_state *state,
                              unsigned int source, unsigned long long *cursor,
                              unsigned int *placed, unsigned int *left,
                              unsigned long long *bytes);

#endif
