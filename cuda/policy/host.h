/* Purpose: Load admitted policy assets and capture finite policy graph nodes.
 * Owns: Native module handles and fixed device argument addresses.
 * Launch shape: Host load and graph glue; policy evaluation stays on the GPU.
 * Lifetime: One boot until every captured graph is destroyed. */
#ifndef AOTX_POLICY_HOST_H
#define AOTX_POLICY_HOST_H
#include "policy/abi.h"
int aotx_policy_open(const char *path, const char *trust);
void aotx_policy_close(void);
unsigned int aotx_policy_capture(void *stream);
unsigned int aotx_policy_rows_capture(void *stream, const aotx_policy_input *input,
    const unsigned char *prior, aotx_policy_output *output, unsigned char *next,
    unsigned int count, unsigned int stride);
#endif
