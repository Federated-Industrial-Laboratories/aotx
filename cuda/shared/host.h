/* Purpose: Allocate persistent shared tables and capture their finite device nodes.
 * Owns: Host handles for the device table batch; no transport grants.
 * Launch shape: Host glue allocates before replay and captures on the pump stream.
 * Lifetime: One compatible complete runtime. */
#ifndef AOTX_SHARED_HOST_H
#define AOTX_SHARED_HOST_H
struct aotx_runtime_shared_profile;
int aotx_shared_open(const struct aotx_runtime_shared_profile *profile);
void aotx_shared_close(void);
void aotx_shared_capture(void *stream);
void aotx_shared_output_capture(void *stream);
#endif
