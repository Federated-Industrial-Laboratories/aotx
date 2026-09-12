/* Purpose: Declare ownership of the decoder matrix module.
 * Owns: No state in this interface.
 * Launch shape: Host graph construction and module release only.
 * Lifetime: One runtime. */
#ifndef AOTX_DECODE_MODULE_HOST_H
#define AOTX_DECODE_MODULE_HOST_H
void aotx_decode_module_open(void);
void aotx_decode_module_close(void);
#endif
