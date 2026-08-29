/* Purpose: Give the node glue and the check the module the loader holds for one entry.
 * Owns: Nothing; the loader owns every module handle.
 * Threading: One thread; the pump and the check are the only callers.
 * Lifetime: From the load of a module to the close. */
#ifndef AOTX_TOOL_MODULE_HOST_H
#define AOTX_TOOL_MODULE_HOST_H

#include <cuda.h>

#include "tool/module.cuh"

/* The kernel of the module the loader holds for one entry, or zero. */
CUfunction aotx_tool_module_function(unsigned int entry);

/* The modules the loader holds, and the entry of one of them in plan order. */
unsigned int aotx_tool_module_held_count(void);
unsigned int aotx_tool_module_entry_of(unsigned int at);

/* The plan row the loader read, and the digest of one file. The check program reads the
 * module file name and the kernel name from the plan, so it parses no manifest. */
int aotx_tool_module_plan_row(unsigned int at, aotx_tool_module_row *out);
int aotx_tool_module_digest(const char *path, unsigned char digest[32]);

/* Launch one module over the batch that stands, outside every graph. The check program
 * takes this path, and the tick graph takes the node instead. */
int aotx_tool_module_launch(unsigned int entry);

#endif
