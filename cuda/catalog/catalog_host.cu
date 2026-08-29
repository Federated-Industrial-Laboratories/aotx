/* Purpose: Put the built-in tools in the catalog before the tick capture starts.
 * Owns: Nothing; the catalog lives on the device.
 * Launch shape: Host glue only; one launch of one thread.
 * Lifetime: From the boot of the run to the first tick. */
#include <cuda_runtime.h>
#include <stddef.h>

#include "boot/check.h"
#include "catalog/catalog.cuh"
#include "tool/module_host.h"

int aotx_catalog_open(void)
{
    /* The built-in tools stand before the first tick. A role that names one of them
     * finds it at the import, and a replay of the journal finds it as well. */
    /* The call comes again at every capture of the tick graph, and the pump thread makes
     * that capture. The launch and the read therefore take the stream of the module path
     * and one wait on it. No wait of this call stands over the whole device. */
    cudaStream_t on = aotx_tool_module_line_of();
    aotx_catalog_boot<<<1, 1, 0, on>>>();
    unsigned int state = AOTX_CATALOG_FREE;
    aotx_check_runtime(cudaMemcpyFromSymbolAsync(&state, aotx_catalog, sizeof state,
                                                 offsetof(aotx_catalog_state, entry)
                                                 + offsetof(aotx_catalog_entry, state),
                                                 cudaMemcpyDeviceToHost, on),
                       "cudaMemcpyFromSymbolAsync");
    aotx_tool_module_wait();
    return (state == AOTX_CATALOG_INSTALLED) ? 0 : 1;
}
