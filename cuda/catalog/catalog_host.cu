/* Purpose: Put the built-in tools in the catalog before the tick capture starts.
 * Owns: Nothing; the catalog lives on the device.
 * Launch shape: Host glue only; one launch of one thread.
 * Lifetime: From the boot of the run to the first tick. */
#include <cuda_runtime.h>
#include <stddef.h>

#include "boot/check.h"
#include "catalog/catalog.cuh"

int aotx_catalog_open(void)
{
    /* The built-in tools stand before the first tick. A role that names one of them
     * finds it at the import, and a replay of the journal finds it as well. */
    aotx_catalog_boot<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int state = AOTX_CATALOG_FREE;
    aotx_check_runtime(cudaMemcpyFromSymbol(&state, aotx_catalog, sizeof state,
                                            offsetof(aotx_catalog_state, entry)
                                            + offsetof(aotx_catalog_entry, state)),
                       "cudaMemcpyFromSymbol");
    return (state == AOTX_CATALOG_INSTALLED) ? 0 : 1;
}
