/* Purpose: Bind the local service transport to runtime startup and shutdown.
 * Owns: Mapping and child-process entry points only.
 * Threading: One runtime owner.
 * Lifetime: Runtime startup through shutdown. */
#ifndef AOTX_SERVICE_HOST_H
#define AOTX_SERVICE_HOST_H
#include "seam/seam.cuh"
int aotx_service_open(aotx_seam_rings *rings, unsigned long long epoch);
void aotx_service_finish(const aotx_seam_rings *rings);
void aotx_service_close(aotx_seam_rings *rings);
void aotx_service_capture(void *stream);
void aotx_service_output_capture(void *stream);
#endif
