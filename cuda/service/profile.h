/* Purpose: Bound service transport and resident request storage.
 * Owns: Build-time service capacities independent of execution slots.
 * Threading: Shared constants for the broker and device.
 * Lifetime: One build. */
#ifndef AOTX_SERVICE_PROFILE_H
#define AOTX_SERVICE_PROFILE_H
#ifndef AOTX_SERVICE_CHANNELS
#define AOTX_SERVICE_CHANNELS 128u
#endif
#ifndef AOTX_SERVICE_PRINCIPALS
#define AOTX_SERVICE_PRINCIPALS 256u
#endif
#ifndef AOTX_SERVICE_REQUESTS
#define AOTX_SERVICE_REQUESTS 128u
#endif
#ifndef AOTX_SERVICE_OUTPUT_BYTES
#define AOTX_SERVICE_OUTPUT_BYTES 65536u
#endif
#ifndef AOTX_SERVICE_REQUEST_SECONDS
#define AOTX_SERVICE_REQUEST_SECONDS 300u
#endif
#ifndef AOTX_SERVICE_UPLOAD_SECONDS
#define AOTX_SERVICE_UPLOAD_SECONDS 120u
#endif
#endif
