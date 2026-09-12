/* Purpose: Select the profile of the build and prove every figure of it is given.
 * Owns: Nothing; the selected header holds the figures.
 * Launch shape: Not applicable; no CUDA symbol.
 * Lifetime: The build.
 *
 * A profile fixes every figure that sizes a device table. A figure that is a compile time
 * constant gives a kernel a known shape and a static allocation. It also gives a check at
 * N=1 and at N=AOTX_SLOTS, with the maximum known to the compiler. A figure that sizes a
 * device table comes from a profile or an explicit build capacity setting.
 *
 * The build sets AOTX_PROFILE_HEADER. A build that sets nothing takes the 12g profile,
 * which is the reference. */
#ifndef AOTX_PROFILE_CUH
#define AOTX_PROFILE_CUH

#ifndef AOTX_PROFILE_HEADER
#define AOTX_PROFILE_HEADER "profile/12g.cuh"
#endif

#include AOTX_PROFILE_HEADER
#if defined(AOTX_KV_SLOT_PAGES) && AOTX_KV_SLOT_PAGES > 0
#undef AOTX_KV_PAGES_EACH
#define AOTX_KV_PAGES_EACH AOTX_KV_SLOT_PAGES
#endif
#if defined(AOTX_KV_POOL_PAGES) && AOTX_KV_POOL_PAGES > 0
#undef AOTX_KV_RANGE_BYTES
#define AOTX_KV_RANGE_BYTES (AOTX_KV_POOL_PAGES * 2097152ull)
#endif

/* Every figure the tree reads. A profile header that gives fewer stops the build here. */
#ifndef AOTX_PROFILE_NAME
#error "the profile header gives no AOTX_PROFILE_NAME"
#endif
#ifndef AOTX_PROFILE_LANGUAGE
#error "the profile header gives no AOTX_PROFILE_LANGUAGE"
#endif
#ifndef AOTX_PROFILE_LANGUAGE_ROLE
#error "the profile header gives no AOTX_PROFILE_LANGUAGE_ROLE"
#endif
#ifndef AOTX_SLOTS
#error "the profile header gives no AOTX_SLOTS"
#endif
#ifndef AOTX_SEQ_MAX_TOKENS
#error "the profile header gives no AOTX_SEQ_MAX_TOKENS"
#endif
#ifndef AOTX_KV_RANGE_BYTES
#error "the profile header gives no AOTX_KV_RANGE_BYTES"
#endif
#ifndef AOTX_KV_PAGES_EACH
#error "the profile header gives no AOTX_KV_PAGES_EACH"
#endif
#ifndef AOTX_DEVICE_RING_SLOTS
#error "the profile header gives no AOTX_DEVICE_RING_SLOTS"
#endif
#ifndef AOTX_HOST_RING_DATA_BYTES
#error "the profile header gives no AOTX_HOST_RING_DATA_BYTES"
#endif
#ifndef AOTX_BULK_RING_DATA_BYTES
#error "the profile header gives no AOTX_BULK_RING_DATA_BYTES"
#endif
#ifndef AOTX_INBOUND_SLOTS
#error "the profile header gives no AOTX_INBOUND_SLOTS"
#endif
#ifndef AOTX_SAY_BYTES
#error "the profile header gives no AOTX_SAY_BYTES"
#endif
#ifndef AOTX_SKILL_BYTES
#error "the profile header gives no AOTX_SKILL_BYTES"
#endif
#ifndef AOTX_TOOL_RESULT_BYTES
#error "the profile header gives no AOTX_TOOL_RESULT_BYTES"
#endif
#ifndef AOTX_TOOL_MODULES
#error "the profile header gives no AOTX_TOOL_MODULES"
#endif
#ifndef AOTX_TOOL_SCRATCH_BYTES
#error "the profile header gives no AOTX_TOOL_SCRATCH_BYTES"
#endif
#ifndef AOTX_MODULE_SLOTS
#error "the profile header gives no AOTX_MODULE_SLOTS"
#endif
#ifndef AOTX_CATALOGUE_BYTES
#error "the profile header gives no AOTX_CATALOGUE_BYTES"
#endif
#ifndef AOTX_MEMORY_TURNS
#error "the profile header gives no AOTX_MEMORY_TURNS"
#endif
#ifndef AOTX_MEMORY_TEXT
#error "the profile header gives no AOTX_MEMORY_TEXT"
#endif
#ifndef AOTX_MODELS_RESIDENT
#error "the profile header gives no AOTX_MODELS_RESIDENT"
#endif
#ifndef AOTX_MEM_WEIGHTS_BYTES
#error "the profile header gives no AOTX_MEM_WEIGHTS_BYTES"
#endif

/* The architecture of the build. The build sets it; a build that sets nothing takes the
 * architecture of the reference card. */
#ifndef AOTX_ARCH
#define AOTX_ARCH 86
#endif

/* The slot count is a power of two from 32 to 256. A scan over the slots of one block
 * takes one block of AOTX_SLOTS threads, and a block holds 1,024 threads at the most. */
typedef char aotx_profile_check_slots[(AOTX_SLOTS >= 32u && AOTX_SLOTS <= 256u
                                       && (AOTX_SLOTS & (AOTX_SLOTS - 1u)) == 0u) ? 1 : -1];

/* The two ring sizes that a mask reads are powers of two. */
typedef char aotx_profile_check_ring[((AOTX_DEVICE_RING_SLOTS
                                       & (AOTX_DEVICE_RING_SLOTS - 1ull)) == 0ull
                                      && (AOTX_INBOUND_SLOTS
                                          & (AOTX_INBOUND_SLOTS - 1ull)) == 0ull) ? 1 : -1];
typedef char aotx_profile_check_bytes[((AOTX_HOST_RING_DATA_BYTES
                                        & (AOTX_HOST_RING_DATA_BYTES - 1ull)) == 0ull
                                       && (AOTX_BULK_RING_DATA_BYTES
                                           & (AOTX_BULK_RING_DATA_BYTES - 1ull)) == 0ull)
                                      ? 1 : -1];

#endif
