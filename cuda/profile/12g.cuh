/* Purpose: Give every figure that sizes a device table for a card of 12 GB.
 * Owns: Nothing; constants only.
 * Launch shape: Not applicable; no CUDA symbol.
 * Lifetime: The build.
 *
 * These figures are measured on a card of this class. This profile is the reference. */
#ifndef AOTX_PROFILE_12G_CUH
#define AOTX_PROFILE_12G_CUH

#define AOTX_PROFILE_NAME          "12g"
#define AOTX_PROFILE_LANGUAGE      "language"
/* The model role of that file, from the role numbers of cuda/model/model.cuh:
 * 2 for the eight bit file and 3 for the four bit file. */
#define AOTX_PROFILE_LANGUAGE_ROLE 2u

/* Agents, sequences, key value cache slots, request slots and bus writers. One name. */
#define AOTX_SLOTS                 64u

/* Prompt and reply together, for one sequence. */
#define AOTX_SEQ_MAX_TOKENS        2048u

/* The key value range is virtual and costs no memory. A slot holds up to this many pages
 * of 2 MB, which is a context of AOTX_SEQ_MAX_TOKENS tokens of the 36 layer model. */
#define AOTX_KV_RANGE_BYTES        (2048ull * 1024ull * 1024ull)
#define AOTX_KV_PAGES_EACH         160u

/* The rings. The record rate follows the slot count, so the ring figures follow it too. */
#define AOTX_DEVICE_RING_SLOTS     65536ull
#define AOTX_HOST_RING_DATA_BYTES  (64ull * 1024ull * 1024ull)
#define AOTX_BULK_RING_DATA_BYTES  (256ull * 1024ull * 1024ull)
#define AOTX_INBOUND_SLOTS         4096ull

/* Bytes of one wrapped prompt, of one skill text and of one tool result. */
#define AOTX_SAY_BYTES             6144u
#define AOTX_SKILL_BYTES           4096u
#define AOTX_TOOL_RESULT_BYTES     4096u

/* Device tool modules the tick graph holds at one time, and the scratch bytes one request
 * row of a module owns for a tick. An import of a device tool past the count is refused
 * with the figure. */
#define AOTX_TOOL_MODULES          8u
#define AOTX_TOOL_SCRATCH_BYTES    (64u * 1024u)

/* The catalog of installed modules. */
#define AOTX_MODULE_SLOTS          64u
#define AOTX_CATALOGUE_BYTES       (4ull * 1024ull * 1024ull)

/* Language models placed at one time. */
#define AOTX_MODELS_RESIDENT       1u

/* The weights region. The range is virtual: physical memory goes behind a part of it only
 * when a tensor of that part arrives. */
#define AOTX_MEM_WEIGHTS_BYTES     (8ull * 1024ull * 1024ull * 1024ull)

#endif
