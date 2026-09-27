/* Purpose: Give every figure that sizes a device table for a card of 24 GB.
 * Owns: Nothing; constants only.
 * Launch shape: Not applicable; no CUDA symbol.
 * Lifetime: The build.
 *
 * These figures are estimated. They come from the 12g figures by proportion. No card of
 * this class has run them. A run on such a card must measure them again. */
#ifndef AOTX_PROFILE_24G_CUH
#define AOTX_PROFILE_24G_CUH

#define AOTX_PROFILE_NAME          "24g"
#define AOTX_PROFILE_LANGUAGE      "language"
/* The model role of that file, from the role numbers of cuda/model/model.cuh:
 * 2 for the eight bit file and 3 for the four bit file. */
#define AOTX_PROFILE_LANGUAGE_ROLE 2u

/* Agents, sequences, key value cache slots, request slots and bus writers. One name. */
#define AOTX_SLOTS                 128u

/* Prompt and reply together, for one sequence. */
#define AOTX_SEQ_MAX_TOKENS        16384u

/* The physical pool bounds the shared cache. Each slot has a separate page limit.
 * Tokens per page depend on the model shape. */
#define AOTX_KV_RANGE_BYTES        (8192ull * 1024ull * 1024ull)
#define AOTX_KV_PAGES_EACH         320u

/* The rings. The record rate follows the slot count, so the ring figures follow it too. */
#define AOTX_DEVICE_RING_SLOTS     131072ull
#define AOTX_HOST_RING_DATA_BYTES  (128ull * 1024ull * 1024ull)
#define AOTX_BULK_RING_DATA_BYTES  (512ull * 1024ull * 1024ull)
#define AOTX_INBOUND_SLOTS         8192ull

/* Bytes of one wrapped prompt, of one skill text and of one tool result. */
#define AOTX_SAY_BYTES             12288u
#define AOTX_SKILL_BYTES           16384u
#define AOTX_TOOL_RESULT_BYTES     16384u

/* Device tool modules the tick graph holds at one time, and the scratch bytes one request
 * row of a module owns for a tick. An import of a device tool past the count is refused
 * with the figure. */
#define AOTX_TOOL_MODULES          16u
#define AOTX_TOOL_SCRATCH_BYTES    (64u * 1024u)

/* The catalog of installed modules. */
#define AOTX_MODULE_SLOTS          128u
#define AOTX_CATALOGUE_BYTES       (8ull * 1024ull * 1024ull)

/* Warm turns kept for each agent. */
#define AOTX_MEMORY_TURNS           512u
#define AOTX_MEMORY_TEXT            2048u

/* Language models placed at one time. */
#define AOTX_MODELS_RESIDENT       2u

/* The weights region. The range is virtual: physical memory goes behind a part of it only
 * when a tensor of that part arrives. */
#define AOTX_MEM_WEIGHTS_BYTES     (16ull * 1024ull * 1024ull * 1024ull)

#endif
