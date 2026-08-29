/* Purpose: Give every figure that sizes a device table for a card of 8 GB.
 * Owns: Nothing; constants only.
 * Launch shape: Not applicable; no CUDA symbol.
 * Lifetime: The build.
 *
 * These figures are measured. A card of 12 GB runs this profile, and the figures come
 * from that run. The default language file is the Q4_0 file. The Q8_0 file of 4.28 GB
 * leaves no key value pages on a card of 8 GB with a desktop. */
#ifndef AOTX_PROFILE_8G_CUH
#define AOTX_PROFILE_8G_CUH

#define AOTX_PROFILE_NAME          "8g"
#define AOTX_PROFILE_LANGUAGE      "language-q4"
/* The model role of that file, from the role numbers of cuda/model/model.cuh:
 * 2 for the eight bit file and 3 for the four bit file. */
#define AOTX_PROFILE_LANGUAGE_ROLE 3u

/* Agents, sequences, key value cache slots, request slots and bus writers. One name. */
#define AOTX_SLOTS                 32u

/* Prompt and reply together, for one sequence. */
#define AOTX_SEQ_MAX_TOKENS        2048u

/* The key value range is virtual and costs no memory. A slot holds up to this many pages
 * of 2 MB, which is a context of 1,228 tokens of the 36 layer model. */
#define AOTX_KV_RANGE_BYTES        (1024ull * 1024ull * 1024ull)
#define AOTX_KV_PAGES_EACH         148u  /* 2,048 tokens at 14 a page; the range still bounds the pool */

/* The rings. The record rate follows the slot count, so the ring figures follow it too. */
#define AOTX_DEVICE_RING_SLOTS     32768ull
#define AOTX_HOST_RING_DATA_BYTES  (32ull * 1024ull * 1024ull)
#define AOTX_BULK_RING_DATA_BYTES  (128ull * 1024ull * 1024ull)
#define AOTX_INBOUND_SLOTS         2048ull

/* Bytes of one wrapped prompt, of one skill text and of one tool result. */
#define AOTX_SAY_BYTES             6144u
#define AOTX_SKILL_BYTES           4096u
#define AOTX_TOOL_RESULT_BYTES     4096u

/* Device tool modules the tick graph holds at one time, and the scratch bytes one request
 * row of a module owns for a tick. An import of a device tool past the count is refused
 * with the figure. */
#define AOTX_TOOL_MODULES          4u
#define AOTX_TOOL_SCRATCH_BYTES    (32u * 1024u)

/* The catalog of installed modules. */
#define AOTX_MODULE_SLOTS          32u
#define AOTX_CATALOGUE_BYTES       (2ull * 1024ull * 1024ull)

/* Language models placed at one time. */
#define AOTX_MODELS_RESIDENT       1u

/* The weights region. The range is virtual: physical memory goes behind a part of it only
 * when a tensor of that part arrives. */
#define AOTX_MEM_WEIGHTS_BYTES     (5ull * 1024ull * 1024ull * 1024ull)

#endif
