/* Purpose: Define explicit memory residency batches and bounded disk reads.
 * Owns: File and transport byte layouts; no process address is stored.
 * Launch shape: Up to 64 input rows and one bounded device-selected read batch.
 * Lifetime: Schema 1 requests, results and cold extents. */
#ifndef AOTX_COGNITIVE_COLD_H
#define AOTX_COGNITIVE_COLD_H
#include "format.h"
#define AOTX_COLD_CONTROL 18u
#define AOTX_COLD_RESULT 19u
#define AOTX_COLD_OFFLOAD 1u
#define AOTX_COLD_FETCH 2u
#define AOTX_COLD_GPU 3u
#define AOTX_COLD_ENABLE 4u
#define AOTX_COLD_HEADER 64u
#define AOTX_COLD_ROW 64u
#define AOTX_COLD_BATCH 64u
#define AOTX_COLD_EXTENT_HEADER 64u
#define AOTX_COLD_EXTENT_ROW (AOTX_COG_OBJECT + 32u)
#define AOTX_COLD_COPY 262144u
/* Control: AOTXTIR1, schema/mode/count/row bytes at 8/12/16/20.
 * Lineage at 24, store sequence at 40, zero at 48..63.
 * Row: ID/version/principal/room at 0/16/24/40, zero at 56..63.
 * Offload and fetch require rows. Enable and GPU mode require no rows.
 *
 * Result: AOTXTIR2, schema/status/mode/count at 8/12/16/20.
 * Lineage at 24, sequence at 40, payload bytes at 48, zero at 56..63.
 * Exact object rows follow, then resident payload bytes for successful reads.
 *
 * Cold extent: AOTXCOLD, schema/count/row bytes at 8/12/16, zero at 20.
 * Payload bytes at 24, lineage at 32, zero at 48..63.
 * Each extent row has object metadata and its 32-byte payload SHA-256.
 * Offsets address the cold extent payload. */
typedef struct aotx_cold_transport {
    uint64_t request, response;
    uint64_t boot, generation, incarnation[2];
    uint32_t count, bytes, status, reserved;
    unsigned char rows[AOTX_COG_OBJECTS][AOTX_COG_OBJECT];
    unsigned char payload[AOTX_COG_PAYLOAD];
} aotx_cold_transport;
#endif
