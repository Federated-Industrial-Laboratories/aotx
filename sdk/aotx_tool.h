/* Purpose: Define the layouts a device tool module reads and writes; the one header a tool
 *   author includes.
 * Owns: Nothing; layouts and constants only, included by the module and by the device.
 * Launch shape: Not applicable; plain C with no CUDA symbol.
 * Lifetime: The tool contract version; a change to any struct increments AOTX_TOOL_ABI. */
#ifndef AOTX_SDK_TOOL_H
#define AOTX_SDK_TOOL_H

#include <stdint.h>

#define AOTX_TOOL_ABI            1u

/* A tool module holds one kernel with two parameters. Both are device addresses:
 *   .entry aotx_tool_<name>(.param .u64 batch, .param .u64 out)
 * The module compiles with the toolkit alone. It includes this header and nothing of the
 * system. The row count and the byte bounds come from the batch, so one module serves
 * every profile. */
#define AOTX_TOOL_ARGS_MAX       4u
#define AOTX_TOOL_KEY_BYTES      32u
#define AOTX_TOOL_VALUE_BYTES    1024u

/* The status a module writes. A status other than ok makes the text a reason. */
#define AOTX_TOOL_STATUS_OK      0u
#define AOTX_TOOL_STATUS_ERROR   1u

typedef struct aotx_tool_argument {
    char     key[AOTX_TOOL_KEY_BYTES];        /* the argument key, end byte included */
    uint32_t length;                          /* bytes of value that carry data */
    char     value[AOTX_TOOL_VALUE_BYTES];
} aotx_tool_argument;

/* One row for each request slot of the system. A module reads the rows whose take is 1
 * and writes the output rows of those and no other. */
typedef struct aotx_tool_row {
    uint32_t take;              /* 1 when this row is a request for this tool in this tick */
    uint32_t request;           /* the request number */
    uint32_t agent;             /* the agent that asked */
    uint32_t arguments;         /* arguments that carry data */
    uint64_t seed;              /* a Philox lane seed for this row and tick */
    aotx_tool_argument argument[AOTX_TOOL_ARGS_MAX];
} aotx_tool_row;

typedef struct aotx_tool_batch {
    uint32_t abi;               /* AOTX_TOOL_ABI */
    uint32_t rows;              /* rows in the table below; the slots of the profile */
    uint64_t tick;
    uint64_t boot_id;
    uint64_t scratch_bytes;     /* bytes of scratch for each row */
    uint64_t out_bytes;         /* bytes of text an output row may carry */
    aotx_tool_row *row;         /* rows entries, in device memory */
    unsigned char *scratch;     /* rows times scratch_bytes, in device memory; row i owns
                                 * the bytes from i * scratch_bytes for the tick */
} aotx_tool_batch;

/* One output row for each request row, out_bytes of text after the head. The module
 * release-stores done last; the tool step reads it. The text of row i starts at
 * out->text + i * out_bytes. */
typedef struct aotx_tool_output_row {
    uint32_t status;            /* AOTX_TOOL_STATUS_* */
    uint32_t length;            /* bytes of text that carry data, at most out_bytes */
    uint32_t done;              /* 1 when the row is written; the module stores it last */
    uint32_t reserved;
} aotx_tool_output_row;

typedef struct aotx_tool_output {
    uint32_t abi;
    uint32_t rows;
    aotx_tool_output_row *head; /* rows entries */
    char *text;                 /* rows times out_bytes */
} aotx_tool_output;

#endif
