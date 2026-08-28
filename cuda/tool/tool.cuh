/* Purpose: Run device tools and request host tools.
 * Owns: The tool table and the pending request table.
 * Launch shape: One thread for each request.
 * Lifetime: The whole run. */
#ifndef TOOL_CUH
#define TOOL_CUH

#include "seam/wire.h"

#define AOTX_REQUEST_SLOTS     64u    /* one pending request for each agent at most */
#define AOTX_TOOL_RESULT_BYTES 4096u  /* bytes a tool result may carry into the next prompt */
#define AOTX_RECALL_COUNT      4u     /* findings a recall returns */

/* The tool call the model writes, as the chat template of the file defines it:
 *   <tool_call>
 *   {"name": "<tool>", "arguments": {"<key>": "<value>", ...}}
 *   </tool_call>
 *
 * The parser takes this shape and no other. One call, string values only. The keys are the
 * ones the tool table names: memory_recall takes text; memory_write takes provenance and
 * text; fs_read takes path. */
typedef struct aotx_tool_call {
    unsigned int tool;          /* AOTX_TOOL_*, or 0 when the reply holds no call */
    unsigned int provenance;    /* AOTX_PROV_* for memory_write, else 0 */
    unsigned int arg_len;
    char         arg[AOTX_TOOL_ARG_BYTES];  /* the text or the path */
} aotx_tool_call;

typedef struct aotx_request {
    unsigned int request;       /* the id, or 0 when the slot is free */
    unsigned int agent;
    unsigned int tool;
    unsigned int auth;          /* AOTX_AUTH_* */
    unsigned int status;        /* AOTX_TOOL_* once replied */
    unsigned int parts_in;      /* reply parts received */
    unsigned int parts;         /* reply parts expected, once the first arrives */
    unsigned int result_len;
    unsigned long long deadline;
    /* The argument the request carries. The panel shows the first bytes of it, so an
     * operator judges a request that waits without a look at the record ring. */
    unsigned int arg_len;
    char arg[AOTX_TOOL_ARG_BYTES];
    char result[AOTX_TOOL_RESULT_BYTES];
} aotx_request;

typedef struct aotx_request_table {
    aotx_request slot[AOTX_REQUEST_SLOTS];
    unsigned int pending_auth;  /* requests that wait for the operator */
    unsigned int refused;
} aotx_request_table;

extern __device__ aotx_request_table aotx_requests;

/* Parse a reply for one tool call. Returns 1 when a well-formed call was found. */
__device__ int aotx_tool_parse(const unsigned char *reply, unsigned int length,
                               aotx_tool_call *call);

/* Open a request for an agent. A device tool is queued for the tool step of the tick. A host
 * tool writes a TOOL_REQUEST record, with auth PENDING when the role needs it. Returns the
 * request id, or 0 when the agent already has one or the table is full. */
__device__ unsigned int aotx_tool_request(unsigned int agent, const aotx_tool_call *call,
                                          unsigned int needs_auth, unsigned long long tick);

/* Apply one TOOL_REPLY record to its request, live or replayed. Returns 0, or 1 when no such
 * request waits (a reply after a deadline, or a duplicate). */
__device__ int aotx_tool_reply_apply(const aotx_tool_reply_body *body);

/* The tool step of the tick. Device tools run over the embed batch: memory_write appends a
 * FINDING with its vector, and memory_recall searches and writes its result. Deadlines pass.
 * A request whose reply is complete hands its result to the agent. One thread for each
 * request slot. */
__global__ void aotx_tool_step(unsigned long long tick);

/* Host glue: capture the embed batch (the embedding model's pass as a child graph) and the
 * tool step into the tick stream. Returns 0, or 1 when no embedding role is loaded. */
int aotx_tool_capture(void *stream);

#endif
