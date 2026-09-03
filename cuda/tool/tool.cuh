/* Purpose: Run device tools and request host tools.
 * Owns: The tool table and the pending request table.
 * Launch shape: One thread for each request.
 * Lifetime: The whole run. */
#ifndef TOOL_CUH
#define TOOL_CUH

#include "catalog/catalog.cuh"
#include "profile/profile.cuh"
#include "seam/wire.h"
#include "settings/settings.cuh"

/* One pending request for each agent at most, so the request count is AOTX_SLOTS. The
 * bytes a tool result may carry into the next prompt come from the profile as well. */
#define AOTX_RECALL_COUNT      4u     /* findings a recall returns */

/* The deadline of a request that waits for the operator. A tool which needs authorization
 * has no deadline while it waits, because a human answers in human time. The deadline of
 * the setting tool.deadline_ticks starts at the tick of the grant. This value is above
 * every tick a run can reach, so no comparison against a tick makes such a request late. */
#define AOTX_TOOL_NO_DEADLINE  0xffffffffffffffffull

/* Ticks a host tool request may take. The setting names the count. */
__device__ __forceinline__ unsigned long long aotx_setting_deadline(void)
{
    return (unsigned long long)aotx_setting_count(AOTX_SET_TOOL_DEADLINE);
}

/* The tool call the model writes, as the chat template of the file defines it:
 *   <tool_call>
 *   {"name": "<tool>", "arguments": {"<key>": "<value>", ...}}
 *   </tool_call>
 *
 * The parser takes this shape and no other. One call, string values only. The name is
 * compared against the entries of the catalog where it stands. The keys are the argument
 * keys the manifest of that entry names.
 *
 * The call keeps the value of every argument key. The pack holds the values one after
 * another and each key names its run in it. The request then writes the values as one line
 * of key=value pairs, in the order the manifest gives the keys.
 *
 * The unit separator byte comes before every pair, the first one included. The byte at the
 * front of the line is thus the mark that says the line holds keys. That line is the
 * argument of the request record and the argument batch of a module tool. */
#define AOTX_TOOL_UNIT   ((char)0x1f)

typedef struct aotx_tool_call {
    unsigned int entry;         /* the catalog entry of the tool, or AOTX_MODULE_SLOTS */
    unsigned int tool;          /* AOTX_TOOL_* of a built-in tool, or 0 */
    unsigned int key;           /* the argument key the value of the call came from */
    unsigned int provenance;    /* AOTX_PROV_* for memory_write, else 0 */
    unsigned int over;          /* 1 when the values do not fit the argument line */
    unsigned int arg_len;
    unsigned int values;        /* argument keys that carry a value */
    unsigned int at[AOTX_CATALOG_ARGS];      /* the run of each key in the pack */
    unsigned int length[AOTX_CATALOG_ARGS];
    unsigned int pack_len;
    char         pack[AOTX_TOOL_ARG_BYTES];  /* the values, one after another */
    char         arg[AOTX_TOOL_ARG_BYTES];   /* the value of the call */
} aotx_tool_call;

typedef struct aotx_request {
    unsigned int request;       /* the id, or 0 when the slot is free */
    unsigned int agent;
    unsigned int entry;         /* the catalog entry of the tool */
    unsigned int tool;
    unsigned int auth;          /* AOTX_AUTH_* */
    unsigned int status;        /* AOTX_TOOL_* once replied */
    unsigned int parts_in;      /* reply parts received */
    unsigned int parts;         /* reply parts expected, once the first arrives */
    unsigned int result_len;
    unsigned long long call_seq;   /* request record of the call, or zero */
    unsigned long long answer_seq; /* input line that granted or refused the call */
    unsigned long long result_seq; /* first reply record of the result */
    /* The tick after which the request fails. A request that waits for the operator holds
     * AOTX_TOOL_NO_DEADLINE, and takes a deadline at the tick of the grant. */
    unsigned long long deadline;
    /* The argument the request carries. The panel shows the first bytes of it, so an
     * operator judges a request that waits without a look at the record ring. */
    unsigned int arg_len;
    char arg[AOTX_TOOL_ARG_BYTES];
    char result[AOTX_TOOL_RESULT_BYTES];
} aotx_request;

typedef struct aotx_request_table {
    aotx_request slot[AOTX_SLOTS];
    unsigned int pending_auth;  /* requests that wait for the operator */
    unsigned int refused;
} aotx_request_table;

extern __device__ aotx_request_table aotx_requests;

/* Parse a reply for one tool call. The return is 1 when a well-formed call was found, and
 * 0 when the reply holds no call. The return is 2 for a call of the correct shape whose
 * argument values do not fit the argument line. Such a call keeps its entry and its tool,
 * carries the over mark and holds no value. It is a call, and the turn ends with a tool
 * error result which names the cause. */
__device__ int aotx_tool_parse(const unsigned char *reply, unsigned int length,
                               aotx_tool_call *call);

/* Open a request for an agent. A device tool is queued for the tool step of the tick. A host
 * tool writes a TOOL_REQUEST record, with auth PENDING when the role needs it. A request
 * that waits for the operator takes no deadline; every other request takes its deadline
 * from this tick. Returns the request id, or 0 when the agent already has one or the table
 * is full. */
__device__ unsigned int aotx_tool_request(unsigned int agent, const aotx_tool_call *call,
                                          unsigned int needs_auth, unsigned long long tick);

/* Write the values of a call as one line of key=value pairs. The keys stand in the order
 * the manifest gives them, and the unit separator byte comes before every pair. The return
 * is the bytes the line took, or zero when the line does not fit the bound. */
__device__ unsigned int aotx_tool_arguments(const aotx_tool_call *call, char *out,
                                            unsigned int max);

/* Give the run of one key of an argument line, into at and length. The return is 1 when
 * the line holds that key. The line is the shape aotx_tool_arguments writes. */
__device__ int aotx_tool_argument_of(const char *line, unsigned int length,
                                     const char *key, unsigned int key_len,
                                     unsigned int *at, unsigned int *span);

/* Apply one TOOL_REPLY record to its request, live or replayed. Returns 0, or 1 when no such
 * request waits (a reply after a deadline, or a duplicate). */
__device__ int aotx_tool_reply_apply(const aotx_tool_reply_body *body,
                                     unsigned long long seq = 0ull);

/* The arm of no result: a call the next turn makes is taken off, with no event and no
 * result. The arm of a call: a call the next turn makes completes at once as ok. A turn
 * with no call then carries no event and no result. */
#define AOTX_TOOL_NO_RESULT     4u
#define AOTX_TOOL_CALL_RESULT   5u

/* Arm the result of the next turn of an agent. The status is AOTX_TOOL_OK, AOTX_TOOL_ERROR,
 * AOTX_TOOL_REFUSED, AOTX_TOOL_NO_RESULT or AOTX_TOOL_CALL_RESULT. The console gives the
 * line. A scripted run can then make the events of a tool result in a known order,
 * whatever the reply holds. */
__device__ void aotx_tool_outcome_arm(unsigned int agent, unsigned int status);

/* Report whether an arm stands for the next turn of an agent. A turn with an arm opens
 * no request and runs no tool: the armed result stands in for the tool. */
__device__ int aotx_tool_outcome_armed(unsigned int agent);

/* Put the armed result of an agent on its request slot, for the tool the turn called, and
 * take the arm off. The request stands as complete at once. The return is the request
 * number. It is 0 for no arm, the arm of no result, a busy slot, or a call to no catalog
 * tool. The next turn carries the result as a real result. */
__device__ unsigned int aotx_tool_outcome_request(unsigned int agent,
                                                  const aotx_tool_call *call,
                                                  unsigned long long tick);

/* Put the armed result of an agent on its request slot and take the arm off, for a turn
 * that made no call. The return is 1 when a result was armed, and 0 for no arm or the arm
 * of no result. The agent reads the slot as the result of its turn. */
__device__ int aotx_tool_outcome_take(unsigned int agent);

/* Put the error result of a call whose values do not fit on the request slot of an agent.
 * No request goes out and no tool runs. The result stands as complete, and the next turn
 * carries it as a real result with the tool error event. The return is the request number,
 * or 0 for a busy slot or a call to no catalog tool. */
__device__ unsigned int aotx_tool_over_request(unsigned int agent,
                                               const aotx_tool_call *call,
                                               unsigned long long tick);

/* The tool step of the tick. Device tools run over the embed batch: memory_write appends a
 * FINDING with its vector, and memory_recall searches and writes its result. Deadlines pass.
 * A request whose reply is complete hands its result to the agent. One thread for each
 * request slot. */
__global__ void aotx_tool_step(unsigned long long tick);

/* Host glue: capture the embed batch (the embedding model's pass as a child graph) and the
 * tool step into the tick stream. Returns 0, or 1 when no embedding role is loaded. */
int aotx_tool_capture(void *stream);

#endif
