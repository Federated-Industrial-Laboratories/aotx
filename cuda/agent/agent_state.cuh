/* Purpose: Hold the private state of the agent module and the helpers of its step.
 * Owns: The reply buffer, the message buffer and the turn marks of each agent slot.
 * Launch shape: Device state; the agent step writes it and reads it.
 * Lifetime: The whole run.
 *
 * The tables here are device state. No host address goes in them. The agent record of
 * agent.cuh carries the state a reader of the seam needs. This block carries the bytes the
 * step works over, which no record holds. */
#ifndef AOTX_AGENT_STATE_CUH
#define AOTX_AGENT_STATE_CUH

#include "agent/agent.cuh"
#include "model/decode.cuh"
#include "model/decode_state.cuh"
#include "model/model.cuh"
#include "text/text.cuh"
#include "tool/tool.cuh"

/* Bytes of the reply of one turn that the step reads back. The reply limit is 256 tokens,
 * and a token of this vocabulary gives eight bytes at the most. */
#define AOTX_AGENT_REPLY_BYTES  2048u

/* Reply tokens one turn of an agent may make. */
#define AOTX_AGENT_REPLY_TOKENS 256u

/* What a turn of an agent answers. The mark goes in the manifest record. */
#define AOTX_AGENT_TURN_TASK    1u   /* the turn works on a task */
#define AOTX_AGENT_TURN_MESSAGE 2u   /* the turn answers a message from the operator */
#define AOTX_AGENT_TURN_VERIFY  3u   /* the turn judges the result of a task */

/* What one agent slot holds between two ticks. */
typedef struct aotx_agent_work {
    unsigned char reply[AOTX_AGENT_REPLY_BYTES];  /* the bytes of the reply of the turn */
    unsigned int reply_len;
    unsigned int prompt_len;      /* bytes of the prompt of the turn */
    unsigned long long input_hash; /* FNV-1a 64 over those prompt bytes */
    unsigned int wrote;           /* 1 when this module wrote the prompt of the slot */
    unsigned int console;         /* 1 when the reply of this agent goes on the console */
    unsigned int kind;            /* AOTX_AGENT_TURN_* of the turn that runs */
    unsigned int result;          /* 1 when a tool result goes in the next prompt */
    unsigned int refused;         /* tool calls this agent made that the role does not hold */
    unsigned int opens;           /* sequences this agent opened since it spawned */
    aotx_tool_call call;          /* the call the reply of the turn holds, or none */
    char line[AOTX_BUS_TEXT_BYTES];  /* the text of one bus message this agent writes */
    unsigned char message[AOTX_TASK_TEXT_BYTES];  /* a message that waits for a prompt */
    unsigned int message_len;
    unsigned int has_message;
} aotx_agent_work;

extern __device__ aotx_agent_work aotx_agent_gear[AOTX_AGENT_SLOTS];

/* The role of each task. The task record of agent.cuh names the assignee and not the
 * role. The engine therefore keeps the role of a task that waits beside the table. */
extern __device__ unsigned int aotx_task_role[AOTX_TASK_SLOTS];

/* The task slots that hold a task. A slot of the table is free while this mark is zero. */
extern __device__ unsigned int aotx_task_used[AOTX_TASK_SLOTS];

/* Why the last call of the command layer was refused. A task open gives one value for
 * every refusal, so the reason stands beside it and the console names it. The message call
 * gives the reason itself: 0 taken, 1 the agent is busy, 2 the text is too long. */
#define AOTX_AGENT_REFUSE_NONE  0u
#define AOTX_AGENT_REFUSE_BUSY  1u   /* the agent is not idle, or it holds a task */
#define AOTX_AGENT_REFUSE_LONG  2u   /* the text is longer than a task text */
#define AOTX_AGENT_REFUSE_FULL  3u   /* the table holds no free slot */
#define AOTX_AGENT_REFUSE_ROLE  4u   /* the role or the agent is not one of the table */

extern __device__ unsigned int aotx_agent_refusal;

/* The counts the agent module keeps for the panel and the tests. */
typedef struct aotx_agent_counts {
    unsigned int spawned;
    unsigned int turns;         /* turns that ended with a reply */
    unsigned int calls;         /* tool calls the parser took */
    unsigned int no_calls;      /* replies with no tool call */
    unsigned int bad_calls;     /* calls to a tool the role does not hold */
    unsigned int done;          /* tasks that ended done */
    unsigned int failed;        /* tasks that ended failed */
    unsigned int verdicts;      /* verdicts a verifier gave */
    unsigned int opens_refused; /* prompts the sequence table refused */
} aotx_agent_counts;

extern __device__ aotx_agent_counts aotx_agent_count;

/* Spawn the conductor on slot 0. The call is idempotent, because the spawn takes slot 0
 * for the conductor alone and refuses a second one. One thread makes the call. */
__global__ void aotx_agent_boot(void);

/* Host glue: put the conductor in the table before the tick capture starts. The conductor
 * is the agent the say command speaks to, so it stands before the first tick and before a
 * replay of the journal. The return is zero when the table holds it. */
int aotx_agent_open(void);

/* The hash of the manifest record. FNV-1a 64 over the bytes, which is the hash the commit
 * of the tick uses over record bodies. */
#define AOTX_AGENT_FNV_BASIS  1469598103934665603ull
#define AOTX_AGENT_FNV_PRIME  1099511628211ull

__device__ __forceinline__ unsigned long long aotx_agent_hash(const unsigned char *bytes,
                                                              unsigned int length)
{
    unsigned long long hash = AOTX_AGENT_FNV_BASIS;
    for (unsigned int i = 0u; i < length; ++i) {
        hash ^= (unsigned long long)bytes[i];
        hash *= AOTX_AGENT_FNV_PRIME;
    }
    return hash;
}

/* Write the three role rows. The call fills a table that is empty and changes nothing when
 * the table is full, so a test which sets another budget keeps it. */
__device__ __forceinline__ void aotx_agent_roles_set(void)
{
    aotx_role *row = aotx_agents.role;
    if (row[AOTX_ROLE_WORKER].budget != 0u) {
        return;
    }
    row[AOTX_ROLE_CONDUCTOR].tools = (1u << AOTX_TOOL_MEMORY_RECALL)
                                   | (1u << AOTX_TOOL_MEMORY_WRITE)
                                   | (1u << AOTX_TOOL_FS_READ);
    row[AOTX_ROLE_CONDUCTOR].needs_auth = (1u << AOTX_TOOL_FS_READ);
    row[AOTX_ROLE_CONDUCTOR].model = AOTX_MODEL_LANGUAGE;
    row[AOTX_ROLE_CONDUCTOR].budget = AOTX_AGENT_BUDGET;
    row[AOTX_ROLE_CONDUCTOR].overlay = 0u;

    row[AOTX_ROLE_WORKER].tools = (1u << AOTX_TOOL_MEMORY_RECALL)
                                | (1u << AOTX_TOOL_MEMORY_WRITE)
                                | (1u << AOTX_TOOL_FS_READ);
    row[AOTX_ROLE_WORKER].needs_auth = (1u << AOTX_TOOL_FS_READ);
    row[AOTX_ROLE_WORKER].model = AOTX_MODEL_LANGUAGE;
    row[AOTX_ROLE_WORKER].budget = AOTX_AGENT_BUDGET;
    row[AOTX_ROLE_WORKER].overlay = 1u;

    row[AOTX_ROLE_VERIFIER].tools = (1u << AOTX_TOOL_MEMORY_RECALL);
    row[AOTX_ROLE_VERIFIER].needs_auth = 0u;
    row[AOTX_ROLE_VERIFIER].model = AOTX_MODEL_LANGUAGE;
    row[AOTX_ROLE_VERIFIER].budget = AOTX_AGENT_BUDGET;
    row[AOTX_ROLE_VERIFIER].overlay = 2u;
}

/* Report whether a role may call a tool. An unknown tool number is not in any mask, so it
 * fails to no tool. */
__device__ __forceinline__ int aotx_agent_may_call(unsigned int role, unsigned int tool)
{
    if (role >= AOTX_ROLE_COUNT || tool == AOTX_TOOL_NONE || tool >= 32u) {
        return 0;
    }
    return (aotx_agents.role[role].tools & (1u << tool)) != 0u;
}

/* Report whether a tool of a role waits for the operator. */
__device__ __forceinline__ int aotx_agent_needs_auth(unsigned int role, unsigned int tool)
{
    if (role >= AOTX_ROLE_COUNT || tool >= 32u) {
        return 0;
    }
    return (aotx_agents.role[role].needs_auth & (1u << tool)) != 0u;
}

/* The bytes that one token gives. The count comes first, so a token that does not fit in
 * the room that is left leaves the buffer as it stands. */
__device__ __forceinline__ unsigned int aotx_agent_token_bytes(unsigned int token,
                                                               unsigned char *out,
                                                               unsigned int room, int write)
{
    const aotx_text_vocab *vocab = &aotx_text_vocab_table;
    if (token >= vocab->tokens) {
        return 0u;
    }
    unsigned long long from = vocab->token_at[token];
    unsigned int span = (unsigned int)(vocab->token_at[token + 1u] - from);
    const unsigned char *text = vocab->token_bytes + from;
    unsigned int walk = 0u;
    unsigned int at = 0u;
    while (walk < span) {
        unsigned int point = 0u;
        walk += aotx_text_decode(text, span, walk, &point);
        unsigned int byte = aotx_text_point_byte(point);
        if (byte < 0x100u) {
            if (write != 0 && at < room) {
                out[at] = (unsigned char)byte;
            }
            at += 1u;
        } else {
            /* A code point the byte map does not hold is written as itself. The count
             * pass and the write pass take the same length, so the two passes agree and
             * no byte of the run is left unwritten. */
            unsigned int span = (point < 0x80u) ? 1u
                              : ((point < 0x800u) ? 2u
                                 : ((point < 0x10000u) ? 3u : 4u));
            if (write != 0 && at + span <= room) {
                aotx_text_encode(point, out + at);
            }
            at += span;
        }
    }
    return at;
}

/* Read the whole reply of a sequence slot into the buffer of an agent. The read starts at
 * the first reply token every time, so a take that the console made does not change it.
 * The return is the byte count. */
__device__ __forceinline__ unsigned int aotx_agent_take_reply(unsigned int slot,
                                                              unsigned char *out,
                                                              unsigned int max)
{
    if (slot >= AOTX_SEQ_SLOTS || max == 0u) {
        return 0u;
    }
    const aotx_seq *seq = &aotx_seqs.slot[slot];
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < seq->sampled; ++i) {
        unsigned int position = seq->prompt + i;
        if (position >= AOTX_SEQ_MAX_TOKENS) {
            break;
        }
        unsigned int token = (unsigned int)aotx_seqs.tokens[slot][position];
        /* The token that ends a reply is not part of the reply. It goes in the journal
         * and in the sequence, and the text of the turn stops in front of it. */
        if (i + 1u == seq->sampled
            && (token == seq->stop || token == AOTX_DECODE_STOP_TEXT)) {
            break;
        }
        unsigned int bytes = aotx_agent_token_bytes(token, out + at, max - at, 0);
        if (at + bytes > max) {
            break;
        }
        aotx_agent_token_bytes(token, out + at, max - at, 1);
        at += bytes;
    }
    return at;
}

/* Give the verdict that the first word of a reply names, or none. The compare skips the
 * space of the start and takes the letters that follow it. */
__device__ __forceinline__ unsigned int aotx_agent_verdict_of(const unsigned char *text,
                                                              unsigned int length)
{
    const char *word[3] = { "uphold", "refute", "uncertain" };
    const unsigned int value[3] = { 1u, 2u, 3u };
    unsigned int at = 0u;
    while (at < length && (text[at] == (unsigned char)' ' || text[at] == (unsigned char)'\n'
                           || text[at] == (unsigned char)'\r'
                           || text[at] == (unsigned char)'\t')) {
        at += 1u;
    }
    for (unsigned int w = 0u; w < 3u; ++w) {
        unsigned int i = 0u;
        while (word[w][i] != '\0') {
            unsigned int byte = (at + i < length) ? (unsigned int)text[at + i] : 0u;
            if (byte >= (unsigned int)'A' && byte <= (unsigned int)'Z') {
                byte += 32u;
            }
            if (byte != (unsigned int)word[w][i]) {
                break;
            }
            i += 1u;
        }
        if (word[w][i] == '\0') {
            return value[w];
        }
    }
    return 0u;
}

#endif
