/* Purpose: Keep each agent transcript and select the memory of a prompt.
 * Owns: The ordered turn index, warm vectors, summary and selection state.
 * Launch shape: Device functions; one call for each agent turn.
 * Lifetime: The whole run. */
#ifndef AOTX_AGENT_TRANSCRIPT_CUH
#define AOTX_AGENT_TRANSCRIPT_CUH

#include "profile/profile.cuh"
#include "seam/wire.h"
#include "tool/tool.cuh"

#define AOTX_TRANSCRIPT_TEXT_BYTES (AOTX_MEMORY_TURNS * AOTX_MEMORY_TEXT)
#define AOTX_TRANSCRIPT_VECTOR      1024u
#define AOTX_TRANSCRIPT_SUMMARY     2048u
#define AOTX_TRANSCRIPT_AUTO        0xffffffffu

#define AOTX_MEMORY_HOT       1u
#define AOTX_MEMORY_WARM      2u
#define AOTX_MEMORY_FOLDED    3u

#define AOTX_MEMORY_EMBED_NONE  0u
#define AOTX_MEMORY_EMBED_TURN  1u
#define AOTX_MEMORY_EMBED_QUERY 2u

typedef struct aotx_transcript_turn {
    unsigned long long seq;          /* first class A record of the turn */
    unsigned long long manifest_seq; /* derived record that closes the turn */
    unsigned long long reply_seq;    /* first console record, or zero for a token range */
    unsigned long long call_seq;     /* tool request record, when the turn made one */
    unsigned long long result_seq;   /* first reply record of the tool result */
    unsigned long long answer_seq;   /* line that granted or refused the tool */
    unsigned int number;
    unsigned int text_at;
    unsigned int text_len;
    unsigned int reply_at;
    unsigned int reply_len;
    unsigned int extra_at;
    unsigned int extra_len;
    unsigned int stored_len;
    unsigned int reply_records;
    unsigned int token_first;
    unsigned int token_count;
    unsigned int tokens;             /* tokens owned by this turn */
    unsigned int tier;               /* AOTX_MEMORY_* */
    unsigned int vector_ready;
    unsigned int text_live;
} aotx_transcript_turn;

typedef struct aotx_transcript_agent {
    aotx_transcript_turn turn[AOTX_MEMORY_TURNS];
    unsigned int count;
    unsigned int first;
    unsigned int text_head;
    unsigned int text_used;
    unsigned int pages;              /* fixed, automatic, or zero for the role */
    unsigned int limit;              /* pages used by the turn that opens */
    unsigned int hot;
    unsigned int warm;
    unsigned int folded;
    unsigned int compact;
    unsigned int compact_first;
    unsigned int compact_count;
    unsigned int compact_left;
    unsigned int force_compact;
    unsigned long long summary_seq;
    unsigned int summary_len;
    unsigned char summary[AOTX_TRANSCRIPT_SUMMARY];
    aotx_selection_body choice;
    aotx_selection_body replay_choice;
    unsigned int choice_pending;
    unsigned int replay_ready;
    unsigned int query_ready;
    unsigned int selected[AOTX_SELECTION_MAX];
    float selected_score[AOTX_SELECTION_MAX];
    unsigned int selected_count;
    unsigned int embed_kind;
    unsigned int embed_turn;
} aotx_transcript_agent;

typedef struct aotx_transcript_counts {
    unsigned long long searches;
    unsigned long long replay_selections;
    unsigned long long embedded;
    unsigned long long compacted;
    unsigned long long text_refused;
    unsigned long long prompt_refused;
    unsigned long long text_released;
} aotx_transcript_counts;

extern __device__ aotx_transcript_agent aotx_transcript[AOTX_SLOTS];
extern __device__ unsigned char
    aotx_transcript_text[AOTX_SLOTS][AOTX_TRANSCRIPT_TEXT_BYTES];
extern __device__ float
    aotx_transcript_vector[AOTX_SLOTS][AOTX_MEMORY_TURNS][AOTX_TRANSCRIPT_VECTOR];
extern __device__ aotx_transcript_counts aotx_transcript_count;
extern __device__ unsigned long long aotx_transcript_source_seq;
extern __device__ unsigned long long aotx_transcript_replay_tick[AOTX_SLOTS];

/* Set the class A record that caused the command which runs. */
__device__ void aotx_transcript_source(unsigned long long seq);

/* Set a fixed or automatic page limit. The next turn reads it. */
__device__ int aotx_transcript_pages(unsigned int agent, unsigned int pages);

/* Ask for a compaction turn. */
__device__ int aotx_transcript_compact(unsigned int agent);

/* Prepare the selection of a turn. Zero means that its embedding still runs. */
__device__ int aotx_transcript_prepare(unsigned int agent, const unsigned char *text,
                                       unsigned int length, unsigned int turn,
                                       unsigned int reserve);

/* Add the summary, recall and hot blocks before the current text. */
__device__ unsigned int aotx_transcript_prompt(unsigned int agent, unsigned char *out,
                                               unsigned int at);

/* Move one oldest hot turn to warm memory. Returns one when a turn moved. */
__device__ int aotx_transcript_give_hot(unsigned int agent);

/* Reduce the current compaction round. Returns one while one turn remains. */
__device__ int aotx_transcript_compact_less(unsigned int agent);

/* Close one turn in the ordered index. */
__device__ void aotx_transcript_finish(unsigned int agent, const unsigned char *text,
                                       unsigned int text_len, const unsigned char *reply,
                                       unsigned int reply_len, unsigned int tokens,
                                       unsigned long long manifest_seq);

/* Attach the result and the authorization answer to the turn that made the call. */
__device__ void aotx_transcript_result(unsigned int agent, const aotx_request *request);

/* Take one vector from the embedding batch. */
__device__ void aotx_transcript_embed_done(unsigned int agent, const float *vector,
                                           unsigned int width);

/* Apply one recorded choice at restore and search nothing. */
__device__ void aotx_transcript_selection_apply(const aotx_selection_body *body);

/* Write staged choices in agent order before the tick commit. */
__device__ void aotx_transcript_commit(unsigned long long tick);

/* Finish a compaction reply as a finding and fold the source turns. */
__device__ void aotx_transcript_summary(unsigned int agent, const unsigned char *text,
                                        unsigned int length, unsigned long long tick);

/* Page and compaction queries for the command layer. */
__device__ unsigned int aotx_transcript_page_limit(unsigned int agent);
__device__ unsigned int aotx_transcript_needs_compact(unsigned int agent);
__device__ int aotx_transcript_maintain(unsigned int agent);

#endif
