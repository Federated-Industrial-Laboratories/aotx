/* Purpose: Hold the conversation quality state and its device entry points.
 * Owns: The vectors, trigram sets, pending rows, phrase table and counts.
 * Launch shape: One block or one thread for each agent.
 * Lifetime: The whole run. */
#ifndef AOTX_QUALITY_CUH
#define AOTX_QUALITY_CUH

#include "model/forward.cuh"
#include "profile/profile.cuh"
#include "settings/settings.cuh"

#define AOTX_QUALITY_ROWS       (2u * AOTX_SLOTS)
#define AOTX_QUALITY_BYTES      512u
#define AOTX_QUALITY_TOKENS     128u
#define AOTX_QUALITY_WIDTH      1024u
#define AOTX_QUALITY_SET_WORDS  256u
#define AOTX_QUALITY_PHRASES    16u
#define AOTX_QUALITY_PHRASE_BYTES 64u
#define AOTX_QUALITY_FILL_THREADS 256u

#define AOTX_QUALITY_ROW_NONE    0u
#define AOTX_QUALITY_ROW_WAIT    1u
#define AOTX_QUALITY_ROW_RUN     2u
#define AOTX_QUALITY_ROW_DONE    3u

typedef struct aotx_quality_slot {
    float previous[AOTX_QUALITY_WIDTH];
    float prompt[AOTX_QUALITY_WIDTH];
    unsigned char text[2][AOTX_QUALITY_BYTES];
    unsigned int length[2];
    unsigned int trigram[AOTX_QUALITY_SET_WORDS];
    unsigned int last[2], last_count, total, distinct, held_total, held_distinct;
    unsigned int row[2], place[2], asked[2];
    unsigned int active, ended, pending, previous_valid, prompt_valid;
    unsigned int turn, tokens, limit, refusal, flags;
    float coherence_prompt, coherence_turn, guard[2], active_guard[2];
    unsigned int active_guard_loaded;
} aotx_quality_slot;

typedef struct aotx_quality_phrase_table {
    unsigned char text[AOTX_QUALITY_PHRASES][AOTX_QUALITY_PHRASE_BYTES];
    unsigned int length[AOTX_QUALITY_PHRASES];
    unsigned int count;
} aotx_quality_phrase_table;

typedef struct aotx_quality_counts {
    unsigned int written, late, filled;
} aotx_quality_counts;

extern __device__ aotx_quality_slot aotx_quality_state[AOTX_SLOTS];
extern __device__ aotx_quality_phrase_table aotx_quality_phrases;
extern __device__ aotx_quality_counts aotx_quality_count;

__device__ void aotx_quality_open(unsigned int agent, aotx_model_how *how);
__device__ void aotx_quality_pick(unsigned int agent, const aotx_model_how *how,
                                  unsigned int token);
__device__ void aotx_quality_end(unsigned int agent);
__device__ void aotx_quality_guard(unsigned int agent, float first, float second,
                                   unsigned int loaded);
__device__ void aotx_quality_fill(unsigned int agent);
__global__ void aotx_quality_turn(void);

int aotx_quality_load(const char *path);
int aotx_quality_load_store(const char *dir);
int aotx_quality_capture(void *stream);

#endif
