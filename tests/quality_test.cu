/* Purpose: Check the quality batch, trigrams, phrases and turn records.
 * Owns: The scripted quality rows, vectors, agent turns and device ring.
 * Launch shape: One thread or one block for each scripted agent.
 * Lifetime: One run of the test program. */
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "agent/agent_state.cuh"
#include "boot/boot.cuh"
#include "boot/check.h"
#include "kvcache/kvcache.cuh"
#include "mem/mem.cuh"
#include "quality/quality.cuh"
#include "sched/sched.cuh"
#include "seam/seam.cuh"
#include "tool/tool_state.cuh"

#include "catalog_feed.h"

#define AOTX_QUALITY_TEST_SLOTS 128u
#define AOTX_QUALITY_TEST_TICK  73ull

static unsigned int cases, bad, skipped;
static unsigned int heads_good;

static void note(const char *name, int good, unsigned int got, unsigned int want)
{
    cases += 1u;
    if (!good) bad += 1u;
    printf("%-58s %-4s %u of %u\n", name, good ? "ok" : "BAD", got, want);
}

#define CLEAR_SYMBOL(symbol, bytes) do { \
    void *at = 0; \
    aotx_check_runtime(cudaGetSymbolAddress(&at, symbol), "cudaGetSymbolAddress"); \
    aotx_check_runtime(cudaMemset(at, 0, bytes), "cudaMemset"); \
} while (0)

__global__ void quality_fill_script(unsigned int count)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count) return;
    aotx_quality_slot *state = &aotx_quality_state[agent];
    state->row[0] = AOTX_QUALITY_ROW_WAIT;
    state->row[1] = AOTX_QUALITY_ROW_WAIT;
    state->length[0] = AOTX_QUALITY_BYTES;
    state->length[1] = AOTX_QUALITY_BYTES;
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    gear->message_len = 600u;
    gear->reply_len = 700u;
    for (unsigned int i = 0u; i < 700u; ++i) {
        if (i < 600u) gear->message[i] = (unsigned char)('a' + agent % 20u);
        gear->reply[i] = (unsigned char)('A' + agent % 20u);
        if (i < AOTX_QUALITY_BYTES) {
            state->text[0][i] = gear->message[i];
            state->text[1][i] = gear->reply[i];
        }
    }
}

static void fill_check(unsigned int count)
{
    CLEAR_SYMBOL(aotx_quality_state, sizeof(aotx_quality_slot) * AOTX_SLOTS);
    CLEAR_SYMBOL(aotx_agent_gear, sizeof(aotx_agent_work) * AOTX_SLOTS);
    CLEAR_SYMBOL(aotx_tool_gear, sizeof(aotx_tool_work));
    quality_fill_script<<<1, AOTX_SLOTS>>>(count);
    aotx_tool_fill<<<AOTX_SLOTS, AOTX_QUALITY_FILL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int length[AOTX_TOOL_BATCH_ROWS];
    aotx_check_runtime(cudaMemcpyFromSymbol(length, aotx_tool_gear, sizeof length,
                       offsetof(aotx_tool_work, length)), "cudaMemcpyFromSymbol");
    unsigned int right = 0u;
    for (unsigned int agent = 0u; agent < count; ++agent) {
        unsigned int row = AOTX_SLOTS + 2u * agent;
        right += (length[row] == AOTX_QUALITY_BYTES
                  && length[row + 1u] == AOTX_QUALITY_BYTES) ? 1u : 0u;
    }
    char label[96];
    snprintf(label, sizeof label, "two quality texts take the first 512 bytes at %u", count);
    note(label, right == count, right, count);
}

__global__ void quality_plan_script(unsigned int count, unsigned int full)
{
    unsigned int agent = threadIdx.x;
    if (agent >= AOTX_SLOTS) return;
    aotx_agents.agent[agent].state = AOTX_AGENT_STATE_IDLE;
    aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_NONE;
    aotx_tool_embed.place[agent] = AOTX_SLOTS;
    aotx_quality_state[agent].row[0] = AOTX_QUALITY_ROW_NONE;
    aotx_quality_state[agent].row[1] = AOTX_QUALITY_ROW_NONE;
    if (agent == 0u) {
        aotx_tool_embed.ready = 1u;
        aotx_tool_embed.role = AOTX_MODEL_EMBEDDING;
        aotx_sched.held = 0ull;
    }
    if (full == 2u) {
        aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_WAIT;
        aotx_tool_gear.count[agent] = 1u;
        if (agent == 0u) {
            aotx_quality_state[agent].row[0] = AOTX_QUALITY_ROW_WAIT;
            aotx_tool_gear.count[AOTX_SLOTS] = 1u;
        }
    } else if (full != 0u) {
        if (agent < 3u) {
            aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_WAIT;
            aotx_tool_gear.count[agent] = 170u;
        }
        if (agent == 3u) {
            aotx_quality_state[agent].row[0] = AOTX_QUALITY_ROW_WAIT;
            aotx_tool_gear.count[AOTX_SLOTS + 2u * agent] = 20u;
        }
    } else {
        unsigned int first = (count == 1u) ? 0u : 1u;
        if (count > 1u && agent == 0u) {
            aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_WAIT;
            aotx_tool_gear.count[agent] = 3u;
        }
        if (agent >= first && agent < min(count, first + 2u)) {
            aotx_quality_state[agent].row[0] = AOTX_QUALITY_ROW_WAIT;
            aotx_tool_gear.count[AOTX_SLOTS + 2u * agent] = 200u;
        }
    }
}

__global__ void quality_drop_tools(void)
{
    unsigned int agent = threadIdx.x;
    if (agent < AOTX_SLOTS) aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_NONE;
}

static void plan_check(unsigned int count)
{
    CLEAR_SYMBOL(aotx_tool_embed, sizeof(aotx_tool_embed_batch));
    CLEAR_SYMBOL(aotx_tool_gear, sizeof(aotx_tool_work));
    CLEAR_SYMBOL(aotx_quality_state, sizeof(aotx_quality_slot) * AOTX_SLOTS);
    quality_plan_script<<<1, AOTX_SLOTS>>>(count, 0u);
    aotx_tool_plan<<<1, AOTX_SLOTS>>>(0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_embed_batch *batch = (aotx_tool_embed_batch *)malloc(sizeof *batch);
    aotx_check_runtime(cudaMemcpyFromSymbol(batch, aotx_tool_embed, sizeof *batch),
                       "cudaMemcpyFromSymbol");
    unsigned int tool = (count == 1u) ? 0u : 1u;
    unsigned int quality = min(2u, count);
    unsigned int right = batch->seqs == tool + quality;
    for (unsigned int i = tool; i < batch->seqs; ++i) {
        right += (batch->live[i] == 0u
                  && batch->offset[i + 1u] - batch->offset[i] == AOTX_QUALITY_TOKENS) ? 1u : 0u;
    }
    char label[96];
    snprintf(label, sizeof label, "quality rows follow tool rows and cut at 128 at %u", count);
    note(label, right == quality + 1u, right, quality + 1u);
    if (count == AOTX_SLOTS) {
        quality_plan_script<<<1, AOTX_SLOTS>>>(count, 1u);
        aotx_tool_plan<<<1, AOTX_SLOTS>>>(0ull);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_quality_slot state;
        aotx_check_runtime(cudaMemcpyFromSymbol(&state, aotx_quality_state, sizeof state,
                           3u * sizeof state), "cudaMemcpyFromSymbol");
        note("a quality row waits behind a full tool budget", state.row[0] == AOTX_QUALITY_ROW_WAIT,
             state.row[0] == AOTX_QUALITY_ROW_WAIT, 1u);
        quality_drop_tools<<<1, AOTX_SLOTS>>>();
        aotx_tool_plan<<<1, AOTX_SLOTS>>>(0ull);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_check_runtime(cudaMemcpyFromSymbol(batch, aotx_tool_embed, sizeof *batch),
                           "cudaMemcpyFromSymbol");
        note("the waiting row lands in a later tick", batch->seqs == 1u && batch->kind[0] == 7u,
             batch->seqs, 1u);
        quality_plan_script<<<1, AOTX_SLOTS>>>(count, 2u);
        aotx_tool_plan<<<1, AOTX_SLOTS>>>(0ull);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_quality_slot first;
        aotx_check_runtime(cudaMemcpyFromSymbol(&first, aotx_quality_state, sizeof first),
                           "cudaMemcpyFromSymbol");
        note("a quality row waits when tool rows fill the sequence batch",
             first.row[0] == AOTX_QUALITY_ROW_WAIT,
             first.row[0] == AOTX_QUALITY_ROW_WAIT, 1u);
    }
    free(batch);
}

__global__ void quality_trigram_script(unsigned int repeated)
{
    aotx_quality_slot *state = &aotx_quality_state[0];
    state->active = 1u;
    aotx_model_how how = {};
    how.quality = 1u;
    unsigned int count = repeated != 0u ? 16u : 10u;
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int token = repeated != 0u ? (i % 4u + 10u) : (i + 100u);
        aotx_quality_pick(0u, &how, token);
    }
}

static void trigram_check(void)
{
    aotx_quality_slot state;
    CLEAR_SYMBOL(aotx_quality_state, sizeof(aotx_quality_slot) * AOTX_SLOTS);
    quality_trigram_script<<<1, 1>>>(1u);
    aotx_check_runtime(cudaMemcpyFromSymbol(&state, aotx_quality_state, sizeof state),
                       "cudaMemcpyFromSymbol");
    float repetition = 1.0f - (float)state.distinct / (float)state.total;
    note("four copies of one token sentence repeat above one half", repetition > 0.5f,
         (unsigned int)(repetition * 1000.0f), 500u);
    CLEAR_SYMBOL(aotx_quality_state, sizeof(aotx_quality_slot) * AOTX_SLOTS);
    quality_trigram_script<<<1, 1>>>(0u);
    aotx_check_runtime(cudaMemcpyFromSymbol(&state, aotx_quality_state, sizeof state),
                       "cudaMemcpyFromSymbol");
    repetition = 1.0f - (float)state.distinct / (float)state.total;
    note("a reply with distinct trigrams has zero repetition", repetition == 0.0f,
         (unsigned int)(repetition * 1000.0f), 0u);
}

typedef struct quality_ring {
    unsigned char *device, *host;
} quality_ring;

static void ring_open(quality_ring *ring, unsigned int replaying)
{
    size_t bytes = AOTX_QUALITY_TEST_SLOTS * AOTX_SLOT_BYTES;
    if (ring->device == 0) {
        aotx_check_runtime(cudaMalloc(&ring->device, bytes), "cudaMalloc");
        ring->host = (unsigned char *)malloc(bytes);
    }
    aotx_check_runtime(cudaMemset(ring->device, 0, bytes), "cudaMemset");
    aotx_seam_state seam = {};
    seam.dev.base = ring->device;
    seam.dev.slot_count = AOTX_QUALITY_TEST_SLOTS;
    seam.dev.mask = AOTX_QUALITY_TEST_SLOTS - 1u;
    seam.boot_id = 0xa02604ull;
    seam.replaying = replaying;
    unsigned long long tick = AOTX_QUALITY_TEST_TICK;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof seam),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_time_tick, &tick, sizeof tick),
                       "cudaMemcpyToSymbol");
}

static unsigned int ring_read(quality_ring *ring, aotx_quality_body *body,
                              unsigned int max)
{
    size_t bytes = AOTX_QUALITY_TEST_SLOTS * AOTX_SLOT_BYTES;
    aotx_check_runtime(cudaMemcpy(ring->host, ring->device, bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    unsigned int count = 0u;
    heads_good = 0u;
    for (unsigned int i = 0u; i < AOTX_QUALITY_TEST_SLOTS; ++i) {
        const aotx_record_header *head =
            (const aotx_record_header *)(ring->host + (size_t)i * AOTX_SLOT_BYTES);
        if (head->magic == AOTX_WIRE_MAGIC && head->type == AOTX_REC_QUALITY && count < max) {
            memcpy(&body[count], (const unsigned char *)head + AOTX_HEADER_BYTES,
                   sizeof body[count]);
            heads_good += head->cls == AOTX_CLASS_B && head->body_len == sizeof body[count]
                       && head->writer == AOTX_WRITER_AGENT_BASE + body[count].agent;
            count += 1u;
        }
    }
    return count;
}

__global__ void quality_off_script(unsigned int count)
{
    unsigned int agent = threadIdx.x;
    if (agent < count) aotx_quality_end(agent);
}

__global__ void quality_turn_script(unsigned int count)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count) return;
    aotx_quality_slot *state = &aotx_quality_state[agent];
    state->active = 1u;
    state->ended = 1u;
    state->total = 10u;
    state->distinct = 4u;
    state->active_guard[0] = 0.25f;
    state->active_guard[1] = -0.5f;
    state->active_guard_loaded = 1u;
    aotx_agents.agent[agent].turn = agent + 1u;
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    const char *yes = "CANNOT HELP WITH THAT. More text.";
    const char *no = "Here is a short answer.";
    const char *text = (agent & 1u) == 0u ? yes : no;
    gear->reply_len = 0u;
    while (text[gear->reply_len] != '\0') gear->reply_len += 1u;
    for (unsigned int i = 0u; i < gear->reply_len; ++i) gear->reply[i] = text[i];
    gear->out_tokens = 7u;
    gear->limit_end = (agent & 1u);
    aotx_seqs.slot[agent].limit = 8u;
}

static int body_takes(const aotx_quality_body *body)
{
    return body->agent < 64u && body->tokens <= body->limit && body->refusal <= 1u
        && isfinite(body->repetition) && body->repetition >= 0.0f && body->repetition <= 1.0f
        && (body->flags & ~0x0fu) == 0u && body->reserved == 0u
        && (((body->flags & 1u) != 0u) || body->coherence_prompt == 0.0f)
        && (((body->flags & 2u) != 0u) || body->coherence_turn == 0.0f);
}

static void turn_check(quality_ring *ring, unsigned int count)
{
    CLEAR_SYMBOL(aotx_quality_state, sizeof(aotx_quality_slot) * AOTX_SLOTS);
    CLEAR_SYMBOL(aotx_tool_embed, sizeof(aotx_tool_embed_batch));
    CLEAR_SYMBOL(aotx_agent_gear, sizeof(aotx_agent_work) * AOTX_SLOTS);
    ring_open(ring, 0u);
    quality_turn_script<<<1, AOTX_SLOTS>>>(count);
    aotx_quality_turn<<<AOTX_SLOTS, AOTX_QUALITY_PHRASES>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_quality_body body[AOTX_SLOTS];
    unsigned int records = ring_read(ring, body, AOTX_SLOTS);
    unsigned int right = 0u, refusal = 0u;
    for (unsigned int i = 0u; i < records; ++i) {
        int good = body[i].tokens == 7u && body[i].limit == 8u
                && (body[i].flags & 3u) == 0u && (body[i].flags & 8u) != 0u
                && body[i].coherence_prompt == 0.0f && body[i].coherence_turn == 0.0f
                && fabsf(body[i].repetition - 0.6f) < 1.0e-5f && body_takes(&body[i]);
        right += good ? 1u : 0u;
        refusal += body[i].refusal == ((body[i].agent & 1u) == 0u) ? 1u : 0u;
    }
    char label[96];
    snprintf(label, sizeof label, "one valid record with absent coherence for %u agents", count);
    note(label, records == count && right == count && heads_good == count, right, count);
    snprintf(label, sizeof label, "case-folded refusal phrases match for %u agents", count);
    note(label, refusal == count, refusal, count);
    aotx_quality_counts counts_out;
    aotx_check_runtime(cudaMemcpyFromSymbol(&counts_out, aotx_quality_count,
                                            sizeof counts_out), "cudaMemcpyFromSymbol");
    if (count == 1u) note("the one-agent turn is not late", counts_out.late == 0u,
                          counts_out.late, 0u);
    ring_open(ring, 0u);
    quality_off_script<<<1, AOTX_SLOTS>>>(count);
    aotx_quality_turn<<<AOTX_SLOTS, AOTX_QUALITY_PHRASES>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    records = ring_read(ring, body, AOTX_SLOTS);
    snprintf(label, sizeof label, "a turn with the quality mark clear writes none at %u", count);
    note(label, records == 0u, records, 0u);
    ring_open(ring, 1u);
    quality_turn_script<<<1, AOTX_SLOTS>>>(count);
    aotx_quality_turn<<<AOTX_SLOTS, AOTX_QUALITY_PHRASES>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    records = ring_read(ring, body, AOTX_SLOTS);
    snprintf(label, sizeof label, "a replay writes no quality record at %u", count);
    note(label, records == 0u, records, 0u);
}

__global__ void quality_vectors(unsigned int count, unsigned int row, unsigned int fresh)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count) return;
    aotx_quality_slot *state = &aotx_quality_state[agent];
    if (fresh != 0u) {
        state->pending = 1u;
        state->turn += 1u;
        state->tokens = 4u;
        state->limit = 8u;
        state->held_total = 0u;
        state->held_distinct = 0u;
        state->flags = 0u;
        state->row[0] = AOTX_QUALITY_ROW_RUN;
        state->row[1] = AOTX_QUALITY_ROW_WAIT;
        state->prompt_valid = 0u;
    } else {
        state->row[1] = AOTX_QUALITY_ROW_RUN;
    }
    state->place[row] = agent;
    aotx_tool_embed.kind[agent] = 2u * agent + row + 1u;
    aotx_tool_embed.live[agent] = 0u;
    for (unsigned int i = 0u; i < 4u; ++i) {
        aotx_tool_embed.vector[agent * 4u + i] = (i == agent % 4u) ? 1.0f : 0.0f;
    }
    if (agent == 0u) {
        aotx_tool_embed.ready = 1u;
        aotx_tool_embed.width = 4u;
        aotx_tool_embed.seqs = count;
    }
}

static void cosine_check(quality_ring *ring, unsigned int count)
{
    CLEAR_SYMBOL(aotx_quality_state, sizeof(aotx_quality_slot) * AOTX_SLOTS);
    CLEAR_SYMBOL(aotx_tool_embed, sizeof(aotx_tool_embed_batch));
    aotx_quality_body body[AOTX_SLOTS];
    for (unsigned int turn = 0u; turn < 2u; ++turn) {
        ring_open(ring, 0u);
        quality_vectors<<<1, AOTX_SLOTS>>>(count, 0u, 1u);
        aotx_quality_turn<<<AOTX_SLOTS, AOTX_QUALITY_PHRASES>>>();
        quality_vectors<<<1, AOTX_SLOTS>>>(count, 1u, 0u);
        aotx_quality_turn<<<AOTX_SLOTS, AOTX_QUALITY_PHRASES>>>();
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        unsigned int records = ring_read(ring, body, AOTX_SLOTS);
        unsigned int right = 0u;
        for (unsigned int i = 0u; i < records; ++i) {
            int flags = (turn == 0u) ? ((body[i].flags & 3u) == 1u)
                                     : ((body[i].flags & 3u) == 3u);
            right += flags && body[i].coherence_prompt > 0.9f
                   && body[i].coherence_prompt <= 1.0f
                   && ((body[i].flags & 2u) == 0u
                       || (body[i].coherence_turn >= -1.0f
                           && body[i].coherence_turn <= 1.0f)) ? 1u : 0u;
        }
        char label[96];
        snprintf(label, sizeof label, "present cosines are in range on turn %u at %u", turn + 1u,
                 count);
        note(label, records == count && right == count, right, count);
    }
}

/* The script of a turn that made a tool call. The agent waits in the tool state with the
 * result in hand, and the two rows of the turn wait. The slot holds the pages the case
 * names. The shape of the embedding role gives the rows a page need. With the stale mark
 * set, the first row holds an ask that a release of the slot took the pages of. */
__global__ void quality_tool_script(unsigned int count, unsigned int held, unsigned int stale)
{
    unsigned int agent = threadIdx.x;
    if (agent >= AOTX_SLOTS) return;
    aotx_kvl_shape *shape = &aotx_model_space[AOTX_MODEL_EMBEDDING].shape;
    if (agent == 0u) {
        aotx_kvl_make(shape, 4u, 2u, 64u);
        aotx_tool_embed.ready = 1u;
        aotx_tool_embed.role = AOTX_MODEL_EMBEDDING;
        aotx_tool_embed.width = 4u;
        aotx_sched.held = 0ull;
        aotx_kv.made = 0u;
        aotx_kv.served = 0u;
        aotx_kv.refused = 0u;
    }
    aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_NONE;
    aotx_tool_embed.place[agent] = AOTX_SLOTS;
    aotx_kv.count[agent] = held;
    aotx_quality_slot *state = &aotx_quality_state[agent];
    state->row[0] = AOTX_QUALITY_ROW_NONE;
    state->row[1] = AOTX_QUALITY_ROW_NONE;
    aotx_agents.agent[agent].state = AOTX_AGENT_STATE_IDLE;
    if (agent >= count) return;
    aotx_agents.agent[agent].state = AOTX_AGENT_STATE_TOOL;
    state->pending = 1u;
    state->waited = 0u;
    state->turn = agent + 1u;
    state->tokens = 4u;
    state->limit = 8u;
    state->held_total = 0u;
    state->held_distinct = 0u;
    state->flags = 0u;
    state->prompt_valid = 0u;
    state->row[0] = AOTX_QUALITY_ROW_WAIT;
    state->row[1] = AOTX_QUALITY_ROW_WAIT;
    state->asked[0] = (stale != 0u) ? aotx_kvl_pages(shape, 20u) : 0u;
    state->asked[1] = 0u;
    aotx_tool_gear.count[AOTX_SLOTS + 2u * agent] = 20u;
    aotx_tool_gear.count[AOTX_SLOTS + 2u * agent + 1u] = 40u;
}

/* The service of the page queue, as the host glue gives it: every slot takes the pages. */
__global__ void quality_tool_serve(unsigned int pages)
{
    unsigned int agent = threadIdx.x;
    if (agent >= AOTX_SLOTS) return;
    aotx_kv.count[agent] = pages;
    if (agent == 0u) aotx_kv.served = aotx_kv.made;
}

/* The pass of the batch: each row that runs this tick takes a vector at its place, as the
 * embedding pass writes it. */
__global__ void quality_tool_land(unsigned int count, unsigned int row)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count) return;
    aotx_quality_slot *state = &aotx_quality_state[agent];
    if (state->row[row] != AOTX_QUALITY_ROW_RUN) return;
    unsigned int place = state->place[row];
    for (unsigned int i = 0u; i < 4u; ++i) {
        aotx_tool_embed.vector[place * 4u + i] = (i == agent % 4u) ? 1.0f : 0.0f;
    }
}

/* The script of a tool text that waits for the batch with a stale ask. The text of the
 * slot has tokens, no page is held, and the ask stands at the need. The query embedding
 * of a transcript at a long history is such a text. */
__global__ void quality_text_script(unsigned int count)
{
    unsigned int agent = threadIdx.x;
    if (agent >= AOTX_SLOTS) return;
    const aotx_kvl_shape *shape = &aotx_model_space[AOTX_MODEL_EMBEDDING].shape;
    aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_NONE;
    aotx_quality_state[agent].row[0] = AOTX_QUALITY_ROW_NONE;
    aotx_quality_state[agent].row[1] = AOTX_QUALITY_ROW_NONE;
    aotx_kv.count[agent] = 0u;
    if (agent == 0u) {
        aotx_kv.made = 0u;
        aotx_kv.served = 0u;
    }
    if (agent >= count) return;
    aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_WAIT;
    aotx_tool_embed.asked[agent] = aotx_kvl_pages(shape, 20u);
    aotx_tool_gear.count[agent] = 20u;
}

__global__ void quality_wait_script(unsigned int count, unsigned int bound, unsigned int *out)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count) return;
    if (bound != 0u) aotx_quality_state[agent].waited = AOTX_QUALITY_WAIT_TICKS;
    out[agent] = (unsigned int)aotx_quality_wait(agent);
}

/* Give the shape of the embedding role, the page counts, the queue marks and the agent
 * states back. The later cases then read the loaded model and a clean table. */
__global__ void quality_tool_reset(void)
{
    unsigned int agent = threadIdx.x;
    if (agent >= AOTX_SLOTS) return;
    aotx_kv.count[agent] = 0u;
    aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_NONE;
    aotx_agents.agent[agent].state = AOTX_AGENT_STATE_IDLE;
    if (agent == 0u) {
        aotx_kvl_make(&aotx_model_space[AOTX_MODEL_EMBEDDING].shape, 0u, 0u, 0u);
        aotx_kv.made = 0u;
        aotx_kv.served = 0u;
    }
}

static unsigned int rows_in(const aotx_quality_slot *state, unsigned int count,
                            unsigned int row, unsigned int wanted)
{
    unsigned int right = 0u;
    for (unsigned int a = 0u; a < count; ++a) right += (state[a].row[row] == wanted) ? 1u : 0u;
    return right;
}

/* A turn that made a tool call, with the result in hand. The rows run while the agent
 * waits in the tool state. A row whose pages went away asks again. The line is written
 * at that turn end and not at the next one. */
static void tool_turn_check(quality_ring *ring, unsigned int count)
{
    aotx_quality_slot *state = (aotx_quality_slot *)malloc(sizeof *state * AOTX_SLOTS);
    aotx_kv_table *table = (aotx_kv_table *)malloc(sizeof *table);
    aotx_kvl_shape shape;
    unsigned int *device_out = 0, waits[AOTX_SLOTS], seen[AOTX_SLOTS];
    char label[96];
    aotx_check_runtime(cudaMalloc(&device_out, sizeof waits), "cudaMalloc");
    CLEAR_SYMBOL(aotx_quality_state, sizeof(aotx_quality_slot) * AOTX_SLOTS);
    CLEAR_SYMBOL(aotx_tool_embed, sizeof(aotx_tool_embed_batch));
    CLEAR_SYMBOL(aotx_tool_gear, sizeof(aotx_tool_work));
    CLEAR_SYMBOL(aotx_agent_gear, sizeof(aotx_agent_work) * AOTX_SLOTS);
    ring_open(ring, 0u);
    /* The stale ask: no page is held and the ask stands at the need. The plan asks again
     * for the need of the first row of every agent. The rows still wait. */
    quality_tool_script<<<1, AOTX_SLOTS>>>(count, 0u, 1u);
    aotx_tool_plan<<<1, AOTX_SLOTS>>>(0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_kv, sizeof *table), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(state, aotx_quality_state, sizeof *state * AOTX_SLOTS),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&shape, aotx_model_space, sizeof shape,
                       (size_t)AOTX_MODEL_EMBEDDING * sizeof(aotx_model_work)
                       + offsetof(aotx_model_work, shape)), "cudaMemcpyFromSymbol");
    unsigned int need = aotx_kvl_pages(&shape, 20u), asked = 0u;
    memset(seen, 0, sizeof seen);
    for (unsigned int i = 0u; i < table->made && i < AOTX_KV_QUEUE_MAX; ++i) {
        const aotx_kv_entry *entry = &table->queue[i];
        if (entry->agent < count && entry->pages == need && seen[entry->agent] == 0u) {
            seen[entry->agent] = 1u;
            asked += 1u;
        }
    }
    snprintf(label, sizeof label, "a row whose pages went away asks again at %u", count);
    note(label, need > 0u && table->made == count && asked == count
         && rows_in(state, count, 0u, AOTX_QUALITY_ROW_WAIT) == count, asked, count);
    /* The agent waits for the rows while they run. */
    quality_wait_script<<<1, AOTX_SLOTS>>>(count, 0u, device_out);
    aotx_check_runtime(cudaMemcpy(waits, device_out, sizeof waits, cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int right = 0u;
    for (unsigned int a = 0u; a < count; ++a) right += waits[a];
    snprintf(label, sizeof label, "a tool result waits for the rows of its turn at %u", count);
    note(label, right == count, right, count);
    /* The service gives the pages. Each round is one tick. The plan takes the rows the
     * row budget holds, the pass lands them, and the node writes the lines whose two
     * rows are done. The agent stays in the tool state throughout. */
    quality_tool_serve<<<1, AOTX_SLOTS>>>(AOTX_KV_PAGES_EACH);
    unsigned int rounds = 0u, left = count;
    while (left != 0u && rounds < 16u) {
        aotx_tool_plan<<<1, AOTX_SLOTS>>>(0ull);
        quality_tool_land<<<1, AOTX_SLOTS>>>(count, 0u);
        quality_tool_land<<<1, AOTX_SLOTS>>>(count, 1u);
        aotx_quality_turn<<<AOTX_SLOTS, AOTX_QUALITY_PHRASES>>>();
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_check_runtime(cudaMemcpyFromSymbol(state, aotx_quality_state, sizeof *state * AOTX_SLOTS),
                           "cudaMemcpyFromSymbol");
        left = 0u;
        for (unsigned int a = 0u; a < count; ++a) left += state[a].pending;
        rounds += 1u;
    }
    quality_wait_script<<<1, AOTX_SLOTS>>>(count, 0u, device_out);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(waits, device_out, sizeof waits, cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_quality_body body[AOTX_SLOTS];
    unsigned int records = ring_read(ring, body, AOTX_SLOTS);
    right = 0u;
    for (unsigned int i = 0u; i < records; ++i) {
        right += ((body[i].flags & 1u) != 0u && body[i].coherence_prompt > 0.9f
                  && body[i].turn == body[i].agent + 1u && body_takes(&body[i])) ? 1u : 0u;
    }
    unsigned int released = 0u;
    for (unsigned int a = 0u; a < count; ++a) released += (state[a].pending == 0u && waits[a] == 0u) ? 1u : 0u;
    snprintf(label, sizeof label, "a turn with a tool result writes its line at its end at %u (%u ticks)", count, rounds);
    note(label, records == count && right == count && released == count && heads_good == count
         && rounds < AOTX_QUALITY_WAIT_TICKS, right, count);
    /* The bound: an agent whose rows cannot run does not wait past it. */
    quality_tool_script<<<1, AOTX_SLOTS>>>(count, 0u, 0u);
    quality_wait_script<<<1, AOTX_SLOTS>>>(count, 1u, device_out);
    aotx_check_runtime(cudaMemcpy(waits, device_out, sizeof waits, cudaMemcpyDeviceToHost), "cudaMemcpy");
    right = 0u;
    for (unsigned int a = 0u; a < count; ++a) right += (waits[a] == 0u) ? 1u : 0u;
    snprintf(label, sizeof label, "the wait for the rows ends at the tick bound at %u", count);
    note(label, right == count, right, count);
    /* A tool text with a stale ask asks again the same way, and joins after the service. */
    quality_text_script<<<1, AOTX_SLOTS>>>(count);
    aotx_tool_plan<<<1, AOTX_SLOTS>>>(0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_kv, sizeof *table), "cudaMemcpyFromSymbol");
    asked = 0u;
    memset(seen, 0, sizeof seen);
    for (unsigned int i = 0u; i < table->made && i < AOTX_KV_QUEUE_MAX; ++i) {
        const aotx_kv_entry *entry = &table->queue[i];
        if (entry->agent < count && entry->pages == need && seen[entry->agent] == 0u) {
            seen[entry->agent] = 1u;
            asked += 1u;
        }
    }
    quality_tool_serve<<<1, AOTX_SLOTS>>>(AOTX_KV_PAGES_EACH);
    aotx_tool_plan<<<1, AOTX_SLOTS>>>(0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int text_state[AOTX_SLOTS];
    aotx_check_runtime(cudaMemcpyFromSymbol(text_state, aotx_tool_embed, sizeof text_state,
                       offsetof(aotx_tool_embed_batch, state)), "cudaMemcpyFromSymbol");
    /* The row budget of one pass takes the texts in slot order. One tick therefore joins
     * at most the texts of 20 tokens that fill 512 rows; the rest join in later ticks. */
    unsigned int fit = (count < AOTX_MODEL_MAX_TOKENS / 20u) ? count : AOTX_MODEL_MAX_TOKENS / 20u;
    right = 0u;
    for (unsigned int a = 0u; a < count; ++a) right += (text_state[a] == AOTX_TOOL_EMBED_RUN) ? 1u : 0u;
    snprintf(label, sizeof label, "a tool text whose pages went away asks again at %u", count);
    note(label, table->made == count && asked == count && right == fit, asked + right, count + fit);
    quality_tool_reset<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaFree(device_out);
    free(state);
    free(table);
}

__global__ void quality_live_script(void)
{
    const char *text = "The harbor report lists calm water and a clear eastern channel.";
    aotx_quality_slot *state = &aotx_quality_state[0];
    state->pending = 1u;
    state->turn = 1u;
    state->tokens = 12u;
    state->limit = 16u;
    state->row[0] = AOTX_QUALITY_ROW_WAIT;
    state->row[1] = AOTX_QUALITY_ROW_WAIT;
    while (text[state->length[0]] != '\0') state->length[0] += 1u;
    state->length[1] = state->length[0];
    for (unsigned int i = 0u; i < state->length[0]; ++i) {
        state->text[0][i] = (unsigned char)text[i];
        state->text[1][i] = (unsigned char)text[i];
    }
}

static void live_embedding_check(const char *models)
{
    char manifest[1024];
    snprintf(manifest, sizeof manifest, "%s/manifest.jsonl", models);
    if (access(manifest, R_OK) != 0) {
        printf("quality: the live embedding case is skipped because the store is absent\n");
        skipped += 1u;
        return;
    }

    CUdevice device;
    CUcontext context;
    aotx_mem_map map = {};
    aotx_seam_rings rings = {};
    aotx_pump pump = {};
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    unsigned long long boot_id = 0xa0260401ull;
    int opened = aotx_mem_reserve(&map) == 0 && aotx_seam_open(&rings, boot_id) == 0;
    if (!opened) {
        note("the live embedding map and rings open", 0, 0u, 1u);
        return;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    aotx_seam_bind_bulk(&rings, map.scratch, AOTX_BULK_STAGE_BYTES);
    if (aotx_test_catalog_setup() != 0 || aotx_boot_models(models, "embedding", 0) != 0
        || aotx_pump_build(&pump, 0ull, 1u) != 0 || pump.embed == 0u) {
        note("the live embedding pass opens", 0, 0u, 1u);
        aotx_pump_close(&pump);
        aotx_boot_models_release();
        aotx_seam_close(&rings);
        aotx_mem_release(&map);
        return;
    }

    CLEAR_SYMBOL(aotx_quality_state, sizeof(aotx_quality_slot) * AOTX_SLOTS);
    quality_live_script<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_quality_slot state = {};
    unsigned int ticks = 0u;
    do {
        aotx_pump_tick(&pump);
        ticks += 1u;
        aotx_check_runtime(cudaMemcpyFromSymbol(&state, aotx_quality_state, sizeof state),
                           "cudaMemcpyFromSymbol");
    } while (state.pending != 0u && ticks < 20u);

    quality_ring ring = {};
    ring.device = (unsigned char *)map.ring;
    ring.host = (unsigned char *)malloc(AOTX_QUALITY_TEST_SLOTS * AOTX_SLOT_BYTES);
    aotx_quality_body body = {};
    unsigned int records = ring_read(&ring, &body, 1u);
    int good = records == 1u && state.pending == 0u && (body.flags & 1u) != 0u
            && body.coherence_prompt > 0.9f && body.coherence_prompt <= 1.0f
            && body_takes(&body);
    note("the live equal texts have prompt coherence above 0.9", good,
         (unsigned int)(body.coherence_prompt * 1000.0f), 900u);
    free(ring.host);
    aotx_pump_close(&pump);
    aotx_boot_models_release();
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
}

int main(int argc, char **argv)
{
    if (argc < 2 || aotx_quality_load(argv[1]) != 0) {
        printf("quality: the refusal phrase fixture did not load\n");
        return 1;
    }
    quality_ring ring = {};
    const unsigned int counts[2] = { 1u, AOTX_SLOTS };
    trigram_check();
    for (unsigned int i = 0u; i < 2u; ++i) {
        fill_check(counts[i]);
        plan_check(counts[i]);
        turn_check(&ring, counts[i]);
        cosine_check(&ring, counts[i]);
        tool_turn_check(&ring, counts[i]);
    }
    cudaFree(ring.device);
    free(ring.host);
    live_embedding_check((argc > 2) ? argv[2] : "models");
    printf("quality: %u cases, %u bad, %u skipped, at 1 and %u\n", cases, bad, skipped,
           AOTX_SLOTS);
    return bad == 0u ? 0 : 1;
}
