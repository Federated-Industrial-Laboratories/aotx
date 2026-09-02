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
    }
    cudaFree(ring.device);
    free(ring.host);
    live_embedding_check((argc > 2) ? argv[2] : "models");
    printf("quality: %u cases, %u bad, %u skipped, at 1 and %u\n", cases, bad, skipped,
           AOTX_SLOTS);
    return bad == 0u ? 0 : 1;
}
