/* Purpose: Check the decode of the tick graph: its records, its states and its rate.
 * Owns: The test consumer of the host ring and the counts of the cases.
 * Launch shape: One thread for each sequence slot; the consumer is a host thread.
 * Lifetime: One run of the test program. */
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "mem/mem.cuh"
#include "model/decode_state.cuh"
#include "model/graph_host.h"
#include "model/roles.h"
#include "sched/sched.cuh"
#include "seam/seam.cuh"

#include "decode_drain.h"

/* Ticks one run may take before the check gives up on it. */
#define AOTX_DECODE_TEST_TICKS   4000u

/* Bytes of one take of the reply text. */
#define AOTX_DECODE_TEST_TEXT    512u

/* The reply of a check run, and the reply of a rate run. */
#define AOTX_DECODE_TEST_REPLY   12u
#define AOTX_DECODE_TEST_RATE    16u

/* The reply of a run under the sanitizer. The sanitizer makes a kernel far slower than the
 * watchdog allows. A run with AOTX_SANITIZER set therefore takes a short reply, one role
 * and no rate case. The batch stays at 64 sequences, because a race between two sequences
 * of one batch is not visible at one sequence. */
#define AOTX_DECODE_TEST_LOW     2u

static unsigned int aotx_decode_test_reply = AOTX_DECODE_TEST_REPLY;
static int aotx_decode_test_lowered = 0;

/* Sequences of the batch case under the sanitizer. Racecheck reads every access to shared
 * memory, and the batch of 64 did not end in 20 minutes on this machine. A race between two
 * sequences of one batch is still visible at eight. */
static unsigned int aotx_decode_test_batch = AOTX_SEQ_SLOTS;

/* The budget of a decode tick at 16 sequences, in nanoseconds. One tick reads every weight
 * of the model one time. At 16 rows the tensor core product pays for the whole 128 rows
 * of its tile. That is about 58 ms for the 4.02 billion weights of this file at the 17.8
 * TFLOPS the product gives. The budget is twice that figure. It covers the plan, the
 * commit, the flush, the page service and the launch of every node of the graph. */
#define AOTX_DECODE_TEST_BUDGET  120000000ull

/* The sample of a rate run, from the model card of the language file. */
#define AOTX_DECODE_TEST_TOP_K   20u
#define AOTX_DECODE_TEST_TOP_P   0.95f
#define AOTX_DECODE_TEST_HEAT    0.7f

/* Open one sequence for each prompt, from one thread, as the command layer does. */
__global__ void aotx_decode_test_open(const int *ids, const unsigned int *start,
                                      const unsigned int *count, unsigned int first,
                                      unsigned int seqs, unsigned int role,
                                      unsigned int limit, unsigned long long seed,
                                      unsigned int top_k, float top_p, float heat,
                                      unsigned int *bad)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int s = first; s < first + seqs; ++s) {
        if (aotx_seq_open(s, role, ids + start[s], count[s], limit, seed + s, top_k,
                          top_p, heat, aotx_time_tick) != 0) {
            *bad += 1u;
        }
    }
}

/* Mark a run of sequences to end at the next commit. */
__global__ void aotx_decode_test_stop(unsigned int seqs)
{
    unsigned int slot = threadIdx.x;
    if (slot < seqs) {
        aotx_seq_stop(slot);
    }
}

/* Apply a run of token records, in order, as the serial apply of a restore does. */
__global__ void aotx_decode_test_apply(const aotx_token_body *body, unsigned int count,
                                       unsigned int *bad)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        if (aotx_seq_apply(&body[i]) != 0) {
            *bad += 1u;
        }
    }
}

/* Put every slot back in the free state and ask for its pages back. */
__global__ void aotx_decode_test_clear(void)
{
    unsigned int slot = threadIdx.x;
    if (slot >= AOTX_SEQ_SLOTS) {
        return;
    }
    aotx_seqs.slot[slot].state = AOTX_SEQ_STATE_FREE;
    aotx_seqs.slot[slot].flags = 0u;
    aotx_seqs.slot[slot].prompt = 0u;
    aotx_seqs.slot[slot].sampled = 0u;
    aotx_seqs.slot[slot].held = 0u;
    aotx_seq_kept[slot] = 0u;
    aotx_seq_shown[slot] = 0u;
    aotx_model_seen[slot] = 0u;
    aotx_model_draw[slot] = 0u;
    aotx_decode.rows[slot] = 0u;
    aotx_kv_release(slot);
    if (slot == 0u) {
        aotx_seqs.live = 0u;
        aotx_seqs.refused = 0u;
    }
}

/* Take the reply bytes of one slot, as the console does. */
__global__ void aotx_decode_test_text(unsigned int slot, unsigned char *out,
                                      unsigned int max, unsigned int *length)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        *length = aotx_seq_take_text(slot, out, max);
    }
}

typedef struct aotx_decode_test_gear {
    int *ids;
    unsigned int *start;
    unsigned int *count;
    unsigned int *bad;
    aotx_token_body *body;
    unsigned char *text;
    unsigned int *length;
    aotx_seq slot[AOTX_SEQ_SLOTS];
} aotx_decode_test_gear;

static void *aotx_decode_test_take(unsigned long long bytes)
{
    void *at = 0;
    aotx_check_runtime(cudaMalloc(&at, (size_t)bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemset(at, 0, (size_t)bytes), "cudaMemset");
    return at;
}

/* Read the sequence table of the device. The token lists stay on the device. */
static void aotx_decode_test_read(aotx_decode_test_gear *gear)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(gear->slot, aotx_seqs, sizeof gear->slot, 0,
                                            cudaMemcpyDeviceToHost),
                       "cudaMemcpyFromSymbol");
}

/* Slots that are not free. */
static unsigned int aotx_decode_test_live(aotx_decode_test_gear *gear)
{
    unsigned int live = 0u;
    aotx_decode_test_read(gear);
    for (unsigned int s = 0u; s < AOTX_SEQ_SLOTS; ++s) {
        if (gear->slot[s].state != AOTX_SEQ_STATE_FREE) {
            live += 1u;
        }
    }
    return live;
}

/* Run ticks until every slot is free again, or until the tick bound. */
static unsigned int aotx_decode_test_drive(aotx_pump *pump, aotx_decode_test_gear *gear,
                                           unsigned int bound)
{
    unsigned int ticks = 0u;
    while (ticks < bound) {
        aotx_pump_tick(pump);
        ticks += 1u;
        if (aotx_decode_test_live(gear) == 0u) {
            break;
        }
    }
    return ticks;
}

/* Open one sequence for each of the first seqs prompts and give the tick count back. */
static void aotx_decode_test_ask(aotx_decode_test_gear *gear, unsigned int seqs,
                                 unsigned int role, unsigned int limit,
                                 unsigned int top_k, float top_p, float heat)
{
    aotx_decode_test_open<<<1, 1>>>(gear->ids, gear->start, gear->count, 0u, seqs, role,
                                    limit, 0x51EEDull, top_k, top_p, heat, gear->bad);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

static void aotx_decode_test_reset(aotx_pump *pump)
{
    aotx_decode_test_clear<<<1, AOTX_SEQ_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_kv_serve(&pump->kv, 0);
}

#include "decode_cases.h"

int main(int argc, char **argv)
{
    const char *models = (argc > 1) ? argv[1] : "models";
    const char *fixtures = (argc > 2) ? argv[2] : "tests/fixtures/tokenizer";
    char path[1024];
    /* The sanitizer gate sets AOTX_SANITIZER. The run then takes a short reply, one role,
     * and no rate case, because the sanitizer makes every kernel far slower. */
    const char *sanitizer = getenv("AOTX_SANITIZER");
    aotx_decode_test_lowered = (sanitizer != NULL) ? 1 : 0;
    if (aotx_decode_test_lowered != 0) {
        aotx_decode_test_reply = AOTX_DECODE_TEST_LOW;
        if (strcmp(sanitizer, "racecheck") == 0) {
            aotx_decode_test_batch = 8u;
        }
        printf("decode: AOTX_SANITIZER is %s: %u reply tokens, one role, no rate case, "
               "a batch of %u\n", sanitizer, aotx_decode_test_reply,
               aotx_decode_test_batch);
    }
    unsigned int applied = 0u;
    unsigned int failed = 0u;
    snprintf(path, sizeof path, "%s/manifest.jsonl", models);
    if (access(path, R_OK) != 0) {
        printf("decode: 0 cases, 0 bad, 16 skipped, no model files in %s\n", models);
        return 0;
    }
    aotx_decode_test_prompt *prompt =
        (aotx_decode_test_prompt *)calloc(1, sizeof *prompt);
    snprintf(path, sizeof path, "%s/golden-language.ids", fixtures);
    if (aotx_decode_test_prompts(path, prompt) != 0) {
        printf("decode: 0 cases, 0 bad, 16 skipped, no prompt list in %s\n", fixtures);
        return 0;
    }

    /* The four bit file joins the run when the model record names it. */
    snprintf(path, sizeof path, "%s/manifest.jsonl", models);
    FILE *record = fopen(path, "rb");
    char text[65536];
    size_t bytes = (record != NULL) ? fread(text, 1u, sizeof text - 1u, record) : 0u;
    text[bytes] = '\0';
    if (record != NULL) {
        fclose(record);
    }
    int four_bit = (strstr(text, "\"name\":\"language-q4\"") != NULL);
    const char *roles = four_bit ? "language,language-q4" : "language";
    /* The sanitizer holds device memory of its own beside the weights. A run under it takes
     * the four bit file when the record names it, because that file is the smaller one. */
    if (aotx_decode_test_lowered != 0 && four_bit) {
        roles = "language-q4";
    }

    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    aotx_pump pump;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    unsigned long long boot_id = 0x0DEC0DEull;
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("decode: the map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    aotx_seam_bind_bulk(&rings, map.scratch, AOTX_BULK_STAGE_BYTES);
    aotx_seam_note_boot<<<1, 1>>>(0ull, 0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double began = aotx_decode_test_now();
    if (aotx_boot_models(models, roles, 0) != 0) {
        printf("decode: the model files did not load\n");
        return 1;
    }
    printf("decode: roles %s loaded in %.1f s\n", roles, aotx_decode_test_now() - began);

    aotx_decode_test_drain *drain = (aotx_decode_test_drain *)calloc(1, sizeof *drain);
    drain->token = (aotx_decode_test_token *)calloc(AOTX_DECODE_TEST_RECORDS,
                                                    sizeof *drain->token);
    drain->map = rings.host_map;
    drain->data = rings.host_map + sizeof(aotx_host_ring_preamble);
    drain->data_bytes = AOTX_HOST_RING_DATA_BYTES;
    drain->mask = AOTX_HOST_RING_DATA_BYTES - 1ull;
    drain->boot_id = boot_id;
    pthread_t thread;
    pthread_create(&thread, NULL, aotx_decode_test_reader, drain);

    aotx_decode_test_gear *gear = (aotx_decode_test_gear *)calloc(1, sizeof *gear);
    gear->ids = (int *)aotx_decode_test_take(sizeof prompt->ids);
    gear->start = (unsigned int *)aotx_decode_test_take(sizeof prompt->start);
    gear->count = (unsigned int *)aotx_decode_test_take(sizeof prompt->count);
    gear->bad = (unsigned int *)aotx_decode_test_take(sizeof(unsigned int));
    gear->text = (unsigned char *)aotx_decode_test_take(1024ull);
    gear->length = (unsigned int *)aotx_decode_test_take(sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(gear->ids, prompt->ids, sizeof prompt->ids,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->start, prompt->start, sizeof prompt->start,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->count, prompt->count, sizeof prompt->count,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    printf("decode: %u prompts of %u to %u tokens\n", prompt->rows, prompt->count[0],
           prompt->count[prompt->rows - 1u]);

    const unsigned int order[2] = {
        (aotx_decode_test_lowered != 0 && four_bit) ? AOTX_MODEL_LANGUAGE_Q4
                                                    : AOTX_MODEL_LANGUAGE,
        AOTX_MODEL_LANGUAGE_Q4
    };
    unsigned int roles_run = (four_bit && aotx_decode_test_lowered == 0) ? 2u : 1u;
    for (unsigned int r = 0u; r < roles_run; ++r) {
        unsigned int role = order[r];
        if (r > 0u) {
            aotx_model_shut(order[r - 1u]);
            if (aotx_model_open(role, AOTX_MODEL_MAX_TOKENS) != 0) {
                printf("decode: the graph of role %u did not capture\n", role);
                return 1;
            }
        }
        if (aotx_pump_build(&pump, 0ull, 1u) != 0 || pump.decode == 0u) {
            printf("decode: the tick graph did not take the decode\n");
            return 1;
        }
        unsigned int vocab = 0u;
        aotx_check_runtime(cudaMemcpyFromSymbol(&vocab, aotx_model, sizeof vocab,
                                                (size_t)role * sizeof(aotx_model_desc)
                                                + offsetof(aotx_model_desc, vocab)),
                           "cudaMemcpyFromSymbol");
        printf("decode: role %s, %u nodes in the tick graph, %u from the decode\n",
               aotx_role_name[role], pump.nodes, pump.decode_nodes);
        /* The run case at one sequence and at 64 covers every node of the decode. A run
         * under the sanitizer takes those and the idle case, and leaves the rest out. */
        aotx_decode_test_case_idle(&pump, drain, &applied, &failed);
        aotx_decode_test_case_run(&pump, gear, drain, 1u, role, &applied, &failed);
        aotx_decode_test_case_run(&pump, gear, drain, aotx_decode_test_batch, role,
                                  &applied, &failed);
        if (aotx_decode_test_lowered != 0) {
            printf("decode: the stop, text, long, replay, feed, rate and graph cases were "
                   "left out\n");
            aotx_pump_close(&pump);
            continue;
        }
        aotx_decode_test_case_stop(&pump, gear, role, 1u, &applied, &failed);
        aotx_decode_test_case_stop(&pump, gear, role, AOTX_SEQ_SLOTS, &applied, &failed);
        aotx_decode_test_case_text(&pump, gear, role, 1u, &applied, &failed);
        aotx_decode_test_case_text(&pump, gear, role, AOTX_SEQ_SLOTS, &applied, &failed);
        aotx_decode_test_case_long(&pump, gear, prompt, role, &applied, &failed);
        aotx_decode_test_case_replay(&pump, gear, drain, role, vocab, 1u, &applied,
                                     &failed);
        aotx_decode_test_case_replay(&pump, gear, drain, role, vocab, AOTX_SEQ_SLOTS,
                                     &applied, &failed);
        aotx_decode_test_case_feed(&pump, gear, drain, prompt, &rings, boot_id, role,
                                   &applied, &failed);
        aotx_decode_test_table(&pump, gear, drain, role, aotx_role_name[role], &applied,
                               &failed);
        if (r == 0u) {
            aotx_decode_test_case_graph(&pump, gear, role, 1000u, &applied, &failed);
        }
        aotx_pump_close(&pump);
    }

    drain->stop = 1;
    pthread_join(thread, NULL);
    unsigned int refused = 0u;
    aotx_check_runtime(cudaMemcpy(&refused, gear->bad, sizeof refused,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    applied += 1u;
    if (refused != 0u || drain->lost != 0u || drain->bad != 0ull) {
        printf("decode: %u calls refused, %u records lost, %llu blocks bad\n", refused,
               drain->lost, drain->bad);
        failed += 1u;
    }
    printf("decode: %u token records and %u sequence events over %llu blocks\n",
           drain->taken, drain->events, drain->blocks);
    printf("decode: %u cases applied, %u failed\n", applied, failed);
    aotx_seam_finish(&rings);
    aotx_seam_close(&rings);
    return (failed == 0u) ? 0 : 1;
}
