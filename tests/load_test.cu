/* Purpose: Check model command judgement, placement and replay state.
 * Owns: The model line fixtures and the placement probes.
 * Launch shape: One thread judges lines and one grid checks model records.
 * Lifetime: One run of the check program. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "cli/cli.cuh"
#include "disk/modelfile/modelfile.h"
#include "mem/mem.cuh"
#include "model/load.cuh"
#include "model/roles.h"
#include "sched/sched.cuh"
#include "seam/seam.cuh"

static unsigned int aotx_load_checks;
static unsigned int aotx_load_failed;

static void aotx_load_check(int good, const char *text)
{
    aotx_load_checks += 1u;
    if (!good) {
        aotx_load_failed += 1u;
        printf("load: FAILED %s\n", text);
    }
}

__global__ void aotx_load_line(const unsigned char *text, unsigned int length)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        aotx_cli_line(text, length, aotx_time_tick);
    }
}

__global__ void aotx_load_sequences(unsigned int count, unsigned int role,
                                    unsigned int state)
{
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at < AOTX_SLOTS) {
        aotx_seq *seq = &aotx_seqs.slot[at];
        seq->state = (at < count) ? state : AOTX_SEQ_STATE_FREE;
        seq->role = role;
        seq->prompt = at + 3u;
        seq->held = at + 7u;
        seq->sampled = at + 11u;
        seq->draw = 0x1000ull + at;
    }
    if (at == 0u) {
        aotx_seqs.live = (state == AOTX_SEQ_STATE_FREE) ? 0u : count;
    }
}

__global__ void aotx_load_commit_test(unsigned long long tick)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        aotx_model_load_commit(tick);
    }
}

typedef struct aotx_load_probe {
    unsigned int stalls;
    unsigned int models;
    unsigned int reason;
    unsigned int writer;
    aotx_model_body body;
} aotx_load_probe;

__global__ void aotx_load_records(aotx_load_probe *probe)
{
    unsigned long long lane = (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned long long stride = (unsigned long long)(gridDim.x * blockDim.x);
    for (unsigned long long seq = lane + 1ull; seq <= aotx_seam.dev.tail; seq += stride) {
        const volatile aotx_record_header *header = aotx_cli_slot(seq);
        if (aotx_cli_holds(header, seq, AOTX_REC_STALL)) {
            const volatile aotx_stall_body *body =
                (const volatile aotx_stall_body *)((const volatile unsigned char *)header
                                                    + AOTX_HEADER_BYTES);
            atomicAdd(&probe->stalls, 1u);
            if ((body->held_count & AOTX_STALL_MODEL_LOAD) != 0ull) {
                atomicExch(&probe->reason, 1u);
            }
        }
        if (aotx_cli_holds(header, seq, AOTX_REC_MODEL)) {
            unsigned int at = atomicAdd(&probe->models, 1u);
            if (at == 0u) {
                probe->writer = header->writer;
                const volatile unsigned char *body =
                    (const volatile unsigned char *)header + AOTX_HEADER_BYTES;
                unsigned char *out = (unsigned char *)&probe->body;
                for (unsigned int b = 0u; b < sizeof(aotx_model_body); ++b) {
                    out[b] = body[b];
                }
            }
        }
    }
}

static void aotx_load_parse(const char *line)
{
    size_t length = strlen(line);
    unsigned char *text = NULL;
    aotx_check_runtime(cudaMalloc(&text, length), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(text, line, length, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_load_line<<<1, 1>>>(text, (unsigned int)length);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaFree(text);
}

static aotx_model_load_state aotx_load_state(void)
{
    aotx_model_load_state state;
    aotx_check_runtime(cudaMemcpyFromSymbol(&state, aotx_model_load, sizeof state),
                       "cudaMemcpyFromSymbol");
    return state;
}

static void aotx_load_place(aotx_pump *pump)
{
    int state = aotx_model_load_step(pump);
    aotx_load_check(state == 0, "a live placement returns success");
}

static int aotx_load_bad_manifest(const char *models, char *dir, size_t dir_bytes)
{
    char form[] = "/tmp/aotx-load-XXXXXX";
    char *made = mkdtemp(form);
    if (made == NULL || strlen(made) + 1u > dir_bytes) {
        return 1;
    }
    strcpy(dir, made);
    char source[1024];
    char manifest[1024];
    char model[1024];
    snprintf(source, sizeof source, "%s/manifest.jsonl", models);
    snprintf(manifest, sizeof manifest, "%s/manifest.jsonl", dir);
    snprintf(model, sizeof model, "%s/qwen3-reranker-0.6b-q8_0.gguf", dir);
    FILE *in = fopen(source, "r");
    FILE *out = fopen(manifest, "w");
    if (in == NULL || out == NULL) {
        if (in != NULL) fclose(in);
        if (out != NULL) fclose(out);
        return 1;
    }
    char line[2048];
    int wrote = 0;
    while (fgets(line, sizeof line, in) != NULL) {
        if (strstr(line, "\"name\":\"reranker\"") == NULL) {
            continue;
        }
        char *digest = strstr(line, "\"sha256\":\"");
        if (digest != NULL) {
            digest += strlen("\"sha256\":\"");
            digest[0] = (digest[0] == '0') ? '1' : '0';
            fputs(line, out);
            wrote = 1;
        }
    }
    fclose(in);
    fclose(out);
    char given[1024];
    snprintf(given, sizeof given, "%s/qwen3-reranker-0.6b-q8_0.gguf", models);
    char *target = realpath(given, NULL);
    if (target == NULL) {
        return 1;
    }
    if (!wrote || symlink(target, model) != 0) {
        free(target);
        return 1;
    }
    free(target);
    return 0;
}

static void aotx_load_judgement(void)
{
    aotx_model_load_state before = aotx_load_state();
    aotx_load_parse("model load");
    aotx_load_parse("model load nowhere reranker");
    aotx_load_parse("model load reranker nowhere");
    aotx_load_parse("model load reranker embedding");
    aotx_model_load_state after = aotx_load_state();
    aotx_load_check(after.pending_count == before.pending_count,
                    "refused lines do not enter the queue");

    aotx_load_sequences<<<1, AOTX_SLOTS>>>(1u, AOTX_MODEL_RERANKER,
                                            AOTX_SEQ_STATE_DECODE);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_load_parse("model load reranker reranker");
    after = aotx_load_state();
    aotx_load_check(after.pending_count == before.pending_count,
                    "a role with one live sequence refuses a load");
    aotx_load_sequences<<<1, AOTX_SLOTS>>>(0u, AOTX_MODEL_RERANKER,
                                            AOTX_SEQ_STATE_FREE);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

int main(int argc, char **argv)
{
    const char *models = (argc > 1) ? argv[1] : "../models";
    char manifest[1024];
    snprintf(manifest, sizeof manifest, "%s/manifest.jsonl", models);
    if (access(manifest, R_OK) != 0) {
        printf("load: 0 failed, skipped because %s is not present\n", manifest);
        return 0;
    }

    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    aotx_pump pump;
    memset(&pump, 0, sizeof pump);
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device),
                      "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, 0x10ad13ull) != 0) {
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, 0x10ad13ull);
    aotx_seam_bind_bulk(&rings, map.scratch, AOTX_BULK_STAGE_BYTES);
    aotx_seam_note_boot<<<1, 1>>>(0ull, 0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    if (aotx_boot_models(models, AOTX_PROFILE_LANGUAGE, 0) != 0) {
        printf("load: the first model did not load\n");
        return 1;
    }
    aotx_check_runtime(cudaStreamCreateWithFlags(&pump.stream, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags");
    aotx_check_runtime(cudaEventCreateWithFlags(&pump.event, cudaEventDisableTiming),
                       "cudaEventCreateWithFlags");

    aotx_load_judgement();

    aotx_load_sequences<<<1, AOTX_SLOTS>>>(AOTX_SLOTS, AOTX_PROFILE_LANGUAGE_ROLE,
                                            AOTX_SEQ_STATE_DECODE);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_seq_table *before = (aotx_seq_table *)malloc(sizeof *before);
    aotx_seq_table *after = (aotx_seq_table *)malloc(sizeof *after);
    aotx_check_runtime(cudaMemcpyFromSymbol(before, aotx_seqs, sizeof *before),
                       "cudaMemcpyFromSymbol");
    aotx_load_parse("model load reranker reranker");
    aotx_model_load_state queued = aotx_load_state();
    aotx_load_check(queued.pending_count == 1u, "an accepted line enters the queue");
    aotx_load_place(&pump);
    aotx_check_runtime(cudaMemcpyFromSymbol(after, aotx_seqs, sizeof *after),
                       "cudaMemcpyFromSymbol");
    aotx_load_check(memcmp(before, after, sizeof *before) == 0,
                    "every sequence survives a load of another role");
    free(before);
    free(after);

    aotx_model_desc desc[AOTX_MODEL_ROLES];
    aotx_check_runtime(cudaMemcpyFromSymbol(desc, aotx_model, sizeof desc),
                       "cudaMemcpyFromSymbol");
    aotx_load_check(desc[AOTX_MODEL_RERANKER].layers != 0u,
                    "placement builds the reranker descriptor");
    aotx_model_load_state placed = aotx_load_state();
    aotx_load_check(placed.placed_bytes >= 600ull * 1024ull * 1024ull,
                    "the placement count includes the reranker megabytes");

    aotx_load_commit_test<<<1, 1>>>(17ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_load_probe *probe = NULL;
    aotx_load_probe got;
    memset(&got, 0, sizeof got);
    aotx_check_runtime(cudaMalloc(&probe, sizeof *probe), "cudaMalloc");
    aotx_check_runtime(cudaMemset(probe, 0, sizeof *probe), "cudaMemset");
    aotx_load_records<<<8, 128>>>(probe);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&got, probe, sizeof got, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_load_check(got.stalls >= 2u && got.reason != 0u,
                    "placement writes the two stall marks with its reason");
    aotx_load_check(got.models == 1u && got.writer == AOTX_WRITER_CONSOLE
                    && got.body.tick == 17ull,
                    "the next commit writes the model placement");
    cudaFree(probe);

    aotx_load_sequences<<<1, AOTX_SLOTS>>>(1u, AOTX_PROFILE_LANGUAGE_ROLE,
                                            AOTX_SEQ_STATE_DECODE);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    placed = aotx_load_state();
    const char *language_line = (AOTX_PROFILE_LANGUAGE_ROLE == AOTX_MODEL_LANGUAGE)
                              ? "model load language language-q4"
                              : "model load language-q4 language";
    aotx_load_parse(language_line);
    queued = aotx_load_state();
    aotx_load_check(queued.pending_count == 0u,
                    "a language load refuses while its reply runs");
    aotx_load_sequences<<<1, AOTX_SLOTS>>>(0u, AOTX_PROFILE_LANGUAGE_ROLE,
                                            AOTX_SEQ_STATE_FREE);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    unsigned long long bytes_before = placed.placed_bytes;
    aotx_load_parse(language_line);
    aotx_load_place(&pump);
    placed = aotx_load_state();
    if (AOTX_PROFILE_LANGUAGE_ROLE == AOTX_MODEL_LANGUAGE_Q4) {
        aotx_load_check(placed.refused > queued.refused
                        && placed.placed_bytes == bytes_before,
                        "the larger language file refuses when the region is full");
    } else {
        aotx_check_runtime(cudaMemcpyFromSymbol(desc, aotx_model, sizeof desc),
                           "cudaMemcpyFromSymbol");
        aotx_load_check(placed.placed_bytes > bytes_before
                        && desc[AOTX_MODEL_LANGUAGE].weight_type == AOTX_TENSOR_Q4_0,
                        "the named language file replaces the active descriptor");
    }

    char bad_dir[1024];
    if (aotx_load_bad_manifest(models, bad_dir, sizeof bad_dir) != 0
        || aotx_model_load_open(bad_dir, "", aotx_mem_weights_held()) != 0) {
        aotx_load_check(0, "the digest fixture opens");
    } else {
        aotx_load_parse("model load reranker reranker");
        aotx_load_place(&pump);
        aotx_model_load_state bad = aotx_load_state();
        aotx_load_check(bad.refused == 1u && bad.placed_bytes == 0ull,
                        "a digest difference refuses before a byte is counted");
        char path[1200];
        snprintf(path, sizeof path, "%s/qwen3-reranker-0.6b-q8_0.gguf", bad_dir);
        unlink(path);
        snprintf(path, sizeof path, "%s/manifest.jsonl", bad_dir);
        unlink(path);
        rmdir(bad_dir);
    }

    cudaEventDestroy(pump.event);
    cudaStreamDestroy(pump.stream);
    aotx_boot_models_release();
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    cuDevicePrimaryCtxRelease(device);
    printf("load: %u checks, %u failed, N=1 and N=%u\n", aotx_load_checks,
           aotx_load_failed, (unsigned int)AOTX_SLOTS);
    return (aotx_load_failed == 0u) ? 0 : 1;
}
