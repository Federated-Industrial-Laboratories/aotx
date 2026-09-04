/* Purpose: Move the tensors of a model file into the weights region.
 * Owns: The pinned buffers, the copy stream, and the placement cursor of one load.
 * Launch shape: Host glue only; the table build holds the one kernel.
 * Lifetime: From the open of the buffers to the close at the end of the load. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "mem/mem.cuh"

extern "C" {
#include "disk/modelfile/modelfile.h"
}

/* Two pinned buffers of 64 MB. One buffer takes the next piece from the disk while the
 * copy of the piece before it runs. The disk and the link therefore work together. */
#define AOTX_WEIGHTS_CHUNK   (64ull * 1024ull * 1024ull)
#define AOTX_WEIGHTS_ALIGN   256ull
#define AOTX_WEIGHTS_BLOCK   128u

/* The events of the two buffers, and one more for the table build. */
#define AOTX_WEIGHTS_EVENTS  3u

/* The tensor types the system reads. A tensor of any other type stays in the file, and the
 * load names it with its numeric type. */
static int aotx_weights_readable(unsigned int type)
{
    return type == AOTX_TENSOR_F32 || type == AOTX_TENSOR_F16
        || type == AOTX_TENSOR_Q4_0 || type == AOTX_TENSOR_Q8_0
        || type == AOTX_TENSOR_Q4_K || type == AOTX_TENSOR_Q5_K
        || type == AOTX_TENSOR_Q6_K;
}

/* Report one tensor that the load leaves in the file. */
static void aotx_weights_left(const aotx_tensor_info *info, unsigned int *left)
{
    fprintf(stderr, "the tensor %s of type %u is not a type this system reads\n",
            info->name, info->type);
    *left += 1u;
}

/* One load runs at a time, so the buffers and the stream are the state of this file. The
 * stream does not block on the stream of the runtime. Every wait of this path is therefore
 * a wait on one event and never a wait on the device. */
static struct {
    unsigned char *host[2];
    cudaEvent_t done[AOTX_WEIGHTS_EVENTS];
    cudaStream_t stream;
    unsigned int turn;
} aotx_weights;

int aotx_model_weights_open(void)
{
    memset(&aotx_weights, 0, sizeof aotx_weights);
    aotx_check_runtime(cudaStreamCreateWithFlags(&aotx_weights.stream,
                                                 cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags");
    for (unsigned int i = 0u; i < 2u; ++i) {
        aotx_check_runtime(cudaMallocHost((void **)&aotx_weights.host[i],
                                          (size_t)AOTX_WEIGHTS_CHUNK), "cudaMallocHost");
    }
    for (unsigned int i = 0u; i < AOTX_WEIGHTS_EVENTS; ++i) {
        aotx_check_runtime(cudaEventCreate(&aotx_weights.done[i]), "cudaEventCreate");
    }
    return 0;
}

void aotx_model_weights_close(void)
{
    for (unsigned int i = 0u; i < AOTX_WEIGHTS_EVENTS; ++i) {
        cudaEventSynchronize(aotx_weights.done[i]);
        cudaEventDestroy(aotx_weights.done[i]);
    }
    for (unsigned int i = 0u; i < 2u; ++i) {
        cudaFreeHost(aotx_weights.host[i]);
    }
    cudaStreamDestroy(aotx_weights.stream);
    memset(&aotx_weights, 0, sizeof aotx_weights);
}

/* Move one tensor from the file into the region. The read of the next piece waits for the
 * copy which last held that buffer. No piece of the file waits for the whole device. */
static int aotx_weights_stream(const aotx_modelfile *file, unsigned long long from,
                               unsigned long long bytes, unsigned long long place)
{
    unsigned long long base = aotx_mem_weights_base();
    for (unsigned long long at = 0ull; at < bytes; at += AOTX_WEIGHTS_CHUNK) {
        unsigned long long piece = bytes - at;
        if (piece > AOTX_WEIGHTS_CHUNK) {
            piece = AOTX_WEIGHTS_CHUNK;
        }
        unsigned int slot = aotx_weights.turn & 1u;
        aotx_check_runtime(cudaEventSynchronize(aotx_weights.done[slot]),
                           "cudaEventSynchronize");
        if (aotx_modelfile_read(file, from + at, piece, aotx_weights.host[slot]) != 0) {
            return 1;
        }
        aotx_check_runtime(cudaMemcpyAsync((void *)(base + place + at),
                                           aotx_weights.host[slot], (size_t)piece,
                                           cudaMemcpyHostToDevice, aotx_weights.stream),
                           "cudaMemcpyAsync");
        aotx_check_runtime(cudaEventRecord(aotx_weights.done[slot], aotx_weights.stream),
                           "cudaEventRecord");
        aotx_weights.turn += 1u;
    }
    return 0;
}

/* Give the tensors of one file to the device table. The copies of the file and this build
 * run on one stream, so the build reads what the copies wrote. The wait for the build is a
 * wait on its event. */
static int aotx_weights_table(const aotx_tensor_info *infos, const unsigned long long *place,
                              unsigned int held, unsigned int model)
{
    void *table = 0;
    void *at = 0;
    aotx_check_runtime(cudaMalloc(&table, (size_t)held * sizeof *infos), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&at, (size_t)held * sizeof *place), "cudaMalloc");
    aotx_check_runtime(cudaMemcpyAsync(table, infos, (size_t)held * sizeof *infos,
                                       cudaMemcpyHostToDevice, aotx_weights.stream),
                       "cudaMemcpyAsync");
    aotx_check_runtime(cudaMemcpyAsync(at, place, (size_t)held * sizeof *place,
                                       cudaMemcpyHostToDevice, aotx_weights.stream),
                       "cudaMemcpyAsync");
    aotx_mem_tensor_replace<<<1, 1, 0, aotx_weights.stream>>>(model);
    unsigned int blocks = (held + AOTX_WEIGHTS_BLOCK - 1u) / AOTX_WEIGHTS_BLOCK;
    aotx_mem_tensor_add<<<blocks, AOTX_WEIGHTS_BLOCK, 0, aotx_weights.stream>>>(
        table, (const unsigned long long *)at, held, model);
    aotx_check_runtime(cudaEventRecord(aotx_weights.done[2], aotx_weights.stream),
                       "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(aotx_weights.done[2]), "cudaEventSynchronize");
    cudaFree(table);
    cudaFree(at);

    /* A table which is full refuses the tensors that come after it, and the load stops. */
    unsigned int state[2] = { 0u, 0u };
    aotx_check_runtime(cudaMemcpyFromSymbol(state, aotx_mem_tensor_list, sizeof state, 0),
                       "cudaMemcpyFromSymbol");
    if (state[1] != 0u) {
        fprintf(stderr, "the tensor table refused %u tensors\n", state[1]);
        return 1;
    }
    return 0;
}

int aotx_model_weights_place(struct aotx_modelfile *file, unsigned int model,
                             unsigned long long *cursor, unsigned int *placed,
                             unsigned int *left)
{
    unsigned long long count = aotx_modelfile_tensor_count(file);
    aotx_tensor_info *infos = (aotx_tensor_info *)malloc((size_t)count * sizeof *infos);
    unsigned long long *place = (unsigned long long *)malloc((size_t)count * sizeof *place);
    if (infos == NULL || place == NULL) {
        free(infos);
        free(place);
        return 1;
    }
    unsigned int held = 0u;
    int bad = 0;
    unsigned long long next = *cursor;
    for (unsigned long long i = 0ull; i < count && bad == 0; ++i) {
        aotx_tensor_info info;
        if (aotx_modelfile_tensor(file, i, &info) != 0) {
            bad = 1;
            break;
        }
        if (!aotx_weights_readable(info.type)) {
            aotx_weights_left(&info, left);
            continue;
        }
        unsigned long long at = (next + AOTX_WEIGHTS_ALIGN - 1ull)
                              / AOTX_WEIGHTS_ALIGN * AOTX_WEIGHTS_ALIGN;
        if (at > AOTX_MEM_WEIGHTS_BYTES || info.bytes > AOTX_MEM_WEIGHTS_BYTES - at) {
            fprintf(stderr, "the weights region is full at %llu bytes\n", at + info.bytes);
            bad = 1;
            break;
        }
        infos[held] = info;
        place[held] = at;
        held += 1u;
        next = at + info.bytes;
    }
    /* The full file is measured before the first byte moves. A region-full refusal
     * therefore leaves the model that runs and the placement cursor unchanged. */
    for (unsigned int i = 0u; i < held && bad == 0; ++i) {
        if (aotx_mem_weights_map(place[i], infos[i].bytes) != 0) {
            fprintf(stderr, "the weights region is full at %llu bytes\n",
                    place[i] + infos[i].bytes);
            bad = 1;
            break;
        }
        if (aotx_weights_stream(file, infos[i].offset, infos[i].bytes, place[i]) != 0) {
            fprintf(stderr, "the tensor %s did not read\n", infos[i].name);
            bad = 1;
            break;
        }
    }
    if (bad == 0 && held != 0u) {
        bad = aotx_weights_table(infos, place, held, model);
    }
    free(infos);
    free(place);
    if (bad == 0) {
        *cursor = next;
        *placed += held;
    }
    return bad;
}

int aotx_model_weights_reload(struct aotx_modelfile *file, unsigned int model,
                              unsigned int *placed, unsigned int *left,
                              unsigned long long *bytes)
{
    unsigned long long count = aotx_modelfile_tensor_count(file);
    aotx_tensor_info *infos = (aotx_tensor_info *)malloc((size_t)count * sizeof *infos);
    unsigned long long *place = (unsigned long long *)malloc((size_t)count * sizeof *place);
    if (infos == NULL || place == NULL) {
        free(infos);
        free(place);
        return 1;
    }
    unsigned int held = 0u;
    int bad = 0;
    for (unsigned long long i = 0ull; i < count && bad == 0; ++i) {
        aotx_tensor_info info;
        bad = aotx_modelfile_tensor(file, i, &info);
        if (bad == 0 && !aotx_weights_readable(info.type)) {
            aotx_weights_left(&info, left);
        } else if (bad == 0) {
            infos[held++] = info;
        }
    }
    void *table = 0;
    unsigned long long *on_place = 0;
    unsigned int *on_found = 0;
    if (bad == 0 && held != 0u) {
        aotx_check_runtime(cudaMalloc(&table, (size_t)held * sizeof *infos), "cudaMalloc");
        aotx_check_runtime(cudaMalloc(&on_place, (size_t)held * sizeof *place), "cudaMalloc");
        aotx_check_runtime(cudaMalloc(&on_found, sizeof *on_found), "cudaMalloc");
        aotx_check_runtime(cudaMemcpy(table, infos, (size_t)held * sizeof *infos,
                                      cudaMemcpyHostToDevice), "cudaMemcpy");
        aotx_check_runtime(cudaMemset(on_place, 0, (size_t)held * sizeof *place),
                           "cudaMemset");
        aotx_check_runtime(cudaMemset(on_found, 0, sizeof *on_found), "cudaMemset");
        unsigned int blocks = (held + AOTX_WEIGHTS_BLOCK - 1u) / AOTX_WEIGHTS_BLOCK;
        aotx_mem_tensor_places<<<blocks, AOTX_WEIGHTS_BLOCK>>>(
            table, held, model, on_place, on_found);
        unsigned int found = 0u;
        aotx_check_runtime(cudaMemcpy(place, on_place, (size_t)held * sizeof *place,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(&found, on_found, sizeof found, cudaMemcpyDeviceToHost),
                           "cudaMemcpy");
        if (found != held) {
            fprintf(stderr, "the resident tensor layout differs at %u of %u tensors\n",
                    found, held);
            bad = 1;
        }
    }
    for (unsigned int i = 0u; i < held && bad == 0; ++i) {
        bad = aotx_weights_stream(file, infos[i].offset, infos[i].bytes, place[i]);
        *bytes += infos[i].bytes;
    }
    cudaFree(table);
    cudaFree(on_place);
    cudaFree(on_found);
    free(infos);
    free(place);
    if (bad == 0) {
        *placed += held;
    }
    return bad;
}
