/* Purpose: Compare the forward pass with the reference lists of the three model files.
 * Owns: The tokenizer buffers, the fixture rows and the counts of the cases.
 * Launch shape: The tokenizer kernels and the forward graph, at one sequence and at
 *                AOTX_SLOTS.
 * Lifetime: The program. */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "embed/embed.cuh"
#include "kvcache/kvcache.cuh"
#include "mem/mem.cuh"
#include "model/blocks.cuh"
#include "model/forward.cuh"
#include "model/roles.h"
#include "rerank/rerank.cuh"
#include "text/text.cuh"

#include "model_text.h"

#define AOTX_GATE_TOP      32u
#define AOTX_GATE_END      151643u
#define AOTX_GATE_RATE_LEN 128u

/* Every tolerance here is twice the largest figure seen on this machine. The error comes
 * from the half precision input of every projection over 28 or 36 layers. The reference
 * holds every value in single precision. The figures seen are 0.2712 for the top 32
 * distance, 0.65 for the whole row, and 0.0148 for the rank value. */
#define AOTX_GATE_L1       5.5e-1

/* The same measure for the four bit file, against the lists that file made. The largest
 * figure seen is 0.2589, and the bound is twice it. */
#define AOTX_GATE_L1_Q4    5.2e-1

/* The margin that makes a position clear. The largest whole row difference seen is 0.65.
 * A reference which separates its two largest logits by more than twice that cannot change
 * its largest logit through the number format alone. */
#define AOTX_GATE_MARGIN   1.3

/* The bound of the whole row comparison, twice the largest difference seen (0.65). */
#define AOTX_GATE_ROW      1.3
/* The cosine the design names for a pooled vector, and the bound of the gate. The largest
 * error seen is 1.133e-3, on the line of three tokens, and the bound is twice it. A short
 * line gives the number format of the activations less to work with than a long one. */
#define AOTX_GATE_FIRM     0.999
#define AOTX_GATE_COSINE   0.9977
#define AOTX_GATE_SCORE    0.03

static unsigned int aotx_gate_cases = 0u;
static unsigned int aotx_gate_bad = 0u;
static unsigned int aotx_gate_skip = 0u;

static void aotx_gate_note(const char *name, int good, const char *how, double value,
                           double bound)
{
    aotx_gate_cases += 1u;
    if (!good) {
        aotx_gate_bad += 1u;
    }
    printf("%-38s %-4s %s %.4f of %.4f\n", name, good ? "ok" : "BAD", how, value, bound);
}

static double aotx_gate_now(void)
{
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (double)at.tv_sec + (double)at.tv_nsec * 1e-9;
}

/* The buffers of one pass of the gate. */
typedef struct aotx_gate_gear {
    int *ids;
    unsigned int *offset;
    unsigned int *agent;
    float *logits;
    float *pooled;
    float *score;
} aotx_gate_gear;

/* Send one group of sequences through the pass. The tokens of the group go in one run. */
static unsigned int aotx_gate_pass(aotx_gate_gear *gear, aotx_kv_map *map, unsigned int role,
                                   unsigned int *const *ids, const unsigned int *counts,
                                   unsigned int first, unsigned int seqs, int want_logits,
                                   unsigned int *offset_out, int reset)
{
    int *run = (int *)malloc((size_t)AOTX_MODEL_MAX_TOKENS * sizeof(int));
    unsigned int *offset = (unsigned int *)malloc((size_t)(seqs + 1u) * sizeof(unsigned int));
    unsigned int *agent = (unsigned int *)malloc((size_t)seqs * sizeof(unsigned int));
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < seqs; ++i) {
        offset[i] = at;
        agent[i] = i;
        for (unsigned int p = 0u; p < counts[first + i]; ++p) {
            run[at + p] = (int)ids[first + i][p];
        }
        at += counts[first + i];
    }
    offset[seqs] = at;
    if (offset_out != 0) {
        memcpy(offset_out, offset, (size_t)(seqs + 1u) * sizeof(unsigned int));
    }
    aotx_check_runtime(cudaMemcpy(gear->ids, run, at * sizeof(int), cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->offset, offset, (seqs + 1u) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->agent, agent, seqs * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    if (reset != 0) {
        aotx_model_forget();
    }
    aotx_model_pages(role, gear->offset, seqs, gear->agent);
    aotx_kv_serve(map, 0);
    int state;
    if (role == AOTX_MODEL_RERANKER) {
        state = aotx_model_rerank(gear->ids, gear->offset, seqs, gear->agent, gear->score);
    } else {
        state = aotx_model_prefill(role, gear->ids, gear->offset, seqs, gear->agent,
                                   want_logits ? gear->logits : 0,
                                   (role == AOTX_MODEL_EMBEDDING) ? gear->pooled : 0);
    }
    if (state != 0) {
        printf("the pass of role %u did not run\n", role);
        exit(1);
    }
    free(run);
    free(offset);
    free(agent);
    return at;
}

/* The mean of every row of a sequence, normed and made a unit vector. The embedding head
 * takes the last row; this head takes the mean, and the gate must tell the two apart. */
__global__ void aotx_gate_mean(unsigned int role, float *out)
{
    __shared__ float part[AOTX_MODEL_ROW_THREADS];
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int r = blockIdx.x;
    if (r >= run->seqs) {
        return;
    }
    const float *weight = (const float *)aotx_block_tensor(work->weights, desc->output_norm);
    float sum[8];
    for (unsigned int i = 0u; i < 8u; ++i) {
        sum[i] = 0.0f;
    }
    unsigned int rows = run->offset[r + 1u] - run->offset[r];
    for (unsigned int t = run->offset[r]; t < run->offset[r + 1u]; ++t) {
        const float *row = work->resid + (unsigned long long)t * desc->hidden;
        float square = 0.0f;
        for (unsigned int d = threadIdx.x; d < desc->hidden; d += blockDim.x) {
            square += row[d] * row[d];
        }
        part[threadIdx.x] = square;
        __syncthreads();
        if (threadIdx.x == 0u) {
            for (unsigned int i = 1u; i < blockDim.x; ++i) {
                part[0] += part[i];
            }
        }
        __syncthreads();
        float scale = rsqrtf(part[0] / (float)desc->hidden + desc->rms_eps);
        for (unsigned int d = threadIdx.x, i = 0u; d < desc->hidden;
             d += blockDim.x, ++i) {
            sum[i] += scale * row[d] * weight[d];
        }
        __syncthreads();
    }
    float length = 0.0f;
    for (unsigned int d = threadIdx.x, i = 0u; d < desc->hidden; d += blockDim.x, ++i) {
        sum[i] /= (float)rows;
        length += sum[i] * sum[i];
    }
    part[threadIdx.x] = length;
    __syncthreads();
    if (threadIdx.x == 0u) {
        for (unsigned int i = 1u; i < blockDim.x; ++i) {
            part[0] += part[i];
        }
    }
    __syncthreads();
    float unit = (part[0] > 0.0f) ? rsqrtf(part[0]) : 0.0f;
    for (unsigned int d = threadIdx.x, i = 0u; d < desc->hidden; d += blockDim.x, ++i) {
        out[(unsigned long long)r * desc->hidden + d] = sum[i] * unit;
    }
}

/* One row of a reference list: the largest identity and the 32 largest logits. */
typedef struct aotx_gate_row {
    unsigned int best;
    unsigned int id[AOTX_GATE_TOP];
    float logit[AOTX_GATE_TOP];
} aotx_gate_row;

/* Read one reference list. The comment lines carry the token identities of the prompt. */
static unsigned int aotx_gate_rows(const char *path, aotx_gate_row *rows, unsigned int max,
                                   unsigned int *ids, unsigned int *count)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0u;
    }
    char *line = (char *)malloc(1u << 16);
    unsigned int held = 0u;
    *count = 0u;
    while (fgets(line, 1 << 16, file) != NULL) {
        if (line[0] == '#') {
            const char *walk = line + 1;
            while (*walk == ' ') {
                ++walk;
            }
            if (*walk >= '0' && *walk <= '9' && strchr(walk, ',') != NULL) {
                while (*walk != '\0' && *walk != '\n') {
                    ids[(*count)++] = (unsigned int)strtoul(walk, (char **)&walk, 10);
                    if (*walk == ',') {
                        ++walk;
                    }
                }
            }
            continue;
        }
        if (held >= max) {
            break;
        }
        const char *walk = line;
        unsigned int position = (unsigned int)strtoul(walk, (char **)&walk, 10);
        (void)position;
        rows[held].best = (unsigned int)strtoul(walk, (char **)&walk, 10);
        for (unsigned int i = 0u; i < AOTX_GATE_TOP; ++i) {
            rows[held].id[i] = (unsigned int)strtoul(walk, (char **)&walk, 10);
            if (*walk == ':') {
                ++walk;
            }
            rows[held].logit[i] = strtof(walk, (char **)&walk);
        }
        held += 1u;
    }
    fclose(file);
    free(line);
    return held;
}

/* The distance between two lists of 32 probabilities. Each list is the softmax over its own
 * 32 logits, so the two are compared over the same identities. */
static double aotx_gate_l1(const float *mine, const float *theirs)
{
    double first[AOTX_GATE_TOP];
    double second[AOTX_GATE_TOP];
    double sum_a = 0.0;
    double sum_b = 0.0;
    double top_a = mine[0];
    double top_b = theirs[0];
    for (unsigned int i = 1u; i < AOTX_GATE_TOP; ++i) {
        top_a = (mine[i] > top_a) ? mine[i] : top_a;
        top_b = (theirs[i] > top_b) ? theirs[i] : top_b;
    }
    for (unsigned int i = 0u; i < AOTX_GATE_TOP; ++i) {
        first[i] = exp((double)mine[i] - top_a);
        second[i] = exp((double)theirs[i] - top_b);
        sum_a += first[i];
        sum_b += second[i];
    }
    double gap = 0.0;
    for (unsigned int i = 0u; i < AOTX_GATE_TOP; ++i) {
        gap += fabs(first[i] / sum_a - second[i] / sum_b);
    }
    return gap;
}

/* Put one prompt in the chat wrap of the model file, with the escapes expanded. */
static void aotx_gate_wrap(const char *raw, unsigned int bytes, char *out, unsigned int max)
{
    char plain[4096];
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < bytes && at + 1u < sizeof plain; ++i) {
        if (raw[i] == '\\' && i + 1u < bytes
            && (raw[i + 1u] == 'n' || raw[i + 1u] == '\\')) {
            plain[at++] = (raw[i + 1u] == 'n') ? '\n' : '\\';
            i += 1u;
            continue;
        }
        plain[at++] = raw[i];
    }
    plain[at] = '\0';
    snprintf(out, max, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", plain);
}

/* The reference lists of the language model, position by position. */
static void aotx_gate_language(aotx_gate_text *text, aotx_gate_gear *gear, aotx_kv_map *map,
                               const char *dir, const aotx_model_desc *desc,
                               unsigned int role, const char *prefix, const char *label,
                               double bound, int extras)
{
    char path[512];
    char note[80];
    unsigned long long bytes = 0ull;
    snprintf(path, sizeof path, "%s/prompts.dat", dir);
    char *file = aotx_gate_slurp(path, &bytes);
    if (file == 0) {
        printf("the prompt file did not read\n");
        exit(1);
    }
    char *raw[16];
    unsigned int prompts = aotx_gate_lines(file, bytes, raw, 16u, 0);

    static char wrapped[16][8192];
    const char *ready[16];
    for (unsigned int i = 0u; i < prompts; ++i) {
        aotx_gate_wrap(raw[i], (unsigned int)strlen(raw[i]), wrapped[i], sizeof wrapped[i]);
        ready[i] = wrapped[i];
    }
    unsigned int *ids = (unsigned int *)malloc((size_t)prompts * AOTX_GATE_STRIDE
                                               * sizeof(unsigned int));
    unsigned int counts[16];
    aotx_gate_tokenize(text, ready, prompts, ids, counts);

    /* The reference lists carry the token identities of each prompt, so the tokenizer of
     * the device is compared with them before the logits are. */
    static aotx_gate_row rows[16][128];
    static unsigned int held[16];
    unsigned int wrong_tokens = 0u;
    for (unsigned int i = 0u; i < prompts; ++i) {
        unsigned int want[512];
        unsigned int want_count = 0u;
        snprintf(path, sizeof path, "%s/%s-%u.pos", dir, prefix, i);
        held[i] = aotx_gate_rows(path, rows[i], 128u, want, &want_count);
        if (held[i] == 0u) {
            printf("the reference list %s did not read\n", path);
            exit(1);
        }
        if (want_count != counts[i]) {
            wrong_tokens += 1u;
        } else {
            for (unsigned int p = 0u; p < want_count; ++p) {
                if (want[p] != ids[i * AOTX_GATE_STRIDE + p]) {
                    wrong_tokens += 1u;
                    break;
                }
            }
        }
    }
    snprintf(note, sizeof note, "%s prompt tokens against the list", label);
    aotx_gate_note(note, wrong_tokens == 0u, "wrong", (double)wrong_tokens, 0.0);

    unsigned int *rowptr[16];
    for (unsigned int i = 0u; i < prompts; ++i) {
        rowptr[i] = ids + i * AOTX_GATE_STRIDE;
    }
    unsigned int offset[17];
    unsigned int tokens = aotx_gate_pass(gear, map, role, rowptr, counts, 0u, prompts, 1,
                                         offset, 1);
    printf("%s gate: %u prompts, %u positions in one pass\n", label, prompts, tokens);

    float *row = (float *)malloc((size_t)desc->vocab * sizeof(float));
    unsigned int moved = 0u;
    unsigned int moved_clear = 0u;
    unsigned int clear = 0u;
    double worst_l1 = 0.0;
    double total_l1 = 0.0;
    unsigned int positions = 0u;
    for (unsigned int i = 0u; i < prompts; ++i) {
        for (unsigned int p = 0u; p < held[i] && p < counts[i]; ++p) {
            aotx_check_runtime(cudaMemcpy(row, gear->logits
                                          + (size_t)(offset[i] + p) * desc->vocab,
                                          (size_t)desc->vocab * sizeof(float),
                                          cudaMemcpyDeviceToHost), "cudaMemcpy");
            unsigned int best = 0u;
            for (unsigned int t = 1u; t < desc->vocab; ++t) {
                if (row[t] > row[best]) {
                    best = t;
                }
            }
            int wide = (rows[i][p].logit[0] - rows[i][p].logit[1]) > AOTX_GATE_MARGIN;
            clear += wide ? 1u : 0u;
            if (best != rows[i][p].best) {
                moved += 1u;
                moved_clear += wide ? 1u : 0u;
            }
            float mine[AOTX_GATE_TOP];
            for (unsigned int t = 0u; t < AOTX_GATE_TOP; ++t) {
                mine[t] = row[rows[i][p].id[t]];
            }
            double gap = aotx_gate_l1(mine, rows[i][p].logit);
            if (gap > worst_l1) {
                worst_l1 = gap;
            }
            total_l1 += gap;
            positions += 1u;
        }
    }
    printf("%s gate: %u positions compared, %u clear, %u of %u agree, mean distance "
           "%.5f\n", label, positions, clear, positions - moved, positions,
           total_l1 / (double)positions);
    snprintf(note, sizeof note, "%s largest logit where the two are apart", label);
    aotx_gate_note(note, moved_clear == 0u, "moved", (double)moved_clear, 0.0);
    snprintf(note, sizeof note, "%s top 32 distance of every position", label);
    aotx_gate_note(note, worst_l1 <= bound, "worst", worst_l1, bound);
    if (extras == 0) {
        free(row);
        free(ids);
        free(file);
        return;
    }

    /* Four prompts keep every logit of the last position, so the whole row is compared. */
    double worst_abs = 0.0;
    unsigned int full = 0u;
    float *want = (float *)malloc((size_t)desc->vocab * sizeof(float));
    for (unsigned int i = 0u; i < 4u && i < prompts; ++i) {
        snprintf(path, sizeof path, "%s/lm-%u-last.f32", dir, i);
        FILE *one = fopen(path, "rb");
        if (one == NULL) {
            continue;
        }
        size_t got = fread(want, sizeof(float), desc->vocab, one);
        fclose(one);
        if (got != desc->vocab) {
            continue;
        }
        aotx_check_runtime(cudaMemcpy(row, gear->logits
                                      + (size_t)(offset[i] + counts[i] - 1u) * desc->vocab,
                                      (size_t)desc->vocab * sizeof(float),
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        for (unsigned int t = 0u; t < desc->vocab; ++t) {
            double gap = fabs((double)row[t] - (double)want[t]);
            if (gap > worst_abs) {
                worst_abs = gap;
            }
        }
        full += 1u;
    }
    printf("%s gate: %u whole rows compared\n", label, full);
    aotx_gate_note("whole row of the last position", worst_abs <= AOTX_GATE_ROW,
                   "worst", worst_abs, AOTX_GATE_ROW);

    /* The gate must be able to fail. The same batch with a turn of the wrong period gives
     * another largest logit at most positions. */
    aotx_model_desc bad = *desc;
    bad.rope_theta = 10000.0f;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &bad, sizeof bad,
                                          (size_t)role * sizeof bad),
                       "cudaMemcpyToSymbol");
    aotx_gate_pass(gear, map, role, rowptr, counts, 0u, prompts, 1, offset, 1);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, desc, sizeof bad,
                                          (size_t)role * sizeof bad),
                       "cudaMemcpyToSymbol");
    unsigned int shifted = 0u;
    double shifted_l1 = 0.0;
    for (unsigned int i = 0u; i < prompts; ++i) {
        for (unsigned int p = 0u; p < held[i] && p < counts[i]; ++p) {
            aotx_check_runtime(cudaMemcpy(row, gear->logits
                                          + (size_t)(offset[i] + p) * desc->vocab,
                                          (size_t)desc->vocab * sizeof(float),
                                          cudaMemcpyDeviceToHost), "cudaMemcpy");
            unsigned int best = 0u;
            for (unsigned int t = 1u; t < desc->vocab; ++t) {
                if (row[t] > row[best]) {
                    best = t;
                }
            }
            if (best != rows[i][p].best
                && (rows[i][p].logit[0] - rows[i][p].logit[1]) > AOTX_GATE_MARGIN) {
                shifted += 1u;
            }
            float mine[AOTX_GATE_TOP];
            for (unsigned int t = 0u; t < AOTX_GATE_TOP; ++t) {
                mine[t] = row[rows[i][p].id[t]];
            }
            double gap = aotx_gate_l1(mine, rows[i][p].logit);
            if (gap > shifted_l1) {
                shifted_l1 = gap;
            }
        }
    }
    printf("%s gate: a wrong period moves %u clear positions of %u and gives a distance "
           "of %.5f\n", label, shifted, clear, shifted_l1);
    aotx_gate_note("a turn of the wrong period", shifted > 0u && shifted_l1 > bound,
                   "moved", (double)shifted, 0.0);
    free(row);
    free(want);
    free(ids);
    free(file);
}

/* One sequence which is longer than a pass goes through in pieces. The cache position of
 * the slot moves on with each piece, so the last piece attends over the whole sequence. */
static void aotx_gate_chunk(aotx_gate_gear *gear, aotx_kv_map *map, unsigned int role,
                            const unsigned int *ids, unsigned int count, float *out,
                            unsigned int width)
{
    int *run = (int *)malloc((size_t)AOTX_MODEL_MAX_TOKENS * sizeof(int));
    unsigned int offset[2];
    unsigned int agent = 0u;
    aotx_model_forget();
    for (unsigned int at = 0u; at < count; ) {
        unsigned int take = count - at;
        if (take > AOTX_MODEL_MAX_TOKENS) {
            take = AOTX_MODEL_MAX_TOKENS;
        }
        for (unsigned int p = 0u; p < take; ++p) {
            run[p] = (int)ids[at + p];
        }
        offset[0] = 0u;
        offset[1] = take;
        aotx_check_runtime(cudaMemcpy(gear->ids, run, take * sizeof(int),
                                      cudaMemcpyHostToDevice), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(gear->offset, offset, sizeof offset,
                                      cudaMemcpyHostToDevice), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(gear->agent, &agent, sizeof agent,
                                      cudaMemcpyHostToDevice), "cudaMemcpy");
        aotx_model_pages(role, gear->offset, 1u, gear->agent);
        aotx_kv_serve(map, 0);
        if (aotx_model_prefill(role, gear->ids, gear->offset, 1u, gear->agent, 0,
                               gear->pooled) != 0) {
            printf("a piece of a long sequence did not run\n");
            exit(1);
        }
        at += take;
    }
    aotx_check_runtime(cudaMemcpy(out, gear->pooled, (size_t)width * sizeof(float),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    free(run);
}

/* Send a list of sequences through the pass in groups. A group holds AOTX_SLOTS sequences
 * and at most the tokens of one pass. */
static void aotx_gate_groups(aotx_gate_gear *gear, aotx_kv_map *map, unsigned int role,
                             unsigned int *const *ids, const unsigned int *counts,
                             unsigned int count, unsigned int width, float *out)
{
    unsigned int first = 0u;
    while (first < count) {
        unsigned int seqs = 0u;
        unsigned int tokens = 0u;
        while (first + seqs < count && seqs < AOTX_SLOTS
               && tokens + counts[first + seqs] <= AOTX_MODEL_MAX_TOKENS) {
            tokens += counts[first + seqs];
            seqs += 1u;
        }
        if (seqs == 0u) {
            aotx_gate_chunk(gear, map, role, ids[first], counts[first],
                            out + (size_t)first * width, width);
            first += 1u;
            continue;
        }
        aotx_gate_pass(gear, map, role, ids, counts, first, seqs, 0, 0, 1);
        const float *from = (role == AOTX_MODEL_RERANKER) ? gear->score : gear->pooled;
        aotx_check_runtime(cudaMemcpy(out + (size_t)first * width, from,
                                      (size_t)seqs * width * sizeof(float),
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        first += seqs;
    }
}

static double aotx_gate_cosine(const float *a, const float *b, unsigned int width)
{
    double dot = 0.0;
    double left = 0.0;
    double right = 0.0;
    for (unsigned int i = 0u; i < width; ++i) {
        dot += (double)a[i] * b[i];
        left += (double)a[i] * a[i];
        right += (double)b[i] * b[i];
    }
    return dot / sqrt(left * right);
}

/* The reference vectors of the embedding model. */
static void aotx_gate_embedding(aotx_gate_text *text, aotx_gate_gear *gear, aotx_kv_map *map,
                                const char *dir, const aotx_model_desc *desc)
{
    char path[512];
    unsigned long long bytes = 0ull;
    snprintf(path, sizeof path, "%s/embed-lines.bin", dir);
    char *file = aotx_gate_slurp(path, &bytes);
    snprintf(path, sizeof path, "%s/embed.f32", dir);
    unsigned long long want_bytes = 0ull;
    char *want_raw = aotx_gate_slurp(path, &want_bytes);
    if (file == 0 || want_raw == 0) {
        printf("the embedding fixture did not read\n");
        exit(1);
    }
    char *line[AOTX_GATE_MAX];
    unsigned int lines = aotx_gate_lines(file, bytes, line, AOTX_GATE_MAX, 1);
    const float *want = (const float *)want_raw;
    unsigned int rows = (unsigned int)(want_bytes / (desc->hidden * sizeof(float)));
    if (rows < lines) {
        lines = rows;
    }

    unsigned int *ids = (unsigned int *)malloc((size_t)lines * AOTX_GATE_STRIDE
                                               * sizeof(unsigned int));
    unsigned int *counts = (unsigned int *)malloc((size_t)lines * sizeof(unsigned int));
    aotx_gate_tokenize(text, (const char **)line, lines, ids, counts);

    /* The model file asks for an end of text token after the text of every input. Every
     * line takes part, the lines that hold a control token as well. The reference reads a
     * control token as one token, which is what the device does. */
    unsigned int **rowptr = (unsigned int **)malloc((size_t)lines * sizeof(unsigned int *));
    unsigned int special = 0u;
    for (unsigned int i = 0u; i < lines; ++i) {
        rowptr[i] = ids + i * AOTX_GATE_STRIDE;
        for (unsigned int p = 0u; p < counts[i]; ++p) {
            if (rowptr[i][p] >= AOTX_GATE_END) {
                special += 1u;
                break;
            }
        }
        rowptr[i][counts[i]] = AOTX_GATE_END;
        counts[i] += 1u;
    }
    float *out = (float *)malloc((size_t)lines * desc->hidden * sizeof(float));
    aotx_gate_groups(gear, map, AOTX_MODEL_EMBEDDING, rowptr, counts, lines, desc->hidden,
                     out);
    double least = 1.0;
    unsigned int firm = 0u;
    for (unsigned int i = 0u; i < lines; ++i) {
        double cosine = aotx_gate_cosine(out + (size_t)i * desc->hidden,
                                         want + (size_t)i * desc->hidden, desc->hidden);
        if (cosine < least) {
            least = cosine;
        }
        if (cosine >= AOTX_GATE_FIRM) {
            firm += 1u;
        } else {
            printf("  line %u: %u tokens, cosine %.6f\n", i, counts[i], cosine);
        }
    }
    printf("embedding gate: %u lines, %u hold a control token, %u at %.3f or better\n",
           lines, special, firm, AOTX_GATE_FIRM);
    aotx_gate_note("vector cosine of every line", least >= AOTX_GATE_COSINE, "least", least,
                   AOTX_GATE_COSINE);

    /* One line alone must give the same vector as the same line in a group. */
    aotx_gate_pass(gear, map, AOTX_MODEL_EMBEDDING, rowptr, counts, 0u, 1u, 0, 0, 1);
    float *one = (float *)malloc((size_t)desc->hidden * sizeof(float));
    aotx_check_runtime(cudaMemcpy(one, gear->pooled, (size_t)desc->hidden * sizeof(float),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    double alone = aotx_gate_cosine(one, want, desc->hidden);
    aotx_gate_note("vector cosine of one line alone", alone >= AOTX_GATE_COSINE, "least",
                   alone, AOTX_GATE_COSINE);

    /* The gate must be able to fail. The mean of every row is another pooling, and it must
     * not pass the same bound. */
    unsigned int seqs = (lines > AOTX_SLOTS) ? AOTX_SLOTS : lines;
    unsigned int tokens = 0u;
    unsigned int taken = 0u;
    while (taken < seqs && tokens + counts[taken] <= AOTX_MODEL_MAX_TOKENS) {
        tokens += counts[taken];
        taken += 1u;
    }
    aotx_gate_pass(gear, map, AOTX_MODEL_EMBEDDING, rowptr, counts, 0u, taken, 0, 0, 1);
    float *mean = (float *)aotx_gate_take((unsigned long long)taken * desc->hidden
                                          * sizeof(float));
    aotx_gate_mean<<<taken, AOTX_MODEL_ROW_THREADS>>>(AOTX_MODEL_EMBEDDING, mean);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    float *copy = (float *)malloc((size_t)taken * desc->hidden * sizeof(float));
    aotx_check_runtime(cudaMemcpy(copy, mean, (size_t)taken * desc->hidden * sizeof(float),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    double worst = 1.0;
    for (unsigned int i = 0u; i < taken; ++i) {
        double cosine = aotx_gate_cosine(copy + (size_t)i * desc->hidden,
                                         want + (size_t)i * desc->hidden, desc->hidden);
        if (cosine < worst) {
            worst = cosine;
        }
    }
    aotx_gate_note("the mean of every row is refused", worst < AOTX_GATE_COSINE, "least",
                   worst, AOTX_GATE_COSINE);
    cudaFree(mean);
    free(copy);
    free(one);
    free(out);
    free(ids);
    free(counts);
    free(rowptr);
    for (unsigned int i = 0u; i < lines; ++i) {
        free(line[i]);
    }
    free(file);
    free(want_raw);
}

/* The reference values of the reranker. The template comes from the model file. */
static void aotx_gate_rerank(aotx_gate_text *text, aotx_gate_gear *gear, aotx_kv_map *map,
                             const char *dir)
{
    char path[512];
    unsigned long long bytes = 0ull;
    snprintf(path, sizeof path, "%s/rerank-pairs.dat", dir);
    char *file = aotx_gate_slurp(path, &bytes);
    snprintf(path, sizeof path, "%s/rerank.f32", dir);
    unsigned long long want_bytes = 0ull;
    char *want_raw = aotx_gate_slurp(path, &want_bytes);
    if (file == 0 || want_raw == 0) {
        printf("the rerank fixture did not read\n");
        exit(1);
    }
    char *line[AOTX_GATE_MAX];
    unsigned int pairs = aotx_gate_lines(file, bytes, line, AOTX_GATE_MAX, 0);
    const float *want = (const float *)want_raw;
    unsigned int rows = (unsigned int)(want_bytes / sizeof(float));
    if (rows < pairs) {
        pairs = rows;
    }

    static char made[AOTX_GATE_MAX][4096];
    const char *ready[AOTX_GATE_MAX];
    for (unsigned int i = 0u; i < pairs; ++i) {
        char *tab = strchr(line[i], '\t');
        if (tab == NULL) {
            printf("a pair line holds no tab byte\n");
            exit(1);
        }
        *tab = '\0';
        snprintf(made[i], sizeof made[i],
                 "<|im_start|>system\nJudge whether the Document meets the requirements "
                 "based on the Query and the Instruct provided. Note that the answer can "
                 "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given "
                 "a web search query, retrieve relevant passages that answer the query\n"
                 "<Query>: %s\n<Document>: %s<|im_end|>\n<|im_start|>assistant\n<think>\n\n"
                 "</think>\n\n", line[i], tab + 1);
        ready[i] = made[i];
    }
    unsigned int *ids = (unsigned int *)malloc((size_t)pairs * AOTX_GATE_STRIDE
                                               * sizeof(unsigned int));
    unsigned int *counts = (unsigned int *)malloc((size_t)pairs * sizeof(unsigned int));
    aotx_gate_tokenize(text, ready, pairs, ids, counts);
    unsigned int **rowptr = (unsigned int **)malloc((size_t)pairs * sizeof(unsigned int *));
    for (unsigned int i = 0u; i < pairs; ++i) {
        rowptr[i] = ids + i * AOTX_GATE_STRIDE;
    }
    float *out = (float *)malloc((size_t)pairs * sizeof(float));
    aotx_gate_groups(gear, map, AOTX_MODEL_RERANKER, rowptr, counts, pairs, 1u, out);
    double worst = 0.0;
    for (unsigned int i = 0u; i < pairs; ++i) {
        double gap = fabs((double)out[i] - (double)want[i]);
        if (gap > worst) {
            worst = gap;
        }
    }
    printf("rerank gate: %u pairs, first %.6f against %.6f, last %.6f against %.6f\n",
           pairs, (double)out[0], (double)want[0], (double)out[pairs - 1u],
           (double)want[pairs - 1u]);
    aotx_gate_note("rank value of every pair", worst <= AOTX_GATE_SCORE, "worst", worst,
                   AOTX_GATE_SCORE);
    free(out);
    free(ids);
    free(counts);
    free(rowptr);
    for (unsigned int i = 0u; i < pairs; ++i) {
        free(line[i]);
    }
    free(file);
    free(want_raw);
}

/* The rate of the prefill. Every sequence takes its own agent slot, so the pages of the
 * whole set stay in the cache to the end of the run. */
static double aotx_gate_rate(aotx_gate_gear *gear, aotx_kv_map *map, unsigned int role,
                             unsigned int total)
{
    unsigned int per = AOTX_MODEL_MAX_TOKENS / AOTX_GATE_RATE_LEN;
    int *run = (int *)malloc((size_t)AOTX_MODEL_MAX_TOKENS * sizeof(int));
    unsigned int offset[AOTX_SLOTS + 1u];
    unsigned int agent[AOTX_SLOTS];
    aotx_model_how how;
    unsigned long long seed = 0ull;
    how.top_k = 1u;
    how.top_p = 1.0f;
    how.temperature = 0.0f;
    how.seed = 0x0102030405060708ull;
    int *token = (int *)aotx_gate_take(AOTX_SLOTS * sizeof(int));

    aotx_model_forget();
    double start = aotx_gate_now();
    for (unsigned int first = 0u; first < total; first += per) {
        unsigned int seqs = (total - first < per) ? (total - first) : per;
        unsigned int at = 0u;
        for (unsigned int i = 0u; i < seqs; ++i) {
            offset[i] = at;
            agent[i] = first + i;
            for (unsigned int p = 0u; p < AOTX_GATE_RATE_LEN; ++p) {
                run[at + p] = (int)(1000u + ((first + i) * 7u + p * 13u) % 100000u);
            }
            at += AOTX_GATE_RATE_LEN;
        }
        offset[seqs] = at;
        aotx_check_runtime(cudaMemcpy(gear->ids, run, at * sizeof(int),
                                      cudaMemcpyHostToDevice), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(gear->offset, offset,
                                      (seqs + 1u) * sizeof(unsigned int),
                                      cudaMemcpyHostToDevice), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(gear->agent, agent, seqs * sizeof(unsigned int),
                                      cudaMemcpyHostToDevice), "cudaMemcpy");
        aotx_model_pages(role, gear->offset, seqs, gear->agent);
        aotx_kv_serve(map, 0);
        if (aotx_model_sample(role, gear->ids, gear->offset, seqs,
                              gear->agent, &how, token, 0, &seed) != 0) {
            printf("the rate pass did not run\n");
            exit(1);
        }
    }
    double spent = aotx_gate_now() - start;
    cudaFree(token);
    free(run);
    return (double)total * AOTX_GATE_RATE_LEN / spent;
}

int main(int argc, char **argv)
{
    const char *fixtures = (argc > 1) ? argv[1] : "tests/fixtures/model";
    const char *models = (argc > 2) ? argv[2] : "models";
    char path[1024];
    snprintf(path, sizeof path, "%s/manifest.jsonl", models);
    if (access(path, R_OK) != 0) {
        /* The count the gate would have applied with every model file in place. */
        aotx_gate_skip = 13u;
        printf("model gate: 0 cases, 0 bad, %u skipped, no model files in %s\n",
               aotx_gate_skip, models);
        return 0;
    }

    /* The four bit file is read when the model record names it and the reference lists for
     * it are there. A run without it costs 5.29 GB of weights, not 7.64 GB. */
    unsigned long long bytes = 0ull;
    char *record = aotx_gate_slurp(path, &bytes);
    snprintf(path, sizeof path, "%s/lm-q4-0.pos", fixtures);
    int four_bit = (record != 0 && strstr(record, "\"name\":\"language-q4\"") != NULL
                    && access(path, R_OK) == 0);
    free(record);
    /* The profile names the language file this build places. A profile whose weights
     * region does not hold both files gates the file it names, and the other file is
     * stated as skipped. */
    int eight_bit = (AOTX_PROFILE_LANGUAGE_ROLE == AOTX_MODEL_LANGUAGE);
    const char *roles = (eight_bit && four_bit) ? "embedding,reranker,language,language-q4"
                      : (eight_bit ? AOTX_ROLES_DEFAULT
                                   : "embedding,reranker," AOTX_PROFILE_LANGUAGE);
    aotx_check_runtime(cudaFree(0), "cudaFree");
    aotx_mem_map map;
    aotx_kv_map pages;
    if (aotx_mem_reserve(&map) != 0 || aotx_kv_open(&pages) != 0) {
        printf("the reservations did not open\n");
        return 1;
    }
    double began = aotx_gate_now();
    if (aotx_boot_models(models, roles, 0) != 0) {
        printf("the model files did not load\n");
        return 1;
    }
    printf("model load: roles %s, %.1f s to the descriptors\n", roles,
           aotx_gate_now() - began);
    aotx_model_desc desc[AOTX_MODEL_ROLES];
    aotx_check_runtime(cudaMemcpyFromSymbol(desc, aotx_model, sizeof desc),
                       "cudaMemcpyFromSymbol");
    const aotx_model_desc *big = &desc[AOTX_PROFILE_LANGUAGE_ROLE];
    printf("language: %u layers %u hidden %u ffn %u heads %u key heads %u vocabulary "
           "tied %u\n", big->layers, big->hidden, big->ffn, big->heads, big->kv_heads,
           big->vocab, big->tied_output);

    aotx_gate_text text;
    aotx_gate_text_open(&text);
    aotx_gate_gear gear;
    memset(&gear, 0, sizeof gear);
    gear.ids = (int *)aotx_gate_take(AOTX_MODEL_MAX_TOKENS * sizeof(int));
    gear.offset = (unsigned int *)aotx_gate_take((AOTX_SLOTS + 1u)
                                                 * sizeof(unsigned int));
    gear.agent = (unsigned int *)aotx_gate_take(AOTX_SLOTS * sizeof(unsigned int));
    gear.logits = (float *)aotx_gate_take((unsigned long long)AOTX_MODEL_MAX_TOKENS
                                          * desc[AOTX_PROFILE_LANGUAGE_ROLE].vocab
                                          * sizeof(float));
    gear.pooled = (float *)aotx_gate_take((unsigned long long)AOTX_SLOTS
                                          * desc[AOTX_MODEL_EMBEDDING].hidden
                                          * sizeof(float));
    gear.score = (float *)aotx_gate_take(AOTX_SLOTS * sizeof(float));

    if (eight_bit) {
        if (aotx_model_open(AOTX_MODEL_LANGUAGE, AOTX_MODEL_MAX_TOKENS) != 0) {
            printf("the language graph did not capture\n");
            return 1;
        }
        aotx_gate_language(&text, &gear, &pages, fixtures, &desc[AOTX_MODEL_LANGUAGE],
                           AOTX_MODEL_LANGUAGE, "lm", "eight bit", AOTX_GATE_L1, 1);
        double one = aotx_gate_rate(&gear, &pages, AOTX_MODEL_LANGUAGE, 1u);
        one = aotx_gate_rate(&gear, &pages, AOTX_MODEL_LANGUAGE, 1u);
        double many = aotx_gate_rate(&gear, &pages, AOTX_MODEL_LANGUAGE, AOTX_SLOTS);
        many = aotx_gate_rate(&gear, &pages, AOTX_MODEL_LANGUAGE, AOTX_SLOTS);
        printf("prefill rate: %.0f tokens a second at one sequence, %.0f at %u "
               "sequences, %u tokens each\n", one, many, (unsigned int)AOTX_SLOTS,
               AOTX_GATE_RATE_LEN);
        aotx_model_shut(AOTX_MODEL_LANGUAGE);
    } else {
        aotx_gate_skip += 1u;
        printf("eight bit gate: skipped, the weights region of the %s profile does not "
               "hold that file\n", AOTX_PROFILE_NAME);
    }

    /* The same eight prompts through the four bit file, against the lists that file made.
     * The four bit weights are the weights the reference read. The difference is again
     * the number format of the activations. */
    if (four_bit) {
        if (aotx_model_open(AOTX_MODEL_LANGUAGE_Q4, AOTX_MODEL_MAX_TOKENS) != 0) {
            printf("the four bit language graph did not capture\n");
            return 1;
        }
        aotx_gate_language(&text, &gear, &pages, fixtures, &desc[AOTX_MODEL_LANGUAGE_Q4],
                           AOTX_MODEL_LANGUAGE_Q4, "lm-q4", "four bit", AOTX_GATE_L1_Q4,
                           0);
        if (!eight_bit) {
            double one = aotx_gate_rate(&gear, &pages, AOTX_MODEL_LANGUAGE_Q4, 1u);
            one = aotx_gate_rate(&gear, &pages, AOTX_MODEL_LANGUAGE_Q4, 1u);
            double many = aotx_gate_rate(&gear, &pages, AOTX_MODEL_LANGUAGE_Q4,
                                         AOTX_SLOTS);
            many = aotx_gate_rate(&gear, &pages, AOTX_MODEL_LANGUAGE_Q4, AOTX_SLOTS);
            printf("prefill rate: %.0f tokens a second at one sequence, %.0f at %u "
                   "sequences, %u tokens each\n", one, many, (unsigned int)AOTX_SLOTS,
                   AOTX_GATE_RATE_LEN);
        }
        aotx_model_shut(AOTX_MODEL_LANGUAGE_Q4);
    } else {
        aotx_gate_skip += 2u;
        printf("four bit gate: skipped, the model record or the reference lists are "
               "not there\n");
    }

    if (aotx_model_open(AOTX_MODEL_EMBEDDING, AOTX_MODEL_MAX_TOKENS) != 0) {
        printf("the embedding graph did not capture\n");
        return 1;
    }
    aotx_gate_embedding(&text, &gear, &pages, fixtures, &desc[AOTX_MODEL_EMBEDDING]);
    aotx_model_shut(AOTX_MODEL_EMBEDDING);

    if (aotx_model_open(AOTX_MODEL_RERANKER, AOTX_MODEL_MAX_TOKENS) != 0) {
        printf("the rank graph did not capture\n");
        return 1;
    }
    aotx_gate_rerank(&text, &gear, &pages, fixtures);
    aotx_model_shut(AOTX_MODEL_RERANKER);

    unsigned int faults = aotx_model_faulted();
    aotx_gate_note("rows without a page", faults == 0u, "count", (double)faults, 0.0);
    aotx_boot_models_release();
    aotx_kv_close(&pages);
    aotx_mem_release(&map);
    printf("model gate: %u cases, %u bad, %u skipped\n", aotx_gate_cases, aotx_gate_bad,
           aotx_gate_skip);
    return (aotx_gate_bad == 0u) ? 0 : 1;
}
