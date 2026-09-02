/* Purpose: Load the text sets of the steer tool and run them through the model in passes.
 * Owns: The texts of a set, their token counts, the pass plan and the batch buffers of a run.
 * Launch shape: Host glue; the tokenizer batch and the pack kernel run for each pass.
 * Lifetime: One tool run. */
#ifndef AOTX_TOOLS_STEER_SET_H
#define AOTX_TOOLS_STEER_SET_H

#include <cuda_runtime.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "kvcache/kvcache.cuh"
#include "mem/mem.cuh"
#include "model/conduct.cuh"
#include "model/forward.cuh"
#include "model/roles.h"
#include "tools/steer_text.h"

/* Texts one set holds at the most, and the longest line of a set file. A pass takes the
 * tokenizer batch of texts, or the agent slots when the profile holds fewer. */
#define AOTX_STEER_SET_TEXTS  256u
#define AOTX_STEER_SET_LINE   8192u
#define AOTX_STEER_PASS_TEXTS ((AOTX_STEER_TEXTS < AOTX_SLOTS) ? AOTX_STEER_TEXTS : AOTX_SLOTS)
#define AOTX_STEER_PIECES     64u
#define AOTX_STEER_PATH       1024u
#define AOTX_STEER_NO_AXIS    0xFFFFFFFFu

/* A text of this many tokens or more is named, so a fixture keeps its sentences short. */
#define AOTX_STEER_LONG_TEXT  24u

/* The axes a probe file can name, by their number in the probe table. */
#define AOTX_STEER_AXES 4u
static const char *aotx_steer_axis_name[AOTX_STEER_AXES] = {
    "valence", "arousal", "sycophancy", "refusal"
};
static const unsigned int aotx_steer_axis_number[AOTX_STEER_AXES] = { 0u, 1u, 4u, 5u };

typedef struct aotx_vector_head {
    char magic[8];
    unsigned int hidden, layers;
    float potency;
    unsigned int reserved;
} aotx_vector_head;

typedef struct aotx_probe_head {
    char magic[8];
    unsigned int hidden, layer, axis;
    float accuracy, agreement, mean, scale;
    unsigned int reserved;
} aotx_probe_head;

/* One text set: the texts, the tokens of each one and the passes the set runs in. A pass
 * holds at most AOTX_STEER_PASS_TEXTS texts and AOTX_MODEL_MAX_TOKENS rows. */
typedef struct aotx_steer_set {
    const char *name;
    char *text[AOTX_STEER_SET_TEXTS];
    unsigned int count[AOTX_STEER_SET_TEXTS];
    unsigned int first[AOTX_STEER_SET_TEXTS + 1u];
    unsigned int texts, pairs, passes, tokens, longest, longest_at;
} aotx_steer_set;

/* The placed model of one run and the device buffers of one pass. */
typedef struct aotx_steer_run {
    unsigned int role;
    aotx_mem_map map;
    aotx_kv_map pages;
    aotx_model_desc desc;
    aotx_steer_text tokenizer;
    int *ids;
    unsigned int *offset;
    unsigned int *agent;
    aotx_model_how *how;
    void *piece[AOTX_STEER_PIECES];
    unsigned int pieces;
} aotx_steer_run;

__global__ void aotx_steer_flat(const unsigned int *, const unsigned int *, unsigned int, int *);
__global__ void aotx_steer_mean(const float *, unsigned int, unsigned int, unsigned int, float *);
__global__ void aotx_steer_kl(const float *, const float *, unsigned int, unsigned int, float *);
__global__ void aotx_steer_last(const float *, const unsigned int *, unsigned int, float *);
__global__ void aotx_steer_nll(const float *, const int *, const unsigned int *, unsigned int,
                               unsigned int, unsigned int, double *, unsigned int *);
__global__ void aotx_steer_gram(const float *, const float *, unsigned int, double *);
__global__ void aotx_steer_compose(const float *, const float *, const double *, unsigned int,
                                   float *, float *);
__global__ void aotx_probe_fit(const float *, unsigned int, unsigned int, unsigned int, float *,
                               float *);
__global__ void aotx_probe_read(const float *, const float *, unsigned int, unsigned int, float *);
__global__ void aotx_probe_scale(const float *, unsigned int, float *, float *);
__global__ void aotx_probe_count(const float *, unsigned int, const float *, float *);
__global__ void aotx_probe_shift(const float *, const float *, unsigned int, float, float, float *);

/* The calibration mode, in its own host file. */
int aotx_steer_calibrate(const char *models, const char *role_name, const char *axes,
                         const char *guards, const char *neutral_path, float dose, float surgical);

static unsigned int aotx_steer_axis_of(const char *name)
{
    for (unsigned int i = 0u; i < AOTX_STEER_AXES; ++i) {
        if (strcmp(name, aotx_steer_axis_name[i]) == 0) return aotx_steer_axis_number[i];
    }
    return AOTX_STEER_NO_AXIS;
}

/* Read a set file. A pair file holds two texts on each line with a tab between them; a
 * plain file holds one text on each line. An empty line separates texts and holds none. */
static int aotx_steer_set_read(aotx_steer_set *set, const char *path, int pair_form)
{
    FILE *in = fopen(path, "r");
    char line[AOTX_STEER_SET_LINE];
    memset(set, 0, sizeof *set);
    set->name = path;
    if (in == 0) { fprintf(stderr, "the set %s does not open\n", path); return 1; }
    while (fgets(line, sizeof line, in) != 0) {
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') continue;
        char *tab = pair_form ? strchr(line, '\t') : 0;
        if (pair_form && tab == 0) {
            fclose(in);
            fprintf(stderr, "the set %s has a line without a tab: %.40s\n", path, line);
            return 1;
        }
        if (set->texts + (pair_form ? 2u : 1u) > AOTX_STEER_SET_TEXTS) {
            fclose(in);
            fprintf(stderr, "the set %s holds more than %u texts\n", path, AOTX_STEER_SET_TEXTS);
            return 1;
        }
        if (tab != 0) *tab++ = '\0';
        set->text[set->texts] = strdup(line);
        if (tab != 0) set->text[set->texts + 1u] = strdup(tab);
        if (set->text[set->texts] == 0 || (tab != 0 && set->text[set->texts + 1u] == 0)) {
            fclose(in);
            fprintf(stderr, "the set %s does not fit in memory\n", path);
            return 1;
        }
        set->texts += (tab != 0) ? 2u : 1u;
    }
    fclose(in);
    set->pairs = set->texts / 2u;
    if (set->texts == 0u) { fprintf(stderr, "the set %s holds no text\n", path); return 1; }
    return 0;
}

/* Count the tokens of every text with the tokenizer batch, one batch at a time. A text
 * with no token, or with more tokens than a pass holds, is refused by its number. */
static int aotx_steer_set_count(aotx_steer_run *run, aotx_steer_set *set)
{
    unsigned int counts[AOTX_STEER_TEXTS];
    set->tokens = 0u; set->longest = 0u; set->longest_at = 0u;
    for (unsigned int at = 0u; at < set->texts; at += AOTX_STEER_TEXTS) {
        unsigned int n = set->texts - at;
        if (n > AOTX_STEER_TEXTS) n = AOTX_STEER_TEXTS;
        if (aotx_steer_tokenize(&run->tokenizer, set->text + at, n, 0, counts) != 0) {
            fprintf(stderr, "the set %s does not fit the tokenizer batch\n", set->name);
            return 1;
        }
        for (unsigned int i = 0u; i < n; ++i) {
            unsigned int c = counts[i];
            if (c == 0u || c > AOTX_MODEL_MAX_TOKENS) {
                fprintf(stderr, "the text %u of %s has %u tokens, a pass holds 1 to %u\n",
                        at + i + 1u, set->name, c, AOTX_MODEL_MAX_TOKENS);
                return 1;
            }
            set->count[at + i] = c;
            set->tokens += c;
            if (c > set->longest) { set->longest = c; set->longest_at = at + i + 1u; }
            if (c >= AOTX_STEER_LONG_TEXT) printf("set %s: text %u has %u tokens\n", set->name, at + i + 1u, c);
        }
    }
    return 0;
}

/* Cut the set into passes. A pass takes texts in their order until the next text does not
 * fit its text count or its row count. */
static void aotx_steer_set_plan(aotx_steer_set *set)
{
    unsigned int rows = 0u, n = 0u;
    set->passes = 0u;
    set->first[0] = 0u;
    for (unsigned int i = 0u; i < set->texts; ++i) {
        if (n == AOTX_STEER_PASS_TEXTS || rows + set->count[i] > AOTX_MODEL_MAX_TOKENS) {
            set->first[++set->passes] = i;
            rows = 0u; n = 0u;
        }
        rows += set->count[i];
        n += 1u;
    }
    set->first[++set->passes] = set->texts;
    printf("set %s: %u texts, %u tokens, longest %u tokens (text %u), %u passes\n", set->name,
           set->texts, set->tokens, set->longest, set->longest_at, set->passes);
}

static unsigned int aotx_steer_set_seqs(const aotx_steer_set *set, unsigned int pass)
{
    return set->first[pass + 1u] - set->first[pass];
}

static unsigned int aotx_steer_set_rows(const aotx_steer_set *set, unsigned int pass)
{
    unsigned int rows = 0u;
    for (unsigned int i = set->first[pass]; i < set->first[pass + 1u]; ++i) rows += set->count[i];
    return rows;
}

/* Take one device buffer of the run and keep it, so the close gives every piece back. */
static void *aotx_steer_run_take(aotx_steer_run *run, size_t bytes)
{
    void *at = 0;
    if (run->pieces >= AOTX_STEER_PIECES) { fprintf(stderr, "the run holds too many buffers\n"); exit(1); }
    aotx_check_runtime(cudaMalloc(&at, bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemset(at, 0, bytes), "cudaMemset");
    run->piece[run->pieces++] = at;
    return at;
}

/* Open the run: place the model of the role, capture its pass and take the batch buffers. */
static int aotx_steer_run_open(aotx_steer_run *run, const char *models, const char *role_name)
{
    unsigned int agent[AOTX_STEER_TEXTS];
    memset(run, 0, sizeof *run);
    run->role = aotx_role_of(role_name);
    if (run->role >= AOTX_MODEL_ROLES || aotx_model_is_language(run->role) == 0) {
        fprintf(stderr, "the role %s is not a language role\n", role_name);
        return 2;
    }
    aotx_check_runtime(cudaFree(0), "cudaFree");
    if (aotx_mem_reserve(&run->map) || aotx_kv_open(&run->pages)
        || aotx_boot_models(models, role_name, 0)
        || aotx_model_open(run->role, AOTX_MODEL_MAX_TOKENS)) return 1;
    aotx_check_runtime(cudaMemcpyFromSymbol(&run->desc, aotx_model, sizeof run->desc,
                                            run->role * sizeof run->desc), "cudaMemcpyFromSymbol");
    aotx_steer_text_open(&run->tokenizer);
    for (unsigned int i = 0u; i < AOTX_STEER_TEXTS; ++i) agent[i] = i;
    run->ids = (int *)aotx_steer_run_take(run, AOTX_MODEL_MAX_TOKENS * sizeof(int));
    run->offset = (unsigned int *)aotx_steer_run_take(run, (AOTX_STEER_TEXTS + 1u) * sizeof(unsigned int));
    run->agent = (unsigned int *)aotx_steer_run_take(run, sizeof agent);
    run->how = (aotx_model_how *)aotx_steer_run_take(run, AOTX_STEER_TEXTS * sizeof(aotx_model_how));
    aotx_check_runtime(cudaMemcpy(run->agent, agent, sizeof agent, cudaMemcpyHostToDevice), "cudaMemcpy");
    return 0;
}

static void aotx_steer_run_close(aotx_steer_run *run)
{
    for (unsigned int i = 0u; i < run->pieces; ++i) cudaFree(run->piece[i]);
    run->pieces = 0u;
    aotx_steer_text_close(&run->tokenizer);
    aotx_model_shut(run->role);
    aotx_boot_models_release();
    aotx_kv_close(&run->pages);
    aotx_mem_release(&run->map);
}

/* A how row that changes nothing: no steer vector, no voice, no affect mark. */
static void aotx_steer_how_plain(aotx_model_how *how)
{
    memset(how, 0, sizeof *how);
    for (unsigned int i = 0u; i < AOTX_MODEL_STEERS; ++i) how->steer[i] = AOTX_MODEL_CONDUCT_NONE;
    how->voice = AOTX_MODEL_CONDUCT_NONE;
    how->voice_scale = 1.0f;
}

/* Give every sequence of a pass the same how row. A null row leaves the pass plain. */
static const aotx_model_how *aotx_steer_run_how(aotx_steer_run *run, const aotx_model_how *one)
{
    aotx_model_how rows[AOTX_STEER_TEXTS];
    if (one == 0) return 0;
    for (unsigned int i = 0u; i < AOTX_STEER_TEXTS; ++i) rows[i] = *one;
    aotx_check_runtime(cudaMemcpy(run->how, rows, sizeof rows, cudaMemcpyHostToDevice), "cudaMemcpy");
    return run->how;
}

/* Run one pass of a set: tokenize its texts, pack them, serve the pages and call the probe
 * pass. The logits and the capture go where the caller asks. */
static int aotx_steer_run_pass(aotx_steer_run *run, const aotx_steer_set *set, unsigned int pass,
                               const aotx_model_how *how, float *logits, unsigned int select,
                               float *capture, const unsigned int *layers, unsigned int layer_count)
{
    unsigned int first = set->first[pass], seqs = aotx_steer_set_seqs(set, pass), total = 0u;
    unsigned int counts[AOTX_STEER_TEXTS], offset[AOTX_STEER_TEXTS + 1u];
    if (aotx_steer_tokenize(&run->tokenizer, (char **)(set->text + first), seqs, 0, counts) != 0) return 1;
    for (unsigned int i = 0u; i < seqs; ++i) { offset[i] = total; total += counts[i]; }
    offset[seqs] = total;
    aotx_check_runtime(cudaMemcpy(run->offset, offset, (seqs + 1u) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_steer_flat<<<seqs, 256u>>>(run->tokenizer.tokens.id, run->offset, AOTX_STEER_STRIDE, run->ids);
    aotx_model_forget();
    if (aotx_model_pages(run->role, run->offset, seqs, run->agent) != 0
        || aotx_kv_serve(&run->pages, 0) < 0) return 1;
    return aotx_model_probe(run->role, run->ids, run->offset, seqs, run->agent, how, logits,
                            select, capture, layers, layer_count);
}

/* Move the capture of one pass into the capture of the whole set. Layer by layer, the rows
 * of the pass go after the rows of the passes before it. */
static void aotx_steer_run_gather(const aotx_steer_run *run, const aotx_steer_set *set,
                                  unsigned int pass, const float *capture_pass,
                                  float *capture_set, unsigned int layer_count)
{
    unsigned int first = set->first[pass], seqs = aotx_steer_set_seqs(set, pass);
    unsigned int hidden = run->desc.hidden;
    for (unsigned int c = 0u; c < layer_count; ++c) {
        aotx_check_runtime(cudaMemcpy(capture_set + ((size_t)c * set->texts + first) * hidden,
                                      capture_pass + (size_t)c * seqs * hidden,
                                      (size_t)seqs * hidden * sizeof(float),
                                      cudaMemcpyDeviceToDevice), "cudaMemcpy");
    }
}

/* Run every pass of a set plain and collect the capture of the named layers for the set. */
static int aotx_steer_run_capture(aotx_steer_run *run, const aotx_steer_set *set,
                                  const unsigned int *layers, unsigned int layer_count,
                                  float *capture_pass, float *capture_set)
{
    for (unsigned int p = 0u; p < set->passes; ++p) {
        if (aotx_steer_run_pass(run, set, p, 0, 0, AOTX_MODEL_ROWS_LAST, capture_pass, layers,
                                layer_count) != 0) return 1;
        aotx_steer_run_gather(run, set, p, capture_pass, capture_set, layer_count);
    }
    return 0;
}

/* The mean divergence of one steered pass from its plain pass over the sequences of the
 * pass, at the last row of each one. */
static float aotx_steer_run_kl(const aotx_steer_run *run, unsigned int seqs, const float *plain,
                               const float *steered, float *sum)
{
    float value = 0.0f;
    aotx_check_runtime(cudaMemset(sum, 0, sizeof(float)), "cudaMemset");
    aotx_steer_kl<<<seqs, 256u>>>(plain, steered, seqs, run->desc.vocab, sum);
    aotx_check_runtime(cudaMemcpy(&value, sum, sizeof value, cudaMemcpyDeviceToHost), "cudaMemcpy");
    return value;
}

/* The potency of each registered vector over a set. It is the divergence of the steered
 * pass from the plain pass at the last row, as a mean over the texts. */
static int aotx_steer_run_potency(aotx_steer_run *run, const aotx_steer_set *set,
                                  const unsigned int *vector_id, unsigned int count,
                                  float strength, float *plain, float *steered, float *sum,
                                  float *potency)
{
    for (unsigned int v = 0u; v < count; ++v) potency[v] = 0.0f;
    for (unsigned int p = 0u; p < set->passes; ++p) {
        unsigned int seqs = aotx_steer_set_seqs(set, p);
        if (aotx_steer_run_pass(run, set, p, 0, plain, AOTX_MODEL_ROWS_LAST, 0, 0, 0u) != 0) return 1;
        for (unsigned int v = 0u; v < count; ++v) {
            aotx_model_how one;
            aotx_steer_how_plain(&one);
            one.steer[0] = vector_id[v];
            one.steer_strength[0] = strength;
            if (aotx_steer_run_pass(run, set, p, aotx_steer_run_how(run, &one), steered,
                                    AOTX_MODEL_ROWS_LAST, 0, 0, 0u) != 0) return 1;
            potency[v] += aotx_steer_run_kl(run, seqs, plain, steered, sum)
                        * (float)seqs / (float)set->texts;
        }
    }
    return 0;
}

/* Write one vector file in the store format: the head, the layer list and the values. */
static int aotx_steer_write_values(const char *path, const unsigned int *layers,
                                   unsigned int layer_count, unsigned int hidden,
                                   const float *values, float potency)
{
    aotx_vector_head head;
    size_t count = (size_t)layer_count * hidden;
    memset(&head, 0, sizeof head);
    memcpy(head.magic, "AOTXSTV1", 8u);
    head.hidden = hidden; head.layers = layer_count; head.potency = potency;
    FILE *out = fopen(path, "wb");
    if (out == 0 || fwrite(&head, sizeof head, 1u, out) != 1u
        || fwrite(layers, sizeof *layers, layer_count, out) != layer_count
        || fwrite(values, sizeof *values, count, out) != count || fclose(out) != 0) {
        fprintf(stderr, "the vector file %s does not write\n", path);
        unlink(path);
        return 1;
    }
    return 0;
}

/* Write the probe file of one axis and its catalog line. The catalog prints the accuracy
 * with nine digits, so the loader reads the float the head holds. A failed catalog line
 * takes the file away again, as a vector file with no catalog line blocks the next run. */
static int aotx_steer_write_probe(const char *dir, const char *axis, unsigned int number,
                                  unsigned int hidden, unsigned int layer, float accuracy,
                                  float agreement, float mean, float scale,
                                  const float *direction)
{
    char path[AOTX_STEER_PATH], line[AOTX_STEER_PATH];
    aotx_probe_head head;
    snprintf(path, sizeof path, "%s/affect", dir);
    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "the directory %s does not open\n", path);
        return 1;
    }
    if (!(scale > 0.0f) || !isfinite(scale) || !isfinite(mean)) {
        fprintf(stderr, "the probe row %s has no spread over the neutral set, no file is written\n",
                axis);
        return 1;
    }
    snprintf(path, sizeof path, "%s/affect/%s.aotxprb", dir, axis);
    memset(&head, 0, sizeof head);
    memcpy(head.magic, "AOTXPRB1", 8u);
    head.hidden = hidden; head.layer = layer; head.axis = number;
    head.accuracy = accuracy; head.agreement = agreement; head.mean = mean; head.scale = scale;
    FILE *out = fopen(path, "wb");
    if (out == 0 || fwrite(&head, sizeof head, 1u, out) != 1u
        || fwrite(direction, sizeof *direction, hidden, out) != hidden || fclose(out) != 0) {
        fprintf(stderr, "the probe file %s does not write\n", path);
        unlink(path);
        return 1;
    }
    snprintf(line, sizeof line, "%s/probes.jsonl", dir);
    out = fopen(line, "a");
    int state = 1;
    if (out != 0) {
        state = fprintf(out, "{\"name\":\"%s\",\"file\":\"affect/%s.aotxprb\",\"axis\":%u,"
                             "\"layer\":%u,\"accuracy\":%.9g}\n",
                        axis, axis, number, layer, (double)accuracy) < 0;
        if (fclose(out) != 0) state = 1;
    }
    if (state) { fprintf(stderr, "the catalog %s does not write\n", line); unlink(path); }
    return state;
}

#endif
