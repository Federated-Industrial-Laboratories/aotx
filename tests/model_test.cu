/* Purpose: Check every kernel of the forward pass against a reference on the processor.
 * Owns: The synthetic weights, the reference buffers and the counts of the cases.
 * Launch shape: The forward graph, at one sequence and at AOTX_SLOTS.
 * Lifetime: The program. */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "embed/embed.cuh"
#include "kvcache/kvcache.cuh"
#include "mem/mem.cuh"
#include "model/forward.cuh"
#include "model/conduct.cuh"
#include "rerank/rerank.cuh"

#include "model_ref.h"

static unsigned int aotx_test_cases = 0u;
static unsigned int aotx_test_bad = 0u;

static void aotx_test_note(const char *name, int good, const char *how, double value,
                           double bound)
{
    aotx_test_cases += 1u;
    if (!good) {
        aotx_test_bad += 1u;
    }
    printf("%-34s %-4s %s %.3e of %.3e\n", name, good ? "ok" : "BAD", how, value, bound);
}

/* Read the key rows and the value rows of one layer out of the pages, one block for each
 * token. The check of the page layout needs the bytes the attention kernel reads. */
__global__ void aotx_test_pages(unsigned int role, unsigned int layer, half *keys,
                                half *values)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int t = blockIdx.x;
    if (t >= run->tokens) {
        return;
    }
    unsigned int s = aotx_model_which(run->offset, run->seqs, t);
    unsigned int position = work->base[s] + (t - run->offset[s]);
    unsigned int agent = run->agent[s];
    unsigned int narrow = desc->kv_heads * desc->head_dim;
    for (unsigned int i = threadIdx.x; i < narrow; i += blockDim.x) {
        unsigned int head = i / desc->head_dim;
        unsigned int d = i % desc->head_dim;
        const half *key = aotx_kvl_key(&work->shape, agent, layer, head, position);
        const half *value = aotx_kvl_value(&work->shape, agent, layer, head, position);
        keys[(unsigned long long)t * narrow + i] = (key != 0) ? key[d] : __float2half(0.0f);
        values[(unsigned long long)t * narrow + i] = (value != 0) ? value[d]
                                                                 : __float2half(0.0f);
    }
}

static void *aotx_test_take(unsigned long long bytes)
{
    void *block = 0;
    aotx_check_runtime(cudaMalloc(&block, (size_t)bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemset(block, 0, (size_t)bytes), "cudaMemset");
    return block;
}

/* The largest difference between a device buffer and the reference, divided by the largest
 * reference value of that buffer. A relative figure states the error in steps of the number
 * format: half precision holds 11 significand bits, so one step is about 5e-4. */
static double aotx_test_worst(const void *device, int is_half, const float *host,
                              unsigned int count)
{
    double worst = 0.0;
    double scale = 1.0e-6;
    float *copy = (float *)malloc((size_t)count * sizeof(float));
    if (is_half) {
        half *raw = (half *)malloc((size_t)count * sizeof(half));
        aotx_check_runtime(cudaMemcpy(raw, device, (size_t)count * sizeof(half),
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        for (unsigned int i = 0u; i < count; ++i) {
            copy[i] = __half2float(raw[i]);
        }
        free(raw);
    } else {
        aotx_check_runtime(cudaMemcpy(copy, device, (size_t)count * sizeof(float),
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
    }
    for (unsigned int i = 0u; i < count; ++i) {
        double gap = fabs((double)copy[i] - (double)host[i]);
        if (gap > worst) {
            worst = gap;
        }
        if (fabs((double)host[i]) > scale) {
            scale = fabs((double)host[i]);
        }
    }
    free(copy);
    return worst / scale;
}

/* One batch of sequences. Every sequence has its own length and its own tokens, so an
 * index that goes wrong cannot give the right answer. */
static unsigned int aotx_test_batch(unsigned int seqs, unsigned int vocab, unsigned int fixed,
                                    int *ids, unsigned int *offset, unsigned int *agent)
{
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < seqs; ++i) {
        offset[i] = at;
        agent[i] = i;
        unsigned int length = (fixed != 0u) ? fixed : (2u + (i % 6u));
        for (unsigned int p = 0u; p < length; ++p) {
            ids[at + p] = (int)((i * 97u + p * 31u + 7u) % vocab);
        }
        at += length;
    }
    offset[seqs] = at;
    return at;
}

/* The tolerance of each buffer. The figure is the largest difference of the buffer,
 * divided by the largest reference value of the buffer. The bound is twice the largest
 * figure seen on this machine, which is 1.7e-3.
 *
 * The error has two sources. The input of every projection is half precision, and the
 * matrix kernel adds the parts of a row in another order. One step of half precision is
 * about 5e-4 of the value, so the bound is about eight steps. A turn of the wrong period
 * gives 6.2e-1, 150 times the bound. */
#define AOTX_TOL_QKV     4.0e-3
#define AOTX_TOL_HEAD    4.0e-3
#define AOTX_TOL_ATT     4.0e-3
#define AOTX_TOL_NORM    4.0e-3
#define AOTX_TOL_FFN     4.0e-3
#define AOTX_TOL_LOGIT   4.0e-3

typedef struct aotx_test_gear {
    int *ids;
    unsigned int *offset;
    unsigned int *agent;
    float *logits;
    float *pooled;
    float *score;
    int *token;
    half *keys;
    half *values;
} aotx_test_gear;

/* One comparison of a device buffer with the reference, named for the batch it came from. */
static void aotx_test_one(const char *label, const char *what, const void *device,
                          int is_half, const float *host, unsigned int count, double bound)
{
    char name[64];
    snprintf(name, sizeof name, "%s %s", label, what);
    double gap = aotx_test_worst(device, is_half, host, count);
    aotx_test_note(name, gap <= bound, "worst", gap, bound);
}

/* The graph of a role holds three nodes before the layers, 15 for each layer, the nodes of
 * the head, and one node after it. The shape never changes, so the count is the shape. */
static void aotx_test_shape_note(unsigned int role, unsigned int layers, const char *label)
{
    unsigned int tail = aotx_model_is_language(role) ? 4u : 1u;
    unsigned int want = 3u + layers * 15u + tail + 1u;
    unsigned int got = aotx_model_nodes(role);
    aotx_test_note(label, got == want, "nodes", (double)got, (double)want);
}

static int aotx_test_probe(aotx_test_gear *gear, aotx_kv_map *map, unsigned int seqs,
                           const aotx_model_how *how, float *logits)
{
    aotx_model_forget();
    if (aotx_model_pages(AOTX_MODEL_LANGUAGE, gear->offset, seqs, gear->agent) != 0
        || aotx_kv_serve(map, 0) < 0) return 1;
    return aotx_model_probe(AOTX_MODEL_LANGUAGE, gear->ids, gear->offset, seqs,
                            gear->agent, how, logits, AOTX_MODEL_ROWS_LAST, 0, 0, 0u);
}

/* The absent selection is a bit-for-bit neutral path. One selected vector changes the
 * residual stream after its named layer and therefore changes at least one output value. */
static void aotx_test_steer(aotx_test_model *model, aotx_test_gear *gear,
                            aotx_kv_map *map, unsigned int seqs)
{
    const aotx_test_shape *s = &model->shape;
    int ids[AOTX_SLOTS * 2u];
    unsigned int offset[AOTX_SLOTS + 1u], agent[AOTX_SLOTS];
    unsigned int tokens = aotx_test_batch(seqs, s->vocab, 2u, ids, offset, agent);
    aotx_check_runtime(cudaMemcpy(gear->ids, ids, tokens * sizeof(int), cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->offset, offset, (seqs + 1u) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->agent, agent, seqs * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    size_t cells = (size_t)seqs * s->vocab;
    float *plain = (float *)aotx_test_take(cells * sizeof(float));
    float *neutral = (float *)aotx_test_take(cells * sizeof(float));
    float *active = (float *)aotx_test_take(cells * sizeof(float));
    aotx_model_how host[AOTX_SLOTS];
    memset(host, 0, sizeof host);
    for (unsigned int i = 0u; i < seqs; ++i) {
        host[i].steer[0] = AOTX_MODEL_CONDUCT_NONE;
        host[i].steer[1] = AOTX_MODEL_CONDUCT_NONE;
        host[i].voice = AOTX_MODEL_CONDUCT_NONE;
    }
    aotx_model_how *how = (aotx_model_how *)aotx_test_take(seqs * sizeof *how);
    aotx_check_runtime(cudaMemcpy(how, host, seqs * sizeof *how, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    if (aotx_test_probe(gear, map, seqs, 0, plain)
        || aotx_test_probe(gear, map, seqs, how, neutral)) exit(1);
    static unsigned int registered = 0u;
    if (!registered) {
        unsigned int layer = 1u;
        float *value = (float *)malloc(s->hidden * sizeof(float));
        for (unsigned int i = 0u; i < s->hidden; ++i) value[i] = 0.25f;
        float *device = (float *)aotx_test_take(s->hidden * sizeof(float));
        aotx_check_runtime(cudaMemcpy(device, value, s->hidden * sizeof(float),
                                      cudaMemcpyHostToDevice), "cudaMemcpy");
        if (aotx_conduct_register_vector("fixture", &layer, 1u, s->hidden, device, 0.1f))
            exit(1);
        cudaFree(device); free(value); registered = 1u;
    }
    for (unsigned int i = 0u; i < seqs; ++i) {
        host[i].steer[0] = 0u;
        host[i].steer_strength[0] = 0.5f;
    }
    aotx_check_runtime(cudaMemcpy(how, host, seqs * sizeof *how, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    if (aotx_test_probe(gear, map, seqs, how, active)) exit(1);
    float mass[AOTX_SLOTS * AOTX_KV_PAGES_EACH];
    aotx_check_runtime(cudaMemcpyFromSymbol(mass, aotx_page_mass, sizeof mass),
                       "cudaMemcpyFromSymbol");
    double mass_sum = 0.0;
    for (unsigned int i = 0u; i < AOTX_SLOTS * AOTX_KV_PAGES_EACH; ++i)
        mass_sum += (double)mass[i];
    float *a = (float *)malloc(cells * sizeof(float));
    float *b = (float *)malloc(cells * sizeof(float));
    float *c = (float *)malloc(cells * sizeof(float));
    aotx_check_runtime(cudaMemcpy(a, plain, cells * sizeof(float), cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(b, neutral, cells * sizeof(float), cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(c, active, cells * sizeof(float), cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    unsigned int moved = 0u;
    for (size_t i = 0u; i < cells; ++i) moved += (c[i] != a[i]) ? 1u : 0u;
    aotx_test_note("absent steer is bit neutral", memcmp(a, b, cells * sizeof(float)) == 0,
                   "different", memcmp(a, b, cells * sizeof(float)) != 0, 0.0);
    aotx_test_note("selected steer changes output", moved > 0u, "values", moved, 0.0);
    aotx_test_note("attention gives plausible page mass", isfinite(mass_sum) && mass_sum > 0.0,
                   "mass", mass_sum, 0.0);
    free(a); free(b); free(c); cudaFree(plain); cudaFree(neutral); cudaFree(active); cudaFree(how);
}

/* Run one batch through the pass and compare every buffer of the last layer. */
static void aotx_test_pass(aotx_test_model *model, aotx_test_gear *gear, aotx_kv_map *map,
                           unsigned int role, unsigned int seqs, unsigned int fixed,
                           const char *label)
{
    const aotx_test_shape *s = &model->shape;
    unsigned int wide = s->heads * s->head_dim;
    unsigned int narrow = s->kv_heads * s->head_dim;
    int *ids = (int *)malloc((size_t)AOTX_MODEL_MAX_TOKENS * sizeof(int));
    unsigned int *offset = (unsigned int *)malloc((size_t)(seqs + 1u) * sizeof(unsigned int));
    unsigned int *agent = (unsigned int *)malloc((size_t)seqs * sizeof(unsigned int));
    unsigned int tokens = aotx_test_batch(seqs, s->vocab, fixed, ids, offset, agent);

    aotx_check_runtime(cudaMemcpy(gear->ids, ids, tokens * sizeof(int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->offset, offset, (seqs + 1u) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->agent, agent, seqs * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_model_forget();
    if (aotx_model_pages(role, gear->offset, seqs, gear->agent) != 0) {
        printf("the page request of %s did not run\n", label);
        exit(1);
    }
    aotx_kv_serve(map, 0);

    float *pooled = (role == AOTX_MODEL_EMBEDDING) ? gear->pooled : 0;
    float *logits = (role == AOTX_MODEL_LANGUAGE) ? gear->logits : 0;
    int state = (role == AOTX_MODEL_RERANKER)
        ? aotx_model_rerank(gear->ids, gear->offset, seqs, gear->agent, gear->score)
        : aotx_model_prefill(role, gear->ids, gear->offset, seqs, gear->agent, logits,
                             pooled);
    if (state != 0) {
        printf("the pass of %s did not run\n", label);
        exit(1);
    }
    aotx_test_pages<<<tokens, 128>>>(role, s->layers - 1u, gear->keys, gear->values);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    aotx_test_ref ref;
    aotx_test_ref_open(&ref, s, tokens);
    aotx_test_reference(model, &ref, ids, offset, seqs);

    aotx_model_work work;
    aotx_check_runtime(cudaMemcpyFromSymbol(&work, aotx_model_space, sizeof work,
                                            (size_t)role * sizeof work),
                       "cudaMemcpyFromSymbol");
    aotx_test_one(label, "query rows", work.q, 0, ref.q, tokens * wide, AOTX_TOL_QKV);
    aotx_test_one(label, "key rows", work.k, 0, ref.k, tokens * narrow, AOTX_TOL_QKV);
    aotx_test_one(label, "value rows", work.v, 0, ref.v, tokens * narrow, AOTX_TOL_QKV);
    aotx_test_one(label, "head norm and angle", work.qh, 1, ref.qh, tokens * wide,
                  AOTX_TOL_HEAD);
    aotx_test_one(label, "page keys", gear->keys, 1, ref.keys, tokens * narrow,
                  AOTX_TOL_HEAD);
    aotx_test_one(label, "page values", gear->values, 1, ref.values, tokens * narrow,
                  AOTX_TOL_HEAD);
    aotx_test_one(label, "attention", work.att, 1, ref.att, tokens * wide, AOTX_TOL_ATT);
    aotx_test_one(label, "feed norm", work.x, 1, ref.x, tokens * s->hidden, AOTX_TOL_NORM);
    aotx_test_one(label, "gate rows", work.gate, 0, ref.gate, tokens * s->ffn,
                  AOTX_TOL_FFN);
    aotx_test_one(label, "gated unit", work.act, 1, ref.act, tokens * s->ffn, AOTX_TOL_FFN);
    aotx_test_one(label, "residual stream", work.resid, 0, ref.resid, tokens * s->hidden,
                  AOTX_TOL_FFN);
    char name[64];
    double gap = 0.0;
    if (role == AOTX_MODEL_LANGUAGE) {
        snprintf(name, sizeof name, "%s logits", label);
        gap = aotx_test_worst(gear->logits, 0, ref.logits, tokens * s->vocab);
        aotx_test_note(name, gap <= AOTX_TOL_LOGIT, "worst", gap, AOTX_TOL_LOGIT);
        unsigned int wrong = 0u;
        float *copy = (float *)malloc((size_t)tokens * s->vocab * sizeof(float));
        aotx_check_runtime(cudaMemcpy(copy, gear->logits,
                                      (size_t)tokens * s->vocab * sizeof(float),
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        for (unsigned int t = 0u; t < tokens; ++t) {
            unsigned int mine = 0u;
            unsigned int theirs = 0u;
            for (unsigned int i = 1u; i < s->vocab; ++i) {
                if (copy[(size_t)t * s->vocab + i] > copy[(size_t)t * s->vocab + mine]) {
                    mine = i;
                }
                if (ref.logits[(size_t)t * s->vocab + i]
                    > ref.logits[(size_t)t * s->vocab + theirs]) {
                    theirs = i;
                }
            }
            if (mine != theirs) {
                wrong += 1u;
            }
        }
        free(copy);
        snprintf(name, sizeof name, "%s largest logit", label);
        aotx_test_note(name, wrong == 0u, "wrong", (double)wrong, 0.0);
    }
    if (role == AOTX_MODEL_EMBEDDING) {
        float *copy = (float *)malloc((size_t)seqs * s->hidden * sizeof(float));
        aotx_check_runtime(cudaMemcpy(copy, gear->pooled,
                                      (size_t)seqs * s->hidden * sizeof(float),
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        double least = 1.0;
        for (unsigned int r = 0u; r < seqs; ++r) {
            const float *want = ref.xnorm + (size_t)(offset[r + 1u] - 1u) * s->hidden;
            double dot = 0.0;
            double length = 0.0;
            for (unsigned int d = 0u; d < s->hidden; ++d) {
                dot += (double)copy[(size_t)r * s->hidden + d] * want[d];
                length += (double)want[d] * want[d];
            }
            double cosine = dot / sqrt(length);
            if (cosine < least) {
                least = cosine;
            }
        }
        free(copy);
        snprintf(name, sizeof name, "%s pooled cosine", label);
        aotx_test_note(name, least >= 0.9999, "least", 1.0 - least, 1.0e-4);
    }
    if (role == AOTX_MODEL_RERANKER) {
        float *copy = (float *)malloc((size_t)seqs * sizeof(float));
        aotx_check_runtime(cudaMemcpy(copy, gear->score, (size_t)seqs * sizeof(float),
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        double worst = 0.0;
        for (unsigned int r = 0u; r < seqs; ++r) {
            const float *want = ref.xnorm + (size_t)(offset[r + 1u] - 1u) * s->hidden;
            float two[AOTX_RERANK_CLASSES];
            aotx_test_matmul(model->cls, want, two, AOTX_RERANK_CLASSES, s->hidden);
            float top = fmaxf(two[0], two[1]);
            double yes = exp((double)two[0] - top);
            double no = exp((double)two[1] - top);
            double gap2 = fabs((double)copy[r] - yes / (yes + no));
            if (gap2 > worst) {
                worst = gap2;
            }
        }
        free(copy);
        snprintf(name, sizeof name, "%s rank score", label);
        aotx_test_note(name, worst <= 1.0e-3, "worst", worst, 1.0e-3);
    }
    free(ids);
    free(offset);
    free(agent);
}

/* The sample kernel. A temperature of zero gives the largest logit. One seed gives one
 * token twice, and two seeds do not give one token for every sequence. */
static void aotx_test_sampling(aotx_test_model *model, aotx_test_gear *gear,
                               aotx_kv_map *map, unsigned int seqs)
{
    const aotx_test_shape *s = &model->shape;
    int *ids = (int *)malloc((size_t)AOTX_MODEL_MAX_TOKENS * sizeof(int));
    unsigned int *offset = (unsigned int *)malloc((size_t)(seqs + 1u) * sizeof(unsigned int));
    unsigned int *agent = (unsigned int *)malloc((size_t)seqs * sizeof(unsigned int));
    unsigned int tokens = aotx_test_batch(seqs, s->vocab, 0u, ids, offset, agent);
    aotx_check_runtime(cudaMemcpy(gear->ids, ids, tokens * sizeof(int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->offset, offset, (seqs + 1u) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->agent, agent, seqs * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_test_ref ref;
    aotx_test_ref_open(&ref, s, tokens);
    aotx_test_reference(model, &ref, ids, offset, seqs);

    unsigned int cases = 5u;
    int *out[5];
    for (unsigned int i = 0u; i < cases; ++i) {
        out[i] = (int *)malloc((size_t)seqs * sizeof(int));
    }
    /* The five runs: the largest logit; one candidate; a full draw; the same draw again;
     * the same draw with another seed. */
    aotx_model_how how[5];
    memset(how, 0, sizeof how);
    for (unsigned int i = 0u; i < cases; ++i) {
        how[i].top_k = 0u;
        how[i].top_p = 1.0f;
        how[i].temperature = 1.0f;
        how[i].repeat_penalty = 1.0f;
        how[i].think_limit = -1;
        how[i].seed = 0xA0A1A2A3A4A5A6A7ull;
    }
    how[0].temperature = 0.0f;
    how[1].top_k = 1u;
    how[4].seed = 0x1234567890ABCDEFull;
    unsigned long long seed = 0ull;
    for (unsigned int i = 0u; i < cases; ++i) {
        /* Every run starts at the first position of the stream of each slot, so two runs
         * of one seed take the same numbers. */
        aotx_model_forget();
        aotx_model_restream();
        aotx_model_pages(AOTX_MODEL_LANGUAGE, gear->offset, seqs, gear->agent);
        aotx_kv_serve(map, 0);
        if (aotx_model_sample(AOTX_MODEL_LANGUAGE, gear->ids, gear->offset, seqs,
                              gear->agent, &how[i], gear->token, 0, &seed) != 0) {
            printf("the sample pass did not run\n");
            exit(1);
        }
        aotx_check_runtime(cudaMemcpy(out[i], gear->token, (size_t)seqs * sizeof(int),
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
    }

    unsigned int wrong = 0u;
    unsigned int single = 0u;
    unsigned int again = 0u;
    unsigned int differ = 0u;
    for (unsigned int r = 0u; r < seqs; ++r) {
        unsigned int last = offset[r + 1u] - 1u;
        unsigned int best = 0u;
        for (unsigned int i = 1u; i < s->vocab; ++i) {
            if (ref.logits[(size_t)last * s->vocab + i]
                > ref.logits[(size_t)last * s->vocab + best]) {
                best = i;
            }
        }
        if (out[0][r] != (int)best) {
            wrong += 1u;
        }
        if (out[1][r] != out[0][r]) {
            single += 1u;
        }
        if (out[3][r] != out[2][r]) {
            again += 1u;
        }
        if (out[4][r] != out[2][r]) {
            differ += 1u;
        }
    }
    aotx_test_note("sample at temperature zero", wrong == 0u, "wrong", (double)wrong, 0.0);
    aotx_test_note("sample with one candidate", single == 0u, "wrong", (double)single, 0.0);
    aotx_test_note("sample with one seed twice", again == 0u, "wrong", (double)again, 0.0);
    aotx_test_note("sample with two seeds", differ > 0u, "differ", (double)differ, 0.0);
    for (unsigned int i = 0u; i < cases; ++i) {
        free(out[i]);
    }
    free(ids);
    free(offset);
    free(agent);
}

/* A name the tensor table does not hold must be refused and counted. The bind kernel forms
 * every name itself, so a model number that no tensor carries makes every name miss. */
static void aotx_test_missing(unsigned int role, unsigned int layers)
{
    unsigned int count = 4u + layers * 11u;
    unsigned int *report = 0;
    unsigned int got[2] = { 0u, ~0u };
    aotx_check_runtime(cudaMalloc((void **)&report, sizeof got), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(report, got, sizeof got, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_model_bind<<<(count + 127u) / 128u, 128>>>(role, 99u, count, report);
    aotx_check_runtime(cudaMemcpy(got, report, sizeof got, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    cudaFree(report);
    unsigned int want = 2u + layers * 11u;
    aotx_test_note("a name the table does not hold", got[0] == want && got[1] == 0u,
                   "missing", (double)got[0], (double)want);
}

/* A sequence of no tokens is refused, so no pass reads a row that is not there. */
static void aotx_test_empty(aotx_test_gear *gear, unsigned int role)
{
    unsigned int offset[3] = { 0u, 0u, 4u };
    unsigned int agent[2] = { 0u, 1u };
    aotx_check_runtime(cudaMemcpy(gear->offset, offset, sizeof offset,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->agent, agent, sizeof agent,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    int state = aotx_model_prefill(role, gear->ids, gear->offset, 2u, gear->agent,
                                   gear->logits, 0);
    aotx_test_note("a sequence of no tokens is refused", state != 0, "return",
                   (double)state, 1.0);
}

/* The angle gate must be able to fail. The same batch with a turn of the wrong period must
 * give a head that is outside the tolerance and a largest logit that moves. */
static void aotx_test_wrong_angle(aotx_test_model *model, aotx_test_gear *gear,
                                  aotx_kv_map *map, unsigned int seqs)
{
    const aotx_test_shape *s = &model->shape;
    unsigned int wide = s->heads * s->head_dim;
    int *ids = (int *)malloc((size_t)AOTX_MODEL_MAX_TOKENS * sizeof(int));
    unsigned int *offset = (unsigned int *)malloc((size_t)(seqs + 1u) * sizeof(unsigned int));
    unsigned int *agent = (unsigned int *)malloc((size_t)seqs * sizeof(unsigned int));
    unsigned int tokens = aotx_test_batch(seqs, s->vocab, 0u, ids, offset, agent);
    aotx_check_runtime(cudaMemcpy(gear->ids, ids, tokens * sizeof(int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->offset, offset, (seqs + 1u) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->agent, agent, seqs * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");

    aotx_model_desc bad = model->desc;
    bad.rope_theta = 10000.0f;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &bad, sizeof bad,
                                          (size_t)AOTX_MODEL_LANGUAGE * sizeof bad),
                       "cudaMemcpyToSymbol");
    aotx_model_forget();
    aotx_model_pages(AOTX_MODEL_LANGUAGE, gear->offset, seqs, gear->agent);
    aotx_kv_serve(map, 0);
    aotx_model_prefill(AOTX_MODEL_LANGUAGE, gear->ids, gear->offset, seqs, gear->agent,
                       gear->logits, 0);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &model->desc, sizeof bad,
                                          (size_t)AOTX_MODEL_LANGUAGE * sizeof bad),
                       "cudaMemcpyToSymbol");

    aotx_test_ref ref;
    aotx_test_ref_open(&ref, s, tokens);
    aotx_test_reference(model, &ref, ids, offset, seqs);
    aotx_model_work work;
    aotx_check_runtime(cudaMemcpyFromSymbol(&work, aotx_model_space, sizeof work,
                                            (size_t)AOTX_MODEL_LANGUAGE * sizeof work),
                       "cudaMemcpyFromSymbol");
    double gap = aotx_test_worst(work.qh, 1, ref.qh, tokens * wide);
    aotx_test_note("a turn of the wrong period", gap > AOTX_TOL_HEAD, "worst", gap,
                   AOTX_TOL_HEAD);
    float *copy = (float *)malloc((size_t)tokens * s->vocab * sizeof(float));
    aotx_check_runtime(cudaMemcpy(copy, gear->logits,
                                  (size_t)tokens * s->vocab * sizeof(float),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int moved = 0u;
    for (unsigned int t = 0u; t < tokens; ++t) {
        unsigned int mine = 0u;
        unsigned int theirs = 0u;
        for (unsigned int i = 1u; i < s->vocab; ++i) {
            if (copy[(size_t)t * s->vocab + i] > copy[(size_t)t * s->vocab + mine]) {
                mine = i;
            }
            if (ref.logits[(size_t)t * s->vocab + i]
                > ref.logits[(size_t)t * s->vocab + theirs]) {
                theirs = i;
            }
        }
        if (mine != theirs) {
            moved += 1u;
        }
    }
    free(copy);
    aotx_test_note("a wrong period moves the logit", moved > 0u, "moved", (double)moved,
                   0.0);
    free(ids);
    free(offset);
    free(agent);
}

int main(void)
{
    aotx_check_runtime(cudaFree(0), "cudaFree");
    aotx_mem_map map;
    if (aotx_mem_reserve(&map) != 0) {
        printf("the memory reservation did not open\n");
        return 1;
    }
    aotx_kv_map pages;
    if (aotx_kv_open(&pages) != 0) {
        printf("the page range did not open\n");
        return 1;
    }
    aotx_test_gear gear;
    memset(&gear, 0, sizeof gear);
    gear.ids = (int *)aotx_test_take(AOTX_MODEL_MAX_TOKENS * sizeof(int));
    gear.offset = (unsigned int *)aotx_test_take((AOTX_SLOTS + 1u)
                                                 * sizeof(unsigned int));
    gear.agent = (unsigned int *)aotx_test_take(AOTX_SLOTS * sizeof(unsigned int));
    gear.logits = (float *)aotx_test_take(512ull * 256ull * sizeof(float));
    gear.pooled = (float *)aotx_test_take((unsigned long long)AOTX_SLOTS * 256ull
                                          * sizeof(float));
    gear.score = (float *)aotx_test_take(AOTX_SLOTS * sizeof(float));
    gear.token = (int *)aotx_test_take(AOTX_SLOTS * sizeof(int));
    gear.keys = (half *)aotx_test_take(512ull * 1024ull * sizeof(half));
    gear.values = (half *)aotx_test_take(512ull * 1024ull * sizeof(half));

    /* The first shape has more query heads than key heads, so the head map is checked. */
    aotx_test_shape narrow_shape = { 4u, 64u, 96u, 4u, 2u, 32u, 128u };

    /* The second shape gives blocks that fill a page in 31 steps, so a sequence of 200
     * positions reads keys and values from two pages. */
    aotx_test_shape wide_shape = { 4u, 128u, 128u, 8u, 8u, 128u, 128u };

    static aotx_test_model model;
    aotx_test_build(&model, &narrow_shape, AOTX_MODEL_LANGUAGE);
    if (aotx_model_open(AOTX_MODEL_LANGUAGE, AOTX_MODEL_MAX_TOKENS) != 0) {
        printf("the language graph did not capture\n");
        return 1;
    }
    aotx_test_shape_note(AOTX_MODEL_LANGUAGE, narrow_shape.layers, "language graph shape");
    aotx_test_pass(&model, &gear, &pages, AOTX_MODEL_LANGUAGE, 1u, 37u, "one sequence");
    aotx_test_pass(&model, &gear, &pages, AOTX_MODEL_LANGUAGE, AOTX_SLOTS, 0u,
                   "every slot");
    aotx_test_empty(&gear, AOTX_MODEL_LANGUAGE);
    aotx_test_sampling(&model, &gear, &pages, AOTX_SLOTS);
    aotx_test_wrong_angle(&model, &gear, &pages, AOTX_SLOTS);
    aotx_test_steer(&model, &gear, &pages, 1u);
    aotx_test_steer(&model, &gear, &pages, AOTX_SLOTS);
    aotx_model_shut(AOTX_MODEL_LANGUAGE);

    static aotx_test_model large;
    aotx_test_build(&large, &wide_shape, AOTX_MODEL_LANGUAGE);
    if (aotx_model_open(AOTX_MODEL_LANGUAGE, AOTX_MODEL_MAX_TOKENS) != 0) {
        printf("the second language graph did not capture\n");
        return 1;
    }
    aotx_test_pass(&large, &gear, &pages, AOTX_MODEL_LANGUAGE, 1u, 200u, "two pages");
    aotx_model_shut(AOTX_MODEL_LANGUAGE);

    static aotx_test_model vectors;
    aotx_test_build(&vectors, &narrow_shape, AOTX_MODEL_EMBEDDING);
    if (aotx_model_open(AOTX_MODEL_EMBEDDING, AOTX_MODEL_MAX_TOKENS) != 0) {
        printf("the embedding graph did not capture\n");
        return 1;
    }
    aotx_test_shape_note(AOTX_MODEL_EMBEDDING, narrow_shape.layers, "vector graph shape");
    aotx_test_pass(&vectors, &gear, &pages, AOTX_MODEL_EMBEDDING, 1u, 37u, "one vector");
    aotx_test_pass(&vectors, &gear, &pages, AOTX_MODEL_EMBEDDING, AOTX_SLOTS, 0u,
                   "every slot of vectors");
    aotx_model_shut(AOTX_MODEL_EMBEDDING);

    static aotx_test_model ranks;
    aotx_test_build(&ranks, &narrow_shape, AOTX_MODEL_RERANKER);
    if (aotx_model_open(AOTX_MODEL_RERANKER, AOTX_MODEL_MAX_TOKENS) != 0) {
        printf("the rank graph did not capture\n");
        return 1;
    }
    aotx_test_shape_note(AOTX_MODEL_RERANKER, narrow_shape.layers, "rank graph shape");
    aotx_test_pass(&ranks, &gear, &pages, AOTX_MODEL_RERANKER, 1u, 37u, "one pair");
    aotx_test_pass(&ranks, &gear, &pages, AOTX_MODEL_RERANKER, AOTX_SLOTS, 0u,
                   "every slot of pairs");
    aotx_model_shut(AOTX_MODEL_RERANKER);

    aotx_test_missing(AOTX_MODEL_EMBEDDING, narrow_shape.layers);
    unsigned int faults = aotx_model_faulted();
    aotx_test_note("rows without a page", faults == 0u, "count", (double)faults, 0.0);
    aotx_kv_close(&pages);
    aotx_mem_release(&map);
    printf("model: %u cases, %u bad, 0 skipped, seed 0x%llx\n", aotx_test_cases,
           aotx_test_bad, (unsigned long long)AOTX_TEST_SEED);
    return (aotx_test_bad == 0u) ? 0 : 1;
}
