/* Purpose: Build a synthetic model and run the whole forward pass on the processor.
 * Owns: The synthetic weights and the reference buffers of one comparison.
 * Threading: One thread; the test calls these one at a time.
 * Lifetime: The program. */
#ifndef AOTX_TEST_MODEL_REF_H
#define AOTX_TEST_MODEL_REF_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "embed/embed.cuh"
#include "mem/mem.cuh"
#include "model/forward.cuh"
#include "rerank/rerank.cuh"

#define AOTX_TEST_SEED    0x5EED1234ull
#define AOTX_TEST_ALIGN   256ull

/* One shape of a synthetic model. Every model of this family has the same kernels, so a
 * small shape checks the same code that the large files take. */
typedef struct aotx_test_shape {
    unsigned int layers;
    unsigned int hidden;
    unsigned int ffn;
    unsigned int heads;
    unsigned int kv_heads;
    unsigned int head_dim;
    unsigned int vocab;
} aotx_test_shape;

/* The weights of one synthetic model, in the memory of the processor. */
typedef struct aotx_test_model {
    aotx_test_shape shape;
    aotx_model_desc desc;
    float *embd;        /* vocab by hidden, the values the blocks hold */
    unsigned char *q8;  /* the same rows as Q8_0 blocks */
    float *out_norm;
    float *cls;
    float *attn_norm[AOTX_MODEL_MAX_LAYERS];
    float *wq[AOTX_MODEL_MAX_LAYERS];
    float *wk[AOTX_MODEL_MAX_LAYERS];
    float *wv[AOTX_MODEL_MAX_LAYERS];
    float *wo[AOTX_MODEL_MAX_LAYERS];
    float *q_norm[AOTX_MODEL_MAX_LAYERS];
    float *k_norm[AOTX_MODEL_MAX_LAYERS];
    float *ffn_norm[AOTX_MODEL_MAX_LAYERS];
    float *gate[AOTX_MODEL_MAX_LAYERS];
    float *up[AOTX_MODEL_MAX_LAYERS];
    float *down[AOTX_MODEL_MAX_LAYERS];
    unsigned long long cursor;
} aotx_test_model;

static unsigned long long aotx_test_state = AOTX_TEST_SEED;

static float aotx_test_unit(void)
{
    aotx_test_state ^= aotx_test_state >> 12;
    aotx_test_state ^= aotx_test_state << 25;
    aotx_test_state ^= aotx_test_state >> 27;
    unsigned int word = (unsigned int)((aotx_test_state * 2685821657736338717ull) >> 32);
    return (float)(word >> 8) * (1.0f / 16777216.0f);
}

/* A half value and back, which is what a projection input holds on the device. */
static float aotx_test_half(float value)
{
    return __half2float(__float2half(value));
}

/* Put one tensor in the weights region and give back its offset. */
static unsigned long long aotx_test_place(aotx_test_model *model, const void *bytes,
                                          unsigned long long size)
{
    unsigned long long at = (model->cursor + AOTX_TEST_ALIGN - 1ull)
                          / AOTX_TEST_ALIGN * AOTX_TEST_ALIGN;
    if (aotx_mem_weights_map(at, size) != 0) {
        printf("the weights region did not take %llu bytes\n", size);
        exit(1);
    }
    aotx_check_runtime(cudaMemcpy((void *)(aotx_mem_weights_base() + at), bytes,
                                  (size_t)size, cudaMemcpyHostToDevice), "cudaMemcpy");
    model->cursor = at + size;
    return at;
}

static float *aotx_test_random(unsigned int count, float span)
{
    float *block = (float *)malloc((size_t)count * sizeof *block);
    for (unsigned int i = 0u; i < count; ++i) {
        block[i] = (aotx_test_unit() - 0.5f) * span;
    }
    return block;
}

/* The norm weights stay near one, so the residual stream keeps a plain range. */
static float *aotx_test_ones(unsigned int count)
{
    float *block = (float *)malloc((size_t)count * sizeof *block);
    for (unsigned int i = 0u; i < count; ++i) {
        block[i] = 1.0f + (aotx_test_unit() - 0.5f) * 0.2f;
    }
    return block;
}

/* Make the token embedding as Q8_0 blocks and keep the values those blocks give. */
static void aotx_test_embedding(aotx_test_model *model)
{
    unsigned int count = model->shape.vocab * model->shape.hidden;
    unsigned int blocks = count / 32u;
    model->embd = (float *)malloc((size_t)count * sizeof *model->embd);
    model->q8 = (unsigned char *)malloc((size_t)blocks * 34u);
    for (unsigned int b = 0u; b < blocks; ++b) {
        float scale = 0.002f + aotx_test_unit() * 0.002f;
        half kept = __float2half(scale);
        unsigned short bits = __half_as_ushort(kept);
        model->q8[b * 34u] = (unsigned char)(bits & 0xFFu);
        model->q8[b * 34u + 1u] = (unsigned char)(bits >> 8);
        for (unsigned int i = 0u; i < 32u; ++i) {
            int q = (int)(aotx_test_unit() * 254.0f) - 127;
            model->q8[b * 34u + 2u + i] = (unsigned char)(signed char)q;
            model->embd[b * 32u + i] = __half2float(kept) * (float)q;
        }
    }
}

static void aotx_test_build(aotx_test_model *model, const aotx_test_shape *shape,
                            unsigned int role)
{
    const aotx_test_shape *s = shape;
    memset(model, 0, sizeof *model);
    model->shape = *shape;
    aotx_test_embedding(model);
    model->out_norm = aotx_test_ones(s->hidden);
    model->cls = aotx_test_random(AOTX_RERANK_CLASSES * s->hidden, 0.2f);

    aotx_model_desc *desc = &model->desc;
    memset(desc, 0, sizeof *desc);
    desc->role = role;
    desc->layers = s->layers;
    desc->hidden = s->hidden;
    desc->ffn = s->ffn;
    desc->heads = s->heads;
    desc->kv_heads = s->kv_heads;
    desc->head_dim = s->head_dim;
    desc->vocab = s->vocab;
    desc->context = 2048u;
    desc->weight_type = AOTX_WEIGHT_F32;
    desc->embd_type = AOTX_WEIGHT_Q8_0;
    desc->tied_output = 1u;
    desc->pooling = (role == AOTX_MODEL_EMBEDDING) ? AOTX_EMBED_POOL_LAST
                  : ((role == AOTX_MODEL_RERANKER) ? AOTX_RERANK_POOL_RANK : 0u);
    desc->rope_theta = 1000000.0f;
    desc->rms_eps = 1e-6f;
    desc->output = AOTX_MODEL_ABSENT;
    desc->cls_output = AOTX_MODEL_ABSENT;

    unsigned int wide = s->heads * s->head_dim;
    unsigned int narrow = s->kv_heads * s->head_dim;
    desc->token_embd = aotx_test_place(model, model->q8,
                                       (unsigned long long)(s->vocab * s->hidden / 32u) * 34u);
    desc->output_norm = aotx_test_place(model, model->out_norm, s->hidden * sizeof(float));
    if (role == AOTX_MODEL_RERANKER) {
        desc->cls_output = aotx_test_place(model, model->cls,
                                           AOTX_RERANK_CLASSES * s->hidden * sizeof(float));
    }
    for (unsigned int l = 0u; l < s->layers; ++l) {
        model->attn_norm[l] = aotx_test_ones(s->hidden);
        model->wq[l] = aotx_test_random(wide * s->hidden, 0.25f);
        model->wk[l] = aotx_test_random(narrow * s->hidden, 0.25f);
        model->wv[l] = aotx_test_random(narrow * s->hidden, 0.25f);
        model->wo[l] = aotx_test_random(s->hidden * wide, 0.25f);
        model->q_norm[l] = aotx_test_ones(s->head_dim);
        model->k_norm[l] = aotx_test_ones(s->head_dim);
        model->ffn_norm[l] = aotx_test_ones(s->hidden);
        model->gate[l] = aotx_test_random(s->ffn * s->hidden, 0.25f);
        model->up[l] = aotx_test_random(s->ffn * s->hidden, 0.25f);
        model->down[l] = aotx_test_random(s->hidden * s->ffn, 0.25f);
        aotx_model_layer *one = &desc->layer[l];
        one->attn_norm = aotx_test_place(model, model->attn_norm[l],
                                         s->hidden * sizeof(float));
        one->attn_q = aotx_test_place(model, model->wq[l], (unsigned long long)wide
                                      * s->hidden * sizeof(float));
        one->attn_k = aotx_test_place(model, model->wk[l], (unsigned long long)narrow
                                      * s->hidden * sizeof(float));
        one->attn_v = aotx_test_place(model, model->wv[l], (unsigned long long)narrow
                                      * s->hidden * sizeof(float));
        one->attn_o = aotx_test_place(model, model->wo[l], (unsigned long long)s->hidden
                                      * wide * sizeof(float));
        one->attn_q_norm = aotx_test_place(model, model->q_norm[l],
                                           s->head_dim * sizeof(float));
        one->attn_k_norm = aotx_test_place(model, model->k_norm[l],
                                           s->head_dim * sizeof(float));
        one->ffn_norm = aotx_test_place(model, model->ffn_norm[l],
                                        s->hidden * sizeof(float));
        one->ffn_gate = aotx_test_place(model, model->gate[l], (unsigned long long)s->ffn
                                        * s->hidden * sizeof(float));
        one->ffn_up = aotx_test_place(model, model->up[l], (unsigned long long)s->ffn
                                      * s->hidden * sizeof(float));
        one->ffn_down = aotx_test_place(model, model->down[l], (unsigned long long)s->hidden
                                        * s->ffn * sizeof(float));
    }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, desc, sizeof *desc,
                                          (size_t)role * sizeof *desc),
                       "cudaMemcpyToSymbol");
}

/* The buffers of the reference. The last layer of the pass keeps its parts, so every
 * kernel of a layer is compared and not only the end of the pass. */
typedef struct aotx_test_ref {
    float *resid;
    float *x;
    float *q;
    float *qh;
    float *k;
    float *v;
    float *att;
    float *gate;
    float *up;
    float *act;
    float *xnorm;
    float *logits;
    float *keys;     /* tokens by kv_heads by head_dim, as the pages hold them */
    float *values;
} aotx_test_ref;

static void aotx_test_ref_open(aotx_test_ref *ref, const aotx_test_shape *s,
                               unsigned int tokens)
{
    unsigned int wide = s->heads * s->head_dim;
    unsigned int narrow = s->kv_heads * s->head_dim;
    ref->resid = (float *)calloc((size_t)tokens * s->hidden, sizeof(float));
    ref->x = (float *)calloc((size_t)tokens * s->hidden, sizeof(float));
    ref->q = (float *)calloc((size_t)tokens * wide, sizeof(float));
    ref->qh = (float *)calloc((size_t)tokens * wide, sizeof(float));
    ref->k = (float *)calloc((size_t)tokens * narrow, sizeof(float));
    ref->v = (float *)calloc((size_t)tokens * narrow, sizeof(float));
    ref->att = (float *)calloc((size_t)tokens * wide, sizeof(float));
    ref->gate = (float *)calloc((size_t)tokens * s->ffn, sizeof(float));
    ref->up = (float *)calloc((size_t)tokens * s->ffn, sizeof(float));
    ref->act = (float *)calloc((size_t)tokens * s->ffn, sizeof(float));
    ref->xnorm = (float *)calloc((size_t)tokens * s->hidden, sizeof(float));
    ref->logits = (float *)calloc((size_t)tokens * s->vocab, sizeof(float));
    ref->keys = (float *)calloc((size_t)tokens * narrow, sizeof(float));
    ref->values = (float *)calloc((size_t)tokens * narrow, sizeof(float));
}

/* The root mean square norm, as the kernel applies it. */
static void aotx_test_rms(const float *in, const float *weight, float *out,
                          unsigned int width, float eps)
{
    float sum = 0.0f;
    for (unsigned int i = 0u; i < width; ++i) {
        sum += in[i] * in[i];
    }
    float scale = 1.0f / sqrtf(sum / (float)width + eps);
    for (unsigned int i = 0u; i < width; ++i) {
        out[i] = scale * in[i] * weight[i];
    }
}

/* One row of a projection. The input is a half row, as the device holds it. */
static void aotx_test_matmul(const float *w, const float *x, float *y, unsigned int n,
                             unsigned int k)
{
    for (unsigned int c = 0u; c < n; ++c) {
        float sum = 0.0f;
        for (unsigned int i = 0u; i < k; ++i) {
            sum += w[(size_t)c * k + i] * aotx_test_half(x[i]);
        }
        y[c] = sum;
    }
}

/* The turn of one head, in the way the reference of llama.cpp states it. */
static void aotx_test_rope(float *head, unsigned int dim, unsigned int position,
                           float theta)
{
    float step = powf(theta, -2.0f / (float)dim);
    for (unsigned int i = 0u; i < dim / 2u; ++i) {
        float angle = (float)position * powf(step, (float)i);
        float cosine = cosf(angle);
        float sine = sinf(angle);
        float low = head[i];
        float high = head[i + dim / 2u];
        head[i] = low * cosine - high * sine;
        head[i + dim / 2u] = low * sine + high * cosine;
    }
}

/* The whole pass on the processor. The batch is the same batch the device takes. */
static void aotx_test_reference(const aotx_test_model *model, aotx_test_ref *ref,
                                const int *ids, const unsigned int *offset,
                                unsigned int seqs)
{
    const aotx_test_shape *s = &model->shape;
    unsigned int tokens = offset[seqs];
    unsigned int wide = s->heads * s->head_dim;
    unsigned int narrow = s->kv_heads * s->head_dim;
    float *proj = (float *)calloc((size_t)tokens * (s->hidden > s->ffn ? s->hidden : s->ffn),
                                  sizeof(float));
    float *row = (float *)calloc(s->hidden > s->ffn ? s->hidden : s->ffn, sizeof(float));

    for (unsigned int t = 0u; t < tokens; ++t) {
        for (unsigned int d = 0u; d < s->hidden; ++d) {
            ref->resid[(size_t)t * s->hidden + d] =
                model->embd[(size_t)ids[t] * s->hidden + d];
        }
    }
    for (unsigned int l = 0u; l < s->layers; ++l) {
        for (unsigned int t = 0u; t < tokens; ++t) {
            aotx_test_rms(ref->resid + (size_t)t * s->hidden, model->attn_norm[l],
                          ref->x + (size_t)t * s->hidden, s->hidden, model->desc.rms_eps);
            aotx_test_matmul(model->wq[l], ref->x + (size_t)t * s->hidden,
                             ref->q + (size_t)t * wide, wide, s->hidden);
            aotx_test_matmul(model->wk[l], ref->x + (size_t)t * s->hidden,
                             ref->k + (size_t)t * narrow, narrow, s->hidden);
            aotx_test_matmul(model->wv[l], ref->x + (size_t)t * s->hidden,
                             ref->v + (size_t)t * narrow, narrow, s->hidden);
        }
        for (unsigned int seq = 0u; seq < seqs; ++seq) {
            for (unsigned int t = offset[seq]; t < offset[seq + 1u]; ++t) {
                unsigned int position = t - offset[seq];
                for (unsigned int h = 0u; h < s->heads; ++h) {
                    float *head = ref->qh + (size_t)t * wide + (size_t)h * s->head_dim;
                    aotx_test_rms(ref->q + (size_t)t * wide + (size_t)h * s->head_dim,
                                  model->q_norm[l], head, s->head_dim, model->desc.rms_eps);
                    aotx_test_rope(head, s->head_dim, position, model->desc.rope_theta);
                    for (unsigned int d = 0u; d < s->head_dim; ++d) {
                        head[d] = aotx_test_half(head[d]);
                    }
                }
                for (unsigned int h = 0u; h < s->kv_heads; ++h) {
                    float *head = ref->keys + (size_t)t * narrow + (size_t)h * s->head_dim;
                    aotx_test_rms(ref->k + (size_t)t * narrow + (size_t)h * s->head_dim,
                                  model->k_norm[l], head, s->head_dim, model->desc.rms_eps);
                    aotx_test_rope(head, s->head_dim, position, model->desc.rope_theta);
                    for (unsigned int d = 0u; d < s->head_dim; ++d) {
                        head[d] = aotx_test_half(head[d]);
                        ref->values[(size_t)t * narrow + (size_t)h * s->head_dim + d] =
                            aotx_test_half(ref->v[(size_t)t * narrow
                                                  + (size_t)h * s->head_dim + d]);
                    }
                }
            }
        }
        for (unsigned int seq = 0u; seq < seqs; ++seq) {
            for (unsigned int t = offset[seq]; t < offset[seq + 1u]; ++t) {
                for (unsigned int h = 0u; h < s->heads; ++h) {
                    unsigned int kv = h / (s->heads / s->kv_heads);
                    const float *query = ref->qh + (size_t)t * wide
                                       + (size_t)h * s->head_dim;
                    float top = -INFINITY;
                    float mass = 0.0f;
                    float *out = ref->att + (size_t)t * wide + (size_t)h * s->head_dim;
                    memset(out, 0, s->head_dim * sizeof(float));
                    for (unsigned int j = offset[seq]; j <= t; ++j) {
                        const float *key = ref->keys + (size_t)j * narrow
                                         + (size_t)kv * s->head_dim;
                        float dot = 0.0f;
                        for (unsigned int d = 0u; d < s->head_dim; ++d) {
                            dot += query[d] * key[d];
                        }
                        dot /= sqrtf((float)s->head_dim);
                        float raised = fmaxf(top, dot);
                        float shift = expf(top - raised);
                        float weight = expf(dot - raised);
                        const float *value = ref->values + (size_t)j * narrow
                                           + (size_t)kv * s->head_dim;
                        for (unsigned int d = 0u; d < s->head_dim; ++d) {
                            out[d] = out[d] * shift + weight * value[d];
                        }
                        mass = mass * shift + weight;
                        top = raised;
                    }
                    for (unsigned int d = 0u; d < s->head_dim; ++d) {
                        out[d] = aotx_test_half(out[d] / mass);
                    }
                }
            }
        }
        for (unsigned int t = 0u; t < tokens; ++t) {
            aotx_test_matmul(model->wo[l], ref->att + (size_t)t * wide,
                             proj + (size_t)t * s->hidden, s->hidden, wide);
        }
        for (unsigned int i = 0u; i < tokens * s->hidden; ++i) {
            ref->resid[i] += proj[i];
        }
        for (unsigned int t = 0u; t < tokens; ++t) {
            aotx_test_rms(ref->resid + (size_t)t * s->hidden, model->ffn_norm[l], row,
                          s->hidden, model->desc.rms_eps);
            memcpy(ref->x + (size_t)t * s->hidden, row, s->hidden * sizeof(float));
            aotx_test_matmul(model->gate[l], row, ref->gate + (size_t)t * s->ffn, s->ffn,
                             s->hidden);
            aotx_test_matmul(model->up[l], row, ref->up + (size_t)t * s->ffn, s->ffn,
                             s->hidden);
            for (unsigned int i = 0u; i < s->ffn; ++i) {
                float gate = ref->gate[(size_t)t * s->ffn + i];
                float unit = gate / (1.0f + expf(-gate));
                ref->act[(size_t)t * s->ffn + i] =
                    aotx_test_half(unit * ref->up[(size_t)t * s->ffn + i]);
            }
            aotx_test_matmul(model->down[l], ref->act + (size_t)t * s->ffn,
                             proj + (size_t)t * s->hidden, s->hidden, s->ffn);
        }
        for (unsigned int i = 0u; i < tokens * s->hidden; ++i) {
            ref->resid[i] += proj[i];
        }
    }
    for (unsigned int t = 0u; t < tokens; ++t) {
        aotx_test_rms(ref->resid + (size_t)t * s->hidden, model->out_norm,
                      ref->xnorm + (size_t)t * s->hidden, s->hidden, model->desc.rms_eps);
        aotx_test_matmul(model->embd, ref->xnorm + (size_t)t * s->hidden,
                         ref->logits + (size_t)t * s->vocab, s->vocab, s->hidden);
    }
    free(proj);
    free(row);
}

#endif
