/* Purpose: Compute the contrast means, the probe figures and the potency of steer vectors.
 * Owns: Nothing; the derivation program owns all input and output buffers.
 * Launch shape: One thread per vector value; one block per distribution, text or layer;
 * one block per text for the pack of the token lists.
 * Lifetime: One derivation run. */
#include <cuda_runtime.h>
#include <math.h>

#include "model/forward.cuh"
#include "model/conduct.cuh"
#include "model/control_position.cuh"
#include "model/wrap.cuh"
#include "tools/steer_text.cuh"

__global__ void aotx_steer_positions(aotx_steer_text tokenizer, unsigned role, unsigned count,
    aotx_model_how *how, unsigned *bad) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    bool response = false;
    for (unsigned j = 0; j < AOTX_MODEL_STEERS; ++j) {
#ifdef AOTX_AFFECT
        if (j == AOTX_MODEL_CONDUCT_AFFECT) continue;
#endif
        unsigned id = how[i].steer[j];
        if (id < aotx_conduct.vectors && how[i].steer_strength[j] != 0 &&
            aotx_conduct.vector[id].positions == AOTX_CONTROL_RESPONSE) response = true;
    }
    how[i].steer_from = 0;
    if (!response) return;
    unsigned p = i * tokenizer.pieces.stride;
    how[i].steer_from = aotx_control_response(&aotx_model_wrap[role], tokenizer.clean,
        tokenizer.clean_start[i], tokenizer.clean_length[i], tokenizer.pieces.start + p,
        tokenizer.pieces.length + p, tokenizer.tokens.chunk + p, tokenizer.pieces.count[i], tokenizer.tokens.count[i]);
    if (!how[i].steer_from) atomicAdd(bad, 1u);
}

/* Pack the token rows of the tokenizer into the flat list the forward pass reads. Each
 * row has one stride. The offset list holds the first index of each text and the total. */
__global__ void aotx_steer_flat(const unsigned int *rows, const unsigned int *offset,
                                unsigned int stride, int *flat)
{
    unsigned int text = blockIdx.x, first = offset[text], count = offset[text + 1u] - first;
    for (unsigned int j = threadIdx.x; j < count; j += blockDim.x) {
        flat[first + j] = (int)rows[(unsigned long long)text * stride + j];
    }
}

__global__ void aotx_steer_mean(const float *capture, unsigned int pairs,
                                unsigned int layers, unsigned int hidden, float *out)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)layers * hidden;
    if (i >= total) return;
    unsigned int layer = (unsigned int)(i / hidden), x = (unsigned int)(i % hidden);
    double sum = 0.0;
    for (unsigned int p = 0u; p < pairs; ++p) {
        unsigned int yes = 2u * p, no = yes + 1u;
        sum += (double)capture[((unsigned long long)layer * pairs * 2u + yes) * hidden + x]
             - (double)capture[((unsigned long long)layer * pairs * 2u + no) * hidden + x];
    }
    out[i] = (float)(sum / (double)pairs);
}

__global__ void aotx_steer_kl(const float *plain, const float *steered, unsigned int probes,
                              unsigned int vocab, float *mean)
{
    __shared__ double a[256], b[256], k[256];
    __shared__ double pmax, qmax;
    unsigned int probe = blockIdx.x;
    if (probe >= probes) return;
    const float *p = plain + (unsigned long long)probe * vocab;
    const float *q = steered + (unsigned long long)probe * vocab;
    float ptop = -INFINITY, qtop = -INFINITY;
    for (unsigned int i = threadIdx.x; i < vocab; i += blockDim.x) {
        ptop = fmaxf(ptop, p[i]); qtop = fmaxf(qtop, q[i]);
    }
    a[threadIdx.x] = ptop; b[threadIdx.x] = qtop; __syncthreads();
    if (threadIdx.x == 0u) for (unsigned int i = 1u; i < blockDim.x; ++i) {
        a[0] = fmax(a[0], a[i]); b[0] = fmax(b[0], b[i]);
    }
    if (threadIdx.x == 0u) { pmax = a[0]; qmax = b[0]; }
    __syncthreads();
    /* The maxima come from their own cells: a thread that writes its sum into a[0] below
     * can run before another thread reads the maximum. */
    double ps = 0.0, qs = 0.0;
    for (unsigned int i = threadIdx.x; i < vocab; i += blockDim.x) {
        ps += exp((double)p[i] - pmax); qs += exp((double)q[i] - qmax);
    }
    a[threadIdx.x] = ps; b[threadIdx.x] = qs; __syncthreads();
    if (threadIdx.x == 0u) for (unsigned int i = 1u; i < blockDim.x; ++i) {
        a[0] += a[i]; b[0] += b[i];
    }
    __syncthreads();
    double logp = log(a[0]) + pmax, logq = log(b[0]) + qmax, kl = 0.0;
    for (unsigned int i = threadIdx.x; i < vocab; i += blockDim.x) {
        double lp = (double)p[i] - logp;
        kl += exp(lp) * (lp - ((double)q[i] - logq));
    }
    k[threadIdx.x] = kl; __syncthreads();
    if (threadIdx.x == 0u) {
        for (unsigned int i = 1u; i < blockDim.x; ++i) k[0] += k[i];
        atomicAdd(mean, (float)(k[0] / (double)probes));
    }
}

/* The sum of one value from every thread of the block. Every thread gets the total. The
 * whole block makes the call, because the block synchronizes here. */
static __device__ double aotx_steer_block_sum(double value, double *part)
{
    part[threadIdx.x] = value;
    __syncthreads();
    if (threadIdx.x == 0u) {
        for (unsigned int i = 1u; i < blockDim.x; ++i) part[0] += part[i];
    }
    __syncthreads();
    double total = part[0];
    __syncthreads();
    return total;
}

static __device__ double aotx_steer_block_max(double value, double *part)
{
    part[threadIdx.x] = value;
    __syncthreads();
    if (threadIdx.x == 0u) {
        for (unsigned int i = 1u; i < blockDim.x; ++i) part[0] = fmax(part[0], part[i]);
    }
    __syncthreads();
    double total = part[0];
    __syncthreads();
    return total;
}

/* The shrinkage discriminant of one layer, one block for each layer. The pooled variance
 * of each width position and the mean difference come from the pair captures. The
 * shrinkage rho is one tenth of the mean variance over the width. The direction is the
 * mean difference divided by the variance plus rho at each position, at unit length. The
 * variance goes to the scratch. */
__global__ void aotx_probe_fit(const float *capture, unsigned int pairs, unsigned int layers,
                               unsigned int hidden, float *variance, float *out)
{
    __shared__ double part[256];
    unsigned int layer = blockIdx.x;
    if (layer >= layers) return;
    const float *rows = capture + (unsigned long long)layer * pairs * 2u * hidden;
    float *var = variance + (unsigned long long)layer * hidden;
    float *r = out + (unsigned long long)layer * hidden;
    double trace = 0.0;
    unsigned int spread = (pairs > 1u) ? (2u * pairs - 2u) : 1u;
    for (unsigned int x = threadIdx.x; x < hidden; x += blockDim.x) {
        double yes = 0.0, no = 0.0, sum = 0.0;
        for (unsigned int p = 0u; p < pairs; ++p) {
            yes += (double)rows[(unsigned long long)(2u * p) * hidden + x];
            no += (double)rows[(unsigned long long)(2u * p + 1u) * hidden + x];
        }
        yes /= (double)pairs; no /= (double)pairs;
        for (unsigned int p = 0u; p < pairs; ++p) {
            double a = (double)rows[(unsigned long long)(2u * p) * hidden + x] - yes;
            double b = (double)rows[(unsigned long long)(2u * p + 1u) * hidden + x] - no;
            sum += a * a + b * b;
        }
        var[x] = (float)(sum / (double)spread);
        r[x] = (float)(yes - no);
        trace += sum / (double)spread;
    }
    trace = aotx_steer_block_sum(trace, part);
    double rho = 0.1 * trace / (double)hidden, norm = 0.0;
    for (unsigned int x = threadIdx.x; x < hidden; x += blockDim.x) {
        double d = (double)var[x] + rho;
        double v = (d > 0.0) ? (double)r[x] / d : (double)r[x];
        r[x] = (float)v;
        norm += v * v;
    }
    norm = sqrt(aotx_steer_block_sum(norm, part));
    for (unsigned int x = threadIdx.x; x < hidden; x += blockDim.x) {
        r[x] = (norm > 0.0) ? (float)((double)r[x] / norm) : 0.0f;
    }
}

/* The readout of every text along one direction: the cosine of its captured row with the
 * direction. That is the dot product over the norm of the row, as the device readout
 * reads it. The direction has unit length. A row of no length reads zero. One block for each text.
 * The second grid axis names the layer, and it moves the capture, the direction and the
 * result on by one layer. */
__global__ void aotx_probe_read(const float *capture, const float *direction, unsigned int texts,
                                unsigned int hidden, float *out)
{
    __shared__ double part[256];
    unsigned int text = blockIdx.x, layer = blockIdx.y;
    if (text >= texts) return;
    const float *row = capture + ((unsigned long long)layer * texts + text) * hidden;
    const float *r = direction + (unsigned long long)layer * hidden;
    double sum = 0.0, square = 0.0;
    for (unsigned int x = threadIdx.x; x < hidden; x += blockDim.x) {
        sum += (double)row[x] * (double)r[x];
        square += (double)row[x] * (double)row[x];
    }
    sum = aotx_steer_block_sum(sum, part);
    square = aotx_steer_block_sum(square, part);
    if (threadIdx.x == 0u) {
        out[(unsigned long long)layer * texts + text] = (square > 0.0) ? (float)(sum / sqrt(square)) : 0.0f;
    }
}

/* The mean and the scale of the readouts of one layer over the neutral texts. The scale is
 * the standard deviation with one degree of freedom taken. One block for each layer. */
__global__ void aotx_probe_scale(const float *readout, unsigned int texts, float *mean,
                                 float *scale)
{
    __shared__ double part[256];
    const float *d = readout + (unsigned long long)blockIdx.x * texts;
    double sum = 0.0, spread = 0.0;
    for (unsigned int t = threadIdx.x; t < texts; t += blockDim.x) sum += (double)d[t];
    double m = aotx_steer_block_sum(sum, part) / (double)texts;
    for (unsigned int t = threadIdx.x; t < texts; t += blockDim.x) {
        double a = (double)d[t] - m;
        spread += a * a;
    }
    spread = aotx_steer_block_sum(spread, part);
    if (threadIdx.x == 0u) {
        mean[blockIdx.x] = (float)m;
        scale[blockIdx.x] = (float)sqrt(spread / (double)((texts > 1u) ? texts - 1u : 1u));
    }
}

/* The two held-out figures of one layer over the pairs. The accuracy is the share of the
 * pairs whose first member reads higher than its second. The agreement is the share of
 * the pairs whose members both read on their own side of the neutral mean. One block for
 * each layer; the result holds the two shares one after the other. */
__global__ void aotx_probe_count(const float *readout, unsigned int pairs, const float *mean,
                                 float *out)
{
    __shared__ double part[256];
    unsigned int layer = blockIdx.x;
    const float *d = readout + (unsigned long long)layer * pairs * 2u;
    float m = mean[layer];
    double right = 0.0, agree = 0.0;
    for (unsigned int p = threadIdx.x; p < pairs; p += blockDim.x) {
        float yes = d[2u * p], no = d[2u * p + 1u];
        right += (yes > no) ? 1.0 : 0.0;
        agree += (yes > m && no < m) ? 1.0 : 0.0;
    }
    right = aotx_steer_block_sum(right, part);
    agree = aotx_steer_block_sum(agree, part);
    if (threadIdx.x == 0u) {
        out[2u * layer] = (float)(right / (double)pairs);
        out[2u * layer + 1u] = (float)(agree / (double)pairs);
    }
}

/* The mean change of the standardized readout for one unit of dose. It is the steered
 * readout less the plain readout over the texts, divided by the scale and the dose. One
 * block. */
__global__ void aotx_probe_shift(const float *plain, const float *steered, unsigned int texts,
                                 float scale, float dose, float *out)
{
    __shared__ double part[256];
    double sum = 0.0;
    for (unsigned int t = threadIdx.x; t < texts; t += blockDim.x) {
        sum += (double)steered[t] - (double)plain[t];
    }
    sum = aotx_steer_block_sum(sum, part);
    if (threadIdx.x == 0u) {
        *out = (float)(sum / ((double)texts * (double)scale * (double)dose));
    }
}

/* Copy the logits of the last row of every sequence into one row for each sequence. */
__global__ void aotx_steer_last(const float *logits, const unsigned int *offset,
                                unsigned int vocab, float *out)
{
    unsigned int seq = blockIdx.x, row = offset[seq + 1u] - 1u;
    for (unsigned int i = threadIdx.x; i < vocab; i += blockDim.x) {
        out[(unsigned long long)seq * vocab + i] = logits[(unsigned long long)row * vocab + i];
    }
}

/* The negative log likelihood of the next token at every row that has one in its own
 * sequence. One block for each row; the sum and the count go to the accumulators. */
__global__ void aotx_steer_nll(const float *logits, const int *ids, const unsigned int *offset,
                               unsigned int seqs, unsigned int rows, unsigned int vocab,
                               double *sum, unsigned int *count)
{
    __shared__ double part[256];
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    unsigned int seq = aotx_model_which(offset, seqs, row);
    if (row + 1u >= offset[seq + 1u]) return;
    const float *p = logits + (unsigned long long)row * vocab;
    double top = -INFINITY, z = 0.0;
    for (unsigned int i = threadIdx.x; i < vocab; i += blockDim.x) top = fmax(top, (double)p[i]);
    top = aotx_steer_block_max(top, part);
    for (unsigned int i = threadIdx.x; i < vocab; i += blockDim.x) z += exp((double)p[i] - top);
    z = aotx_steer_block_sum(z, part);
    if (threadIdx.x == 0u) {
        int next = ids[row + 1u];
        atomicAdd(sum, log(z) + top - (double)p[next]);
        atomicAdd(count, 1u);
    }
}

/* The Gram matrix of two directions: a.a, a.b and b.b. One block. */
__global__ void aotx_steer_gram(const float *a, const float *b, unsigned int hidden, double *gram)
{
    __shared__ double part[256];
    double aa = 0.0, ab = 0.0, bb = 0.0;
    for (unsigned int x = threadIdx.x; x < hidden; x += blockDim.x) {
        double u = (double)a[x], v = (double)b[x];
        aa += u * u; ab += u * v; bb += v * v;
    }
    aa = aotx_steer_block_sum(aa, part);
    ab = aotx_steer_block_sum(ab, part);
    bb = aotx_steer_block_sum(bb, part);
    if (threadIdx.x == 0u) { gram[0] = aa; gram[1] = ab; gram[2] = bb; }
}

/* The symmetric orthogonalization of two directions, W = V (V^T V)^(-1/2). Each column
 * then takes the length of its own direction back, so one unit of dose keeps its meaning.
 * Two directions that are parallel or empty stay as they are. One thread for each width
 * position; every thread forms the same two by two matrix. */
__global__ void aotx_steer_compose(const float *a, const float *b, const double *gram,
                                   unsigned int hidden, float *wa, float *wb)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= hidden) return;
    double g0 = gram[0], g1 = gram[1], g2 = gram[2];
    double half = 0.5 * (g0 + g2), det = g0 * g2 - g1 * g1;
    double disc = sqrt(fmax(0.0, half * half - det));
    double l1 = half + disc, l2 = half - disc;
    if (!(l2 > 1.0e-9 * l1) || !(l1 > 0.0)) { wa[x] = a[x]; wb[x] = b[x]; return; }
    /* The eigenvector of the larger value comes from a row of the matrix less that value
     * on its diagonal. The longer of the two rows is taken, because one row is zero when
     * the directions are orthogonal. Two equal values leave any direction, so the first
     * axis is taken. */
    double e0 = l1 - g2, e1 = g1, h0 = g1, h1 = l1 - g0;
    if (h0 * h0 + h1 * h1 > e0 * e0 + e1 * e1) { e0 = h0; e1 = h1; }
    if (e0 == 0.0 && e1 == 0.0) { e0 = 1.0; }
    double length = sqrt(e0 * e0 + e1 * e1);
    e0 /= length; e1 /= length;
    double f0 = -e1, f1 = e0, s1 = 1.0 / sqrt(l1), s2 = 1.0 / sqrt(l2);
    double m00 = e0 * e0 * s1 + f0 * f0 * s2;
    double m01 = e0 * e1 * s1 + f0 * f1 * s2;
    double m11 = e1 * e1 * s1 + f1 * f1 * s2;
    double u = (double)a[x], v = (double)b[x];
    wa[x] = (float)((u * m00 + v * m01) * sqrt(g0));
    wb[x] = (float)((u * m01 + v * m11) * sqrt(g2));
}
