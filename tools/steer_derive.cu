/* Purpose: Compute mean contrast residuals and the potency of a steer vector.
 * Owns: Nothing; the derivation program owns all input and output buffers.
 * Launch shape: One thread per vector value; one block per probe distribution; one block
 * per text for the pack of the token lists.
 * Lifetime: One derivation run. */
#include <cuda_runtime.h>
#include <math.h>

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
