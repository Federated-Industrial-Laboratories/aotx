/* Purpose: Allocate image storage and capture its finite device work.
 * Owns: Device allocations and their host allocation handles.
 * Launch shape: Host glue only; all source and feature work runs in graph nodes.
 * Lifetime: From validated component load to runtime close. */
#include "media/runtime.cuh"
#include <stdio.h>
#include <string.h>

static aotx_media_state aotx_media_host;
static unsigned char *aotx_media_allocation, *aotx_media_weights;
static aotx_vision_desc *aotx_media_descriptor;
static unsigned long long aotx_media_round(unsigned long long n) { return (n + 255u) & ~255ull; }
static bool aotx_media_add(unsigned long long &sum, unsigned long long n, unsigned long long count = 1)
{
    if (n > (~0ull - 255u) / count) return false;
    unsigned long long bytes = aotx_media_round(n * count);
    if (bytes > ~0ull - sum) return false;
    sum += bytes; return true;
}
static unsigned char *aotx_media_span(unsigned char *&at, unsigned long long n)
{
    unsigned char *out = at; at += aotx_media_round(n); return out;
}
void aotx_media_close(void)
{
    cudaFree(aotx_media_allocation); aotx_media_allocation = 0;
    cudaFree(aotx_media_weights); aotx_media_weights = 0;
    cudaFree(aotx_media_descriptor); aotx_media_descriptor = 0;
    memset(&aotx_media_host, 0, sizeof aotx_media_host);
    cudaMemcpyToSymbol(aotx_media, &aotx_media_host, sizeof aotx_media_host);
}
int aotx_media_allocate(const aotx_media_profile *p, const aotx_vision_desc *desc,
                         unsigned role, unsigned char **weights)
{
    if (aotx_media_allocation) return 1;
    unsigned long long coefficients = 3ull * (p->pixels + 30ull * p->dimension + 256u);
    unsigned long long work = 0, bytes = 0;
    if (!aotx_media_add(work, coefficients, 4) || !aotx_media_add(work, coefficients) ||
        !aotx_media_add(work, p->pixels, 3) || !aotx_media_add(work, p->horizontal, 4) ||
        !aotx_media_add(work, p->patches, 768) || !aotx_media_add(work, p->patches, 3072) ||
        !aotx_media_add(work, p->patches, 12288) || !aotx_media_add(work, p->patches, 6144) ||
        !aotx_media_add(work, p->patches, 6144) || !aotx_media_add(work, p->patches, 9216) ||
        !aotx_media_add(bytes, p->objects, sizeof(aotx_media_object)) ||
        !aotx_media_add(bytes, p->objects, sizeof(aotx_media_hash)) ||
        !aotx_media_add(bytes, p->workers, sizeof(aotx_image_job)) ||
        !aotx_media_add(bytes, p->workers, sizeof(aotx_vision_job)) ||
        !aotx_media_add(bytes, p->workers, sizeof(unsigned)) ||
        !aotx_media_add(bytes, p->bytes) || !aotx_media_add(bytes, p->feature_rows, 4096) ||
        !aotx_media_add(bytes, work, p->workers)) return 2;
    unsigned long long total = bytes;
    if (!aotx_media_add(total, desc->bytes) || !aotx_media_add(total, sizeof *desc)) return 2;
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || total > free_bytes) {
        fprintf(stderr, "image: storage needs %llu bytes; %zu bytes free\n", total, free_bytes);
        return 2;
    }
    if (cudaMalloc(&aotx_media_allocation, (size_t)bytes) != cudaSuccess ||
        cudaMalloc(&aotx_media_weights, (size_t)desc->bytes) != cudaSuccess ||
        cudaMalloc(&aotx_media_descriptor, sizeof *desc) != cudaSuccess ||
        cudaMemcpy(aotx_media_descriptor, desc, sizeof *desc, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(aotx_media_allocation, 0, (size_t)bytes) != cudaSuccess) {
        aotx_media_close(); return 1;
    }
    aotx_media_state &s = aotx_media_host;
    /* Preserve the ring binding if it was installed before the component. */
    if (cudaMemcpyFromSymbol(&s, aotx_media, sizeof s) != cudaSuccess) { aotx_media_close(); return 1; }
    s.profile = *p; s.coefficient_count = coefficients; s.workspace_each = work;
    s.allocated = total; s.role = role; s.enabled = 1;
    unsigned char *at = aotx_media_allocation;
    s.objects = (aotx_media_object *)aotx_media_span(at, (unsigned long long)p->objects * sizeof *s.objects);
    s.hash = (aotx_media_hash *)aotx_media_span(at, (unsigned long long)p->objects * sizeof *s.hash);
    s.image = (aotx_image_job *)aotx_media_span(at, (unsigned long long)p->workers * sizeof *s.image);
    s.vision = (aotx_vision_job *)aotx_media_span(at, (unsigned long long)p->workers * sizeof *s.vision);
    s.owner = (unsigned *)aotx_media_span(at, (unsigned long long)p->workers * sizeof *s.owner);
    s.source = aotx_media_span(at, p->bytes);
    s.features = (float *)aotx_media_span(at, (unsigned long long)p->feature_rows * 4096u);
    s.workspace = aotx_media_span(at, work * p->workers);
    if (cudaMemcpyToSymbol(aotx_media, &s, sizeof s) != cudaSuccess) { aotx_media_close(); return 1; }
    aotx_media_initialize<<<(p->workers + 63u) / 64u,64>>>();
    if (cudaDeviceSynchronize() != cudaSuccess) { aotx_media_close(); return 1; }
    *weights = aotx_media_weights;
    printf("image: %u sources, %llu source bytes, %u feature rows, %u workspaces, %llu device bytes\n",
        p->objects, (unsigned long long)p->bytes, p->feature_rows, p->workers, total);
    return 0;
}
void aotx_media_capture(cudaStream_t on)
{
    aotx_media_ingest<<<1,1,0,on>>>();
    if (!aotx_media_host.enabled) return;
    aotx_media_state &s = aotx_media_host;
    aotx_media_hash_step<<<(s.profile.objects+63u)/64u,64,0,on>>>(s.hash, s.profile.objects, 128);
    aotx_media_schedule<<<1,1,0,on>>>();
    aotx_image_capture(on, s.image, s.profile.workers, 256);
    aotx_vision_capture(on, s.vision, s.profile.workers, aotx_media_weights,
        aotx_media_descriptor, s.profile.patches, 256);
    aotx_media_complete<<<1,1,0,on>>>();
}
