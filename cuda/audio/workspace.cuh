/* Purpose: Name aligned numerical storage inside one audio workspace.
 * Owns: No allocation; pointer arithmetic describes the caller's device extent.
 * Launch shape: Host allocation glue and one device initializer per workspace.
 * Lifetime: One runtime allocation. */
#ifndef AOTX_AUDIO_WORKSPACE_CUH
#define AOTX_AUDIO_WORKSPACE_CUH
#include "audio/audio.cuh"
__host__ __device__ static inline unsigned long long aotx_audio_round(unsigned long long n)
{
    return (n+255u)&~255ull;
}
__host__ __device__ static inline unsigned char *aotx_audio_span(unsigned char *&p,unsigned long long n)
{
    unsigned char *out=p;p+=aotx_audio_round(n);return out;
}
__host__ __device__ static inline unsigned long long aotx_audio_work_bytes(unsigned frames)
{
    return aotx_audio_round((unsigned long long)frames*4u)+
        aotx_audio_round(480000ull*4u)+aotx_audio_round(3000ull*400u*4u)+
        aotx_audio_round(3000ull*201u*8u)+aotx_audio_round(128ull*3000u*4u)+
        aotx_audio_round(3000ull*1280u*4u)+aotx_audio_round(1500ull*1280u*4u)+
        aotx_audio_round(1500ull*5120u*4u)+aotx_audio_round(1500ull*3840u*4u)+
        aotx_audio_round(1500ull*5120u*4u);
}
__host__ __device__ static inline void aotx_audio_layout(aotx_audio_job &j,unsigned char *p,unsigned frames)
{
    j.decoded=(float*)aotx_audio_span(p,(unsigned long long)frames*4u);
    j.samples_out=(float*)aotx_audio_span(p,480000ull*4u);
    j.fft_input=(float*)aotx_audio_span(p,3000ull*400u*4u);
    j.spectrum=(cufftComplex*)aotx_audio_span(p,3000ull*201u*8u);
    j.mel=(float*)aotx_audio_span(p,128ull*3000u*4u);
    j.conv=(float*)aotx_audio_span(p,3000ull*1280u*4u);
    j.residual=(float*)aotx_audio_span(p,1500ull*1280u*4u);
    j.product=(float*)aotx_audio_span(p,1500ull*5120u*4u);
    j.qkv=(float*)aotx_audio_span(p,1500ull*3840u*4u);
    j.input=(float*)aotx_audio_span(p,1500ull*5120u*4u);
}
#endif
