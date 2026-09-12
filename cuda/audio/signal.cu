/* Purpose: Decode PCM and resample complete timelines without transport boundary resets.
 * Owns: Decoded and canonical mono sample values in each job's spans.
 * Launch shape: Samples in x and independent jobs in y.
 * Lifetime: Two finite frontend steps. */
#include "audio/audio.cuh"
#include "media/wire.h"
__global__ void aotx_audio_decode(aotx_audio_job *jobs, unsigned count)
{
    if (blockIdx.y>=count) return; aotx_audio_job &j=jobs[blockIdx.y];
    if (j.phase!=AOTX_AUDIO_DECODE) return;
    const unsigned char *p=j.source+j.data_offset; unsigned size=j.encoding==AOTX_AUDIO_S16?2u:4u;
    for (unsigned at=blockIdx.x*blockDim.x+threadIdx.x;at<j.source_frames;at+=gridDim.x*blockDim.x) {
        float value=0.0f;
        for (unsigned c=0;c<j.channels;++c) {
            unsigned v=(unsigned)aotx_media_get(p+((unsigned long long)at*j.channels+c)*size,size);
            float sample=size==2u?(float)(int16_t)v*(1.0f/32768.0f):__uint_as_float(v);
            sample=aotx_audio_finite(j,sample); value += j.channels==1u?sample:sample*0.5f;
        }
        j.decoded[at]=aotx_audio_finite(j,value);
    }
}
__global__ void aotx_audio_resample(aotx_audio_job *jobs, unsigned count, aotx_audio_coefficients c)
{
    if (blockIdx.y>=count) return; aotx_audio_job &j=jobs[blockIdx.y];
    if (j.phase!=AOTX_AUDIO_RESAMPLE) return;
    for (unsigned at=blockIdx.x*blockDim.x+threadIdx.x;at<AOTX_AUDIO_SAMPLES;at+=gridDim.x*blockDim.x) {
        float value=0.0f;
        if (at<j.samples) {
            if (j.rate==16000u) value=j.decoded[at];
            else {
                unsigned phases=j.rate==44100u?160u:1u, orig=j.rate==44100u?441u:3u;
                unsigned width=j.rate==44100u?187u:203u, taps=2u*width+orig;
                const float *weights=(j.rate==44100u?c.resample441:c.resample48)+(at%phases)*taps;
                int first=(int)((at/phases)*orig)-(int)width; double sum=0.0;
                for (unsigned k=0;k<taps;++k) {
                    int index=first+(int)k;
                    if (index>=0 && (unsigned)index<j.source_frames) sum += (double)weights[k]*j.decoded[index];
                }
                value=(float)sum;
            }
            value=aotx_audio_finite(j,value); atomicMax(&j.peak,__float_as_uint(fabsf(value)));
        }
        j.samples_out[at]=value;
    }
}
