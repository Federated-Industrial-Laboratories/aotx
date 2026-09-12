/* Purpose: Make trained Whisper log-mel features from canonical samples.
 * Owns: Windowed FFT inputs, mel powers and one maximum per item.
 * Launch shape: Independent frames and bins in x, jobs in y.
 * Lifetime: One finite spectral step around the captured CUDA FFT. */
#include "audio/audio.cuh"
__global__ void aotx_audio_window(aotx_audio_job *jobs,unsigned count)
{
    if(blockIdx.y>=count)return; aotx_audio_job &j=jobs[blockIdx.y];
    if(j.phase!=AOTX_AUDIO_MEL)return;
    for(unsigned at=blockIdx.x*blockDim.x+threadIdx.x;at<3000u*400u;at+=gridDim.x*blockDim.x){
        unsigned frame=at/400u,k=at%400u; int sample=(int)(frame*160u+k)-200;
        if(sample<0)sample=-sample;
        if(sample>=480000)sample=959998-sample;
        double window=0.5-0.5*cos(6.2831853071795864769*(double)k/400.0);
        j.fft_input[at]=aotx_audio_finite(j,(float)((double)j.samples_out[sample]*window));
    }
}
__global__ void aotx_audio_mel(aotx_audio_job *jobs,unsigned count,aotx_audio_coefficients c)
{
    if(blockIdx.y>=count)return; aotx_audio_job &j=jobs[blockIdx.y];
    if(j.phase!=AOTX_AUDIO_MEL)return;
    unsigned lane=threadIdx.x%32u;
    for(unsigned at=blockIdx.x*4u+threadIdx.x/32u;at<128u*3000u;at+=gridDim.x*4u){
        unsigned mel=at/3000u,frame=at%3000u; float sum=0.0f;
        for(unsigned k=lane;k<201u;k+=32u){
            cufftComplex z=j.spectrum[frame*201u+k];
            float power=aotx_audio_finite(j,z.x*z.x+z.y*z.y);
            sum+=power*c.mel[mel*201u+k];
        }
        sum=aotx_audio_sum(sum);
        if(!lane)j.mel[at]=aotx_audio_finite(j,log10f(fmaxf(sum,1.0e-10f)));
    }
}
__global__ void aotx_audio_log(aotx_audio_job *jobs,unsigned count)
{
    if(blockIdx.x>=count)return; aotx_audio_job &j=jobs[blockIdx.x];
    if(j.phase!=AOTX_AUDIO_MEL)return;
    __shared__ float peak[8]; float high=-10.0f;
    for(unsigned at=threadIdx.x;at<128u*3000u;at+=blockDim.x)high=fmaxf(high,j.mel[at]);
    for(unsigned d=16;d;d>>=1)high=fmaxf(high,__shfl_xor_sync(0xffffffffu,high,d));
    if(!(threadIdx.x%32u))peak[threadIdx.x/32u]=high; __syncthreads();
    if(threadIdx.x==0){for(unsigned i=0;i<8;++i)high=fmaxf(high,peak[i]);j.log_peak=high;}
}
__global__ void aotx_audio_floor(aotx_audio_job *jobs,unsigned count)
{
    if(blockIdx.y>=count)return; aotx_audio_job &j=jobs[blockIdx.y];
    if(j.phase!=AOTX_AUDIO_MEL)return;
    for(unsigned at=blockIdx.x*blockDim.x+threadIdx.x;at<128u*3000u;at+=gridDim.x*blockDim.x)
        j.mel[at]=(fmaxf(j.mel[at],j.log_peak-8.0f)+4.0f)*0.25f;
}
