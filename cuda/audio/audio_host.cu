/* Purpose: Capture finite sound preparation and encoder nodes.
 * Owns: No allocation; callers supply FFT plans and all device spans.
 * Launch shape: Host graph glue; samples, transforms and rows execute on CUDA.
 * Lifetime: Graph construction; job progress remains on the device. */
#include "audio/audio.cuh"
#include "boot/check.h"
void aotx_audio_capture(cudaStream_t on,aotx_audio_job *jobs,unsigned count,
    const unsigned char *weights,const aotx_audio_desc *desc,const aotx_audio_coefficients *coeff,
    const cufftHandle *plans,float *const *fft_input,cufftComplex *const *spectrum,unsigned quantum)
{
    if(!count || !quantum)return;
    unsigned groups=(count+63u)/64u; dim3 elements(256,count),norms(128,count);
    dim3 matrix(160,(3000u+31u)/32u,count);
    aotx_audio_step<<<groups,64,0,on>>>(jobs,count);
    aotx_audio_decode<<<elements,256,0,on>>>(jobs,count);
    aotx_audio_resample<<<elements,256,0,on>>>(jobs,count,*coeff);
    aotx_audio_window<<<elements,256,0,on>>>(jobs,count);
    for(unsigned i=0;i<count;++i){
        aotx_check_runtime(cufftSetStream(plans[i],on)==CUFFT_SUCCESS?cudaSuccess:cudaErrorUnknown,"cufftSetStream");
        aotx_check_runtime(cufftExecR2C(plans[i],fft_input[i],spectrum[i])==CUFFT_SUCCESS?cudaSuccess:cudaErrorUnknown,"cufftExecR2C");
    }
    aotx_audio_mel<<<dim3(1024,count),128,0,on>>>(jobs,count,*coeff);
    aotx_audio_log<<<count,256,0,on>>>(jobs,count);
    aotx_audio_floor<<<elements,256,0,on>>>(jobs,count);
    for(unsigned op=0;op<2;++op){
        aotx_audio_convolution<<<elements,256,0,on>>>(jobs,count,op);
        aotx_audio_product<<<matrix,256,0,on>>>(jobs,count,weights,desc,op);
        aotx_audio_element<<<elements,256,0,on>>>(jobs,count,weights,desc,op);
    }
    aotx_audio_norm<<<norms,128,0,on>>>(jobs,count,weights,desc,0);
    for(unsigned op=2;op<5;++op){
        aotx_audio_product<<<matrix,256,0,on>>>(jobs,count,weights,desc,op);
        aotx_audio_element<<<elements,256,0,on>>>(jobs,count,weights,desc,op);
    }
    aotx_audio_attention<<<dim3((quantum+3u)/4u,20,count),128,0,on>>>(jobs,count,quantum);
    for(unsigned op=5;op<8;++op){
        if(op==6)aotx_audio_norm<<<norms,128,0,on>>>(jobs,count,weights,desc,1);
        aotx_audio_product<<<matrix,256,0,on>>>(jobs,count,weights,desc,op);
        aotx_audio_element<<<elements,256,0,on>>>(jobs,count,weights,desc,op);
    }
    aotx_audio_element<<<elements,256,0,on>>>(jobs,count,weights,desc,8);
    aotx_audio_norm<<<norms,128,0,on>>>(jobs,count,weights,desc,2);
    aotx_audio_product<<<matrix,256,0,on>>>(jobs,count,weights,desc,8);
    aotx_audio_element<<<elements,256,0,on>>>(jobs,count,weights,desc,9);
    aotx_audio_finish<<<groups,64,0,on>>>(jobs,count,quantum);
}
