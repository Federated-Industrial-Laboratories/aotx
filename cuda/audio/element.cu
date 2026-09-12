/* Purpose: Apply sound encoder biases, activations, residuals and ordered pooling.
 * Owns: Job row values; model parameters remain immutable.
 * Launch shape: Independent elements in x and jobs in y.
 * Lifetime: One finite encoder step. */
#include "audio/audio.cuh"
__global__ void aotx_audio_element(aotx_audio_job *jobs,unsigned count,const unsigned char *weights,
    const aotx_audio_desc *desc,unsigned operation)
{
    if(blockIdx.y>=count)return; aotx_audio_job &j=jobs[blockIdx.y];
    unsigned phase=operation==0?AOTX_AUDIO_FIRST:operation==1?AOTX_AUDIO_SECOND:
        operation<5?AOTX_AUDIO_PREPARE:operation<8?AOTX_AUDIO_BLOCK:AOTX_AUDIO_POOL;
    if(j.phase!=phase)return;
    unsigned width=operation==6?5120u:operation==9?4096u:1280u;
    unsigned rows=operation==0?3000u:operation>=8?j.rows:j.keys;
    const float *bias=0;
    if(operation<2)bias=(const float *)(weights+desc->base[operation?AOTX_AUDIO_CONV2_BIAS:AOTX_AUDIO_CONV1_BIAS]);
    else if(operation==9)bias=(const float *)(weights+desc->base[AOTX_AUDIO_PROJECT_BIAS]);
    else if(operation!=3 && operation<8){
        unsigned item=operation==2?AOTX_AUDIO_Q_BIAS:operation==4?AOTX_AUDIO_V_BIAS:
            operation==5?AOTX_AUDIO_OUT_BIAS:operation==6?AOTX_AUDIO_UP_BIAS:AOTX_AUDIO_DOWN_BIAS;
        bias=(const float *)(weights+desc->layer[j.layer][item]);
    }
    const float *position=(const float *)(weights+desc->base[AOTX_AUDIO_POSITION]);
    for(unsigned at=blockIdx.x*blockDim.x+threadIdx.x;at<rows*width;at+=gridDim.x*blockDim.x){
        unsigned row=at/width,c=at%width;
        if(operation==8){j.conv[at]=(j.residual[(2u*row)*1280u+c]+j.residual[(2u*row+1u)*1280u+c])*0.5f;continue;}
        float v=j.product[at]+(bias?bias[c]:0.0f);
        if(operation<2 || operation==6)v=0.5f*v*(1.0f+erff(v*0.7071067811865475244f));
        if(operation==0)j.conv[at]=aotx_audio_finite(j,v);
        else if(operation==1)j.residual[at]=aotx_audio_finite(j,v+position[at]);
        else if(operation<5)j.qkv[row*3840u+(operation-2u)*1280u+c]=aotx_audio_finite(j,v);
        else if(operation==5 || operation==7)j.residual[at]=aotx_audio_finite(j,j.residual[at]+v);
        else if(operation==6)aotx_audio_input(j,at,v);
        else if(operation==9)j.features[at]=aotx_audio_finite(j,v);
    }
}
