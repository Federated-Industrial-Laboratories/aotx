/* Purpose: Apply trained affine LayerNorm to sound rows.
 * Owns: Full precision matrix inputs; residual rows remain in full precision.
 * Launch shape: One warp per row, independent jobs in y.
 * Lifetime: One encoder graph node. */
#include "audio/audio.cuh"
__global__ void aotx_audio_norm(aotx_audio_job *jobs,unsigned count,const unsigned char *weights,
    const aotx_audio_desc *desc,unsigned operation)
{
    if(blockIdx.y>=count)return; aotx_audio_job &j=jobs[blockIdx.y];
    unsigned phase=operation==0?AOTX_AUDIO_PREPARE:operation==1?AOTX_AUDIO_BLOCK:AOTX_AUDIO_POOL;
    if(j.phase!=phase)return;
    const float *scale=(const float *)(weights+(operation==2?desc->base[AOTX_AUDIO_NORM]:
        desc->layer[j.layer][operation?AOTX_AUDIO_LN2:AOTX_AUDIO_LN1]));
    const float *bias=(const float *)(weights+(operation==2?desc->base[AOTX_AUDIO_NORM_BIAS]:
        desc->layer[j.layer][operation?AOTX_AUDIO_LN2_BIAS:AOTX_AUDIO_LN1_BIAS]));
    unsigned lane=threadIdx.x%32u,rows=operation==2?j.rows:j.keys;
    for(unsigned p=blockIdx.x*4u+threadIdx.x/32u;p<rows;p+=gridDim.x*4u){
        const float *row=(operation==2?j.conv:j.residual)+p*1280u; float sum=0;
        for(unsigned c=lane;c<1280u;c+=32u)sum+=row[c];
        float mean=aotx_audio_sum(sum)/1280.0f,square=0;
        for(unsigned c=lane;c<1280u;c+=32u){float d=row[c]-mean;square+=d*d;}
        float inverse=rsqrtf(aotx_audio_sum(square)/1280.0f+1.0e-5f);
        for(unsigned c=lane;c<1280u;c+=32u)aotx_audio_input(j,p*1280u+c,(row[c]-mean)*inverse*scale[c]+bias[c]);
    }
}
