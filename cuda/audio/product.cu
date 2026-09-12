/* Purpose: Apply trained matrices to batches of native sound rows.
 * Owns: Shared matrix tiles; each job owns its product and full precision input spans.
 * Launch shape: Output tiles in x and y, independent jobs in z.
 * Lifetime: One product node of an encoder graph. */
#include "audio/audio.cuh"

__global__ __launch_bounds__(256,3) void aotx_audio_product(
    aotx_audio_job *jobs,unsigned count,const unsigned char *weights,
    const aotx_audio_desc *desc,unsigned operation)
{
    __shared__ float sx[32][32], sw[32][32];
    if(blockIdx.z>=count)return; aotx_audio_job &j=jobs[blockIdx.z];
    unsigned phase=operation==0?AOTX_AUDIO_FIRST:operation==1?AOTX_AUDIO_SECOND:
        operation<5?AOTX_AUDIO_PREPARE:operation<8?AOTX_AUDIO_BLOCK:AOTX_AUDIO_POOL;
    if(j.phase!=phase)return;
    unsigned m=j.keys,n=1280,k=1280; unsigned long long offset;
    if(operation==0){m=3000;k=384;offset=desc->base[AOTX_AUDIO_CONV1];}
    else if(operation==1){k=3840;offset=desc->base[AOTX_AUDIO_CONV2];}
    else if(operation==2)offset=desc->layer[j.layer][AOTX_AUDIO_Q];
    else if(operation==3)offset=desc->layer[j.layer][AOTX_AUDIO_K];
    else if(operation==4)offset=desc->layer[j.layer][AOTX_AUDIO_V];
    else if(operation==5)offset=desc->layer[j.layer][AOTX_AUDIO_OUT];
    else if(operation==6){n=5120;offset=desc->layer[j.layer][AOTX_AUDIO_UP];}
    else if(operation==7){k=5120;offset=desc->layer[j.layer][AOTX_AUDIO_DOWN];}
    else {m=j.rows;n=4096;offset=desc->base[AOTX_AUDIO_PROJECT];}
    if(blockIdx.y*32u>=m || blockIdx.x*32u>=n)return;
    unsigned r=(threadIdx.x/16u)*2u,c=(threadIdx.x%16u)*2u;
    unsigned row=blockIdx.y*32u+r,col=blockIdx.x*32u+c;
    const half *w=(const half *)(weights+offset);
    float a00=0.0f,a01=0.0f,a10=0.0f,a11=0.0f;
    for(unsigned base=0;base<k;base+=32u){
        for(unsigned at=threadIdx.x;at<1024u;at+=256u){
            unsigned y=at/32u,x=at%32u,source=blockIdx.y*32u+y,out=blockIdx.x*32u+y;
            sx[y][x]=source<m?j.input[(unsigned long long)source*k+base+x]:0.0f;
            sw[y][x]=out<n?__half2float(w[(unsigned long long)out*k+base+x]):0.0f;
        }
        __syncthreads();
        #pragma unroll
        for(unsigned i=0;i<32u;++i){
            float x0=sx[r][i],x1=sx[r+1u][i],w0=sw[c][i],w1=sw[c+1u][i];
            a00=__fmaf_rn(x0,w0,a00);a01=__fmaf_rn(x0,w1,a01);
            a10=__fmaf_rn(x1,w0,a10);a11=__fmaf_rn(x1,w1,a11);
        }
        __syncthreads();
    }
    if(row<m&&col<n)j.product[(unsigned long long)row*n+col]=a00;
    if(row<m&&col+1u<n)j.product[(unsigned long long)row*n+col+1u]=a01;
    if(row+1u<m&&col<n)j.product[(unsigned long long)(row+1u)*n+col]=a10;
    if(row+1u<m&&col+1u<n)j.product[(unsigned long long)(row+1u)*n+col+1u]=a11;
}
__global__ void aotx_audio_convolution(aotx_audio_job *jobs,unsigned count,unsigned operation)
{
    if(blockIdx.y>=count)return; aotx_audio_job &j=jobs[blockIdx.y];
    if(j.phase!=(operation?AOTX_AUDIO_SECOND:AOTX_AUDIO_FIRST))return;
    unsigned width=operation?1280u:128u, rows=operation?j.keys:3000u, k=width*3u;
    for(unsigned at=blockIdx.x*blockDim.x+threadIdx.x;at<rows*k;at+=gridDim.x*blockDim.x){
        unsigned row=at/k,c=(at%k)/3u; int t=(int)(row*(operation?2u:1u)+at%3u)-1;
        float v=t<0 || t>=3000?0.0f:operation?j.conv[(unsigned)t*1280u+c]:j.mel[c*3000u+(unsigned)t];
        aotx_audio_input(j,at,v);
    }
}
