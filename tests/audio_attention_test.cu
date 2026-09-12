/* Purpose: Check sound attention across finite slices and the full trained extent.
 * Owns: Distinct query, key and value data, output guards and independent references.
 * Launch shape: N=1 and N=64 jobs with different valid lengths and row contents.
 * Lifetime: One test process; each job completes at most six attention slices. */
#include "audio/audio.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static unsigned checks,failures;
static constexpr unsigned maximum=1500,width=1280,guard=8,quantum=256;
static constexpr float untouched=-1000.0f;
static void check(bool good,const char *name)
{
    ++checks;if(!good){++failures;fprintf(stderr,"FAIL %s\n",name);}
}
static void cu(cudaError_t rc)
{
    if(rc!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(rc));exit(1);}
}
static __host__ __device__ float value(unsigned job,unsigned row,unsigned col,unsigned part)
{
    if(part==0)return 0.5f+((int)((29u*row+7u*col+31u*job)%257u)-128)*(1.0f/512.0f);
    if(part==1)return ((int)((11u*row+13u*col+43u*job)%263u)-131)*(1.0f/512.0f);
    return ((int)((19u*row+17u*col+59u*job)%269u)-134)*(1.0f/256.0f)+job*(1.0f/128.0f);
}
__global__ void aotx_audio_attention_fill(aotx_audio_job *jobs,unsigned count)
{
    unsigned job=blockIdx.y;if(job>=count)return;
    aotx_audio_job &j=jobs[job];
    for(unsigned at=blockIdx.x*blockDim.x+threadIdx.x;at<maximum*3840u;at+=gridDim.x*blockDim.x){
        unsigned row=at/3840u,part=(at%3840u)/width,col=at%width;
        j.qkv[at]=row<j.keys?value(job,row,col,part):(part==1?20.0f:40.0f);
    }
    for(unsigned at=blockIdx.x*blockDim.x+threadIdx.x;at<(maximum+guard)*width;at+=gridDim.x*blockDim.x)
        j.input[at]=untouched;
}

/* Double dot products and a two-pass softmax do not use the device reduction. */
static void reference(unsigned job,unsigned row,unsigned head,unsigned keys,double *out)
{
    const unsigned columns[5]={0,17,31,32,63};
    std::vector<double> weights(keys);double top=-INFINITY,mass=0;
    for(unsigned k=0;k<keys;++k){
        double dot=0;
        for(unsigned col=0;col<64;++col)
            dot+=(double)value(job,row,head*64u+col,0)*value(job,k,head*64u+col,1);
        weights[k]=dot*0.125;top=std::max(top,weights[k]);
    }
    for(unsigned k=0;k<keys;++k){weights[k]=exp(weights[k]-top);mass+=weights[k];}
    for(unsigned c=0;c<5;++c){
        double sum=0;
        for(unsigned k=0;k<keys;++k)sum+=weights[k]*value(job,k,head*64u+columns[c],2);
        out[c]=sum/mass;
    }
}
static void exercise(unsigned count)
{
    const unsigned lengths[]={1,2,255,256,257,511,512,513,767,768,769,1023,1024,1025,1279,1280,1281,1499,1500};
    const unsigned probes[]={0,1,254,255,256,257,510,511,512,513,767,768,1023,1024,1279,1280,1498,1499};
    const unsigned heads[]={0,9,19},columns[]={0,17,31,32,63};
    const size_t input_span=(maximum+guard)*width,qkv_span=maximum*3840u;
    float *qkv=nullptr,*input=nullptr;aotx_audio_job *device=nullptr;
    cu(cudaMalloc(&qkv,count*qkv_span*sizeof(float)));cu(cudaMalloc(&input,count*input_span*sizeof(float)));
    cu(cudaMalloc(&device,count*sizeof(aotx_audio_job)));
    std::vector<aotx_audio_job> jobs(count);
    std::vector<unsigned> completed(count,0);
    for(unsigned i=0;i<count;++i){
        auto &j=jobs[i];j.keys=count==1?maximum:lengths[i%(sizeof(lengths)/sizeof(lengths[0]))];
        j.phase=AOTX_AUDIO_ATTEND;j.layer=7;j.qkv=qkv+i*qkv_span;j.input=input+i*input_span;
    }
    cu(cudaMemcpy(device,jobs.data(),count*sizeof(aotx_audio_job),cudaMemcpyHostToDevice));
    aotx_audio_attention_fill<<<dim3(128,count),256>>>(device,count);cu(cudaDeviceSynchronize());
    std::vector<float> actual(input_span);unsigned compared=0,expected=0;
    for(const auto &j:jobs)for(unsigned row:probes)if(row<j.keys)expected+=15;
    for(unsigned slice=0;slice<6;++slice){
        auto prior=jobs;
        aotx_audio_attention<<<dim3((quantum+3u)/4u,20,count),128>>>(device,count,quantum);
        aotx_audio_finish<<<(count+63u)/64u,64>>>(device,count,quantum);
        cu(cudaDeviceSynchronize());cu(cudaMemcpy(jobs.data(),device,count*sizeof(aotx_audio_job),cudaMemcpyDeviceToHost));
        for(unsigned i=0;i<count;++i){
            auto &j=jobs[i];const auto &old=prior[i];
            unsigned next=old.phase==AOTX_AUDIO_ATTEND?old.query+quantum:old.query;
            unsigned phase=old.phase==AOTX_AUDIO_ATTEND?(next>=j.keys?AOTX_AUDIO_BLOCK:AOTX_AUDIO_ATTEND):AOTX_AUDIO_READY;
            check(j.query==next&&j.phase==phase&&j.layer==7&&!j.status,"slice progress belongs to its job");
            unsigned end=std::min(next,j.keys);bool live=true,clear=true;
            cu(cudaMemcpy(actual.data(),j.input,input_span*sizeof(float),cudaMemcpyDeviceToHost));
            for(size_t at=0;at<input_span;++at){
                if(at<(size_t)end*width)live&=std::isfinite(actual[at])&&fabsf(actual[at])<1.1f;
                else clear&=actual[at]==untouched;
            }
            check(live,"all completed attention rows are finite and written");
            check(clear,"future and padding rows retain their guards");
            for(unsigned row:probes)if(row>=completed[i]&&row<end){
                for(unsigned head:heads){
                    double wanted[5];reference(i,row,head,j.keys,wanted);
                    for(unsigned c=0;c<5;++c){
                        double got=actual[(size_t)row*width+head*64u+columns[c]];
                        // The absolute bound covers 1500 F32 accumulation terms of bounded magnitude.
                        check(fabs(got-wanted[c])<=2.0e-4+2.0e-5*fabs(wanted[c]),"attention agrees with independent double arithmetic");
                        ++compared;
                    }
                }
            }
            completed[i]=end;
            if(j.phase==AOTX_AUDIO_BLOCK)j.phase=AOTX_AUDIO_READY;
        }
        cu(cudaMemcpy(device,jobs.data(),count*sizeof(aotx_audio_job),cudaMemcpyHostToDevice));
    }
    for(unsigned i=0;i<count;++i)check(completed[i]==jobs[i].keys&&jobs[i].phase==AOTX_AUDIO_READY,"every valid row completes");
    check(compared==expected&&compared>0,"every selected reference value is checked");
    printf("audio attention N=%u reference_values=%u checks=%u failures=%u\n",count,compared,checks,failures);
    cu(cudaFree(device));cu(cudaFree(input));cu(cudaFree(qkv));
}
int main(void)
{
    exercise(1);exercise(64);return failures?1:0;
}
