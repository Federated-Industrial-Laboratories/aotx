/* Purpose: Compare batched native sound features with independent reference tensors.
 * Owns: Model bytes, reference cases, CUDA work spans and one captured graph.
 * Launch shape: N=1 or N=64 distinct source jobs in three input orders.
 * Lifetime: One test process. */
#include "audio/audio.cuh"
#include "disk/modelfile/audio.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
static unsigned checks,failures;
static void check(bool good,const char *what) { ++checks;if(!good){++failures;fprintf(stderr,"FAIL %s\n",what);} }
static void cu(cudaError_t rc) { if(rc!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(rc));exit(1);} }
static void fft(cufftResult rc) { if(rc!=CUFFT_SUCCESS){fprintf(stderr,"FFT %d\n",(int)rc);exit(1);} }
static std::vector<void*> allocations;
template<class T> static void take(T **p,size_t count) {
    cu(cudaMalloc(p,count*sizeof(T)));cu(cudaMemset(*p,0,count*sizeof(T)));allocations.push_back(*p);
}
template<class T> static std::vector<T> read(const std::string &path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f){fprintf(stderr,"cannot read %s\n",path.c_str());exit(1);}
    size_t n=(size_t)f.tellg();if(n%sizeof(T))exit(1);
    std::vector<T> v(n/sizeof(T));f.seekg(0);f.read((char*)v.data(),n);if(!f)exit(1);return v;
}
struct audio_case { std::string name;unsigned kind,samples,keys,rows,seen=0;bool no_signal=false; };
static void compare(const std::string &root,const std::string &out,const audio_case &c,
    const char *name,const float *device,unsigned rows,unsigned width,unsigned transpose=0,const float *bias=nullptr)
{
    auto expected=read<float>(root+"/"+c.name+"."+name+".f32");size_t count=(size_t)rows*width;
    check(expected.size()>=count,"reference extent contains the valid rows");if(expected.size()<count)return;
    std::vector<float> got(count);cu(cudaMemcpy(got.data(),device,count*4,cudaMemcpyDeviceToHost));
    double square=0,ref=0,maximum=0,cosine=1;size_t outside=0,bad=0;
    bool samples=std::string(name)=="samples",mel=std::string(name)=="mel";
    for(unsigned r=0;r<rows;++r){double a=0,b=0,dot=0;
        for(unsigned col=0;col<width;++col){size_t at=(size_t)r*width+col;
            double x=got[at]+(bias?bias[col]:0.0f),y=expected[transpose?(size_t)col*transpose+r:at];
            bad+=!std::isfinite(x)||!std::isfinite(y);double error=x-y;
            square+=error*error;ref+=y*y;maximum=std::max(maximum,fabs(error));
            outside+=fabs(error)>(samples?3.0e-6:mel?1.0e-4:0.02+0.01*fabs(y));
            a+=x*x;b+=y*y;dot+=x*y;
        }
        cosine=std::min(cosine,a>0&&b>0?dot/sqrt(a*b):a==b?1.0:0.0);
    }
    double relative=ref>0?sqrt(square/ref):square==0?0:INFINITY;
    bool good=!bad&&!outside&&relative<=(samples?1.0e-5:mel?2.0e-5:0.005)&&
        (std::string(name)!="features"||cosine>=0.9995);
    printf("case=%s stage=%s values=%zu relative=%.9g maximum=%.9g cosine=%.9g outside=%zu nonfinite=%zu\n",
        c.name.c_str(),name,count,relative,maximum,cosine,outside,bad);
    check(good,"native tensor meets the declared precision limits");
    if(!good){std::ofstream f(out+"/"+c.name+"."+name+".actual",std::ios::binary);f.write((char*)got.data(),count*4);}
}
int main(int argc,char **argv)
{
    if(argc!=6){fprintf(stderr,"usage: audio_test MODEL CASES COUNT ORDER OUTPUT\n");return 2;}
    unsigned count=(unsigned)strtoul(argv[3],nullptr,10),order=(unsigned)strtoul(argv[4],nullptr,10);
    if((count!=1&&count!=64)||order>2)return 2;
    aotx_modelfile *file=nullptr;aotx_audio_desc descriptor{};
    if(aotx_modelfile_open(argv[1],&file)||aotx_audio_file(file,&descriptor))return 1;
    unsigned char *weights=nullptr;aotx_audio_desc *desc=nullptr;
    take(&weights,descriptor.bytes);take(&desc,1);cu(cudaMemcpy(desc,&descriptor,sizeof descriptor,cudaMemcpyHostToDevice));
    std::vector<unsigned char> buffer(2u*1024u*1024u);
    for(uint64_t at=0;at<descriptor.bytes;at+=buffer.size()){
        size_t n=(size_t)std::min<uint64_t>(buffer.size(),descriptor.bytes-at);
        if(aotx_modelfile_read(file,at,n,buffer.data()))return 1;
        cu(cudaMemcpy(weights+at,buffer.data(),n,cudaMemcpyHostToDevice));
    }
    aotx_modelfile_close(file);
    aotx_audio_coefficients coeff{};take(&coeff.resample441,160u*815u);take(&coeff.resample48,409u);take(&coeff.mel,128u*201u);
    aotx_audio_coefficients_make<<<128,256>>>(coeff);cu(cudaDeviceSynchronize());
    std::ifstream list(std::string(argv[2])+"/cases.tsv");std::vector<audio_case> all(64),cases(count);
    for(auto &c:all)if(!(list>>c.name>>c.kind>>c.samples>>c.keys>>c.rows))return 1;
    std::vector<aotx_audio_job> jobs(count);std::vector<cufftHandle> plans(count);
    std::vector<float*> inputs(count);std::vector<cufftComplex*> spectra(count);
    for(unsigned i=0;i<count;++i){
        cases[i]=all[order==0?i:order==1?63u-i:(17u*i+7u)%64u];auto &c=cases[i];auto &j=jobs[i];
        auto canonical=read<float>(std::string(argv[2])+"/"+c.name+".samples.f32");
        c.no_signal=std::all_of(canonical.begin(),canonical.end(),[](float v){return v==0.0f;});
        auto bytes=read<unsigned char>(std::string(argv[2])+"/"+c.name+".source");
        unsigned char *source=nullptr;take(&source,bytes.size());cu(cudaMemcpy(source,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
        j.source=source;j.source_bytes=bytes.size();j.format=c.kind;j.source_capacity=1440000;j.feature_capacity=750;
        take(&j.decoded,1440000);take(&j.samples_out,480000);take(&j.fft_input,3000u*400u);
        take(&j.spectrum,3000u*201u);take(&j.mel,128u*3000u);take(&j.conv,3000u*1280u);
        take(&j.residual,1500u*1280u);take(&j.product,1500u*5120u);take(&j.qkv,1500u*3840u);
        take(&j.input,1500u*5120u);take(&j.features,750u*4096u);
        fft(cufftCreate(&plans[i]));fft(cufftSetAutoAllocation(plans[i],0));size_t work=0;
        fft(cufftMakePlan1d(plans[i],400,CUFFT_R2C,3000,&work));unsigned char *workspace=nullptr;
        if(work){take(&workspace,work);fft(cufftSetWorkArea(plans[i],workspace));}
        inputs[i]=j.fft_input;spectra[i]=j.spectrum;
    }
    auto bias1=read<float>(std::string(argv[2])+"/conv1.bias.f32"),bias2=read<float>(std::string(argv[2])+"/conv2.bias.f32");
    check(bias1.size()==1280&&bias2.size()==1280,"convolution bias reference widths");if(failures)return 1;
    aotx_audio_job *device=nullptr;take(&device,count);cu(cudaMemcpy(device,jobs.data(),count*sizeof(*device),cudaMemcpyHostToDevice));
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;cu(cudaStreamCreate(&stream));
    cu(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    aotx_audio_capture(stream,device,count,weights,desc,&coeff,plans.data(),inputs.data(),spectra.data(),256);
    cu(cudaStreamEndCapture(stream,&graph));cu(cudaGraphInstantiate(&exec,graph,0));
    auto start=std::chrono::steady_clock::now();bool done=false;unsigned steps=0;
    for(;steps<1000&&!done;++steps){
        cu(cudaGraphLaunch(exec,stream));cu(cudaStreamSynchronize(stream));
        cu(cudaMemcpy(jobs.data(),device,count*sizeof(*device),cudaMemcpyDeviceToHost));done=true;
        for(unsigned i=0;i<count;++i){auto &j=jobs[i];auto &c=cases[i];
            if(j.phase!=AOTX_AUDIO_READY&&j.phase!=AOTX_AUDIO_REFUSED)done=false;
            if(j.phase==AOTX_AUDIO_MEL&&!(c.seen&1)){
                check(j.samples==c.samples&&j.keys==c.keys&&j.rows==c.rows,"valid audio dimensions match");
                compare(argv[2],argv[5],c,"samples",j.samples_out,j.samples,1);c.seen|=1;
            }
            if(j.phase==AOTX_AUDIO_FIRST&&!(c.seen&2)){
                compare(argv[2],argv[5],c,"mel",j.mel,128,3000);c.seen|=2;
            }
            if(j.phase==AOTX_AUDIO_SECOND&&!(c.seen&4)){
                compare(argv[2],argv[5],c,"conv1",j.product,3000,1280,3000,bias1.data());c.seen|=4;
            }
            if(j.phase==AOTX_AUDIO_PREPARE&&j.layer==0&&!(c.seen&8)){
                compare(argv[2],argv[5],c,"conv2",j.product,j.keys,1280,1500,bias2.data());c.seen|=8;
            }
            for(unsigned k=0;k<3;++k){unsigned layer=k==0?1u:k==1?16u:32u,flag=16u<<k;
                if(j.layer==layer&&(j.phase==AOTX_AUDIO_PREPARE||j.phase==AOTX_AUDIO_POOL)&&!(c.seen&flag)){
                    const char *name=k==0?"block0":k==1?"block15":"block31";
                    compare(argv[2],argv[5],c,name,j.residual,j.keys,1280);c.seen|=flag;
                }
            }
            if(j.phase==AOTX_AUDIO_READY&&!(c.seen&128)){
                compare(argv[2],argv[5],c,"norm",j.input,j.rows,1280);
                compare(argv[2],argv[5],c,"features",j.features,j.rows,4096);c.seen|=128;
            }
        }
    }
    for(unsigned i=0;i<count;++i){
        printf("case=%s final_phase=%u status=%u seen=%u zero_signal=%u\n",cases[i].name.c_str(),jobs[i].phase,jobs[i].status,cases[i].seen,cases[i].no_signal?1u:0u);
        if(cases[i].no_signal){check(jobs[i].phase==AOTX_AUDIO_REFUSED&&jobs[i].status==AOTX_AUDIO_NO_SIGNAL,"exactly zero signal refuses before inference");}
        else {check(jobs[i].phase==AOTX_AUDIO_READY&&jobs[i].status==0,"audio job completes without status");check(cases[i].seen==255,"all reference stages were compared");}
    }
    check(done,"the bounded batch completes");
    printf("audio N=%u order=%u checks=%u failures=%u steps=%u seconds=%.6f\n",count,order,checks,failures,steps,
        std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    cu(cudaGraphExecDestroy(exec));cu(cudaGraphDestroy(graph));cu(cudaStreamDestroy(stream));
    for(auto p:plans)fft(cufftDestroy(p));for(void *p:allocations)cu(cudaFree(p));return failures?1:0;
}
