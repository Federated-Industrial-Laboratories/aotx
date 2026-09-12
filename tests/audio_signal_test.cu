/* Purpose: Check PCM, resampling and spectral values against independent arithmetic.
 * Owns: Distinct source waves, host reference values and bounded CUDA job buffers.
 * Launch shape: Complete frontend batches at N=1 and N=64.
 * Lifetime: One test process; no language model is loaded. */
#include "audio/audio.cuh"
#include "media/wire.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static unsigned checks,failures;
static constexpr double pi=3.1415926535897932384626433832795;
static void check(bool v,const char *s) { ++checks;if(!v){++failures;fprintf(stderr,"FAIL %s\n",s);} }
static void cu(cudaError_t r) { if(r!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(r));exit(1);} }
static void fft(cufftResult r) { if(r!=CUFFT_SUCCESS){fprintf(stderr,"FFT %d\n",(int)r);exit(1);} }
static std::vector<void*> allocations;
template<class T> static void take(T **p,size_t n) { cu(cudaMalloc(p,n*sizeof(T)));allocations.push_back(*p); }
static double coefficient(unsigned out,int input,unsigned rate) {
    double t=(input-(double)out*rate/16000)*16000*.9475937167399596/rate;
    t=std::max(-64.0,std::min(64.0,t));
    double window=std::cyl_bessel_i(0.0,14.769656459379492*sqrt(std::max(0.0,1-t*t/4096)))/
                  std::cyl_bessel_i(0.0,14.769656459379492);
    return (t==0?1:sin(pi*t)/(pi*t))*window*16000*.9475937167399596/rate;
}
static double hz(double mel) { return mel<15?mel*(200.0/3.0):1000*exp((mel-15)*log(6.4)/27); }
static void compare(const std::vector<float> &got,const std::vector<double> &expected,double absolute,double relative,const char *name) {
    double square=0,norm=0,maximum=0;unsigned bad=0;
    check(got.size()==expected.size(),"reference and output extents agree");
    for(size_t i=0;i<got.size();++i){double d=got[i]-expected[i];bad+=!std::isfinite(got[i])||fabs(d)>absolute;
        square+=d*d;norm+=expected[i]*expected[i];maximum=std::max(maximum,fabs(d));}
    double error=sqrt(square/std::max(norm,1e-300));
    if(bad||error>relative)printf("stage=%s maximum=%.9g relative=%.9g outside=%u\n",name,maximum,error,bad);
    check(!bad&&error<=relative,name);
}
struct reference { std::vector<unsigned char> source;std::vector<double> decoded,samples;unsigned frames,rate,channels,encoding; };
static reference make(unsigned id) {
    reference r{};unsigned rates[3]={16000,44100,48000};r.rate=rates[id%3];r.channels=1+(id/3)%2;
    r.encoding=(id/6)%2?3:1;r.frames=1200+id*17;unsigned bytes=r.encoding==1?2:4;
    r.source.resize(32+(size_t)r.frames*r.channels*bytes);memcpy(r.source.data(),"AOTXPCM1",8);
    aotx_media_put(r.source.data()+8,r.encoding,4);aotx_media_put(r.source.data()+12,r.rate,4);
    aotx_media_put(r.source.data()+16,r.channels,4);aotx_media_put(r.source.data()+24,r.frames,8);
    for(unsigned j=0;j<r.frames;++j){float mean=0;
        for(unsigned c=0;c<r.channels;++c){double signal=.17*sin(2*pi*(311+id*31)*j/r.rate+.17*c)+
            .09*cos(2*pi*(1371+id*7)*j/r.rate+.31*c);
            if(r.rate>16000)signal+=.04*sin(2*pi*12000*j/r.rate+.09*c);
            if(j==id*7)signal+=.21;
            float value=(float)signal;unsigned bits=0;
            if(bytes==2){int16_t q=(int16_t)lround(signal*32768);bits=(uint16_t)q;value=q/32768.0f;}
            else memcpy(&bits,&value,4);
            aotx_media_put(r.source.data()+32+((size_t)j*r.channels+c)*bytes,bits,bytes);
            mean+=r.channels==1?value:value*.5f;
        }r.decoded.push_back(mean);
    }
    unsigned samples=(r.frames*16000u+r.rate-1)/r.rate;r.samples.resize(samples);
    for(unsigned j=0;j<samples;++j){
        if(r.rate==16000){r.samples[j]=r.decoded[j];continue;}
        unsigned orig=r.rate==44100?441:3,phases=r.rate==44100?160:1,width=r.rate==44100?187:203;
        int first=(int)(j/phases*orig)-(int)width;double value=0;
        for(unsigned k=0;k<2*width+orig;++k){int in=first+(int)k;
            if(in>=0&&(unsigned)in<r.frames)value+=coefficient(j,in,r.rate)*r.decoded[in];}
        r.samples[j]=value;
    }
    return r;
}
static std::vector<double> spectral(const reference &r) {
    unsigned live=(r.samples.size()+359)/160;std::vector<double> power((size_t)live*201);
    for(unsigned frame=0;frame<live;++frame)for(unsigned bin=0;bin<=200;++bin){double re=0,im=0;
        for(unsigned k=0;k<400;++k){int at=(int)(frame*160+k)-200;if(at<0)at=-at;
            double x=(unsigned)at<r.samples.size()?r.samples[at]:0;
            double window=(1-cos(2*pi*k/400))*.5,angle=2*pi*bin*k/400;
            re+=x*window*cos(angle);im-=x*window*sin(angle);}
        power[(size_t)frame*201+bin]=re*re+im*im;
    }
    std::vector<double> result(128u*3000u,-10.0);double high=-10,top=15+log(8)*27/log(6.4);
    for(unsigned m=0;m<128;++m){double left=hz(top*m/129),centre=hz(top*(m+1)/129),right=hz(top*(m+2)/129);
        for(unsigned frame=0;frame<live;++frame){double sum=0;
            for(unsigned k=0;k<=200;++k){double weight=std::max(0.0,std::min((k*40-left)/(centre-left),(right-k*40)/(right-centre)))*2/(right-left);
                sum+=power[(size_t)frame*201+k]*weight;}
            double value=log10(std::max(sum,1e-10));result[(size_t)m*3000+frame]=value;high=std::max(high,value);}
    }
    for(auto &v:result)v=(std::max(v,high-8)+4)*.25;return result;
}
static void run(unsigned count) {
    aotx_audio_coefficients coeff{};take(&coeff.resample441,160u*815u);take(&coeff.resample48,409u);take(&coeff.mel,128u*201u);
    aotx_audio_coefficients_make<<<128,256>>>(coeff);cu(cudaDeviceSynchronize());
    std::vector<aotx_audio_job> jobs(count);std::vector<cufftHandle> plans(count);aotx_audio_job *device=nullptr;take(&device,count);
    std::vector<unsigned char*> source(count);
    for(unsigned j=0;j<count;++j){auto &v=jobs[j];take(&source[j],32+2400u*8u);take(&v.decoded,2400);
        take(&v.samples_out,480000);take(&v.fft_input,3000u*400u);take(&v.spectrum,3000u*201u);take(&v.mel,128u*3000u);
        fft(cufftPlan1d(&plans[j],400,CUFFT_R2C,3000));}
    for(unsigned first=0;first<64;first+=count){std::vector<reference> refs;
        for(unsigned j=0;j<count;++j){refs.push_back(make(first+j));auto &r=refs.back();auto &v=jobs[j];
            cu(cudaMemcpy(source[j],r.source.data(),r.source.size(),cudaMemcpyHostToDevice));v.source=source[j];v.source_bytes=r.source.size();
            v.format=4;v.phase=0;v.status=v.cancel=v.peak=0;v.source_capacity=2400;v.feature_capacity=750;}
        cu(cudaMemcpy(device,jobs.data(),count*sizeof(*device),cudaMemcpyHostToDevice));
        aotx_audio_step<<<1,64>>>(device,count);aotx_audio_decode<<<dim3(16,count),256>>>(device,count);
        aotx_audio_finish<<<1,64>>>(device,count,256);
        aotx_audio_resample<<<dim3(64,count),256>>>(device,count,coeff);aotx_audio_finish<<<1,64>>>(device,count,256);
        aotx_audio_window<<<dim3(128,count),256>>>(device,count);
        for(unsigned j=0;j<count;++j)fft(cufftExecR2C(plans[j],jobs[j].fft_input,jobs[j].spectrum));
        aotx_audio_mel<<<dim3(128,count),128>>>(device,count,coeff);aotx_audio_log<<<count,256>>>(device,count);
        aotx_audio_floor<<<dim3(128,count),256>>>(device,count);cu(cudaGetLastError());cu(cudaDeviceSynchronize());
        cu(cudaMemcpy(jobs.data(),device,count*sizeof(*device),cudaMemcpyDeviceToHost));
        for(unsigned j=0;j<count;++j){auto &r=refs[j];auto &v=jobs[j];check(!v.status&&v.phase==AOTX_AUDIO_MEL,"complete finite frontend is active");
            std::vector<float> decoded(r.frames),samples(r.samples.size()),mel(128u*3000u);
            cu(cudaMemcpy(decoded.data(),v.decoded,decoded.size()*4,cudaMemcpyDeviceToHost));
            cu(cudaMemcpy(samples.data(),v.samples_out,samples.size()*4,cudaMemcpyDeviceToHost));
            cu(cudaMemcpy(mel.data(),v.mel,mel.size()*4,cudaMemcpyDeviceToHost));
            bool exact=true;for(unsigned k=0;k<r.frames;++k)exact&=decoded[k]==r.decoded[k];check(exact,"PCM decoding and stereo mean are exact");
            if(r.rate==16000){check(memcmp(decoded.data(),samples.data(),samples.size()*4)==0,"16 kHz input bypass is bit exact");}
            compare(samples,r.samples,3e-6,1e-5,"resampling meets independent filter limits");
            compare(mel,spectral(r),1e-4,2e-5,"log mel meets independent direct Fourier limits");
        }
    }
    for(unsigned variant=0;variant<8;++variant){
        for(unsigned j=0;j<count;++j){auto &v=jobs[j];unsigned frames=800+j*13,channels=variant==6?2:1;
            std::vector<unsigned char> raw(32+(size_t)frames*channels*4,0);memcpy(raw.data(),"AOTXPCM1",8);
            aotx_media_put(raw.data()+8,3,4);aotx_media_put(raw.data()+12,16000,4);
            aotx_media_put(raw.data()+16,channels,4);aotx_media_put(raw.data()+24,frames,8);
            for(unsigned k=0;k<frames;++k)for(unsigned c=0;c<channels;++c){
                float value=variant==7?1e-20f:variant==6?(c?-.125f:.125f):0;
                unsigned bits;memcpy(&bits,&value,4);aotx_media_put(raw.data()+32+((size_t)k*channels+c)*4,bits,4);}
            if(variant<4){unsigned patterns[4]={0x7fc00001u,0x7f800000u,0xff800000u,0x7f7fffffu};
                aotx_media_put(raw.data()+32+(size_t)(311+j)*4,patterns[variant],4);}
            cu(cudaMemcpy(source[j],raw.data(),raw.size(),cudaMemcpyHostToDevice));
            v.source_bytes=raw.size();v.phase=v.status=v.peak=0;v.cancel=variant==5;
        }
        cu(cudaMemcpy(device,jobs.data(),count*sizeof(*device),cudaMemcpyHostToDevice));
        aotx_audio_step<<<1,64>>>(device,count);aotx_audio_decode<<<dim3(16,count),256>>>(device,count);
        aotx_audio_finish<<<1,64>>>(device,count,256);aotx_audio_resample<<<dim3(64,count),256>>>(device,count,coeff);
        aotx_audio_finish<<<1,64>>>(device,count,256);aotx_audio_window<<<dim3(128,count),256>>>(device,count);
        for(unsigned j=0;j<count;++j)fft(cufftExecR2C(plans[j],jobs[j].fft_input,jobs[j].spectrum));
        aotx_audio_mel<<<dim3(128,count),128>>>(device,count,coeff);aotx_audio_finish<<<1,64>>>(device,count,256);
        cu(cudaGetLastError());cu(cudaDeviceSynchronize());cu(cudaMemcpy(jobs.data(),device,count*sizeof(*device),cudaMemcpyDeviceToHost));
        for(auto &v:jobs){unsigned status=variant<4?AOTX_AUDIO_NONFINITE:variant==5?AOTX_AUDIO_CANCELLED:
            variant==7?0:AOTX_AUDIO_NO_SIGNAL;
            check(v.status==status,"nonfinite samples, power overflow, zero signals and cancellation have distinct statuses");
            check(v.phase==(status?AOTX_AUDIO_REFUSED:AOTX_AUDIO_FIRST),"very quiet nonzero input remains distinct from exact silence");}
    }
    for(auto p:plans)fft(cufftDestroy(p));for(void *p:allocations)cu(cudaFree(p));allocations.clear();
    printf("audio signal N=%u checks=%u failures=%u\n",count,checks,failures);
}
int main() { run(1);run(64);return failures?1:0; }
