/* Purpose: Check complete sound headers, independent capacities and terminal states.
 * Owns: Distinct source bytes and bounded device job batches.
 * Launch shape: Each source case runs at N=1 and N=64.
 * Lifetime: One test process. */
#include "audio/audio.cuh"
#include "media/wire.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static unsigned checks,failures;
static void check(bool v,const char *s) { ++checks;if(!v){++failures;fprintf(stderr,"FAIL %s\n",s);} }
static void cu(cudaError_t r) { if(r!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(r));exit(1);} }
using bytes=std::vector<unsigned char>;
static void put(bytes &b,size_t at,unsigned long long v,unsigned n) { aotx_media_put(b.data()+at,v,n); }
static void tag(bytes &b,size_t at,const char *v) { memcpy(b.data()+at,v,4); }
static void chunk(bytes &b,const char *name,const bytes &data) {
    size_t at=b.size();b.resize(at+8+data.size()+(data.size()%2));tag(b,at,name);put(b,at+4,data.size(),4);
    memcpy(b.data()+at+8,data.data(),data.size());
}
static bytes format(unsigned encoding,unsigned rate,unsigned channels,unsigned extended) {
    unsigned width=encoding==1?16:32,align=channels*width/8;
    bytes b(extended?40:16);put(b,0,extended?65534:encoding,2);put(b,2,channels,2);
    put(b,4,rate,4);put(b,8,rate*align,4);put(b,12,align,2);put(b,14,width,2);
    if(extended){put(b,16,22,2);put(b,18,width,2);put(b,20,channels==1?4:3,4);put(b,24,encoding,4);
        const unsigned char guid[12]={0,0,16,0,128,0,0,170,0,56,155,113};memcpy(b.data()+28,guid,12);}
    return b;
}
struct source { bytes data;unsigned kind,status=0,frames,rate,channels,encoding,source_cap=1440000,feature_cap=750,cancel=0; };
static source make(unsigned i,unsigned mutation) {
    unsigned rate[3]={16000,44100,48000};source s{};
    s.rate=rate[i%3];s.channels=1+(i/3)%2;s.encoding=(i/6)%2?3:1;s.frames=1200+i*13;
    if(mutation==50||mutation==51)s.frames=s.rate*30+(mutation==51);
    if(mutation==52||mutation==53)s.frames=320*s.rate/16000+(mutation==53);
    unsigned stride=s.channels*(s.encoding==1?2:4);bytes pcm((size_t)s.frames*stride,(unsigned char)(i+1));
    s.kind=mutation>=40?4:3;
    if(s.kind==4){s.data.resize(32);memcpy(s.data.data(),"AOTXPCM1",8);put(s.data,8,s.encoding,4);
        put(s.data,12,s.rate,4);put(s.data,16,s.channels,4);put(s.data,24,s.frames,8);
        s.data.insert(s.data.end(),pcm.begin(),pcm.end());
        switch(mutation){case 41:s.data[0]^=1;break;case 42:put(s.data,20,1,4);break;
            case 43:put(s.data,24,s.frames+1,8);break;case 44:put(s.data,8,2,4);break;
            case 45:put(s.data,12,22050,4);break;case 46:put(s.data,16,3,4);break;
            case 47:s.data.pop_back();break;case 48:s.data.resize(31);break;case 49:s.data.resize(32);break;}
        if(mutation>40&&mutation<50)s.status=1;
        if(mutation==51||mutation==52)s.status=2;
        return s;
    }
    s.data.resize(12);tag(s.data,0,"RIFF");tag(s.data,8,"WAVE");
    auto fmt=format(s.encoding,s.rate,s.channels,i%2);bytes fact(4);put(fact,0,s.frames,4);
    if(mutation==1){fmt=format(s.encoding,s.rate,s.channels,0);fmt.resize(18);}
    if(mutation==2){chunk(s.data,"JUNK",bytes(3,19));chunk(s.data,"LIST",bytes{'I','N','F','O'});}
    if(mutation==3)chunk(s.data,"fact",fact);
    if(mutation==4)chunk(s.data,"data",pcm);
    if(mutation==5)put(fmt,0,2,2);
    if(mutation==6)put(fmt,2,3,2);
    if(mutation==7)put(fmt,4,22050,4);
    if(mutation==8)put(fmt,8,s.rate*stride+1,4);
    if(mutation==9)put(fmt,12,stride+1,2);
    if(mutation==10)put(fmt,14,24,2);
    if(mutation>=11&&mutation<=15){fmt=format(s.encoding,s.rate,s.channels,1);
        if(mutation==11)put(fmt,16,21,2);if(mutation==12)put(fmt,18,8,2);
        if(mutation==13)put(fmt,20,1,4);if(mutation==14)fmt[39]^=1;if(mutation==15)put(fmt,24,4,4);}
    if(mutation==16){fmt=format(s.encoding,s.rate,s.channels,0);fmt.resize(18);put(fmt,16,1,2);}
    if(mutation==17)fmt.resize(17);
    chunk(s.data,"fmt ",fmt);if(mutation==18)chunk(s.data,"fmt ",fmt);
    if(mutation==19)pcm.clear();if(mutation==20)pcm.pop_back();
    chunk(s.data,"data",pcm);if(mutation==21)chunk(s.data,"data",pcm);
    if(mutation==22){put(fact,0,s.frames+1,4);chunk(s.data,"fact",fact);}
    if(mutation==23){chunk(s.data,"fact",fact);chunk(s.data,"fact",fact);}
    if(mutation==24)chunk(s.data,"fact",bytes(3));
    if(mutation==25)chunk(s.data,"plst",bytes(4));if(mutation==26)chunk(s.data,"slnt",bytes(4));
    if(mutation==27)chunk(s.data,"LIST",bytes{'w','a','v','l'});
    if(mutation==28)chunk(s.data,"LIST",bytes(3));
    if(mutation==29){chunk(s.data,"JUNK",bytes(3));s.data.pop_back();}
    if(mutation==30)s.data.push_back(0);
    put(s.data,4,s.data.size()-8,4);
    if(mutation==31)s.data[0]^=1;if(mutation==32)s.data[8]^=1;
    if(mutation==33)put(s.data,4,s.data.size()-9,4);
    if(mutation==34)put(s.data,16,0xffffffffu,4);
    if(mutation>=4&&mutation<=34)s.status=1;
    if(mutation==35){s.source_cap=s.frames-1;s.status=2;}
    unsigned samples=(s.frames*16000u+s.rate-1)/s.rate;
    unsigned rows=(((samples+159)/160+1)/2)/2;
    if(mutation==36){s.feature_cap=rows-1;s.status=2;}
    if(mutation==37){s.source_cap=s.frames;s.feature_cap=rows;}
    if(mutation==38){s.cancel=1;s.status=3;}
    if(mutation==39){s.source_cap=0;s.status=2;}
    return s;
}
static void run(unsigned count) {
    aotx_audio_job *device=nullptr;cu(cudaMalloc(&device,count*sizeof(*device)));
    for(unsigned mutation=0;mutation<54;++mutation)for(unsigned first=0;first<64;first+=count){
        std::vector<source> sources;std::vector<aotx_audio_job> jobs(count);std::vector<unsigned char*> data(count);
        for(unsigned j=0;j<count;++j){sources.push_back(make(first+j,mutation));auto &s=sources.back();auto &v=jobs[j];
            cu(cudaMalloc(&data[j],s.data.size()));cu(cudaMemcpy(data[j],s.data.data(),s.data.size(),cudaMemcpyHostToDevice));
            v.source=data[j];v.source_bytes=s.data.size();v.format=s.kind;v.source_capacity=s.source_cap;
            v.feature_capacity=s.feature_cap;v.cancel=s.cancel;}
        cu(cudaMemcpy(device,jobs.data(),count*sizeof(*device),cudaMemcpyHostToDevice));
        aotx_audio_step<<<1,64>>>(device,count);cu(cudaGetLastError());cu(cudaDeviceSynchronize());
        cu(cudaMemcpy(jobs.data(),device,count*sizeof(*device),cudaMemcpyDeviceToHost));
        for(unsigned j=0;j<count;++j){auto &s=sources[j];auto &v=jobs[j];
            if(v.status!=s.status)fprintf(stderr,"case %u mutation %u status %u expected %u\n",first+j,mutation,v.status,s.status);
            check(v.status==s.status,"source has its independent expected status");
            check(v.phase==(s.status?AOTX_AUDIO_REFUSED:AOTX_AUDIO_DECODE),"source stops at the required phase");
            if(!s.status){check(v.rate==s.rate&&v.encoding==s.encoding&&v.channels==s.channels&&v.source_frames==s.frames,
                "accepted metadata matches the complete source");check(v.data_offset+v.data_bytes<=s.data.size(),"accepted PCM extent stays inside the source");}
            cu(cudaFree(data[j]));}
    }
    cu(cudaFree(device));printf("audio header N=%u checks=%u failures=%u\n",count,checks,failures);
}
int main() { run(1);run(64);return failures?1:0; }
