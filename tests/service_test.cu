/* Purpose: Check scoped service admission, complete prompts and exact request ownership.
 * Owns: Small mapped transport fixtures and isolated device tables.
 * Launch shape: Real service mailbox batches at N=1 and N=64.
 * Lifetime: Each fixture releases all of its mapped and device allocations. */
#include "service/internal.cuh"
#include "model/load.cuh"
#include "model/wrap.cuh"
#include "model/decode.cuh"
#include "media/runtime.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
static unsigned checks, failures;
static void check(bool good, const char *label)
{ ++checks; if (!good) { ++failures; fprintf(stderr,"FAIL %s\n",label); } }
static void cu(cudaError_t rc)
{ if (rc != cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(rc)); exit(1); } }
static void put(std::vector<unsigned char> &p, unsigned at, unsigned long long value, unsigned n=4)
{ for (unsigned i=0;i<n;++i) p[at+i]=(unsigned char)(value>>(8*i)); }
static void add(std::vector<unsigned char> &p, unsigned value)
{ unsigned at=(unsigned)p.size(); p.resize(at+4); put(p,at,value); }
static std::vector<unsigned char> frame(unsigned op,unsigned principal=0,unsigned request=0)
{
    std::vector<unsigned char> p(AOTX_SERVICE_HEAD,0);
    memcpy(p.data(),AOTX_SERVICE_MAGIC,8); put(p,8,op); put(p,16,principal); put(p,32,1,8);
    put(p,48,request);
    if (op==AOTX_SERVICE_SUBMIT || op==AOTX_SERVICE_READ || op==AOTX_SERVICE_CANCEL) put(p,40,17,8);
    return p;
}
static std::vector<unsigned char> submit(unsigned principal,unsigned request,const std::string &text)
{
    auto p=frame(AOTX_SERVICE_SUBMIT,principal,request);
    put(p,72,AOTX_MODEL_LANGUAGE);put(p,76,16);put(p,84,0x3f800000);
    add(p,4);
    const unsigned roles[]={0,1,2,1};
    const std::string words[] = {"rule",text,"answer","question"};
    for (unsigned i=0;i<4;++i) {
        add(p,roles[i]);add(p,1);add(p,0);add(p,(unsigned)words[i].size());
        p.insert(p.end(),words[i].begin(),words[i].end());
    }
    put(p,88,p.size()-AOTX_SERVICE_HEAD);return p;
}
static std::vector<unsigned char> text_parts(unsigned principal, unsigned request,
    unsigned role, const std::vector<std::string> &parts)
{
    auto p=frame(AOTX_SERVICE_SUBMIT,principal,request);
    put(p,72,AOTX_MODEL_LANGUAGE);put(p,76,16);put(p,84,0x3f800000);
    add(p,1);add(p,role);add(p,(unsigned)parts.size());
    for(const auto &text:parts) {
        add(p,0);add(p,(unsigned)text.size());p.insert(p.end(),text.begin(),text.end());
    }
    put(p,88,p.size()-AOTX_SERVICE_HEAD);return p;
}
__global__ void aotx_service_test_reset(void)
{
    aotx_sched.held=0;aotx_sched.start_ns=100;aotx_seam.replaying=0;
    aotx_model_load.pending_count=0;
    for(unsigned i=0;i<AOTX_MODEL_ROLES;++i) aotx_model_load.resident[i]={};
    aotx_model_load.resident[AOTX_MODEL_LANGUAGE].active=1;
    for(unsigned i=0;i<AOTX_SLOTS;++i) aotx_seqs.slot[i]={};
}
__global__ void aotx_service_test_terminal(unsigned count)
{
    for(unsigned i=0;i<count;++i) {
        auto &j=aotx_service.jobs[i];j.phase=AOTX_SERVICE_DONE;j.output=3;j.result[0]='a';j.result[1]='b';j.result[2]='c';
        j.prompt=12;j.sampled=2;j.finish=1;j.changed=i+1;
    }
}
__global__ void aotx_service_test_active(void)
{
    auto &j=aotx_service.jobs[0];j.phase=AOTX_SERVICE_RUNNING;j.slot=1;j.cancel=0;j.status=0;
    aotx_service.slot[1]=1;aotx_seqs.slot[1].state=AOTX_SEQ_STATE_DECODE;aotx_seqs.slot[1].flags=0;
}
__global__ void aotx_service_test_reused(void)
{
    auto &old=aotx_service.jobs[0];old.phase=AOTX_SERVICE_DONE;old.slot=AOTX_SLOTS;
    auto &next=aotx_service.jobs[AOTX_SERVICE_REQUESTS-1];next.phase=AOTX_SERVICE_RUNNING;next.slot=1;
    next.id[0]=222;next.principal[0]=1;next.revision=1;
    aotx_service.slot[1]=AOTX_SERVICE_REQUESTS;
    aotx_seqs.slot[1].flags=0;
}
__global__ void aotx_service_test_lease(unsigned *out)
{
    auto &j=aotx_service.jobs[0];j.phase=AOTX_SERVICE_QUEUED;j.media_count=1;j.media[0]={7,99};
    out[0]=aotx_service_media_leased(7,99);out[1]=aotx_service_media_leased(7,100);
    j.phase=AOTX_SERVICE_DONE;out[2]=aotx_service_media_leased(7,99);
}
struct fixture {
    aotx_service_state s={};aotx_service_mailbox *host=nullptr;
    fixture(unsigned count) {
        cu(cudaHostAlloc(&host,AOTX_SERVICE_CHANNELS*sizeof(*host),cudaHostAllocMapped));
        memset(host,0,AOTX_SERVICE_CHANNELS*sizeof(*host));
        cu(cudaHostGetDevicePointer(&s.mailbox,host,0));
        cu(cudaMalloc(&s.frames,(size_t)AOTX_SERVICE_CHANNELS*AOTX_SERVICE_FRAME));
        cu(cudaMalloc(&s.ready,AOTX_SERVICE_CHANNELS*sizeof(unsigned)));
        cu(cudaMemset(s.ready,0,AOTX_SERVICE_CHANNELS*sizeof(unsigned)));
        cu(cudaMalloc(&s.grants,AOTX_SERVICE_PRINCIPALS*sizeof(aotx_service_grant)));
        cu(cudaMalloc(&s.jobs,AOTX_SERVICE_REQUESTS*sizeof(aotx_service_job)));
        cu(cudaMemset(s.jobs,0,AOTX_SERVICE_REQUESTS*sizeof(aotx_service_job)));
        s.enabled=1;s.epoch=17;cu(cudaMemcpyToSymbol(aotx_service,&s,sizeof s));
        aotx_wrap wrap={};const char *spans[]={"PS","s","U","u","A","a","G","(",")"};
        unsigned at=0;
        for(unsigned i=0;i<AOTX_WRAP_SPANS;++i) {
            wrap.offset[i]=(unsigned short)at;wrap.length[i]=(unsigned char)strlen(spans[i]);
            memcpy(wrap.bytes+at,spans[i],wrap.length[i]);at+=wrap.length[i];
        }
        wrap.usable=1;wrap.prefix_length=1;
        cu(cudaMemcpyToSymbol(aotx_model_wrap,&wrap,sizeof wrap,AOTX_MODEL_LANGUAGE*sizeof wrap));
        aotx_service_test_reset<<<1,1>>>();cu(cudaDeviceSynchronize());
        auto grants=frame(AOTX_SERVICE_GRANTS);put(grants,76,count+1);
        grants.resize(AOTX_SERVICE_HEAD+(count+1)*64);put(grants,88,grants.size()-AOTX_SERVICE_HEAD);
        for(unsigned i=0;i<=count;++i) {
            unsigned at0=AOTX_SERVICE_HEAD+i*64;
            put(grants,at0,i+1);put(grants,at0+16,1,8);put(grants,at0+24,15);put(grants,at0+28,1u<<AOTX_MODEL_LANGUAGE);
            put(grants,at0+36,32);put(grants,at0+40,2);put(grants,at0+44,2);put(grants,at0+48,1024,8);
        }
        send(0,grants);tick();check(status(0)==200,"complete grants install before client admission");
    }
    ~fixture() {
        aotx_service_state empty={};cu(cudaMemcpyToSymbol(aotx_service,&empty,sizeof empty));
        cudaFree(s.jobs);cudaFree(s.grants);cudaFree(s.ready);cudaFree(s.frames);cudaFreeHost(host);
    }
    void send(unsigned channel,const std::vector<unsigned char> &p) {
        check(p.size()<=AOTX_SERVICE_FRAME,"fixture packet fits the mailbox");
        memcpy(host[channel].bytes,p.data(),p.size());host[channel].length=p.size();
        __atomic_store_n(&host[channel].state,1ull,__ATOMIC_RELEASE);
    }
    void tick() {
        aotx_service_copy<<<AOTX_SERVICE_CHANNELS,256>>>();aotx_service_admit<<<1,1>>>();cu(cudaDeviceSynchronize());
    }
    unsigned status(unsigned channel) {
        check(host[channel].state==2,"device publishes a complete reply");
        return (unsigned)aotx_service_get(host[channel].bytes+8,4);
    }
    std::vector<aotx_service_job> jobs() {
        std::vector<aotx_service_job> out(AOTX_SERVICE_REQUESTS);
        cu(cudaMemcpy(out.data(),s.jobs,out.size()*sizeof(out[0]),cudaMemcpyDeviceToHost));return out;
    }
};
static void run(unsigned n)
{
    fixture f(n);
    for(unsigned i=0;i<n;++i) f.send(i+1,submit(i+1,i+101,"case"+std::to_string(i)));
    f.tick();auto jobs=f.jobs();
    for(unsigned i=0;i<n;++i) {
        check(f.status(i+1)==202,"distinct principal request admitted");
        std::string expected="PSrulesUcase"+std::to_string(i)+"uAansweraUquestionuG()";
        check(jobs[i].length==expected.size() && !memcmp(jobs[i].text,expected.data(),expected.size()),
            "complete roles and prefix render once without an operator overlay");
        check(jobs[i].sample.temperature==0 && jobs[i].sample.top_p==1 && jobs[i].sample.repeat_penalty==1,
            "sampling parameters are frozen with neutral defaults");
        check(jobs[i].slot==AOTX_SLOTS && jobs[i].pages>0,"admission does not claim an execution slot");
    }
    auto before=f.jobs();
    for(unsigned i=0;i<n;++i) {
        auto p=frame(AOTX_SERVICE_READ,i+1==n+1?1:i+2,i+101);f.send(i+1,p);
    }
    f.tick();for(unsigned i=0;i<n;++i)check(f.status(i+1)==404,"foreign request read refused");
    for(unsigned i=0;i<n;++i) {
        auto p=submit(i+1,i+1001,"[image:untyped]");f.send(i+1,p);
    }
    f.tick();for(unsigned i=0;i<n;++i)check(f.status(i+1)==400,"untyped media marker refused");
    auto after=f.jobs();check(!memcmp(before.data(),after.data(),before.size()*sizeof(before[0])),
        "failed admission preserves every existing result");
    for(unsigned i=0;i<n;++i) {
        auto p=frame(AOTX_SERVICE_INFO,i+1);put(p,80,1);f.send(i+1,p);
    }
    f.tick();for(unsigned i=0;i<n;++i)check(f.status(i+1)==400,"unused input fields must be zero");
    for(unsigned i=0;i<n;++i) {
        auto p=frame(AOTX_SERVICE_READ,i+1,i+101);put(p,32,2,8);f.send(i+1,p);
    }
    f.tick();for(unsigned i=0;i<n;++i)check(f.status(i+1)==403,"uninstalled grant revision refused");
    aotx_service_test_terminal<<<1,1>>>(n);cu(cudaDeviceSynchronize());
    for(unsigned i=0;i<n;++i) {
        auto p=frame(AOTX_SERVICE_READ,i+1,i+101);put(p,64,1,8);f.send(i+1,p);
    }
    f.tick();
    for(unsigned i=0;i<n;++i) {
        check(f.status(i+1)==200,"owned terminal window read");
        const unsigned char *p=f.host[i+1].bytes;
        check(aotx_service_get(p+88,4)==2 && p[128]=='b' && p[129]=='c' && aotx_service_get(p+76,4)==3,
            "cursor returns exact bytes and total length");
    }
    aotx_service_test_active<<<1,1>>>();cu(cudaDeviceSynchronize());
    auto cancel=frame(AOTX_SERVICE_CANCEL,1,101);put(cancel,64,99,8);f.send(1,cancel);f.tick();
    check(f.status(1)==409,"invalid cancellation cursor refused");
    jobs=f.jobs();check(!jobs[0].cancel,"invalid cancellation has no side effect");
    cancel=frame(AOTX_SERVICE_CANCEL,1,101);f.send(1,cancel);f.tick();
    jobs=f.jobs();check(jobs[0].cancel && jobs[0].phase==AOTX_SERVICE_RUNNING,"cancel request does not claim early completion");
    aotx_service_test_reused<<<1,1>>>();cu(cudaDeviceSynchronize());
    f.send(1,cancel);f.tick();
    unsigned flags=0;cu(cudaMemcpyFromSymbol(&flags,aotx_seqs,sizeof flags,
        offsetof(aotx_seq_table,slot)+sizeof(aotx_seq)+offsetof(aotx_seq,flags)));
    check(flags==0,"old request cancellation cannot stop the reused slot");
    unsigned *out=nullptr,lease[3]={};cu(cudaMalloc(&out,sizeof lease));
    aotx_service_test_lease<<<1,1>>>(out);cu(cudaDeviceSynchronize());
    cu(cudaMemcpy(lease,out,sizeof lease,cudaMemcpyDeviceToHost));cudaFree(out);
    check(lease[0]==1 && lease[1]==0 && lease[2]==0,"queued media leases bind the exact generation and end at completion");
    auto revoke=frame(AOTX_SERVICE_GRANTS);put(revoke,32,2,8);f.send(0,revoke);f.tick();
    check(f.status(0)==200,"empty newer grant table revokes all principals");
    f.send(1,frame(AOTX_SERVICE_READ,1,101));f.tick();check(f.status(1)==403,"revocation blocks retained output reads");
    f.send(0,revoke);f.tick();check(f.status(0)==400,"grant revision cannot be replayed");
}
static void text_boundaries(unsigned n)
{
    fixture f(n);auto before=f.jobs();
    for(const char *word:{"[image:","[audio:"}) {
        std::string prefix=word;
        for(unsigned split=1;split<=7;++split) {
            for(unsigned i=0;i<n;++i) {
                char suffix[3];snprintf(suffix,sizeof suffix,"%02x",i);
                std::string digest=std::string(62,'a')+suffix+"]";
                std::vector<std::string> parts;
                if(split<7) parts={prefix.substr(0,split),"",prefix.substr(split)+digest};
                else {
                    for(char c:prefix) parts.push_back(std::string(1,c));
                    parts.push_back(digest);
                }
                f.send(i+1,text_parts(i+1,i+1001,(i+split)%3,parts));
            }
            f.tick();
            for(unsigned i=0;i<n;++i) check(f.status(i+1)==400,"joined text cannot create a native media prefix");
            auto after=f.jobs();
            check(!memcmp(before.data(),after.data(),before.size()*sizeof(before[0])),
                "refused text parts preserve every request and source lease");
        }
    }
    for(unsigned i=0;i<n;++i) f.send(i+1,text_parts(i+1,i+2001,1,{"case"+std::to_string(i)+"[im","","age?"}));
    f.tick();auto jobs=f.jobs();
    for(unsigned i=0;i<n;++i) {
        std::string expected="PUcase"+std::to_string(i)+"[image?uG()";
        check(f.status(i+1)==202,"ordinary joined text remains available");
        check(jobs[i].length==expected.size() && !memcmp(jobs[i].text,expected.data(),expected.size()) && !jobs[i].media_count,
            "joined text retains its exact bytes without a media lease");
    }
}
static void result_revisions(unsigned n)
{
    for(unsigned models:{1u<<AOTX_MODEL_LANGUAGE,0u}) {
        fixture f(n);
        for(unsigned i=0;i<n;++i) f.send(i+1,submit(i+1,i+101,"owner"+std::to_string(i)));
        f.tick();for(unsigned i=0;i<n;++i) check(f.status(i+1)==202,"original revision admits each owner");
        aotx_service_test_terminal<<<1,1>>>(n);cu(cudaDeviceSynchronize());
        auto grants=frame(AOTX_SERVICE_GRANTS);put(grants,32,2,8);put(grants,76,n);
        grants.resize(AOTX_SERVICE_HEAD+n*64);put(grants,88,n*64);
        for(unsigned i=0;i<n;++i) {
            unsigned at=AOTX_SERVICE_HEAD+i*64;
            put(grants,at,i+1);put(grants,at+16,2,8);put(grants,at+24,1);put(grants,at+28,models);
            put(grants,at+36,32);put(grants,at+40,2);
        }
        f.send(0,grants);f.tick();check(f.status(0)==200,"current grants replace the admission revision");
        auto before=f.jobs();
        for(unsigned op:{AOTX_SERVICE_READ,AOTX_SERVICE_CANCEL}) {
            for(unsigned i=0;i<n;++i) {
                auto p=frame(op,i+1,i+101);put(p,32,2,8);f.send(i+1,p);
            }
            f.tick();for(unsigned i=0;i<n;++i) check(f.status(i+1)==404,"new grants cannot recover prior revision results");
        }
        auto after=f.jobs();check(!memcmp(before.data(),after.data(),before.size()*sizeof(before[0])),
            "refused prior revision operations preserve retained results");
        for(unsigned i=0;i<n;++i) {
            auto p=submit(i+1,i+1001,"current"+std::to_string(i));put(p,32,2,8);f.send(i+1,p);
        }
        f.tick();for(unsigned i=0;i<n;++i) check(f.status(i+1)==(models?202u:404u),"new submissions use the current model grant");
    }
}
static void concurrent(unsigned n)
{
    fixture f(n);
    std::atomic<bool> done(false), stop(false);
    unsigned verified=0, bad=0;
    std::thread producer([&]() {
        std::vector<unsigned> sent(n,0),received(n,0);
        while(!stop.load()) {
            bool complete=true;
            for(unsigned i=0;i<n;++i) {
                auto &m=f.host[i+1];
                if(sent[i]!=received[i] && __atomic_load_n(&m.state,__ATOMIC_ACQUIRE)==2) {
                    unsigned expected=100000u+sent[i]*AOTX_SLOTS+i;
                    if(aotx_service_get(m.bytes+48,4)!=expected || aotx_service_get(m.bytes+16,4)!=i+1 ||
                        aotx_service_get(m.bytes+8,4)!=404 || m.length!=AOTX_SERVICE_HEAD) ++bad;
                    ++verified;++received[i];__atomic_store_n(&m.state,0ull,__ATOMIC_RELEASE);
                }
                if(sent[i]==received[i] && sent[i]<1000) {
                    ++sent[i];auto p=frame(AOTX_SERVICE_READ,i+1,100000u+sent[i]*AOTX_SLOTS+i);
                    memcpy(m.bytes,p.data(),p.size());m.length=p.size();
                    __atomic_store_n(&m.state,1ull,__ATOMIC_RELEASE);
                }
                complete &= received[i]==1000;
            }
            if(complete) break;
            std::this_thread::yield();
        }
        done.store(true);
    });
    unsigned ticks=0;
    while(!done.load() && ticks++<200000) f.tick();
    stop.store(true);producer.join();
    check(verified==n*1000,"concurrent producer exchanges the complete request batch");
    check(!bad,"concurrent publication cannot mix prior and current headers");
    printf("service concurrent: %u responses, %u invalid headers\n",verified,bad);
}
__global__ void aotx_service_test_sources(unsigned count,unsigned owners)
{
    for(unsigned i=0;i<count;++i) {
        auto &o=aotx_media.objects[i];o={};o.phase=AOTX_MEDIA_READY;o.scope=AOTX_MEDIA_PRIVATE;
        o.principal[0]=(unsigned char)(1+i%owners);o.format=AOTX_IMAGE_JPEG;o.bytes=10+i;
        aotx_service_put(o.transfer,i+1,4);o.digest[0]=(unsigned char)i;
    }
}
static void inventory(unsigned n)
{
    fixture f(n);aotx_media_state media={};media.enabled=1;media.profile.objects=n==1?1000:n*2+3;
    cu(cudaMalloc(&media.objects,media.profile.objects*sizeof(aotx_media_object)));
    cu(cudaMemcpyToSymbol(aotx_media,&media,sizeof media));
    aotx_service_test_sources<<<1,1>>>(media.profile.objects,n);cu(cudaDeviceSynchronize());
    for(unsigned principal=1;principal<=n;++principal) {
        unsigned long long cursor=0;unsigned count=0;
        do {
            auto p=frame(AOTX_SERVICE_MEDIA_LIST,principal);put(p,64,cursor,8);f.send(1,p);f.tick();
            check(f.status(1)==200,"owned source inventory read");
            const unsigned char *r=f.host[1].bytes;
            unsigned bytes=(unsigned)aotx_service_get(r+88,4);
            check(bytes%80==0,"source inventory contains complete rows");
            for(unsigned at=0;at<bytes;at+=80) {
                unsigned id=(unsigned)aotx_service_get(r+AOTX_SERVICE_HEAD+at,4);
                check(id && (id-1)%n+1==principal,"source inventory excludes every foreign principal");++count;
            }
            cursor=aotx_service_get(r+64,8);
        } while(cursor);
        check(count==(media.profile.objects+n-principal)/n,"source inventory pages cover all owned rows");
    }
    auto p=frame(AOTX_SERVICE_MEDIA_LIST,n+1);f.send(1,p);f.tick();
    check(f.status(1)==200 && aotx_service_get(f.host[1].bytes+88,4)==0,"unrelated principal receives an empty inventory");
    cudaFree(media.objects);media={};cu(cudaMemcpyToSymbol(aotx_media,&media,sizeof media));
}
int main(void)
{
    run(1);run(AOTX_SLOTS);text_boundaries(1);text_boundaries(AOTX_SLOTS);result_revisions(1);result_revisions(AOTX_SLOTS);concurrent(1);concurrent(AOTX_SLOTS);inventory(1);inventory(AOTX_SLOTS);
    printf("service: %u checks, %u failures\n",checks,failures);return failures?1:0;
}
