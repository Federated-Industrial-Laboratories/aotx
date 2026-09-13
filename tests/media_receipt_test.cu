/* Purpose: Keep exact upload refusals available after source descriptor reuse.
 * Owns: Distinct mapped requests, codec boundary states and receipt assertions.
 * Launch shape: Real service and source batches at N=1 and N=64 with one worker.
 * Lifetime: One case; no codec, signal or trained model arithmetic is run. */
#include "media_pressure_fixture.h"
#include <array>

__global__ void aotx_receipt_clock(unsigned long long now)
{ aotx_sched.start_ns=now; aotx_sched.held=0; }
__global__ void aotx_receipt_initialize(void)
{
    for (unsigned i=0;i<aotx_media.profile.objects;++i) aotx_media.objects[i].worker=~0u;
    aotx_media.owner[0]=aotx_audio_runtime.owner[0]=~0u;
    aotx_media.image[0].phase=AOTX_IMAGE_REFUSED;
    aotx_media.vision[0].phase=AOTX_VISION_REFUSED;
    aotx_audio_runtime.jobs[0].phase=AOTX_AUDIO_REFUSED;
}
__device__ static unsigned aotx_receipt_find(unsigned i, unsigned epoch)
{
    unsigned char id[16]={}; identity(id,i,epoch);
    for (unsigned at=0;at<aotx_media.profile.objects;++at)
        if (aotx_media.objects[at].phase && aotx_service_equal(id,aotx_media.objects[at].transfer,16)) return at;
    return aotx_media.profile.objects;
}
__host__ __device__ static unsigned aotx_receipt_status(unsigned i, bool audio)
{
    if (!audio) return i%2?AOTX_MEDIA_ENCODER:AOTX_MEDIA_CODEC;
    const unsigned status[]={AOTX_MEDIA_NO_SIGNAL,AOTX_MEDIA_AUDIO_NUMERIC,AOTX_MEDIA_LIMIT,AOTX_MEDIA_AUDIO_FORMAT};
    return status[i%4];
}
__global__ void aotx_receipt_boundary(unsigned i, unsigned epoch, bool audio, bool ready)
{
    unsigned at=aotx_receipt_find(i,epoch); if (at==aotx_media.profile.objects) return;
    auto &o=aotx_media.objects[at]; o.worker=0; o.samples=1000+i; o.rows=1+i%5;
    aotx_media.hash[at].active=0;
    o.phase=ready || i%2?AOTX_MEDIA_ENCODE:AOTX_MEDIA_DECODE;
    o.feature=at*64; o.span=audio?audio_rows(i):64;
    if (audio) {
        o.feature=0;
        for (unsigned j=0;j<at;++j) o.feature+=audio_rows(j%(aotx_media.profile.objects/2));
    }
    if (audio) {
        aotx_audio_runtime.owner[0]=at; auto &j=aotx_audio_runtime.jobs[0];
        const unsigned statuses[]={AOTX_AUDIO_NO_SIGNAL,AOTX_AUDIO_NONFINITE,AOTX_AUDIO_LIMIT,AOTX_AUDIO_INVALID};
        j.phase=ready?AOTX_AUDIO_READY:AOTX_AUDIO_REFUSED; j.status=ready?0:statuses[i%4]; j.rows=audio_rows(i);
        if (ready) o.phase=AOTX_MEDIA_ENCODE;
    } else {
        aotx_media.owner[0]=at; aotx_media.image[0].phase=AOTX_IMAGE_REFUSED;
        auto &v=aotx_media.vision[0]; v.phase=ready?AOTX_VISION_READY:AOTX_VISION_REFUSED;
        v.rows=64; v.resized_width=v.resized_height=256;
    }
    aotx_sched.held=1;
}
__global__ void aotx_receipt_replace(aotx_pressure_result *out, unsigned n,
    unsigned epoch, bool audio, bool cancel)
{
    for (unsigned i=0;i<n;++i) {
        unsigned char p[AOTX_BODY_BYTES]; begin_body(p,i,epoch,source_bytes(i),audio);
        identity(p+80,i,9);
        if (cancel) aotx_media_put(p+4,AOTX_MEDIA_CANCEL,4);
        auto &r=out[i]; r={}; r.valid=aotx_media_part(p,cancel?24:AOTX_MEDIA_BEGIN_BYTES,70000ull+epoch*100+i);
        r.found=aotx_receipt_find(i,epoch);
        if (r.found<aotx_media.profile.objects) r.object=aotx_media.objects[r.found];
    }
}
struct receipt_fixture:fixture {
    unsigned long long now=100;
    bool audio;
    receipt_fixture(unsigned n, bool sound):fixture(n,true,true),audio(sound) {
        cudaFree(m.features); m.profile.feature_rows*=2;
        storage(&m.features,(size_t)m.profile.feature_rows*AOTX_VISION_OUTPUT);
        cudaFree(a.features); a.profile.feature_rows*=2; storage(&a.features,(size_t)a.profile.feature_rows*4096);
        profiles(m.profile,a.profile);
        storage(&s.jobs,AOTX_SERVICE_REQUESTS);
        cudaFree(s.grants); storage(&s.grants,n+1); s.grant_count=0;
        cu(cudaMemcpyToSymbol(aotx_media,&m,sizeof m));
        cu(cudaMemcpyToSymbol(aotx_audio_runtime,&a,sizeof a));
        cu(cudaMemcpyToSymbol(aotx_service,&s,sizeof s));
        aotx_receipt_initialize<<<1,1>>>(); aotx_receipt_clock<<<1,1>>>(now); aotx_pressure_sync();
        auto &box=mailbox[0]; unsigned char *f=box.bytes; unsigned bytes=(n+1)*AOTX_SERVICE_GRANT_BYTES;
        memset(f,0,AOTX_SERVICE_HEAD+bytes); memcpy(f,AOTX_SERVICE_MAGIC,8);
        aotx_media_put(f+8,AOTX_SERVICE_GRANTS,4); aotx_media_put(f+32,1,8);
        aotx_media_put(f+76,n+1,4); aotx_media_put(f+88,bytes,4);
        for (unsigned i=0;i<=n;++i) {
            auto *p=f+AOTX_SERVICE_HEAD+i*AOTX_SERVICE_GRANT_BYTES; identity(p,i,8);
            aotx_media_put(p+16,1,8); aotx_media_put(p+24,AOTX_SERVICE_UPLOAD,4);
            aotx_media_put(p+28,1u<<(audio?AOTX_MODEL_LANGUAGE_AUDIO:AOTX_MODEL_LANGUAGE),4);
            aotx_media_put(p+32,1,4); aotx_media_put(p+36,64,4); aotx_media_put(p+40,1,4);
            aotx_media_put(p+44,2*n+2,4); aotx_media_put(p+48,4*m.profile.bytes,8);
        }
        box.length=AOTX_SERVICE_HEAD+bytes; box.state=1; tick();
        check(box.state==2 && aotx_media_get(box.bytes+8,4)==200,"mapped operator frames install the complete valid grant batch");
    }
    ~receipt_fixture() { cudaFree(s.jobs); }
    void tick(void) {
        aotx_receipt_clock<<<1,1>>>(now);
        aotx_service_copy<<<AOTX_SERVICE_CHANNELS,64>>>(); aotx_service_admit<<<1,1>>>(); aotx_pressure_sync();
    }
    void status(unsigned expected, const char *label) {
        for (unsigned i=0;i<count;++i) check(mailbox[i+1].state==2 &&
            aotx_media_get(mailbox[i+1].bytes+8,4)==expected,label);
    }
    void begin(unsigned epoch, unsigned expected=200) {
        aotx_receipt_clock<<<1,1>>>(now); send(epoch,0,audio); status(expected,"mapped begin returns its exact status");
    }
    void request(unsigned epoch, unsigned op=0, bool foreign=false) {
        for (unsigned i=0;i<count;++i) {
            auto &box=mailbox[i+1]; unsigned char *f=box.bytes; unsigned payload=op==AOTX_MEDIA_CHUNK?source_bytes(i):0;
            unsigned length=AOTX_SERVICE_HEAD+(op?AOTX_MEDIA_FRAME_HEAD+payload:0);
            memset(f,0,length); memcpy(f,AOTX_SERVICE_MAGIC,8);
            aotx_media_put(f+8,op?AOTX_SERVICE_MEDIA:AOTX_SERVICE_MEDIA_READ,4);
            identity(f+16,foreign?count:i,8); aotx_media_put(f+32,1,8); identity(f+48,i,epoch);
            if (op) {
                aotx_media_put(f+88,length-AOTX_SERVICE_HEAD,4); unsigned char *p=f+AOTX_SERVICE_HEAD;
                aotx_media_put(p,AOTX_MEDIA_SCHEMA,4); aotx_media_put(p+4,op,4); identity(p+8,i,epoch);
                aotx_media_put(p+24,source_bytes(i),8);
                aotx_media_put(p+32,op==AOTX_MEDIA_END?source_bytes(i):0,8); aotx_media_put(p+40,payload,4);
                for (unsigned j=0;j<payload;++j) p[AOTX_MEDIA_FRAME_HEAD+j]=(unsigned char)(i*7+epoch+j);
            }
            box.length=length; box.state=1;
        }
        tick();
    }
    void finish(unsigned epoch, bool ready=false) {
        request(epoch,AOTX_MEDIA_CHUNK); status(200,"source chunks enter the canonical store");
        request(epoch,AOTX_MEDIA_END); status(200,"complete source bytes reach hash admission");
        for (unsigned i=0;i<count;++i) {
            aotx_receipt_boundary<<<1,1>>>(i,epoch,audio,ready);
            if (audio) aotx_audio_complete<<<1,1>>>(); else aotx_media_complete<<<1,1>>>();
            aotx_receipt_clock<<<1,1>>>(now);
        }
        aotx_pressure_sync();
    }
    void replace(unsigned epoch, bool cancel=false) {
        aotx_receipt_replace<<<1,1>>>(result,count,epoch,audio,cancel); aotx_pressure_sync();
        for (unsigned i=0;i<count;++i) {
            const auto r=copy(result+i,1)[0];
            check(r.valid && r.found<m.profile.objects && r.object.phase==(cancel?AOTX_MEDIA_REFUSED:AOTX_MEDIA_RECEIVE) &&
                aotx_media_get(r.object.transfer+4,4)==epoch && aotx_media_get(r.object.principal+4,4)==9,
                "unrelated canonical sources use exact replacement identities");
        }
    }
    std::vector<std::array<unsigned char,64>> expected(unsigned epoch, bool ready=false, bool timeout=false) {
        std::vector<std::array<unsigned char,64>> rows(count);
        for (unsigned i=0;i<count;++i) {
            auto *p=rows[i].data();
            for (unsigned j=0;j<32;++j) p[j]=(unsigned char)(i*13u+epoch*5u+j);
            aotx_media_put(p+32,source_bytes(i),8); aotx_media_put(p+40,ready?AOTX_MEDIA_READY:AOTX_MEDIA_REFUSED,4);
            aotx_media_put(p+44,ready?0:timeout?AOTX_MEDIA_CANCELLED:aotx_receipt_status(i,audio),4);
            aotx_media_put(p+48,audio?AOTX_AUDIO_WAV:AOTX_IMAGE_JPEG,4); aotx_media_put(p+52,timeout?0:1000+i,4);
            aotx_media_put(p+56,timeout?0:ready?(audio?audio_rows(i):64):1+i%5,4);
        }
        return rows;
    }
    void matches(unsigned epoch, bool ready=false, bool timeout=false) {
        auto rows=expected(epoch,ready,timeout);
        for (unsigned i=0;i<count;++i) {
            const auto &box=mailbox[i+1];
            check(box.length==AOTX_SERVICE_HEAD+64 && !memcmp(box.bytes+AOTX_SERVICE_HEAD,rows[i].data(),64),
                "all retained upload metadata bytes match the original source");
            check(aotx_media_get(box.bytes+48,4)==i+1 && aotx_media_get(box.bytes+52,4)==epoch &&
                aotx_media_get(box.bytes+16,4)==i+1,"read replies preserve each transfer and principal");
        }
    }
    void read(unsigned epoch, bool ready=false, bool timeout=false) {
        request(epoch); status(200,"owned upload status remains readable"); matches(epoch,ready,timeout);
    }
    unsigned long long applied(void) {
        aotx_seam_state value; cu(cudaMemcpyFromSymbol(&value,aotx_seam,sizeof value)); return value.apply.applied_count;
    }
    void refused(unsigned epoch) {
        auto before=copy(m.objects,m.profile.objects); auto records=applied(); begin(epoch,429);
        auto after=copy(m.objects,m.profile.objects);
        check(!memcmp(before.data(),after.data(),before.size()*sizeof(before[0])) && applied()==records,
            "receipt capacity refuses admission without changing sources or canonical records");
    }
    void cache(unsigned epoch, unsigned replacement) {
        begin(epoch); finish(epoch); auto before=copy(m.objects,m.profile.objects);
        replace(replacement); auto after=copy(m.objects,m.profile.objects); unsigned changed=0;
        for (unsigned at=0;at<m.profile.objects;++at)
            if (aotx_media_get(before[at].transfer+4,4)==epoch && before[at].phase==AOTX_MEDIA_REFUSED) {
                check(aotx_media_get(after[at].transfer+4,4)==replacement && after[at].generation!=before[at].generation,
                    "each failed descriptor is reused before the first status poll"); ++changed;
            }
        check(changed==count,"the full distinct failure batch is replaced");
        read(epoch); replace(replacement,true);
    }
};
static void cached_cases(unsigned n, bool audio)
{
    receipt_fixture f(n,audio); f.cache(1,11);
    auto records=f.applied(); f.request(1,0,true); f.status(404,"a foreign principal cannot read a cached refusal");
    for (unsigned i=0;i<n;++i) check(f.mailbox[i+1].length==AOTX_SERVICE_HEAD,"foreign reads return no cached metadata");
    f.request(1,AOTX_MEDIA_CANCEL,true); f.status(404,"a foreign principal cannot remove a cached refusal");
    check(f.applied()==records,"foreign cache operations publish no canonical mutation"); f.read(1);
    f.cache(2,12); f.refused(3); f.read(1); f.read(2);
    records=f.applied(); f.request(1,AOTX_MEDIA_CANCEL); f.status(200,"the owner removes a cache-only refusal");
    f.matches(1);
    check(f.applied()==records,"cache-only removal does not alter an unrelated source");
    f.request(1); f.status(404,"removed cached handles remain absent after descriptor reuse"); f.read(2);
    f.begin(3); f.request(3); f.status(200,"freed receipt capacity accepts a new pending upload");
    f.refused(4); f.read(2); f.request(3,AOTX_MEDIA_CANCEL); f.status(200,"explicit pending cancellation frees its receipt");
    auto deadline=100ull+AOTX_SERVICE_UPLOAD_SECONDS*1000000000ull;
    f.now=deadline-1; f.read(2); records=f.applied();
    f.now=deadline; f.request(2); f.status(404,"cached refusals expire at their exact deadline");
    check(f.applied()==records,"cache expiry does not publish a source mutation");
    f.begin(4); f.begin(5); f.refused(6);
    printf("cached receipts N=%u audio=%u checks=%u failures=%u\n",n,audio,checks,failures);
}
static void ready_cases(unsigned n, bool audio)
{
    receipt_fixture f(n,audio); f.begin(1); f.finish(1,true); f.read(1,true);
    f.cache(2,12); f.begin(3); f.refused(4); f.read(1,true); f.read(2);
    f.request(3); f.status(200,"ready source trackers release capacity for later pending uploads");
    f.request(1,0,true); f.status(404,"ready sources retain their original private scope");
    printf("ready receipts N=%u audio=%u checks=%u failures=%u\n",n,audio,checks,failures);
}
static void timeout_cases(unsigned n, bool audio)
{
    receipt_fixture f(n,audio); f.begin(1);
    auto deadline=100ull+AOTX_SERVICE_UPLOAD_SECONDS*1000000000ull;
    f.now=deadline-1; f.request(1); f.status(200,"pending upload exists before the deadline");
    for (unsigned i=0;i<n;++i) check(aotx_media_get(f.mailbox[i+1].bytes+AOTX_SERVICE_HEAD+40,4)==AOTX_MEDIA_RECEIVE,
        "receive timeout does not cancel early");
    auto records=f.applied(); f.now=deadline; f.read(1,false,true);
    check(f.applied()-records==n,"receive timeout records one canonical cancellation for every upload");
    f.replace(11); f.read(1,false,true); f.replace(11,true);
    f.now=deadline+AOTX_SERVICE_UPLOAD_SECONDS*1000000000ull-1; f.read(1,false,true);
    f.now++; f.request(1); f.status(404,"timeout receipts expire after their own retention interval");
    f.begin(2); f.begin(3); f.refused(4);
    printf("timeout receipts N=%u audio=%u checks=%u failures=%u\n",n,audio,checks,failures);
}
int main(void)
{
    for (unsigned n:{1u,64u}) for (bool audio:{false,true}) {
        cached_cases(n,audio); ready_cases(n,audio); timeout_cases(n,audio);
    }
    printf("media receipts: checks=%u failures=%u\n",checks,failures); return failures?1:0;
}
