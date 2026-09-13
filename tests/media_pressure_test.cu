/* Purpose: Distinguish occupied media pools from fixed source and feature limits.
 * Owns: Exact refusal, release, reservation and mapped response assertions.
 * Launch shape: Real ordered admission and scheduler nodes at N=1 and N=64.
 * Lifetime: One process; no codec, trained encoder or language model is run. */
#include "media_pressure_fixture.h"

__global__ void aotx_pressure_initialize(void)
{
    aotx_sched.held=0; aotx_sched.start_ns=100;
    for (unsigned i=0;i<aotx_media.profile.objects;++i) aotx_media.objects[i].worker=~0u;
    for (unsigned w=0;w<aotx_media.profile.workers;++w) {
        aotx_media.owner[w]=aotx_audio_runtime.owner[w]=~0u;
        aotx_media.image[w].phase=AOTX_IMAGE_REFUSED;
        aotx_media.image[w].rgb=aotx_media.workspace;
        aotx_media.image[w].rgb_bytes=(unsigned long long)aotx_media.profile.pixels*3;
        aotx_media.image[w].pixel_limit=aotx_media.profile.pixels;
        aotx_media.image[w].dimension_limit=aotx_media.profile.dimension;
        aotx_media.vision[w].phase=AOTX_VISION_REFUSED;
        aotx_audio_runtime.jobs[w].phase=AOTX_AUDIO_REFUSED;
        aotx_audio_runtime.jobs[w].source_capacity=aotx_audio_runtime.profile.source_frames;
    }
}
__device__ static unsigned locate(const unsigned char *id)
{
    for (unsigned i=0;i<aotx_media.profile.objects;++i)
        if (aotx_media.objects[i].phase && aotx_service_equal(id,aotx_media.objects[i].transfer,16)) return i;
    return aotx_media.profile.objects;
}
__global__ void aotx_pressure_sources(aotx_pressure_result *out, unsigned n,
    unsigned epoch, unsigned mode, bool audio)
{
    for (unsigned i=0;i<n;++i) {
        unsigned char p[AOTX_BODY_BYTES];
        begin_body(p,i,epoch,mode==2?aotx_media.profile.bytes+1:source_bytes(i),audio);
        bool valid=aotx_media_part(p,AOTX_MEDIA_BEGIN_BYTES,epoch*1000ull+i+1);
        if (mode==1) {
            aotx_media_put(p+4,AOTX_MEDIA_CHUNK,4); aotx_media_put(p+32,0,8);
            for (unsigned j=0;j<source_bytes(i);++j) p[AOTX_MEDIA_PART+j]=(unsigned char)(i*11u+j*3u);
            valid &= aotx_media_part(p,AOTX_MEDIA_PART+source_bytes(i),epoch*1000ull+n+i+1);
        }
        auto &r=out[i]; r={}; r.valid=valid; r.found=locate(p+8);
        if (r.found<aotx_media.profile.objects) r.object=aotx_media.objects[r.found];
        r.accepted=aotx_media.accepted; r.refused=aotx_media.refused;
    }
}
__global__ void aotx_pressure_cancel(aotx_pressure_result *out, unsigned n, unsigned epoch)
{
    for (unsigned i=0;i<n;++i) {
        unsigned char p[24]={}; aotx_media_put(p,AOTX_MEDIA_SCHEMA,4);
        aotx_media_put(p+4,AOTX_MEDIA_CANCEL,4); identity(p+8,i,epoch);
        auto &r=out[i]; r={}; r.valid=aotx_media_part(p,24,90000ull+i);
        r.found=locate(p+8); if (r.found<aotx_media.profile.objects) r.object=aotx_media.objects[r.found];
    }
}
static void released(fixture &f, unsigned epoch)
{
    aotx_pressure_cancel<<<1,1>>>(f.result,f.count,epoch); aotx_pressure_sync();
    for (const auto &r:copy(f.result,f.count)) check(r.valid && r.object.phase==AOTX_MEDIA_REFUSED &&
        r.object.status==AOTX_MEDIA_CANCELLED && r.object.worker==~0u,"source cancellation releases each owner");
}
static void source_cases(unsigned n, bool audio)
{
    fixture f(n); aotx_pressure_initialize<<<1,1>>>();
    aotx_pressure_sources<<<1,1>>>(f.result,n,1,1,audio); aotx_pressure_sync();
    auto baseline=copy(f.m.objects,n); auto bytes=copy(f.m.source,f.m.profile.bytes);
    unsigned offset=0;
    for (unsigned i=0;i<n;++i) {
        const auto &o=baseline[i];
        check(o.phase==AOTX_MEDIA_RECEIVE && o.offset==offset && o.received==source_bytes(i) &&
            o.generation==1001+i && o.principal[0]==i+1,"distinct source spans fill the byte pool");
        for (unsigned j=0;j<source_bytes(i);++j) check(bytes[offset+j]==(unsigned char)(i*11u+j*3u),
            "source bytes remain exact before pressure");
        offset+=source_bytes(i);
    }
    check(offset==f.m.profile.bytes,"the occupied byte pool has no free span");
    for (unsigned mode:{0u,2u}) {
        aotx_pressure_sources<<<1,1>>>(f.result,n,2+mode,mode,audio); aotx_pressure_sync();
        auto results=copy(f.result,n);
        for (unsigned i=0;i<n;++i) {
            const auto &r=results[i];
            check(r.valid && r.found<f.m.profile.objects && r.object.phase==AOTX_MEDIA_REFUSED &&
                r.object.status==(mode?2u:12u),"source capacity and occupied storage have distinct refusal codes");
            check(r.object.offset==~0ull && !r.object.received && !r.object.span && r.object.worker==~0u &&
                r.accepted==n && r.object.generation==(2+mode)*1000ull+i+1,
                "refused source has no byte, feature or worker reservation");
        }
        auto current=copy(f.m.objects,n);
        check(!memcmp(current.data(),baseline.data(),n*sizeof(current[0])) && bytes==copy(f.m.source,f.m.profile.bytes),
            "refusals preserve every occupied descriptor and source byte");
    }
    released(f,1); aotx_pressure_sources<<<1,1>>>(f.result,n,5,1,audio); aotx_pressure_sync();
    auto fresh=copy(f.result,n); offset=0;
    for (unsigned i=0;i<n;++i) {
        const auto &o=fresh[i].object;
        check(fresh[i].valid && o.phase==AOTX_MEDIA_RECEIVE && o.offset==offset &&
            o.generation==5001+i && o.received==source_bytes(i),"fresh begins reuse only released source ranges");
        offset+=source_bytes(i);
    }
    printf("source N=%u audio=%u checks=%u failures=%u\n",n,audio,checks,failures);
}
__global__ void aotx_pressure_features(unsigned n, bool audio, unsigned stage)
{
    if (stage==3) {
        if (audio) aotx_audio_runtime.profile.feature_rows=1;
        else aotx_media.profile.feature_rows=63;
    }
    unsigned at=0;
    for (unsigned i=0;i<n;++i) {
        if (!stage) {
            auto &o=aotx_media.objects[i]; o={}; identity(o.transfer,i,1); identity(o.principal,i,8);
            o.generation=100+i; o.scope=AOTX_MEDIA_PRIVATE; o.format=audio?AOTX_AUDIO_WAV:AOTX_IMAGE_JPEG;
            o.offset=i*128; o.bytes=source_bytes(i); o.phase=AOTX_MEDIA_READY; o.worker=~0u;
            o.rows=o.span=audio?audio_rows(i):64; o.feature=at; at+=o.span;
            for (unsigned j=0;j<32;++j) o.digest[j]=(unsigned char)(i*7+j);
        }
        auto &o=aotx_media.objects[n+i]; o={}; identity(o.transfer,i,stage+2); identity(o.principal,i,8);
        o.scope=AOTX_MEDIA_PRIVATE; o.format=audio?AOTX_AUDIO_WAV:AOTX_IMAGE_JPEG;
        o.generation=2000ull+stage*1000+n-i; o.offset=(n+i)*128; o.bytes=source_bytes(i);
        o.phase=AOTX_MEDIA_WAIT; o.worker=~0u;
        for (unsigned j=0;j<32;++j) o.digest[j]=(unsigned char)(i*17+stage+j);
    }
}
__global__ void aotx_pressure_audio_boundary(unsigned n, bool ready)
{
    for (unsigned w=0;w<aotx_audio_runtime.profile.workers;++w) {
        unsigned index=aotx_audio_runtime.owner[w]; if (index==~0u) continue;
        auto &j=aotx_audio_runtime.jobs[w]; j.phase=ready?AOTX_AUDIO_READY:AOTX_AUDIO_RESAMPLE;
        j.rows=audio_rows(index-n); j.rate=16000; j.channels=1+(index-n)%2;
        j.samples=j.rows*640; j.source_frames=j.samples; j.encoding=1;
    }
}
__global__ void aotx_pressure_image_boundary(unsigned n, bool ready)
{
    for (unsigned w=0;w<aotx_media.profile.workers;++w) {
        if (aotx_media.owner[w]==~0u) continue;
        if (ready) {
            auto &v=aotx_media.vision[w]; v.phase=AOTX_VISION_READY; v.rows=64;
            v.resized_width=v.resized_height=256;
        } else {
            auto &d=aotx_media.image[w]; d.phase=AOTX_IMAGE_READY; d.width=d.height=256;
        }
    }
}
static void schedule(unsigned n, bool audio)
{
    if (audio) {
        aotx_audio_schedule<<<1,1>>>(); aotx_pressure_audio_boundary<<<1,1>>>(n,false);
        aotx_audio_complete<<<1,1>>>();
    } else aotx_media_schedule<<<1,1>>>();
    aotx_pressure_sync();
}
static void feature_cases(unsigned n, bool audio)
{
    fixture f(n,false,true); aotx_pressure_initialize<<<1,1>>>();
    aotx_pressure_features<<<1,1>>>(n,audio,0); aotx_pressure_sync();
    auto baseline=copy(f.m.objects,n); auto idle=copy(f.m.image,1);
    for (unsigned step=0;step<n;++step) schedule(n,audio);
    auto refused=copy(f.m.objects+n,n); auto owners=copy(audio?f.a.owner:f.m.owner,1);
    auto refused_jobs=copy(f.a.jobs,1); auto untouched=copy(f.m.image,1);
    for (unsigned i=0;i<n;++i) {
        check(refused[i].phase==AOTX_MEDIA_REFUSED && refused[i].status==12,
            "occupied feature storage returns pressure for every distinct source");
        check(!refused[i].span && refused[i].worker==~0u && owners[0]==~0u,
            "feature refusal releases all worker owners without a row lease");
    }
    if (audio) check(refused_jobs[0].phase==AOTX_AUDIO_REFUSED && !refused_jobs[0].features,
        "refused audio work has no feature pointer");
    else check(!memcmp(idle.data(),untouched.data(),sizeof(idle[0])),"image pressure does not take a decode workspace");
    auto occupied=copy(f.m.objects,n);
    check(!memcmp(occupied.data(),baseline.data(),n*sizeof(occupied[0])),"feature pressure preserves every ready source");
    released(f,1); aotx_pressure_features<<<1,1>>>(n,audio,1);
    std::vector<aotx_media_object> fresh(n);
    unsigned at=0;
    for (unsigned step=0;step<n;++step) {
        schedule(n,audio); unsigned i=n-1-step;
        fresh[i]=copy(f.m.objects+n+i,1)[0]; const auto &o=fresh[i]; unsigned rows=audio?audio_rows(i):64;
        owners=copy(audio?f.a.owner:f.m.owner,1);
        auto jobs=copy(f.a.jobs,1); auto images=copy(f.m.image,1); auto vision=copy(f.m.vision,1);
        check(o.phase==(audio?AOTX_MEDIA_ENCODE:AOTX_MEDIA_DECODE) && !o.status && o.feature==at &&
            o.span==rows && o.worker==0 && owners[0]==n+i && o.generation==3000ull+n-i &&
            o.principal[0]==i+1,"fresh feature ranges are distinct and follow the recorded generation order");
        if (audio) check(jobs[0].features==f.a.features+(size_t)at*4096 && jobs[0].source==f.m.source+(n+i)*128 &&
            o.samples==rows*640 && o.channels==1+i%2,"audio work binds the exact source and reserved feature range");
        else check(vision[0].features==f.m.features+(size_t)at*AOTX_VISION_OUTPUT && vision[0].feature_capacity==rows &&
            images[0].source==f.m.source+(n+i)*128 && images[0].bytes==source_bytes(i),
            "image work binds the exact source and reserved feature range");
        at+=rows;
        if (audio) { aotx_pressure_audio_boundary<<<1,1>>>(n,true); aotx_audio_complete<<<1,1>>>(); }
        else {
            aotx_pressure_image_boundary<<<1,1>>>(n,false); aotx_media_complete<<<1,1>>>();
            aotx_pressure_image_boundary<<<1,1>>>(n,true); aotx_media_complete<<<1,1>>>();
        }
        aotx_pressure_sync(); owners=copy(audio?f.a.owner:f.m.owner,1);
        check(owners[0]==~0u,"each completion releases the workspace before the next source");
    }
    check(at==(audio?f.rows:64*n),"fresh reservations cover the released pool without gaps");
    auto ready=copy(f.m.objects+n,n);
    for (unsigned i=0;i<n;++i) check(ready[i].phase==AOTX_MEDIA_READY && !ready[i].status &&
        ready[i].worker==~0u && ready[i].rows==fresh[i].span && ready[i].span==fresh[i].span &&
        ready[i].feature==fresh[i].feature && ready[i].generation==fresh[i].generation,
        "ready publication keeps exact features after the worker is released");
    released(f,3); aotx_pressure_features<<<1,1>>>(n,audio,3); aotx_pressure_sync();
    aotx_media_state m; aotx_audio_runtime_state a;
    cu(cudaMemcpyFromSymbol(&m,aotx_media,sizeof m)); cu(cudaMemcpyFromSymbol(&a,aotx_audio_runtime,sizeof a));
    fixture::profiles(m.profile,a.profile);
    for (unsigned step=0;step<n;++step) schedule(n,audio);
    auto large=copy(f.m.objects+n,n); owners=copy(audio?f.a.owner:f.m.owner,1);
    for (unsigned i=0;i<n;++i) check(large[i].phase==AOTX_MEDIA_REFUSED && large[i].status==2 &&
        !large[i].span && large[i].worker==~0u && owners[0]==~0u,
        "a feature request larger than the total pool retains the fixed limit status");
    printf("features N=%u audio=%u checks=%u failures=%u\n",n,audio,checks,failures);
}
__global__ void aotx_pressure_table(unsigned n)
{ aotx_media.profile.objects=n; aotx_service.media_count=n; }
static void service_cases(unsigned n, bool audio)
{
    fixture f(n,true); aotx_pressure_initialize<<<1,1>>>();
    aotx_pressure_sources<<<1,1>>>(f.result,n,1,1,audio); aotx_pressure_sync();
    auto baseline=copy(f.m.objects,n); auto bytes=copy(f.m.source,f.m.profile.bytes);
    aotx_pressure_table<<<1,1>>>(n); f.send(9,0,audio);
    for (unsigned i=0;i<n;++i) check(f.mailbox[i+1].state==2 &&
        aotx_media_get(f.mailbox[i+1].bytes+8,4)==429 && f.mailbox[i+1].length==AOTX_SERVICE_HEAD,
        "a full descriptor table returns pressure without an admitted source");
    auto full=copy(f.m.objects,n);
    check(!memcmp(full.data(),baseline.data(),n*sizeof(full[0])),"descriptor pressure preserves all prior identities");
    aotx_pressure_table<<<1,1>>>(2*n);
    for (unsigned mode:{0u,1u,2u}) {
        aotx_seam_state before; cu(cudaMemcpyFromSymbol(&before,aotx_seam,sizeof before));
        f.send(2+mode,mode,audio);
        for (unsigned i=0;i<n;++i) {
            auto &box=f.mailbox[i+1]; unsigned status=(unsigned)aotx_media_get(box.bytes+8,4);
            check(box.state==2 && status==(mode==0?429u:mode==1?413u:400u),
                "mapped media replies distinguish pressure, fixed limits and invalid frames");
            check(aotx_media_get(box.bytes+48,4)==i+1 && aotx_media_get(box.bytes+52,4)==2+mode &&
                aotx_media_get(box.bytes+16,4)==i+1,"each mapped refusal keeps its own transfer and principal");
            if (mode<2) check(box.length==AOTX_SERVICE_HEAD+64 &&
                aotx_media_get(box.bytes+AOTX_SERVICE_HEAD+40,4)==AOTX_MEDIA_REFUSED &&
                aotx_media_get(box.bytes+AOTX_SERVICE_HEAD+44,4)==(mode?2u:12u),
                "mapped capacity refusals expose the exact device result");
        }
        auto current=copy(f.m.objects,n); aotx_media_state m; aotx_seam_state after;
        cu(cudaMemcpyFromSymbol(&m,aotx_media,sizeof m)); cu(cudaMemcpyFromSymbol(&after,aotx_seam,sizeof after));
        check(m.accepted==n && !memcmp(current.data(),baseline.data(),n*sizeof(current[0])) &&
            bytes==copy(f.m.source,f.m.profile.bytes),"refused service frames cannot admit or alter source data");
        check(after.apply.applied_count-before.apply.applied_count==(mode==2?0u:n),
            "only valid source operations enter canonical records");
    }
    released(f,1); f.send(6,0,audio);
    for (unsigned i=0;i<n;++i) check(f.mailbox[i+1].state==2 &&
        aotx_media_get(f.mailbox[i+1].bytes+8,4)==200 &&
        aotx_media_get(f.mailbox[i+1].bytes+AOTX_SERVICE_HEAD+40,4)==AOTX_MEDIA_RECEIVE,
        "fresh mapped begins succeed after explicit source release");
    printf("service N=%u audio=%u checks=%u failures=%u\n",n,audio,checks,failures);
}
__global__ void aotx_pressure_clear_pending(unsigned n)
{
    for (unsigned i=n;i<2*n;++i) { aotx_media.objects[i]={}; aotx_media.objects[i].worker=~0u; }
}
static void admission_cases(unsigned n, bool audio)
{
    fixture f(n,true,true); aotx_pressure_initialize<<<1,1>>>();
    storage(&f.s.jobs,AOTX_SERVICE_REQUESTS); cu(cudaMemcpyToSymbol(aotx_service,&f.s,sizeof f.s));
    aotx_pressure_features<<<1,1>>>(n,audio,0); aotx_pressure_clear_pending<<<1,1>>>(n); aotx_pressure_sync();
    auto baseline=copy(f.m.objects,2*n); auto bytes=copy(f.m.source,f.m.profile.bytes);
    auto receipts=copy(f.s.uploads,2*n);
    aotx_seam_state before; cu(cudaMemcpyFromSymbol(&before,aotx_seam,sizeof before));
    for (unsigned repeat=0;repeat<3;++repeat) {
        f.send(10+repeat,0,audio);
        for (unsigned i=0;i<n;++i) {
            const auto &box=f.mailbox[i+1];
            check(box.state==2 && aotx_media_get(box.bytes+8,4)==429 && box.length==AOTX_SERVICE_HEAD,
                "occupied feature capacity refuses upload before source publication");
            check(aotx_media_get(box.bytes+48,4)==i+1 && aotx_media_get(box.bytes+52,4)==10+repeat &&
                aotx_media_get(box.bytes+16,4)==i+1,"early pressure preserves each distinct transfer and principal");
        }
        auto current=copy(f.m.objects,2*n); auto uploads=copy(f.s.uploads,2*n);
        aotx_seam_state after; aotx_media_state media;
        cu(cudaMemcpyFromSymbol(&after,aotx_seam,sizeof after));cu(cudaMemcpyFromSymbol(&media,aotx_media,sizeof media));
        check(!memcmp(current.data(),baseline.data(),current.size()*sizeof(current[0])) &&
            !memcmp(uploads.data(),receipts.data(),uploads.size()*sizeof(uploads[0])) &&
            bytes==copy(f.m.source,f.m.profile.bytes),"repeated feature pressure preserves sources and upload receipts");
        check(after.dev.tail==before.dev.tail && after.apply.applied_count==before.apply.applied_count &&
            after.apply.state_hash==before.apply.state_hash && !media.accepted && !media.refused,
            "early pressure cannot add canonical records or source counters");
    }
    f.send(20,0,!audio);
    for (unsigned i=0;i<n;++i) check(f.mailbox[i+1].state==2 &&
        aotx_media_get(f.mailbox[i+1].bytes+8,4)==200,
        "a full feature pool does not block the other media family");
    released(f,20);released(f,1);f.send(21,0,audio);
    for (unsigned i=0;i<n;++i) check(f.mailbox[i+1].state==2 &&
        aotx_media_get(f.mailbox[i+1].bytes+8,4)==200,
        "feature capacity release permits a fresh upload admission");
    printf("admission N=%u audio=%u checks=%u failures=%u\n",n,audio,checks,failures);
    cudaFree(f.s.jobs);
}
int main(void)
{
    for (unsigned n:{1u,64u}) for (bool audio:{false,true}) {
        source_cases(n,audio); feature_cases(n,audio); service_cases(n,audio); admission_cases(n,audio);
    }
    printf("media pressure: checks=%u failures=%u\n",checks,failures); return failures?1:0;
}
