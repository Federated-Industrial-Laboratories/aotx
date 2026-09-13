/* Purpose: Keep scoped service reads available during a scheduler hold.
 * Owns: Exact mapped replies, mutation non-effects, grant changes and expiry checks.
 * Launch shape: Real service nodes process distinct N=1 and N=64 request batches.
 * Lifetime: A hold, suppressed replay traffic and resumed ordinary execution. */
#include "service_hold_fixture.h"
#include "shared/bridge.cuh"
#include "cognitive/checkpoint.cuh"

template<class T> static void append(std::vector<unsigned char> &out, const T *p, size_t count)
{
    const auto *bytes=(const unsigned char *)p; out.insert(out.end(),bytes,bytes+count*sizeof(T));
}
template<class T> static void capture(std::vector<unsigned char> &out, const T *p, size_t count)
{ auto rows=copy(p,count); append(out,rows.data(),rows.size()); }
static std::vector<unsigned char> snapshot(hold_fixture &f)
{
    std::vector<unsigned char> out;
    capture(out,f.s.jobs,AOTX_SERVICE_REQUESTS); capture(out,f.s.uploads,f.s.media_count);
    capture(out,f.m.objects,f.m.profile.objects); capture(out,f.m.hash,f.m.profile.objects);
    capture(out,f.m.source,(size_t)f.m.profile.bytes);
    capture(out,f.shared.participants,f.count+1); capture(out,f.shared.spaces,f.count);
    capture(out,f.shared.conversations,f.count); capture(out,f.shared.members,1);
    capture(out,f.shared.receipts,2*f.count);
    aotx_shared_state shared; aotx_service_state service; aotx_seam_state seam; aotx_media_state media;
    cu(cudaMemcpyFromSymbol(&shared,aotx_shared,sizeof shared)); append(out,&shared,1);
    cu(cudaMemcpyFromSymbol(&service,aotx_service,sizeof service)); append(out,service.slot,AOTX_SLOTS);
    cu(cudaMemcpyFromSymbol(&seam,aotx_seam,sizeof seam)); append(out,&seam.apply,1);
    cu(cudaMemcpyFromSymbol(&media,aotx_media,sizeof media)); append(out,&media,1);
    return out;
}
static void put(std::array<unsigned char,AOTX_SHARED_REPLY_HEAD> &p, unsigned at,
                unsigned long long value, unsigned bytes=8)
{ aotx_service_put(p.data()+at,value,bytes); }
static std::array<unsigned char,AOTX_SHARED_REPLY_HEAD> shared_head(unsigned n, unsigned i,
                                                                 unsigned kind, unsigned cursor)
{
    std::array<unsigned char,AOTX_SHARED_REPLY_HEAD> p={};
    memcpy(p.data(),AOTX_SHARED_MAGIC,8); put(p,8,kind,4); identity(p.data()+16,0,77);
    put(p,136,300+n); put(p,144,12); identity(p.data()+152,0,78);
    put(p,216,123456); put(p,224,28,4); put(p,256,19);
    for (unsigned j=0;j<32;++j) p[264+j]=150+j;
    identity(p.data()+64,i,8);
    if (kind==AOTX_SHARED_OPERATION_READ) {
        put(p,12,AOTX_SHARED_DONE,4); identity(p.data()+32,i,51); identity(p.data()+48,i,31);
        put(p,80,999+i); put(p,88,1000+i); put(p,96,500+i); put(p,104,4+i); put(p,112,1+i);
        put(p,120,100+i); put(p,128,(i%2?400:200)+i); put(p,168,200,4);
        put(p,172,i%2?11:7,4); put(p,176,8,4); put(p,180,10+i,4);
        put(p,184,3+i,4); put(p,188,1,4); put(p,208,cursor);
        put(p,236,AOTX_SHARED_INPUT,4); identity(p.data()+240,i,61); identity(p.data()+296,i,41);
    }
    if (kind==AOTX_SHARED_CONVERSATION_READ) {
        put(p,12,AOTX_SHARED_DONE,4); identity(p.data()+32,i,51); identity(p.data()+48,i,31);
        put(p,104,5+i); put(p,112,1+i); put(p,232,7,4);
    }
    if (kind==AOTX_SHARED_SPACES_READ) { put(p,192,1,4); put(p,196,64,4); }
    return p;
}
static void read_shared(hold_fixture &f, unsigned kind, unsigned cursor=0)
{
    f.requests(11,kind,false,true,cursor); f.status(200,"shared status is readable while held");
    for (unsigned i=0;i<f.count;++i) {
        auto head=shared_head(f.count,i,kind,cursor); const auto &box=f.mailbox[i+1];
        unsigned tail=kind==AOTX_SHARED_OPERATION_READ?8-cursor:kind==AOTX_SHARED_SPACES_READ?64:0;
        check(box.length==AOTX_SERVICE_HEAD+head.size()+tail &&
            !memcmp(box.bytes+AOTX_SERVICE_HEAD,head.data(),head.size()),
            "each shared reply preserves exact scope, sequence, saved proof and pressure fields");
        if (kind==AOTX_SHARED_OPERATION_READ) {
            unsigned char text[8]; hold_text(text,i);
            check(!memcmp(box.bytes+AOTX_SERVICE_HEAD+head.size(),text+cursor,8-cursor),
                "shared byte cursors return the exact distinct UTF-8 suffix");
        }
        if (kind==AOTX_SHARED_SPACES_READ) {
            unsigned char row[64]={}; identity(row,i,31); identity(row+16,i,8); aotx_service_put(row+36,7,4);
            check(!memcmp(box.bytes+AOTX_SERVICE_HEAD+head.size(),row,sizeof row),
                "the space list contains exactly one owned private row");
        }
    }
}
static void read_media(hold_fixture &f, unsigned epoch, unsigned phase, unsigned status=0)
{
    f.requests(AOTX_SERVICE_MEDIA_READ,epoch); f.status(200,"owned source status remains readable");
    for (unsigned i=0;i<f.count;++i) {
        unsigned char expected[64]={}; for (unsigned j=0;j<32;++j) expected[j]=i*13+epoch*5+j;
        aotx_service_put(expected+32,source_bytes(i),8); aotx_service_put(expected+40,phase,4);
        aotx_service_put(expected+44,status,4); aotx_service_put(expected+48,AOTX_IMAGE_JPEG,4);
        if (epoch==40) { aotx_service_put(expected+52,2000+i,4); aotx_service_put(expected+56,64,4); }
        const auto &box=f.mailbox[i+1];
        check(box.length==AOTX_SERVICE_HEAD+64 && !memcmp(box.bytes+AOTX_SERVICE_HEAD,expected,sizeof expected),
            "source replies preserve all distinct metadata bytes and the exact phase");
    }
}
static void read_ordinary(hold_fixture &f)
{
    f.requests(AOTX_SERVICE_READ,90,false,true,4); f.status(200,"ordinary results are readable while held");
    for (unsigned i=0;i<f.count;++i) {
        const auto &box=f.mailbox[i+1]; const auto *p=box.bytes; unsigned char text[8]; hold_text(text,i);
        check(box.length==AOTX_SERVICE_HEAD+4 && !memcmp(p+AOTX_SERVICE_HEAD,text+4,4),
            "ordinary byte cursors preserve each exact UTF-8 result suffix");
        check(aotx_service_get(p+12,4)==AOTX_SERVICE_DONE && aotx_service_get(p+76,4)==8 &&
            aotx_service_get(p+80,4)==20+i && aotx_service_get(p+84,4)==4+i &&
            aotx_service_get(p+92,4)==1 && !aotx_service_get(p+96,4) && !aotx_service_get(p+100,4),
            "ordinary status retains each exact token count and terminal cause");
    }
}
static void read_information(hold_fixture &f)
{
    for (unsigned op:{AOTX_SERVICE_INFO,AOTX_SERVICE_METRICS}) {
        f.requests(op); f.status(200,"information and metrics remain readable while held");
        for (unsigned i=0;i<f.count;++i) {
            const auto &box=f.mailbox[i+1]; const auto *p=box.bytes+AOTX_SERVICE_HEAD;
            check(box.length==AOTX_SERVICE_HEAD+232 && aotx_service_get(p+28,4)==64+i &&
                aotx_service_get(p+32,4)==1 && aotx_service_get(p+36,4)==2 &&
                aotx_service_get(p+40,4)==127 && aotx_service_get(p+44,4)==1 &&
                aotx_service_get(p+48,4)==2*f.count && aotx_service_get(p+152,4)==1,
                "held capabilities retain the exact principal grant and enabled profiles");
            unsigned char digest[32]; for (unsigned j=0;j<32;++j) digest[j]=90+j;
            check(aotx_service_get(p+192,4)==AOTX_MODEL_LANGUAGE && !memcmp(p+200,digest,sizeof digest),
                "held capabilities return the frozen visible model identity");
            check(aotx_service_get(p+80,8)==(op==AOTX_SERVICE_METRICS?271:0) &&
                aotx_service_get(p+88,8)==(op==AOTX_SERVICE_METRICS?f.now:0),
                "held metrics advance the transport clock without running work");
        }
    }
}
static void isolated(hold_fixture &f)
{
    f.requests(AOTX_SERVICE_READ,90,true); f.status(404,"foreign ordinary handles remain hidden during a hold");
    f.requests(AOTX_SERVICE_MEDIA_READ,40,true); f.status(404,"foreign media handles remain hidden during a hold");
    f.requests(11,AOTX_SHARED_OPERATION_READ,true); f.status(404,"foreign private receipt counters remain hidden");
    f.requests(11,AOTX_SHARED_CONVERSATION_READ,true); f.status(404,"foreign private conversations remain hidden");
    f.requests(AOTX_SERVICE_MEDIA_LIST,0,true); f.status(200,"foreign principals can read their own empty media lists");
    for (unsigned i=0;i<f.count;++i) check(f.mailbox[i+1].length==AOTX_SERVICE_HEAD,
        "empty foreign media lists expose no owned source rows");
}
static void mutations(hold_fixture &f, unsigned expected)
{
    auto before=snapshot(f);
    const unsigned ops[]={AOTX_SERVICE_SUBMIT,AOTX_SERVICE_CANCEL,AOTX_SERVICE_MEDIA,10};
    const unsigned ids[]={92,91,80,0};
    for (unsigned i=0;i<4;++i) {
        f.requests(ops[i],ids[i]); f.status(expected,"held mutations return the exact current-grant status");
        check(snapshot(f)==before,"held mutations leave jobs, uploads, shared tables and journal records unchanged");
    }
    aotx_service_work<<<1,1>>>(); aotx_pressure_sync();
    check(snapshot(f)==before,"the ordinary work node leaves queued requests unchanged during a hold");
}
static void replay_suppressed(hold_fixture &f)
{
    auto before=snapshot(f); auto grants=copy(f.s.grants,f.count+1); auto ready=copy(f.s.ready,AOTX_SERVICE_CHANNELS);
    aotx_service_state service; cu(cudaMemcpyFromSymbol(&service,aotx_service,sizeof service));
    f.replay=true;
    for (unsigned op:{(unsigned)AOTX_SERVICE_METRICS,10u}) {
        f.held=op==10u;
        f.requests(op,0,false,false); f.grants(2,127,false,false); f.tick();
        for (unsigned i=0;i<=f.count;++i) check(f.mailbox[i].state==1,
            "replay leaves mapped read, mutation and operator requests pending");
        check(copy(f.s.ready,AOTX_SERVICE_CHANNELS)==ready && snapshot(f)==before,
            "replay suppresses copy, admission, expiry and recorded state changes");
        auto current=copy(f.s.grants,f.count+1); aotx_service_state state;
        cu(cudaMemcpyFromSymbol(&state,aotx_service,sizeof state));
        check(!memcmp(grants.data(),current.data(),grants.size()*sizeof(grants[0])) &&
            state.revision==service.revision && state.clock==service.clock,
            "replay does not install grants or advance the service clock");
    }
    for (unsigned i=0;i<AOTX_SERVICE_CHANNELS;++i) f.mailbox[i].state=0;
    f.replay=false;
}
static void resume(hold_fixture &f)
{
    auto records=f.applied(); f.held=false; f.tick();
    check(f.applied()==records+f.count,"resumed admission records one cancellation for each expired RECEIVE upload");
    read_media(f,80,AOTX_MEDIA_REFUSED,AOTX_MEDIA_CANCELLED); read_media(f,40,AOTX_MEDIA_READY);
    unsigned total=0;
    for (unsigned wave=0;wave<(f.count+62)/63;++wave) {
        aotx_service_work<<<1,1>>>(); aotx_hold_slots<<<1,AOTX_SLOTS>>>(f.slots,false); aotx_pressure_sync();
        auto rows=copy(f.slots,AOTX_SLOTS); unsigned count=0;
        for (unsigned slot=0;slot<AOTX_SLOTS;++slot) if (rows[slot].index) {
            ++count; unsigned i=rows[slot].index-1-f.count; unsigned char text[8]; hold_text(text,i);
            check(slot>0 && i<f.count && rows[slot].phase==AOTX_SERVICE_PREPARE && rows[slot].wanted==1 &&
                rows[slot].length==8 && !memcmp(rows[slot].text,text,8),
                "resumed work leases each distinct queued prompt onto a free non-operator slot");
        }
        check(count==((f.count-total)>63?63:f.count-total),"resumed work uses the exact available slot count"); total+=count;
        aotx_hold_slots<<<1,AOTX_SLOTS>>>(f.slots,true); aotx_service_reply<<<1,AOTX_SLOTS>>>(); aotx_pressure_sync();
    }
    auto jobs=copy(f.s.jobs+f.count,f.count);
    for (const auto &j:jobs) check(j.phase==AOTX_SERVICE_DONE && j.slot==AOTX_SLOTS && !j.cancel && !j.status && j.finish==1,
        "ordinary terminal publication releases every resumed request slot");
    aotx_service_state state; cu(cudaMemcpyFromSymbol(&state,aotx_service,sizeof state));
    for (unsigned slot=0;slot<AOTX_SLOTS;++slot) check(!state.slot[slot],"terminal requests release all service ownership");
}
static void held_case(unsigned n)
{
    hold_fixture f(n); f.held=true; f.now+=AOTX_SERVICE_UPLOAD_SECONDS*1000000000ull;
    auto before=snapshot(f);
    read_information(f); read_ordinary(f); read_media(f,40,AOTX_MEDIA_READY); read_media(f,80,AOTX_MEDIA_RECEIVE);
    for (unsigned kind:{AOTX_SHARED_OPERATION_READ,AOTX_SHARED_SAVE_READ,AOTX_SHARED_SPACES_READ,AOTX_SHARED_CONVERSATION_READ})
        read_shared(f,kind,kind==AOTX_SHARED_OPERATION_READ?4:0);
    f.requests(AOTX_SERVICE_MEDIA_LIST); f.status(200,"held media lists remain scoped to the authenticated principal");
    for (unsigned i=0;i<n;++i) {
        const auto &box=f.mailbox[i+1]; const auto *p=box.bytes+AOTX_SERVICE_HEAD;
        check(box.length==AOTX_SERVICE_HEAD+160 && !aotx_service_get(box.bytes+64,8) &&
            aotx_service_get(p,4)==i+1 && aotx_service_get(p+4,4)==80 &&
            aotx_service_get(p+80,4)==i+1 && aotx_service_get(p+84,4)==40,
            "owned media lists return exactly the pending and ready source identities");
    }
    isolated(f); check(snapshot(f)==before,"held reads preserve all recorded and cognitive state beyond the upload deadline");
    aotx_hold_clear_pressure<<<1,1>>>(); aotx_pressure_sync(); mutations(f,429);
    replay_suppressed(f); resume(f);
    printf("N=%u hold checks=%u failures=%u\n",n,checks,failures);
}
static void grant_case(unsigned n)
{
    hold_fixture f(n); f.held=true; f.clock(); aotx_pressure_sync();
    aotx_hold_clear_pressure<<<1,1>>>(); aotx_pressure_sync(); auto before=snapshot(f);
    unsigned actions=AOTX_SERVICE_TELEMETRY|AOTX_SHARED_READ_ACTION;
    f.grants(2,actions); check(snapshot(f)==before,"held grant replacement changes transport authority without changing retained state");
    f.revision=1; f.requests(11,AOTX_SHARED_OPERATION_READ); f.status(403,"the old grant revision is refused during a hold");
    f.revision=2; f.requests(11,AOTX_SHARED_OPERATION_READ); f.status(200,"fresh read grants can read saved shared receipts");
    mutations(f,403);
    f.requests(AOTX_SERVICE_READ,90); f.status(403,"removed inference rights refuse ordinary reads during a hold");
    f.requests(AOTX_SERVICE_MEDIA_READ,40); f.status(403,"removed upload rights refuse source reads during a hold");
    f.grants(3,127,true); f.requests(11,AOTX_SHARED_OPERATION_READ); f.status(403,"removed principals cannot read with a current revision");
    f.grants(4,127); f.requests(11,AOTX_SHARED_OPERATION_READ); f.status(200,"restored current grants permit retained shared reads");
    f.requests(AOTX_SERVICE_READ,90); f.status(404,"ordinary handles keep their original grant revision after replacement");
    check(snapshot(f)==before,"grant revocation and restoration preserve shared receipts and cognitive state");
    for (unsigned i=0;i<AOTX_SERVICE_CHANNELS;++i) f.mailbox[i].state=0;
    f.held=false; f.tick(); aotx_service_work<<<1,1>>>(); aotx_pressure_sync();
    auto jobs=copy(f.s.jobs+n,n);
    for (const auto &j:jobs) check(j.phase==AOTX_SERVICE_CANCELLED && j.cancel && j.status==403 && j.slot==AOTX_SLOTS,
        "resumed work cancels queued inputs whose original grant revision is no longer current");
    printf("N=%u grants checks=%u failures=%u\n",n,checks,failures);
}
__global__ void aotx_hold_checkpoint(aotx_checkpoint_ring *ring, unsigned n)
{
    aotx_checkpoint={}; aotx_checkpoint.ring=ring; aotx_checkpoint.head=2;
    aotx_checkpoint.pending_bytes=1200+2*n;
    aotx_checkpoint.cuts[0]={77,12,300+n,500+n};
    aotx_checkpoint.cuts[1]={88,13,999+n,700+n};
    aotx_seam.boot_id=19; aotx_shared.source=1100+n;
    aotx_shared.pending_bytes=1200+2*n; aotx_shared.disk_error=0; aotx_shared.pressure=1;
}
static void saved_projection(hold_fixture &f, unsigned long long source,
                             unsigned long long pending, unsigned error, unsigned generation)
{
    f.clock(); aotx_shared_work<<<1,1>>>(); aotx_pressure_sync();
    for (unsigned kind:{AOTX_SHARED_SAVE_READ,AOTX_SHARED_OPERATION_READ}) {
        f.requests(11,kind); f.status(200,"current save state is readable during a hold");
        for (unsigned i=0;i<f.count;++i) {
            const unsigned char *p=f.mailbox[i+1].bytes+AOTX_SERVICE_HEAD;
            unsigned char incarnation[16]={}; identity(incarnation,0,78);
            check(aotx_service_get(p+136,8)==source && aotx_service_get(p+144,8)==generation &&
                !memcmp(p+152,incarnation,sizeof incarnation) && aotx_service_get(p+256,8)==19,
                "held replies include the exact current saved boundary and file identity");
            unsigned char digest[32]; for (unsigned j=0;j<32;++j) digest[j]=180+generation+j;
            check(!memcmp(p+264,digest,sizeof digest) && aotx_service_get(p+216,8)==pending &&
                aotx_service_get(p+224,4)==error,
                "held replies update the commit digest, pending bytes and disk error");
            if (kind==AOTX_SHARED_OPERATION_READ) {
                bool terminal=(i%2?400:200)+i<=source;
                unsigned flags=3u+(terminal?4u:0u)+(i%2?8u:0u);
                check(aotx_service_get(p+172,4)==flags && aotx_service_get(p+12,4)==AOTX_SHARED_DONE,
                    "each receipt reports its exact saved flags without changing execution state");
            }
        }
    }
}
static void checkpoint_case(unsigned n)
{
    hold_fixture f(n); aotx_checkpoint_ring *host=nullptr,*device=nullptr;
    cu(cudaHostAlloc(&host,sizeof(*host),cudaHostAllocMapped)); memset(host,0,sizeof(*host));
    cu(cudaHostGetDevicePointer(&device,host,0));
    host->magic=AOTX_CP_MAGIC; host->layout=AOTX_CP_LAYOUT; host->boot=host->ack_boot=19;
    host->head=2; host->consumed=1; host->ack_serial=2;
    host->durable_sequence=77; host->durable_revision=12; host->reserved[1]=300+n; host->generation=12;
    identity((unsigned char *)host->incarnation,0,78);
    for (unsigned i=0;i<32;++i) ((unsigned char *)host->commit_digest)[i]=192+i;
    aotx_hold_checkpoint<<<1,1>>>(device,n); aotx_pressure_sync(); f.held=true;
    auto records=f.applied();
    saved_projection(f,300+n,700+n,0,12);
    __atomic_store_n(&host->ack_serial,3ull,__ATOMIC_RELEASE);
    host->consumed=2; host->durable_sequence=88; host->durable_revision=13;
    host->reserved[1]=999+n; host->generation=13;
    for (unsigned i=0;i<32;++i) ((unsigned char *)host->commit_digest)[i]=193+i;
    __atomic_store_n(&host->ack_serial,4ull,__ATOMIC_RELEASE);
    saved_projection(f,999+n,0,0,13);
    __atomic_store_n(&host->error,1ull,__ATOMIC_RELEASE);
    saved_projection(f,999+n,0,1,13);
    mutations(f,429);
    __atomic_store_n(&host->error,0ull,__ATOMIC_RELEASE);
    saved_projection(f,999+n,0,0,13);
    check(f.applied()==records,"save projections do not emit canonical records while held");
    auto before=snapshot(f); f.replay=true; f.clock(); aotx_shared_work<<<1,1>>>(); aotx_pressure_sync();
    check(snapshot(f)==before,"replay suppresses shared state refresh and execution");
    f.replay=false; f.held=false; saved_projection(f,999+n,0,0,13);
    aotx_checkpoint_state clear={}; cu(cudaMemcpyToSymbol(aotx_checkpoint,&clear,sizeof clear));
    cu(cudaFreeHost(host));
    printf("N=%u checkpoint checks=%u failures=%u\n",n,checks,failures);
}

int main(void)
{
    cu(cudaSetDeviceFlags(cudaDeviceMapHost));
    for (unsigned n:{1u,64u}) { held_case(n); grant_case(n); checkpoint_case(n); }
    printf("service hold: %u checks, %u failures\n",checks,failures); return failures?1:0;
}
