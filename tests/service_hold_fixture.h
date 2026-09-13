/* Purpose: Supply distinct mapped requests and retained state for service hold checks.
 * Owns: Valid deployment grants, bounded device tables and exact reply bytes.
 * Launch shape: N=1 and N=64 requests through the real copy and admission nodes.
 * Lifetime: One case; retained results and model completion are explicit test inputs. */
#ifndef AOTX_SERVICE_HOLD_FIXTURE_H
#define AOTX_SERVICE_HOLD_FIXTURE_H
#include "media_pressure_fixture.h"
#include "shared/state.cuh"
#include "model/load.cuh"
#include "model/decode_state.cuh"
#include "model/wrap.cuh"
#include "agent/agent_state.cuh"
#include "cli/prompt.cuh"
#include <array>

__host__ __device__ static void hold_text(unsigned char *p, unsigned i)
{
    p[0]='r'; p[1]='0'+i/10; p[2]='0'+i%10; p[3]=':';
    p[4]=0xc3; p[5]=0xa9; p[6]='!'; p[7]='\n';
}
__global__ void aotx_hold_clock(unsigned long long now, bool held, bool replay)
{ aotx_sched.start_ns=now; aotx_sched.held=held; aotx_seam.replaying=replay; }
__global__ void aotx_hold_seed(unsigned n)
{
    for (unsigned slot=0;slot<AOTX_SLOTS;++slot) {
        aotx_agents.agent[slot]={}; aotx_agents.agent[slot].state=AOTX_AGENT_STATE_FREE;
        aotx_seqs.slot[slot]={}; aotx_live_bindings[slot]={}; aotx_say.slot[slot]={};
        aotx_seq_shown[slot]=aotx_seq_kept[slot]=0; aotx_media_prompts[slot]={};
    }
    aotx_time_tick=271; aotx_live.ready=1; identity(aotx_live_store.lineage,0,77);
    aotx_model_load.pending_count=0;
    for (unsigned role=0;role<AOTX_MODEL_ROLES;++role) {
        aotx_model_load.resident[role].active=0; aotx_model_wrap[role]={};
    }
    auto &model=aotx_model_load.resident[AOTX_MODEL_LANGUAGE]; model.active=1;
    for (unsigned j=0;j<32;++j) model.body.digest[j]=90+j;
    auto &wrap=aotx_model_wrap[AOTX_MODEL_LANGUAGE]; wrap.usable=1;
    aotx_media.owner[0]=aotx_audio_runtime.owner[0]=~0u;
    for (unsigned i=0;i<aotx_media.profile.objects;++i) aotx_media.objects[i].worker=~0u;
    for (unsigned i=0;i<=n;++i) {
        auto &p=aotx_shared.participants[i]; p.active=1; identity(p.id,i,8);
        p.next=1000+i; p.floor=500+i;
        if (i==n) continue;
        auto &s=aotx_shared.spaces[i]; s.active=1; identity(s.id,i,31); identity(s.owner,i,8);
        auto &c=aotx_shared.conversations[i]; c.active=1; c.space=i;
        identity(c.id,i,51); c.next_order=5+i; c.event_floor=1+i;
        auto &r=aotx_shared.receipts[i]; r.phase=AOTX_SHARED_DONE; r.operation=AOTX_SHARED_INPUT;
        identity(r.actor,i,8); identity(r.id,i,41); identity(r.key,i,61); identity(r.command+56,i,51);
        r.sequence=999+i; r.revision=1; r.order=4+i; r.admission_source=100+i;
        r.terminal_source=(i%2?400:200)+i; r.status=200;
        r.participant=r.space=r.conversation=i; r.slot=AOTX_SLOTS; r.role=AOTX_MODEL_LANGUAGE;
        r.output=8; r.prompt=10+i; r.sampled=3+i; r.finish=1; r.gap=i%2;
        r.limit=64+i; r.pages=1; r.length=AOTX_SHARED_COMMAND_HEAD+8; r.input_committed=1;
        for (unsigned j=0;j<32;++j) r.model_digest[j]=90+j;
        aotx_service_bytes(r.command,(const unsigned char *)AOTX_SHARED_MAGIC,8);
        aotx_service_put(r.command+8,AOTX_SHARED_INPUT,4); aotx_service_put(r.command+16,r.sequence,8);
        identity(r.command+24,i,61); identity(r.command+40,0,77); identity(r.command+72,i,31);
        aotx_service_put(r.command+104,r.role,4); aotx_service_put(r.command+108,r.limit,4);
        aotx_service_put(r.command+112,1,4); aotx_service_put(r.command+124,0x3f800000u,4);
        aotx_service_put(r.command+136,8,4); hold_text(r.command+AOTX_SHARED_COMMAND_HEAD,i); hold_text(r.result,i);
        auto &j=aotx_service.jobs[i]; identity(j.id,i,90); identity(j.principal,i,8);
        j.revision=1; j.opened=j.changed=100; j.phase=AOTX_SERVICE_DONE;
        j.role=AOTX_MODEL_LANGUAGE; j.slot=AOTX_SLOTS; j.output=8; j.prompt=20+i;
        j.sampled=4+i; j.finish=1; j.limit=64+i; j.pages=1; j.length=8;
        for (unsigned k=0;k<32;++k) j.model_digest[k]=90+k;
        hold_text(j.text,i); hold_text(j.result,i);
    }
    unsigned char incarnation[16]={}, digest[32]; identity(incarnation,0,78);
    for (unsigned i=0;i<32;++i) digest[i]=150+i;
    aotx_shared.source=500+n; aotx_shared_ack(300+n,12,incarnation,19,digest);
    aotx_shared.pending_bytes=123456; aotx_shared.disk_error=28; aotx_shared.pressure=1;
}
__global__ void aotx_hold_ready(unsigned n)
{
    for (unsigned i=0;i<n;++i) {
        auto &o=aotx_media.objects[n+i]; o={}; o.worker=~0u;
        identity(o.transfer,i,40); identity(o.principal,i,8);
        for (unsigned j=0;j<32;++j) o.digest[j]=i*13+40*5+j;
        o.phase=AOTX_MEDIA_READY; o.scope=AOTX_MEDIA_PRIVATE; o.format=AOTX_IMAGE_JPEG;
        o.bytes=o.received=source_bytes(i); o.offset=(n+i)*128; o.generation=900+i;
        o.rows=o.span=64; o.feature=i*64; o.samples=2000+i;
    }
}
__global__ void aotx_hold_clear_pressure(void)
{ aotx_shared.pressure=aotx_shared.disk_error=0; }
struct aotx_hold_slot { unsigned index, phase, wanted, length; unsigned char text[8]; };
__global__ void aotx_hold_slots(aotx_hold_slot *rows, bool complete)
{
    unsigned slot=threadIdx.x; auto &out=rows[slot]; out={};
    out.index=aotx_service.slot[slot]; if (!out.index) return;
    auto &j=aotx_service.jobs[out.index-1]; out.phase=j.phase;
    out.wanted=aotx_say.slot[slot].wanted; out.length=aotx_say.slot[slot].length;
    for (unsigned i=0;i<8;++i) out.text[i]=aotx_say.prompt[slot][i];
    if (!complete) return;
    /* The decoder boundary supplies DONE with no new token bytes. */
    aotx_service_start_result(slot,0); auto &seq=aotx_seqs.slot[slot]; seq={};
    seq.state=AOTX_SEQ_STATE_DONE; seq.role=j.role; seq.prompt=30+out.index;
    seq.last=seq.stop=42; aotx_seq_shown[slot]=aotx_seq_kept[slot]=0;
}
struct hold_fixture:fixture {
    aotx_shared_state shared={}; aotx_hold_slot *slots=nullptr;
    unsigned long long now=100, revision=1;
    bool held=false, replay=false;
    explicit hold_fixture(unsigned n):fixture(n,true,true) {
        cudaFree(m.features); m.profile.feature_rows*=2;
        storage(&m.features,(size_t)m.profile.feature_rows*AOTX_VISION_OUTPUT); profiles(m.profile,a.profile);
        storage(&s.jobs,AOTX_SERVICE_REQUESTS); cudaFree(s.grants); storage(&s.grants,n+1); s.grant_count=0;
        shared.enabled=1; shared.participant_capacity=n+1; shared.space_capacity=n;
        shared.conversation_capacity=n; shared.member_capacity=1; shared.receipt_capacity=2*n;
        storage(&shared.participants,n+1); storage(&shared.spaces,n); storage(&shared.conversations,n);
        storage(&shared.members,1); storage(&shared.receipts,2*n); storage(&slots,AOTX_SLOTS);
        cu(cudaMemcpyToSymbol(aotx_media,&m,sizeof m)); cu(cudaMemcpyToSymbol(aotx_service,&s,sizeof s));
        cu(cudaMemcpyToSymbol(aotx_shared,&shared,sizeof shared));
        aotx_hold_seed<<<1,1>>>(n); aotx_pressure_sync(); grants(1,127);
        clock(); send(80,0,false); status(200,"valid uploads enter RECEIVE before the hold");
        aotx_hold_ready<<<1,1>>>(n); aotx_pressure_sync();
        requests(AOTX_SERVICE_SUBMIT,91); status(202,"valid distinct inputs enter the real ordinary queue");
    }
    ~hold_fixture() {
        cudaFree(s.jobs); cudaFree(shared.participants); cudaFree(shared.spaces); cudaFree(shared.members);
        cudaFree(shared.conversations); cudaFree(shared.receipts); cudaFree(slots); shared={};
        cu(cudaMemcpyToSymbol(aotx_shared,&shared,sizeof shared));
    }
    void clock(void) { aotx_hold_clock<<<1,1>>>(now,held,replay); }
    void tick(void) {
        clock(); aotx_service_copy<<<AOTX_SERVICE_CHANNELS,64>>>();
        aotx_service_admit<<<1,1>>>(); aotx_pressure_sync();
    }
    void status(unsigned expected, const char *label) {
        for (unsigned i=0;i<count;++i) check(mailbox[i+1].state==2 &&
            aotx_service_get(mailbox[i+1].bytes+8,4)==expected,label);
    }
    unsigned char *frame(unsigned channel, unsigned op, unsigned actor, unsigned bytes) {
        auto &box=mailbox[channel]; memset(box.bytes,0,AOTX_SERVICE_HEAD+bytes);
        memcpy(box.bytes,AOTX_SERVICE_MAGIC,8); aotx_service_put(box.bytes+8,op,4);
        identity(box.bytes+16,actor,8); aotx_service_put(box.bytes+32,revision,8);
        aotx_service_put(box.bytes+88,bytes,4); box.length=AOTX_SERVICE_HEAD+bytes; box.state=1;
        return box.bytes;
    }
    void grants(unsigned long long next, unsigned actions, bool revoke=false, bool run=true) {
        unsigned n=revoke?1:count+1, bytes=n*AOTX_SERVICE_GRANT_BYTES;
        auto *f=frame(0,AOTX_SERVICE_GRANTS,0,bytes); aotx_service_put(f+32,next,8); aotx_service_put(f+76,n,4);
        for (unsigned i=0;i<n;++i) {
            unsigned actor=revoke?count:i; auto *p=f+AOTX_SERVICE_HEAD+i*AOTX_SERVICE_GRANT_BYTES;
            identity(p,actor,8); aotx_service_put(p+16,next,8); aotx_service_put(p+24,actions,4);
            aotx_service_put(p+28,1u<<AOTX_MODEL_LANGUAGE,4); aotx_service_put(p+32,1,4);
            aotx_service_put(p+36,64+actor,4); aotx_service_put(p+40,2,4);
            aotx_service_put(p+44,4,4); aotx_service_put(p+48,4*m.profile.bytes,8);
        }
        if (!run) return;
        tick(); check(mailbox[0].state==2 && aotx_service_get(mailbox[0].bytes+8,4)==200,
            "the real operator installer accepts the complete valid grant batch"); revision=next;
    }
    void requests(unsigned op, unsigned target=0, bool foreign=false, bool run=true, unsigned cursor=0) {
        for (unsigned i=0;i<count;++i) {
            unsigned bytes=op==AOTX_SERVICE_SUBMIT?28:op==AOTX_SERVICE_MEDIA?64:op==11?96:op==10?192:0;
            auto *f=frame(i+1,op,foreign?count:i,bytes); auto *p=f+AOTX_SERVICE_HEAD;
            if (op==AOTX_SERVICE_READ || op==AOTX_SERVICE_CANCEL || op==AOTX_SERVICE_SUBMIT) {
                aotx_service_put(f+40,17,8); identity(f+48,i,target); aotx_service_put(f+64,cursor,8);
            }
            if (op==AOTX_SERVICE_SUBMIT) {
                aotx_service_put(f+72,AOTX_MODEL_LANGUAGE,4); aotx_service_put(f+76,4+i%3,4);
                aotx_service_put(f+84,0x3f800000u,4); aotx_service_put(p,1,4);
                aotx_service_put(p+4,1,4); aotx_service_put(p+8,1,4); aotx_service_put(p+16,8,4); hold_text(p+20,i);
            }
            if (op==AOTX_SERVICE_MEDIA_READ || op==AOTX_SERVICE_MEDIA) identity(f+48,i,target);
            if (op==AOTX_SERVICE_MEDIA) {
                aotx_service_put(p,AOTX_MEDIA_SCHEMA,4); aotx_service_put(p+4,AOTX_MEDIA_CANCEL,4); identity(p+8,i,target);
            }
            if (op==11) {
                memcpy(p,AOTX_SHARED_MAGIC,8); aotx_service_put(p+8,target,4); identity(p+16,0,77);
                if (target==AOTX_SHARED_OPERATION_READ) identity(p+32,i,41);
                if (target==AOTX_SHARED_CONVERSATION_READ) identity(p+32,i,51);
                aotx_service_put(p+72,cursor,8); aotx_service_put(p+80,64,4);
            }
            if (op==10) {
                memcpy(p,AOTX_SHARED_MAGIC,8); aotx_service_put(p+8,AOTX_SHARED_SAVE,4);
                aotx_service_put(p+16,1000+i,8); identity(p+24,i,62); identity(p+40,0,77);
            }
        }
        if (run) tick();
    }
    unsigned long long applied(void) {
        aotx_seam_state state; cu(cudaMemcpyFromSymbol(&state,aotx_seam,sizeof state)); return state.apply.applied_count;
    }
};
#endif
