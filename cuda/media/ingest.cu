/* Purpose: Convert producer frames to canonical source-byte records on the device.
 * Owns: The producer cursor and one bounded frame expansion offset.
 * Launch shape: One ordered thread, at most 64 canonical records per tick.
 * Lifetime: One mapped producer ring. */
#include "media/runtime.cuh"
#include "cognitive/live.cuh"
#include "sched/sched.cuh"
#include "cli/cli.cuh"

static __device__ unsigned char aotx_media_body[AOTX_BODY_BYTES];
static __device__ char aotx_media_notice_text[160];
static __device__ unsigned aotx_media_transfer(const unsigned char *id)
{
    for (unsigned i=0; aotx_media.enabled && i<aotx_media.profile.objects; ++i) {
        const aotx_media_object &o=aotx_media.objects[i];
        bool same=o.phase!=AOTX_MEDIA_FREE;
        for (unsigned j=0;j<16;++j) same &= o.transfer[j]==id[j];
        if (same) return i;
    }
    return aotx_media.profile.objects;
}
static __device__ bool aotx_media_owner(unsigned index,unsigned slot)
{
    if (index>=aotx_media.profile.objects) return false;
    const aotx_media_object &o=aotx_media.objects[index];
    const aotx_live_binding &b=aotx_live_bindings[slot];
    if (o.slot!=slot) return false;
    if (o.scope==AOTX_MEDIA_SHARED) return true;
    if (o.scope==AOTX_MEDIA_LOCAL) return !b.active;
    if (!b.active) return false;
    for (unsigned j=0;j<16;++j) {
        if (o.room[j]!=b.room[j]) return false;
        if (o.scope==AOTX_MEDIA_PRIVATE && o.principal[j]!=b.principal[j]) return false;
    }
    return true;
}
static __device__ void aotx_media_record(unsigned n)
{
    unsigned long long seq=aotx_seam_write(AOTX_WRITER_FEEDER,AOTX_CLASS_A,AOTX_REC_MEDIA,0,aotx_media_body,n);
    aotx_seam.apply.state_hash=aotx_seam_fnv1a(aotx_seam.apply.state_hash,aotx_media_body,n);
    ++aotx_seam.apply.applied_count;
    aotx_media_part(aotx_media_body,n,seq);
}
__device__ void aotx_media_report(unsigned index,unsigned status,unsigned op)
{
    unsigned at=0;
    const char *prefix=status ? "image: refused " : op==AOTX_MEDIA_CANCEL ? "image: canceled " :
        op==AOTX_MEDIA_END ? "image: ready [image:" : "image: [image:";
    for (unsigned i=0;prefix[i];++i) aotx_media_notice_text[at++]=prefix[i];
    const char *hex="0123456789abcdef";
    if (!status && (op==AOTX_MEDIA_BEGIN || op==AOTX_MEDIA_END) && index<aotx_media.profile.objects) {
        for (unsigned i=0;i<32;++i) {
            unsigned c=aotx_media.objects[index].digest[i];
            aotx_media_notice_text[at++]=hex[c>>4];aotx_media_notice_text[at++]=hex[c&15];
        }
        const char *middle="] transfer ";
        for(unsigned i=0;middle[i];++i) aotx_media_notice_text[at++]=middle[i];
    }
    for(unsigned i=0;i<16;++i) {
        unsigned c=index<aotx_media.profile.objects ? aotx_media.objects[index].transfer[i] : aotx_media_body[8+i];
        aotx_media_notice_text[at++]=hex[c>>4];aotx_media_notice_text[at++]=hex[c&15];
    }
    if (status) {aotx_media_notice_text[at++]=' ';aotx_media_notice_text[at++]=hex[status&15];}
    aotx_console_write(aotx_media_notice_text,at);
}
__global__ void aotx_media_ingest(void)
{
    if (!aotx_media.ring || aotx_sched.held || aotx_seam.replaying) return;
    unsigned long long head=aotx_media_acquire(&aotx_media.ring->head);
    if (head<aotx_media.consumed || head-aotx_media.consumed>AOTX_MEDIA_RING_SLOTS) {
        aotx_media.fatal=1;
        aotx_media_release(&aotx_media.ring->status,AOTX_MEDIA_INVALID);
        aotx_media_release(&aotx_media.ring->closed,1);return;
    }
    for (unsigned made=0;made<AOTX_MEDIA_EMIT && aotx_media.consumed<head;++made) {
        const unsigned char *f=aotx_media.frames+(aotx_media.consumed%AOTX_MEDIA_RING_SLOTS)*AOTX_MEDIA_FRAME_BYTES;
        unsigned op=(unsigned)aotx_media_get(f+4,4),payload=(unsigned)aotx_media_get(f+40,4);
        unsigned slot=(unsigned)aotx_media_get(f+44,4),scope=(unsigned)aotx_media_get(f+48,4);
        bool valid=aotx_media_get(f,4)==AOTX_MEDIA_SCHEMA && slot<AOTX_SLOTS &&
            scope<=AOTX_MEDIA_SHARED && payload<=AOTX_MEDIA_FRAME_DATA;
        unsigned nonzero=0;for(unsigned j=8;j<24;++j) nonzero|=f[j];valid &= nonzero!=0;
        for(unsigned j=52;j<64;++j) valid &= !f[j];
        unsigned index=aotx_media_transfer(f+8),status=0;
        bool owner=valid && aotx_media_owner(index,slot);
        for(unsigned j=0;j<AOTX_BODY_BYTES;++j) aotx_media_body[j]=0;
        for(unsigned j=0;j<AOTX_MEDIA_PART;++j) aotx_media_body[j]=f[j];
        unsigned n=0;
        if (!aotx_media.enabled) {valid=false;status=AOTX_MEDIA_UNAVAILABLE;}
        if (valid && op!=AOTX_MEDIA_BEGIN && !owner) {valid=false;status=AOTX_MEDIA_UNAVAILABLE;}
        if (valid && op==AOTX_MEDIA_BEGIN && payload==48 && !aotx_media.frame_at) {
            unsigned format=(unsigned)aotx_media_get(f+64,4),width=(unsigned)aotx_media_get(f+68,4);
            unsigned height=(unsigned)aotx_media_get(f+72,4);
            unsigned long long bytes=aotx_media_get(f+24,8);
            valid=index==aotx_media.profile.objects && !aotx_media_get(f+32,8) &&
                !aotx_media_get(f+76,4) && bytes && bytes<=~0ull/8u &&
                ((format==AOTX_IMAGE_JPEG && !width && !height) ||
                 (format==AOTX_IMAGE_RGB8 && width && height && bytes>=AOTX_MEDIA_RGB_HEAD &&
                  (bytes-AOTX_MEDIA_RGB_HEAD)%3u==0 &&
                  (unsigned long long)width*height==(bytes-AOTX_MEDIA_RGB_HEAD)/3u));
            const aotx_live_binding &b=aotx_live_bindings[slot];
            if (!b.active && scope==AOTX_MEDIA_ROOM) valid=false;
            if (!b.active && scope==AOTX_MEDIA_PRIVATE) scope=AOTX_MEDIA_LOCAL;
            aotx_media_put(aotx_media_body+40,slot,4);aotx_media_put(aotx_media_body+44,scope,4);
            for(unsigned j=0;j<16;++j) aotx_media_body[48+j]=f[64+j];
            if(scope==AOTX_MEDIA_PRIVATE || scope==AOTX_MEDIA_ROOM)
                for(unsigned j=0;j<16;++j) aotx_media_body[64+j]=b.room[j];
            if(scope==AOTX_MEDIA_PRIVATE)
                for(unsigned j=0;j<16;++j) aotx_media_body[80+j]=b.principal[j];
            for(unsigned j=0;j<32;++j) aotx_media_body[96+j]=f[80+j];
            n=AOTX_MEDIA_BEGIN_BYTES;
        } else if(valid && op==AOTX_MEDIA_CHUNK && payload && aotx_media.frame_at<payload) {
            unsigned take=min(AOTX_MEDIA_DATA,payload-aotx_media.frame_at);
            unsigned long long offset=aotx_media_get(f+32,8);
            valid=offset<=~0ull-aotx_media.frame_at && aotx_media.objects[index].phase==AOTX_MEDIA_RECEIVE;
            aotx_media_put(aotx_media_body+32,offset+aotx_media.frame_at,8);
            for(unsigned j=0;j<take;++j) aotx_media_body[AOTX_MEDIA_PART+j]=f[AOTX_MEDIA_FRAME_HEAD+aotx_media.frame_at+j];
            aotx_media.frame_at+=take;n=AOTX_MEDIA_PART+take;
        } else if(valid && op==AOTX_MEDIA_END && !payload) n=AOTX_MEDIA_PART;
        else if(valid && op==AOTX_MEDIA_CANCEL && !payload) n=24;
        else valid=false;
        if(valid) {
            aotx_media_record(n);index=aotx_media_transfer(f+8);
            if(index==aotx_media.profile.objects) status=AOTX_MEDIA_LIMIT;
            else {
                const aotx_media_object &o=aotx_media.objects[index];
                if(o.phase==AOTX_MEDIA_REFUSED && !(op==AOTX_MEDIA_CANCEL && o.status==AOTX_MEDIA_CANCELLED)) status=o.status;
                if(op==AOTX_MEDIA_CANCEL && o.phase!=AOTX_MEDIA_REFUSED) status=AOTX_MEDIA_LEASED;
            }
        } else {
            if(!status) status=AOTX_MEDIA_INVALID;
            ++aotx_media.refused;
            if(owner && op!=AOTX_MEDIA_BEGIN && aotx_media.objects[index].phase==AOTX_MEDIA_RECEIVE) {
                aotx_media_put(aotx_media_body,AOTX_MEDIA_SCHEMA,4);
                aotx_media_put(aotx_media_body+4,AOTX_MEDIA_CANCEL,4);aotx_media_record(24);
            }
        }
        if(status || op==AOTX_MEDIA_BEGIN || op==AOTX_MEDIA_CANCEL) aotx_media_report(index,status,op);
        if(status || op!=AOTX_MEDIA_CHUNK || aotx_media.frame_at==payload) {
            aotx_media.frame_at=0;++aotx_media.consumed;
            aotx_media_release(&aotx_media.ring->status,status);
            aotx_media_release(&aotx_media.ring->consumed,aotx_media.consumed);
        }
    }
    if(aotx_media.consumed==head && aotx_media.enabled && aotx_media_acquire(&aotx_media.ring->closed)) {
        for(unsigned i=0;i<aotx_media.profile.objects;++i) {
            const aotx_media_object &o=aotx_media.objects[i];
            if(o.phase!=AOTX_MEDIA_RECEIVE) continue;
            aotx_media_put(aotx_media_body,AOTX_MEDIA_SCHEMA,4);
            aotx_media_put(aotx_media_body+4,AOTX_MEDIA_CANCEL,4);
            for(unsigned j=0;j<16;++j) aotx_media_body[8+j]=o.transfer[j];
            aotx_media_record(24);break;
        }
    }
}
