/* Purpose: Schedule native audio over scoped immutable media sources.
 * Owns: Worker assignments, exact feature reservations and terminal publication.
 * Launch shape: One initializer thread per worker and one ordered scheduler thread.
 * Lifetime: One runtime allocation. */
#include "audio/runtime.cuh"
#include "audio/workspace.cuh"
#include "media/runtime.cuh"
#include "sched/sched.cuh"
__device__ aotx_audio_runtime_state aotx_audio_runtime;
__global__ void aotx_audio_initialize(void)
{
    unsigned w=blockIdx.x*blockDim.x+threadIdx.x;if(w>=aotx_audio_runtime.profile.workers)return;
    aotx_audio_runtime.owner[w]=~0u;aotx_audio_job &j=aotx_audio_runtime.jobs[w];j={};
    aotx_audio_layout(j,aotx_audio_runtime.workspace+w*aotx_audio_runtime.workspace_each,aotx_audio_runtime.profile.source_frames);
    j.source_capacity=aotx_audio_runtime.profile.source_frames;j.phase=AOTX_AUDIO_REFUSED;
}
__global__ void aotx_audio_schedule(void)
{
    if(!aotx_audio_runtime.enabled || aotx_sched.held)return;
    for(unsigned w=0;w<aotx_audio_runtime.profile.workers;++w){
        if(aotx_audio_runtime.owner[w]!=~0u)continue;
        unsigned next=~0u;
        for(unsigned i=0;i<aotx_media.profile.objects;++i){
            const aotx_media_object &o=aotx_media.objects[i];
            if(o.phase==AOTX_MEDIA_WAIT && aotx_media_is_audio(o.format) &&
                (next==~0u || o.generation<aotx_media.objects[next].generation))next=i;
        }
        if(next==~0u)break;
        aotx_media_object &o=aotx_media.objects[next];aotx_audio_job &j=aotx_audio_runtime.jobs[w];
        o.worker=w;o.phase=AOTX_MEDIA_DECODE;aotx_audio_runtime.owner[w]=next;
        j.source=aotx_media.source+o.offset;j.source_bytes=o.bytes;j.format=o.format;
        j.phase=AOTX_AUDIO_NEW;j.status=j.cancel=0;j.features=0;j.feature_capacity=AOTX_AUDIO_ROWS;
    }
}
__global__ void aotx_audio_complete(void)
{
    if(!aotx_audio_runtime.enabled)return;
    for(unsigned w=0;w<aotx_audio_runtime.profile.workers;++w){
        unsigned i=aotx_audio_runtime.owner[w];if(i==~0u)continue;
        aotx_media_object &o=aotx_media.objects[i];aotx_audio_job &j=aotx_audio_runtime.jobs[w];
        if(o.phase==AOTX_MEDIA_DECODE && j.phase==AOTX_AUDIO_RESAMPLE){
            unsigned at=aotx_media_rows(j.rows,true);
            if(at==~0u){
                o.phase=AOTX_MEDIA_REFUSED;
                o.status=j.rows>aotx_audio_runtime.profile.feature_rows?AOTX_MEDIA_LIMIT:AOTX_MEDIA_PRESSURE;
                j.phase=AOTX_AUDIO_REFUSED;++aotx_media.refused;
            }
            else {
                o.feature=at;o.span=j.rows;o.phase=AOTX_MEDIA_ENCODE;
                o.rate=j.rate;o.channels=j.channels;o.source_frames=j.source_frames;o.samples=j.samples;o.encoding=j.encoding;
                j.features=aotx_audio_runtime.features+(unsigned long long)at*4096u;
            }
        }
        if(o.phase!=AOTX_MEDIA_REFUSED && j.phase==AOTX_AUDIO_REFUSED){
            o.phase=AOTX_MEDIA_REFUSED;
            o.status=j.status==AOTX_AUDIO_NO_SIGNAL?AOTX_MEDIA_NO_SIGNAL:j.status==AOTX_AUDIO_NONFINITE?
                AOTX_MEDIA_AUDIO_NUMERIC:j.status==AOTX_AUDIO_LIMIT?AOTX_MEDIA_LIMIT:AOTX_MEDIA_AUDIO_FORMAT;
            ++aotx_media.refused;
        } else if(o.phase==AOTX_MEDIA_ENCODE && j.phase==AOTX_AUDIO_READY){
            o.phase=AOTX_MEDIA_READY;o.rows=j.rows;o.span=j.rows;o.columns=1;o.lines=j.rows;
        }
        if(o.phase==AOTX_MEDIA_REFUSED || o.phase==AOTX_MEDIA_READY){
            o.worker=~0u;aotx_audio_runtime.owner[w]=~0u;j.phase=AOTX_AUDIO_REFUSED;
        }
    }
}
