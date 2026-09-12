/* Purpose: Check audio prompt ownership, feature positions and source lifetime.
 * Owns: Distinct scoped sources and independently checked native row maps.
 * Launch shape: One prompt per slot at N=1 and N=64, with ordered mutation batches.
 * Lifetime: One test process; trained encoder numerics have separate checks. */
#include "audio/runtime.cuh"
#include "media/runtime.cuh"
#include "media/prompt.cuh"
#include "cli/prompt.cuh"
#include "cognitive/live.cuh"
#include "model/decode_state.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static unsigned checks, failures;
static void check(bool good,const char *name) { ++checks;if(!good){++failures;fprintf(stderr,"FAIL %s\n",name);} }
static void cu(cudaError_t rc) { if(rc!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(rc));exit(1);} }
struct result { unsigned role,stage,refs,extra,count,lease,retired,second_lease,second_retired; unsigned char text[256]; };
__global__ void aotx_audio_registry_seed(unsigned count,unsigned mode)
{
    unsigned slot=threadIdx.x;if(slot>=AOTX_SLOTS)return;
    aotx_say.slot[slot]={};aotx_seqs.slot[slot]={};aotx_media_prompts[slot]={};
    aotx_live_bindings[slot]={};aotx_prompt_roles[slot]=AOTX_MODEL_LANGUAGE_AUDIO;
    if(slot==0){
        aotx_model[AOTX_MODEL_LANGUAGE].layers=1;
        aotx_model[AOTX_MODEL_LANGUAGE].hidden=1024;
        aotx_model[AOTX_MODEL_LANGUAGE_AUDIO].layers=1;
        aotx_model[AOTX_MODEL_LANGUAGE_AUDIO].hidden=4096;
    }
    if(slot>=count)return;
    aotx_media_object &o=aotx_media.objects[slot];o={};
    o.phase=mode==5?AOTX_MEDIA_WAIT:AOTX_MEDIA_READY;
    o.format=mode==4?1u:4u;o.rows=o.span=1u+slot%7u;o.feature=slot*8u;
    o.columns=1;o.lines=o.rows;o.scope=AOTX_MEDIA_LOCAL;o.slot=mode==2?(slot+1u)%64u:slot;
    o.generation=100u+slot;o.worker=~0u;
    for(unsigned j=0;j<32;++j)o.digest[j]=(unsigned char)(slot+3u*j);
    aotx_media_object &v=aotx_media.objects[count+slot];v=o;v.format=mode==1?1u:4u;v.digest[0]^=128;
    v.rows=v.span=2u+slot%5u;v.lines=v.rows;v.feature=(count+slot)*8u;v.generation=500u+slot;
    unsigned char *out=aotx_say.prompt[slot];unsigned at=0;
    const char *hex="0123456789abcdef";
    for(unsigned ref=0;ref<2;++ref){
        const char *head=mode==1&&ref?"[image:":"[audio:";
        const aotx_media_object &source=ref?v:o;
        for(unsigned j=0;head[j];++j)out[at++]=(unsigned char)head[j];
        for(unsigned j=0;j<32;++j){out[at++]=hex[source.digest[j]>>4];out[at++]=hex[source.digest[j]&15];}
        out[at++]=']';out[at++]=' ';
    }
    aotx_say.slot[slot].length=at;aotx_say.slot[slot].wanted=1;
}
__global__ void aotx_audio_registry_role(result *out,unsigned count)
{
    unsigned slot=threadIdx.x;if(slot<count)out[slot].role=aotx_prompt_select(aotx_say.prompt[slot],aotx_say.slot[slot].length);
}
__global__ void aotx_audio_registry_expand(result *out,unsigned count,unsigned mode)
{
    unsigned slot=threadIdx.x;if(slot>=count)return;
    const aotx_media_prompt_state &m=aotx_media_prompts[slot];
    out[slot].stage=m.stage;out[slot].refs=m.count;out[slot].extra=m.extra;
    for(unsigned j=0;j<256;++j)out[slot].text[j]=j<aotx_say.slot[slot].length?aotx_say.prompt[slot][j]:0;
    const unsigned ids[]={10,151647,151646,151648,20,151647,151646,151648,30};
    for(unsigned j=0;j<9;++j)aotx_say_id[slot*AOTX_SAY_TOKENS+j]=ids[j];
    if(mode==3||mode==6)++aotx_media.objects[(mode==6?count:0u)+slot].generation;
    out[slot].count=aotx_media_expand(slot,9);
}
__global__ void aotx_audio_registry_install(const result *out,unsigned count)
{
    unsigned slot=threadIdx.x;if(slot>=count)return;
    aotx_say.slot[slot].wanted=0;
    aotx_seq &s=aotx_seqs.slot[slot];s.state=AOTX_SEQ_STATE_PREFILL;s.input_count=out[slot].count;
    for(unsigned j=0;j<s.input_count;++j)aotx_seq_input[slot][j]=aotx_media_input[slot][j];
}
__global__ void aotx_audio_registry_lease(result *out,unsigned count,unsigned retire)
{
    if(threadIdx.x)return;
    if(retire)for(unsigned slot=0;slot<count;++slot)aotx_seqs.slot[slot].state=AOTX_SEQ_STATE_DONE;
    for(unsigned slot=0;slot<count;++slot){
        unsigned value=aotx_media_leased(slot)?1u:0u;
        if(retire)out[slot].retired=value;else out[slot].lease=value;
        value=aotx_media_leased(count+slot)?1u:0u;
        if(retire)out[slot].second_retired=value;else out[slot].second_lease=value;
    }
}
static void run(unsigned count)
{
    aotx_media_state media={};aotx_audio_runtime_state audio={};
    media.enabled=media.image_enabled=1;media.role=AOTX_MODEL_LANGUAGE;
    media.profile.objects=count*2u;media.profile.feature_rows=count*16u;
    audio.enabled=1;audio.role=AOTX_MODEL_LANGUAGE_AUDIO;audio.profile.feature_rows=count*16u;
    cu(cudaMalloc(&media.objects,media.profile.objects*sizeof(*media.objects)));
    cu(cudaMalloc(&media.features,count*16u*1024u*sizeof(float)));
    cu(cudaMalloc(&audio.features,count*16u*4096u*sizeof(float)));
    cu(cudaMemcpyToSymbol(aotx_media,&media,sizeof media));
    cu(cudaMemcpyToSymbol(aotx_audio_runtime,&audio,sizeof audio));
    result *device;cu(cudaMalloc(&device,count*sizeof(*device)));
    std::vector<result> out(count);std::vector<aotx_model_input> input(AOTX_SEQ_MAX_TOKENS);
    for(unsigned mode=0;mode<7;++mode){
        cu(cudaMemset(device,0,count*sizeof(*device)));
        aotx_audio_registry_seed<<<1,64>>>(count,mode);
        aotx_audio_registry_role<<<1,64>>>(device,count);
        aotx_media_prepare<<<1,64>>>();
        aotx_audio_registry_expand<<<1,64>>>(device,count,mode);
        if(!mode){
            aotx_audio_registry_install<<<1,64>>>(device,count);
            aotx_audio_registry_lease<<<1,1>>>(device,count,0);
            aotx_audio_registry_lease<<<1,1>>>(device,count,1);
        }
        cu(cudaMemcpy(out.data(),device,count*sizeof(*device),cudaMemcpyDeviceToHost));
        for(unsigned slot=0;slot<count;++slot){
            const result &r=out[slot];unsigned rows=1u+slot%7u,second=2u+slot%5u;
            check(r.role==(mode==1?AOTX_MODEL_ROLES:AOTX_MODEL_LANGUAGE_AUDIO),"model ownership follows all source links");
            if(mode==1||mode==2||mode==4){check(r.stage==2&&r.count==0,"invalid source dependencies refuse complete prompts");continue;}
            if(mode==5){check(r.stage==3&&r.count==0,"pending sources preserve a waiting prompt");continue;}
            if(mode==3||mode==6){check(r.stage==1&&r.count==0,"changed generation cannot supply typed rows");continue;}
            check(r.stage==1&&r.refs==2&&r.extra==rows+second-2u,"two complete source expansions are counted");
            check(!strcmp((const char *)r.text,"Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n Audio 2: <|audio_bos|><|AUDIO|><|audio_eos|>\n "),"native labels and markers are ordered");
            check(r.count==rows+second+7u,"expanded token count includes every feature row");
            check(r.lease==1&&r.retired==0&&r.second_lease==1&&r.second_retired==0,"active sequence owns features until retirement");
            cu(cudaMemcpyFromSymbol(input.data(),aotx_media_input,input.size()*sizeof(input[0]),slot*input.size()*sizeof(input[0])));
            for(unsigned j=0;j<r.count;++j){
                check(input[j].position[0]==j&&input[j].position[1]==j&&input[j].position[2]==j,"audio rotary positions are linear");
                unsigned row=j>=2u&&j<2u+rows?j-2u:j>=rows+5u&&j<rows+second+5u?j-rows-5u:~0u;
                check(row==~0u?!input[j].feature:input[j].feature==audio.features+((slot+(j>=rows+5u?count:0u))*8ull+row)*4096u&&
                    input[j].width==4096u&&input[j].generation==(j>=rows+5u?500u:100u)+slot,"features retain source order width and generation");
            }
        }
    }
    cudaFree(device);cudaFree(media.objects);cudaFree(media.features);cudaFree(audio.features);
    media={};audio={};cu(cudaMemcpyToSymbol(aotx_media,&media,sizeof media));cu(cudaMemcpyToSymbol(aotx_audio_runtime,&audio,sizeof audio));
}
int main(){run(1);run(64);printf("audio registry checks=%u failures=%u\n",checks,failures);return failures?1:0;}
