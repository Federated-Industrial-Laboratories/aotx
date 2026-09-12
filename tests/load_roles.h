/* Purpose: Check language load admission before placement and during replay.
 * Owns: Distinct file names, digests and preserved resident identities.
 * Launch shape: One ordered device thread judges batches of 1 and 64 commands.
 * Lifetime: Each case resets only its synthetic model load state. */
#ifndef AOTX_TEST_LOAD_ROLES_H
#define AOTX_TEST_LOAD_ROLES_H
#include "model/load.cuh"
#include "media/runtime.cuh"
struct aotx_load_roles_result {
    unsigned queued, refused, slot, replay_queued, replay_bad, replay_result, preserved;
};
__device__ aotx_load_roles_result aotx_load_roles_results[64];
__global__ void aotx_load_roles_batch(unsigned n,unsigned mode) {
    if(blockIdx.x || threadIdx.x)return;
    const unsigned targets[]={AOTX_MODEL_LANGUAGE,AOTX_MODEL_LANGUAGE_AUDIO,
        AOTX_MODEL_LANGUAGE,AOTX_MODEL_LANGUAGE_Q4,AOTX_MODEL_LANGUAGE,AOTX_MODEL_LANGUAGE_AUDIO};
    const unsigned residents[]={AOTX_MODEL_LANGUAGE_AUDIO,AOTX_MODEL_LANGUAGE,
        AOTX_MODEL_LANGUAGE,AOTX_MODEL_LANGUAGE,AOTX_MODEL_LANGUAGE_Q4,AOTX_MODEL_LANGUAGE_AUDIO};
    const char *names[]={"embedding","reranker","language","language-q4","language-audio"};
    unsigned target=targets[mode],resident=residents[mode];
    aotx_media.image_enabled=0;aotx_audio_runtime.enabled=0;
    for(unsigned i=0;i<n;++i) {
        aotx_model_load={};aotx_seam.replaying=0;aotx_sched.held=0;
        aotx_model_load.files=1;
        auto &file=aotx_model_load.file[0];file.role=target;
        file.name[0]='m';file.name[1]='0'+i/10;file.name[2]='0'+i%10;
        for(unsigned j=0;j<64;++j)file.file[j]=file.name[j];
        for(unsigned j=0;j<32;++j)file.digest[j]=(unsigned char)(17*i+3*j+1);
        auto &old=aotx_model_load.resident[resident];
        old.active=1;old.slot=resident;old.source=i+1;
        for(unsigned j=0;j<32;++j)old.body.digest[j]=(unsigned char)(i+7*j);
        unsigned length=0;while(names[target][length])++length;
        aotx_cli_out out={};unsigned refused=aotx_cli_count.refused;
        aotx_model_load_command(&out,names[target],length,file.name,3,0,i+1);
        auto &result=aotx_load_roles_results[i];result={};
        result.queued=aotx_model_load.pending_count;
        result.refused=aotx_cli_count.refused-refused;
        result.slot=aotx_model_load.pending[0].slot;
        aotx_model_load.pending_count=0;
        aotx_model_body body={};body.tick=i+3;
        for(unsigned j=0;j<length;++j)body.role[j]=names[target][j];
        for(unsigned j=0;j<64;++j)body.file[j]=file.file[j];
        for(unsigned j=0;j<32;++j)body.digest[j]=file.digest[j];
        aotx_seam.replaying=1;
        result.replay_result=aotx_model_load_apply(&body);
        result.replay_queued=aotx_model_load.pending_count;
        result.replay_bad=aotx_model_load.replay_bad;
        result.preserved=old.active==1 && old.slot==resident && old.source==i+1;
        for(unsigned j=0;j<32;++j)result.preserved&=old.body.digest[j]==(unsigned char)(i+7*j);
    }
}
static void aotx_load_roles_check(unsigned n) {
    aotx_live_device device(n);AOTX_LIVE_CLEAR(aotx_seqs);
    for(unsigned mode=0;mode<6;++mode) {
        aotx_load_roles_batch<<<1,1>>>(n,mode);AOTX_CUDA(cudaDeviceSynchronize());
        aotx_load_roles_result results[64];
        AOTX_CUDA(cudaMemcpyFromSymbol(results,aotx_load_roles_results,sizeof results));
        for(unsigned i=0;i<n;++i) {
            const auto &r=results[i];bool allowed=mode>=2;
            aotx_check(r.queued==allowed && r.refused==!allowed,
                "only configured language loads enter the live queue");
            aotx_check(r.replay_queued==allowed && r.replay_bad==!allowed && r.replay_result==!allowed,
                "unconfigured language records cannot enter replay placement");
            unsigned slot=mode==4?AOTX_MODEL_LANGUAGE_Q4:mode==5?AOTX_MODEL_LANGUAGE_AUDIO:AOTX_MODEL_LANGUAGE;
            aotx_check(!allowed || r.slot==slot,"base replacement preserves the configured descriptor slot");
            aotx_check(r.preserved,"admission preserves each prior resident identity");
        }
    }
}
#endif
