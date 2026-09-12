/* Purpose: Check role isolation, total tick budgets and bounded sequence service.
 * Owns: Distinct token lists and synthetic page ownership without model arithmetic.
 * Launch shape: The real plan and commit kernels at N=1 and N=64.
 * Lifetime: One device test with no model weights. */
#include "live_fixture.h"
#include "model/decode_state.cuh"
#include <memory>
#include "load_roles.h"
__global__ void aotx_roles_seed(unsigned n,unsigned budget,unsigned mode) {
    unsigned i=threadIdx.x;
    if(!i){aotx_decode.ready=1;aotx_decode.default_role=AOTX_MODEL_LANGUAGE;
        aotx_decode.roles=(1u<<AOTX_MODEL_LANGUAGE)|(1u<<AOTX_MODEL_LANGUAGE_AUDIO);
        aotx_decode.next_role=0;aotx_decode.role=AOTX_MODEL_LANGUAGE;
        aotx_setting_table.row[AOTX_SET_PREFILL_TOKENS].value=budget;
        aotx_sched.held=0;aotx_seam.replaying=0;
        aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape,24,2,128);
        aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE_AUDIO].shape,32,32,128);}
    if(i>=n)return;auto &s=aotx_seqs.slot[i];
    s.role=n==1||i%2?AOTX_MODEL_LANGUAGE_AUDIO:AOTX_MODEL_LANGUAGE;
    s.state=mode==1?AOTX_SEQ_STATE_PREFILL:AOTX_SEQ_STATE_DECODE;
    s.prompt=32+i;s.sampled=1;s.limit=128;s.page_limit=640;s.seed=710+i;
    aotx_model_seen[i]=mode==1?i:32+i;aotx_kv.count[i]=mode==2?0:640;
    for(unsigned k=0;k<128;++k)aotx_seqs.tokens[i][k]=1000+i*128+k;
}
__global__ void aotx_roles_done(unsigned n) {
    unsigned i=threadIdx.x;if(i>=n)return;
    aotx_seqs.slot[i].state=AOTX_SEQ_STATE_DONE;
    aotx_seqs.slot[i].sampled=17+i;aotx_decode.rows[i]=0;
    if(!i)aotx_seqs.live=n;
}
static void run(unsigned n) {
    aotx_live_device d(n);
    for(unsigned mode=0;mode<2;++mode)for(unsigned budget:{1u,17u,256u}){
        AOTX_LIVE_CLEAR(aotx_seqs);AOTX_LIVE_CLEAR(aotx_decode);AOTX_LIVE_CLEAR(aotx_kv);AOTX_LIVE_CLEAR(aotx_seq_asked);
        aotx_roles_seed<<<1,64>>>(n,budget,mode);AOTX_CUDA(cudaDeviceSynchronize());
        unsigned served[64]={0},previous=AOTX_MODEL_ROLES;
        for(unsigned tick=0;tick<128;++tick){aotx_decode_begin<<<1,1>>>();AOTX_CUDA(cudaDeviceSynchronize());
            aotx_decode_state initial;AOTX_CUDA(cudaMemcpyFromSymbol(&initial,aotx_decode,sizeof initial));
            aotx_check(initial.selected==(n==1?AOTX_MODEL_LANGUAGE_AUDIO:previous==AOTX_MODEL_LANGUAGE?
                AOTX_MODEL_LANGUAGE_AUDIO:AOTX_MODEL_LANGUAGE),"ready language roles receive alternating ticks");
            previous=initial.selected;unsigned total=0;
            for(unsigned role:{AOTX_MODEL_LANGUAGE,AOTX_MODEL_LANGUAGE_AUDIO}){
                aotx_decode_select<<<1,1>>>(role);aotx_decode_plan<<<1,64>>>(tick+1);AOTX_CUDA(cudaDeviceSynchronize());
                aotx_decode_state plan;AOTX_CUDA(cudaMemcpyFromSymbol(&plan,aotx_decode,sizeof plan));total+=plan.tokens;
                aotx_check(plan.tokens<=budget&&plan.seqs<=n,"a role never exceeds the total row or sequence capacity");
                aotx_check(role==initial.selected||!plan.tokens,"the other role has no forward rows");
                unsigned rows=0;
                for(unsigned i=0;i<n;++i)if(plan.rows[i]){
                    unsigned expected=n==1||i%2?AOTX_MODEL_LANGUAGE_AUDIO:AOTX_MODEL_LANGUAGE;
                    aotx_check(expected==role,"only the current role contributes a sequence");
                    aotx_check(plan.place[i]<plan.seqs&&plan.agent[plan.place[i]]==i,"physical slot ownership survives rotated scheduling");
                    unsigned at=plan.offset[plan.place[i]];
                    for(unsigned k=0;k<plan.rows[i];++k)
                        aotx_check(plan.ids[at+k]==1000+(int)i*128+(int)plan.first[i]+(int)k,"each planned token belongs to its exact source slot");
                    ++served[i];rows+=plan.rows[i];
                }
                aotx_check(rows==plan.tokens,"every emitted row has one slot owner");
            }
            aotx_check(total<=budget&&total>0,"the complete tick has one bounded nonempty batch");
        }
        for(unsigned i=0;i<n;++i)aotx_check(served[i]>0,"every ready distinct slot receives bounded service");
    }
    AOTX_LIVE_CLEAR(aotx_seqs);AOTX_LIVE_CLEAR(aotx_decode);AOTX_LIVE_CLEAR(aotx_kv);AOTX_LIVE_CLEAR(aotx_seq_asked);
    aotx_roles_seed<<<1,64>>>(n,1,2);aotx_decode_begin<<<1,1>>>();
    for(unsigned role:{AOTX_MODEL_LANGUAGE,AOTX_MODEL_LANGUAGE_AUDIO}){aotx_decode_select<<<1,1>>>(role);aotx_decode_plan<<<1,64>>>(1);}
    AOTX_CUDA(cudaDeviceSynchronize());aotx_kv_table kv;AOTX_CUDA(cudaMemcpyFromSymbol(&kv,aotx_kv,sizeof kv));
    unsigned expected_n=n==1?1:n/2;aotx_check(kv.made==expected_n,"only the selected role requests missing pages");
    for(unsigned j=0;j<kv.made;++j){unsigned i=kv.queue[j].agent;aotx_kvl_shape shape;
        aotx_kvl_make(&shape,n==1?32:24,n==1?32:2,128);
        aotx_check(i<n&&kv.queue[j].pages==aotx_kvl_pages(&shape,32+i+128),"page requests cover the full prompt and reply");}
    aotx_roles_done<<<1,64>>>(n);AOTX_CUDA(cudaDeviceSynchronize());
    auto before=std::make_unique<aotx_seq_table>(),after=std::make_unique<aotx_seq_table>();
    AOTX_CUDA(cudaMemcpyFromSymbol(before.get(),aotx_seqs,sizeof(*before)));
    aotx_decode_select<<<1,1>>>(AOTX_MODEL_LANGUAGE);aotx_decode_commit<<<1,64>>>(2);AOTX_CUDA(cudaDeviceSynchronize());
    AOTX_CUDA(cudaMemcpyFromSymbol(after.get(),aotx_seqs,sizeof(*after)));
    for(unsigned i=0;i<n;++i){bool audio=n==1||i%2;
        aotx_check(audio?!memcmp(&before->slot[i],&after->slot[i],sizeof(aotx_seq)):after->slot[i].state==AOTX_SEQ_STATE_FREE,
            "a commit changes only its owned completion states");}
}
int main() {
    run(1);run(64);
    aotx_load_roles_check(1);aotx_load_roles_check(64);
    printf("decode roles: %u checks, %u failed\n",aotx_checks,aotx_failures);
    return aotx_failures || !aotx_checks ? 1 : 0;
}
