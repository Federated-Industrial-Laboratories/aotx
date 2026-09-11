/* Purpose: Check source assembly, scope, cancellation, identity and typed image positions.
 * Owns: Distinct byte fixtures, small device arenas and independent expected row maps.
 * Launch shape: Ordered record batches and one prompt thread per slot, at N=1 and N=64.
 * Lifetime: One test process; no trained weights are needed for registry invariants. */
#include "media/runtime.cuh"
#include "media/prompt.cuh"
#include "cli/prompt.cuh"
#include "cognitive/live.cuh"
#include "model/decode.cuh"
#include "model/decode_state.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstddef>
extern "C" {
#include "disk/wire/diskwire.h"
}
static unsigned checks, failures;
static void check(bool good, const char *name) {
    ++checks; if (!good) { ++failures; fprintf(stderr,"FAIL %s\n",name); }
}
static void cu(cudaError_t rc) {
    if (rc != cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(rc)); exit(1); }
}
struct record { unsigned char body[192]; unsigned bytes, accepted; };
__global__ void aotx_media_test_apply(record *rows, unsigned count, unsigned long long sequence) {
    for (unsigned i=0;i<count;++i) rows[i].accepted=aotx_media_part(rows[i].body,rows[i].bytes,sequence+i);
}
__global__ void aotx_media_test_bind(unsigned count, unsigned kind) {
    for (unsigned i=0;i<AOTX_SLOTS;++i) {
        aotx_say.slot[i]={}; aotx_media_prompts[i]={}; aotx_seqs.slot[i]={};
        aotx_live_bindings[i]={};
    }
    for (unsigned i=0;i<count;++i) {
        aotx_live_bindings[i] = {};
        if (kind) {
            aotx_live_bindings[i].active=1;
            aotx_live_bindings[i].principal[0]=(unsigned char)(i+1);
            aotx_live_bindings[i].room[0]=(unsigned char)(1+i/2);
        }
    }
}
__global__ void aotx_media_test_ready(unsigned count) {
    for (unsigned i=0;i<count;++i) {
        aotx_media_object &o=aotx_media.objects[i];
        o.phase=AOTX_MEDIA_READY; o.feature=i*6; o.rows=o.span=6; o.columns=3; o.lines=2;
    }
    aotx_model[AOTX_MODEL_LANGUAGE].layers=1;
    aotx_model[AOTX_MODEL_LANGUAGE].hidden=1024;
    aotx_model[AOTX_MODEL_LANGUAGE].delta_dim=1;
}
__global__ void aotx_media_test_find(int *out,unsigned count) {
    unsigned i=threadIdx.x;
    if(i<count) {
        out[3*i]=aotx_media_find(aotx_media.objects[i].digest,i);
        out[3*i+1]=aotx_media_find(aotx_media.objects[(i+1)%count].digest,i);
        out[3*i+2]=aotx_media_quiet();
    }
}
__global__ void aotx_media_test_expand(unsigned *out,unsigned count) {
    unsigned i=threadIdx.x;
    if(i>=count) return;
    unsigned *ids=aotx_say_id+i*AOTX_SAY_TOKENS;
    const unsigned words[]={10,248053,248056,248054,20,248053,248056,248054,30};
    for(unsigned j=0;j<9;++j) ids[j]=words[j];
    out[i]=aotx_media_expand(i,9);
}
__global__ void aotx_media_test_lease(unsigned slot,unsigned active) {
    aotx_say.slot[slot].wanted=0;
    aotx_seq &s=aotx_seqs.slot[slot];
    s.state=active ? AOTX_SEQ_STATE_PREFILL : AOTX_SEQ_STATE_DONE;
    s.input_count=19;
    for(unsigned j=0;j<s.input_count;++j) aotx_seq_input[slot][j]=aotx_media_input[slot][j];
}
__global__ void aotx_media_test_generation(unsigned *out,unsigned count) {
    unsigned i=threadIdx.x;if(i>=count)return;
    aotx_seq &s=aotx_seqs.slot[i];s.state=AOTX_SEQ_STATE_PREFILL;
    s.prompt=s.input_count=19;s.input_set=1;aotx_seq_kept[i]=1;
    for(unsigned j=0;j<19;++j) {
        aotx_seqs.tokens[i][j]=(int)aotx_say_id[i*AOTX_SAY_TOKENS+j];
        aotx_seq_input[i][j]=aotx_media_input[i][j];
    }
    ++aotx_media_input[i][2].generation;
    aotx_model_how sample={};
    out[i]=aotx_seq_open(i,AOTX_MODEL_LANGUAGE,aotx_seqs.tokens[i],19,1,1,&sample,100,aotx_media_input[i]);
    --aotx_media_input[i][2].generation;
}
static void apply_records(std::vector<record> &rows,unsigned long long sequence=100) {
    record *device; cu(cudaMalloc(&device,rows.size()*sizeof(record)));
    cu(cudaMemcpy(device,rows.data(),rows.size()*sizeof(record),cudaMemcpyHostToDevice));
    aotx_media_test_apply<<<1,1>>>(device,(unsigned)rows.size(),sequence); cu(cudaDeviceSynchronize());
    cu(cudaMemcpy(rows.data(),device,rows.size()*sizeof(record),cudaMemcpyDeviceToHost)); cudaFree(device);
}
static record begin(unsigned i,unsigned scope,const std::vector<unsigned char> &source) {
    record r={};r.bytes=128;
    aotx_media_put(r.body,1,4);aotx_media_put(r.body+4,1,4);r.body[8]=(unsigned char)(i+1);
    aotx_media_put(r.body+24,source.size(),8);aotx_media_put(r.body+40,i,4);
    aotx_media_put(r.body+44,scope,4);aotx_media_put(r.body+48,AOTX_IMAGE_JPEG,4);
    if(scope==AOTX_MEDIA_PRIVATE || scope==AOTX_MEDIA_ROOM) r.body[64]=(unsigned char)(1+i/2);
    if(scope==AOTX_MEDIA_PRIVATE) r.body[80]=(unsigned char)(i+1);
    aotx_sha256 sha;aotx_sha256_init(&sha);aotx_sha256_update(&sha,source.data(),source.size());
    aotx_sha256_final(&sha,r.body+96);return r;
}
static record chunk(const record &head,const std::vector<unsigned char> &source,unsigned at,unsigned n) {
    record r=head;r.bytes=40+n;aotx_media_put(r.body+4,n?2:3,4);aotx_media_put(r.body+32,at,8);
    if(n) memcpy(r.body+40,source.data()+at,n);return r;
}
static void run(unsigned count,unsigned scope) {
    aotx_media_state state={};state.enabled=1;state.role=AOTX_MODEL_LANGUAGE;
    state.profile.objects=count+1;state.profile.bytes=(count+1)*512u;
    state.profile.feature_rows=(count+1)*6;
    cu(cudaMalloc(&state.objects,state.profile.objects*sizeof(*state.objects)));
    cu(cudaMalloc(&state.hash,state.profile.objects*sizeof(*state.hash)));
    cu(cudaMalloc(&state.source,state.profile.bytes));
    cu(cudaMalloc(&state.features,(size_t)state.profile.feature_rows*4096));
    cu(cudaMemset(state.objects,0,state.profile.objects*sizeof(*state.objects)));
    cu(cudaMemset(state.hash,0,state.profile.objects*sizeof(*state.hash)));
    cu(cudaMemcpyToSymbol(aotx_media,&state,sizeof state));
    aotx_media_initialize<<<1,64>>>();aotx_media_test_bind<<<1,1>>>(count,scope!=AOTX_MEDIA_LOCAL);
    cu(cudaDeviceSynchronize());
    std::vector<std::vector<unsigned char>> sources(count);
    std::vector<record> heads,records;
    for(unsigned i=0;i<count;++i) {
        sources[i].resize(192+i);
        for(unsigned j=0;j<sources[i].size();++j) sources[i][j]=(unsigned char)(i*13+j*7);
        heads.push_back(begin(i,scope,sources[i]));
    }
    records=heads;apply_records(records);
    for(const auto &r:records) check(r.accepted,"distinct source begins are admitted");
    records.clear();
    for(unsigned part=0;part<2;++part) for(unsigned i=count;i--;) {
        unsigned at=part*152u,n=(unsigned)sources[i].size()-at;if(n>152)n=152;
        records.push_back(chunk(heads[i],sources[i],at,n));
    }
    for(unsigned i=0;i<count;++i) records.push_back(chunk(heads[i],sources[i],(unsigned)sources[i].size(),0));
    apply_records(records,1000);
    for(const auto &r:records) check(r.accepted,"interleaved streams preserve their own offsets");
    aotx_media_hash_step<<<(count+63)/64,64>>>(state.hash,count,16);cu(cudaDeviceSynchronize());
    std::vector<aotx_media_object> objects(count);std::vector<aotx_media_hash> hash(count);
    cu(cudaMemcpy(objects.data(),state.objects,count*sizeof(objects[0]),cudaMemcpyDeviceToHost));
    cu(cudaMemcpy(hash.data(),state.hash,count*sizeof(hash[0]),cudaMemcpyDeviceToHost));
    for(unsigned i=0;i<count;++i) {
        std::vector<unsigned char> got(sources[i].size());
        cu(cudaMemcpy(got.data(),state.source+objects[i].offset,got.size(),cudaMemcpyDeviceToHost));
        check(got==sources[i] && objects[i].received==got.size(),"assembled source bytes are exact");
        check(hash[i].done && !hash[i].status && !memcmp(hash[i].digest,heads[i].body+96,32),"source identity covers every byte");
    }
    aotx_media_test_ready<<<1,1>>>(count);
    int *found;cu(cudaMalloc(&found,count*3*sizeof(int)));
    aotx_media_test_find<<<1,64>>>(found,count);cu(cudaDeviceSynchronize());
    std::vector<int> values(count*3);cu(cudaMemcpy(values.data(),found,values.size()*sizeof(int),cudaMemcpyDeviceToHost));
    for(unsigned i=0;i<count;++i) {
        check(values[3*i]==(int)i,"the source owner can resolve its image");
        check(values[3*i+2],"completed media permits a checkpoint");
        bool shared=scope==AOTX_MEDIA_SHARED || (scope==AOTX_MEDIA_ROOM && i/2==((i+1)%count)/2);
        check(values[3*i+1]==(count==1 || shared ? (int)((i+1)%count) : -1),"private, room and shared scopes remain distinct");
    }
    for(unsigned i=0;i<count;++i) {
        unsigned second=scope==AOTX_MEDIA_SHARED ? (i+1)%count : i;
        char digest[65],other[65],prompt[192];aotx_sha256_text(heads[i].body+96,digest);
        aotx_sha256_text(heads[second].body+96,other);
        int n=snprintf(prompt,sizeof prompt,"A [image:%s] B [image:%s] C",digest,other);
        aotx_say_slot say={};say.wanted=1;say.length=(unsigned)n;
        cu(cudaMemcpyToSymbol(aotx_say,&say,sizeof say,offsetof(aotx_say_state,slot)+i*sizeof say));
        cu(cudaMemcpyToSymbol(aotx_say,prompt,(size_t)n,offsetof(aotx_say_state,prompt)+i*AOTX_SAY_BYTES));
    }
    /* Symbol clearing is explicit because every profile case reuses these slots. */
    std::vector<aotx_media_prompt_state> empty(AOTX_SLOTS);
    cu(cudaMemcpyToSymbol(aotx_media_prompts,empty.data(),empty.size()*sizeof(empty[0])));
    aotx_media_prepare<<<1,64>>>();
    unsigned *expanded;cu(cudaMalloc(&expanded,count*sizeof(unsigned)));
    aotx_media_test_expand<<<1,64>>>(expanded,count);cu(cudaDeviceSynchronize());
    std::vector<unsigned> sizes(count);cu(cudaMemcpy(sizes.data(),expanded,count*sizeof(unsigned),cudaMemcpyDeviceToHost));
    for(unsigned i=0;i<count;++i) {
        aotx_model_input inputs[19];
        cu(cudaMemcpyFromSymbol(inputs,aotx_media_input,sizeof inputs,(size_t)i*AOTX_SEQ_MAX_TOKENS*sizeof(aotx_model_input)));
        check(sizes[i]==19,"two image placeholders expand into both full grids");
        const unsigned expected[19][3]={{0,0,0},{1,1,1},{2,2,2},{2,2,3},{2,2,4},{2,3,2},{2,3,3},{2,3,4},
            {5,5,5},{6,6,6},{7,7,7},{8,8,8},{8,8,9},{8,8,10},{8,9,8},{8,9,9},{8,9,10},{11,11,11},{12,12,12}};
        for(unsigned j=0;j<19;++j) check(!memcmp(inputs[j].position,expected[j],sizeof expected[j]),"image axes and following text positions match the independent map");
        for(unsigned j=2;j<8;++j) check(inputs[j].feature==state.features+((size_t)i*6+j-2)*1024 &&
            inputs[j].generation==objects[i].generation,"typed rows bind exact feature addresses and source generations");
        unsigned second=scope==AOTX_MEDIA_SHARED ? (i+1)%count : i;
        for(unsigned j=11;j<17;++j) check(inputs[j].feature==state.features+((size_t)second*6+j-11)*1024 &&
            inputs[j].generation==objects[second].generation,"the second image keeps its own source identity and row order");
    }
    aotx_media_test_generation<<<1,64>>>(expanded,count);cu(cudaDeviceSynchronize());
    cu(cudaMemcpy(sizes.data(),expanded,count*sizeof(unsigned),cudaMemcpyDeviceToHost));
    for(unsigned size:sizes) check(size==1,"reused feature addresses cannot take over a different source generation");
    /* Active sequences hold features; completed sequences permit explicit retirement. */
    for(unsigned i=0;i<count;++i) aotx_media_test_lease<<<1,1>>>(i,1);
    records.clear();for(unsigned i=0;i<count;++i) { record r=heads[i];r.bytes=24;aotx_media_put(r.body+4,4,4);records.push_back(r); }
    apply_records(records,2000);cu(cudaMemcpy(objects.data(),state.objects,count*sizeof(objects[0]),cudaMemcpyDeviceToHost));
    for(const auto &o:objects) check(o.phase==AOTX_MEDIA_READY,"cancel cannot reclaim an active feature lease");
    for(unsigned i=0;i<count;++i) aotx_media_test_lease<<<1,1>>>(i,0);
    apply_records(records,3000);cu(cudaMemcpy(objects.data(),state.objects,count*sizeof(objects[0]),cudaMemcpyDeviceToHost));
    for(const auto &o:objects) check(o.phase==AOTX_MEDIA_REFUSED && o.status==AOTX_MEDIA_CANCELLED,"completed sources can be explicitly retired");
    /* Retired spans are reusable. Old handles cannot cancel their replacements. */
    std::vector<record> old=records;
    records=heads;for(auto &r:records) r.body[9]=1;
    apply_records(records,4000);
    apply_records(old,5000);
    cu(cudaMemcpy(objects.data(),state.objects,count*sizeof(objects[0]),cudaMemcpyDeviceToHost));
    for(unsigned i=0;i<count;++i) check(objects[i].phase==AOTX_MEDIA_RECEIVE &&
        objects[i].generation==4000+i,"new generations survive stale cancellation");
    for(unsigned i=0;i<count;++i) {
        record r=chunk(records[i],sources[i],i%2 ? 0 : 1,i%2 ? 0 : 16);
        std::vector<record> one={r};apply_records(one,6000+i);
        check(one[0].accepted,"a valid request with a bad stream has a recorded terminal result");
    }
    cu(cudaMemcpy(objects.data(),state.objects,count*sizeof(objects[0]),cudaMemcpyDeviceToHost));
    for(const auto &o:objects) check(o.phase==AOTX_MEDIA_REFUSED && o.status==AOTX_MEDIA_INVALID,
        "gaps and premature end release source reservations");
    records=heads;
    for(auto &r:records) {r.body[9]=2;aotx_media_put(r.body+24,state.profile.bytes+1,8);}
    apply_records(records,7000);
    cu(cudaMemcpy(objects.data(),state.objects,count*sizeof(objects[0]),cudaMemcpyDeviceToHost));
    /* A refused slot may be reused by the next request in the same batch. */
    check(objects[0].phase==AOTX_MEDIA_REFUSED && objects[0].status==AOTX_MEDIA_LIMIT,
        "one byte beyond source capacity is refused without allocation");
    for(const auto &r:records) check(r.accepted,"capacity refusal is a valid replayable request");
    for(unsigned i=0;i<count;++i) {
        record r=heads[i];r.body[9]=3;r.bytes=127;
        std::vector<record> one={r};apply_records(one,8000+i);
        check(!one[0].accepted,"truncated canonical headers are rejected");
        r.bytes=128;r.body[60]=1;one={r};apply_records(one,9000+i);
        check(!one[0].accepted,"reserved canonical header bits are rejected");
    }
    cudaFree(expanded);cudaFree(found);cudaFree(state.objects);cudaFree(state.hash);cudaFree(state.source);cudaFree(state.features);
    state={};cu(cudaMemcpyToSymbol(aotx_media,&state,sizeof state));
}
int main(void) {
    for(unsigned scope=0;scope<4;++scope) {run(1,scope);run(64,scope);}
    printf("checks=%u failures=%u\n",checks,failures);return failures?1:0;
}
