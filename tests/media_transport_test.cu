/* Purpose: Check real file commands, producer backpressure and device admission.
 * Owns: New files, distinct slot bindings and small mapped/device test rings.
 * Launch shape: One producer thread and ordered CUDA frame batches at N=1 and N=64.
 * Lifetime: Each case closes its files, thread, maps and device allocations. */
#include "media/runtime.cuh"
#include "cognitive/live.cuh"
#include "cli/cli.cuh"
extern "C" {
#include "disk/feed/media_io.h"
}
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
static unsigned checks,failures;
static void check(bool good,const char *name) {
    ++checks;if(!good){++failures;fprintf(stderr,"FAIL %s\n",name);}
}
static void cu(cudaError_t rc) {
    if(rc!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(rc));exit(1);}
}
__global__ void aotx_media_transport_bind(unsigned count,unsigned changed) {
    for(unsigned i=0;i<AOTX_SLOTS;++i) {
        aotx_live_bindings[i]={};
        if(i<count) {
            auto &b=aotx_live_bindings[i];b.active=1;b.room[0]=1;
            b.principal[0]=(unsigned char)(i+1+changed);
        }
    }
}
struct fixture {
    aotx_seam_rings rings={};aotx_media_state state={};aotx_seam_state seam={};
    aotx_media_producer producer={};aotx_inbound_ring control={};aotx_inbound_preamble pre={};
    explicit fixture(unsigned count) {
        check(!aotx_media_ring_open(&rings),"mapped device producer ring opens");
        check(!aotx_media_producer_open(rings.media_fd,&producer),"disk producer maps the same ring");
        cu(cudaMemcpyFromSymbol(&state,aotx_media,sizeof state));
        state.enabled=1;state.profile.objects=count+1;state.profile.bytes=count*2048u;
        cu(cudaMalloc(&state.objects,state.profile.objects*sizeof(*state.objects)));
        cu(cudaMalloc(&state.hash,state.profile.objects*sizeof(*state.hash)));
        cu(cudaMalloc(&state.source,state.profile.bytes));
        cu(cudaMemset(state.objects,0,state.profile.objects*sizeof(*state.objects)));
        cu(cudaMemset(state.hash,0,state.profile.objects*sizeof(*state.hash)));
        cu(cudaMemcpyToSymbol(aotx_media,&state,sizeof state));
        seam.dev.slot_count=8192;seam.dev.mask=8191;seam.apply.state_hash=AOTX_FNV_BASIS;
        cu(cudaMalloc(&seam.dev.base,8192*AOTX_SLOT_BYTES));cu(cudaMemset(seam.dev.base,0,8192*AOTX_SLOT_BYTES));
        cu(cudaMemcpyToSymbol(aotx_seam,&seam,sizeof seam));
        aotx_media_initialize<<<1,64>>>();aotx_media_transport_bind<<<1,1>>>(count,0);cu(cudaDeviceSynchronize());
        control.pre=&pre;
    }
    ~fixture() {
        aotx_media_producer_close(&producer);aotx_media_ring_close(&rings);
        cudaFree(state.objects);cudaFree(state.hash);cudaFree(state.source);cudaFree(seam.dev.base);
        state={};cu(cudaMemcpyToSymbol(aotx_media,&state,sizeof state));
        seam={};cu(cudaMemcpyToSymbol(aotx_seam,&seam,sizeof seam));
    }
    int command(const std::string &line) {
        std::atomic<bool> done(false);int rc=0;volatile sig_atomic_t stop=0;
        std::thread feeder([&](){rc=aotx_media_feed_line(&producer,(const unsigned char *)line.data(),
            (unsigned)line.size(),&control,&stop);done.store(true);});
        unsigned ticks=0;
        while(!done.load() && ticks++<200000) {
            aotx_media_ingest<<<1,1>>>();cu(cudaDeviceSynchronize());std::this_thread::yield();
        }
        if(!done.load()) stop=1;
        feeder.join();check(ticks<200000,"producer and device acknowledgments terminate");return rc;
    }
    std::vector<aotx_media_object> objects() {
        std::vector<aotx_media_object> out(state.profile.objects);
        cu(cudaMemcpy(out.data(),state.objects,out.size()*sizeof(out[0]),cudaMemcpyDeviceToHost));return out;
    }
    unsigned long long applied() {
        aotx_seam_state s;cu(cudaMemcpyFromSymbol(&s,aotx_seam,sizeof s));return s.apply.applied_count;
    }
};
static void run(unsigned count) {
    fixture f(count);char directory[]="/tmp/aotx-media-XXXXXX";check(mkdtemp(directory)!=nullptr,"temporary input directory opens");
    std::vector<std::vector<unsigned char>> sources(count);
    for(unsigned i=0;i<count;++i) {
        auto &data=sources[i];data.resize(1000+i);
        for(unsigned j=0;j<data.size();++j)data[j]=(unsigned char)(i*13+j*7);
        std::string path=std::string(directory)+"/image "+std::to_string(i);
        FILE *out=fopen(path.c_str(),"wb");if(!out)exit(1);
        check(fwrite(data.data(),1,data.size(),out)==data.size(),"distinct source file is written");fclose(out);
        check(f.command("image load "+std::to_string(i)+" private jpeg "+path)==1,"file command is handled");
        check(!f.producer.pre->status,"device accepts each complete transfer");unlink(path.c_str());
    }
    auto objects=f.objects();
    for(unsigned i=0;i<count;++i) {
        const auto &o=objects[i];std::vector<unsigned char> data(o.received);
        cu(cudaMemcpy(data.data(),f.state.source+o.offset,data.size(),cudaMemcpyDeviceToHost));
        aotx_sha256 sha;unsigned char digest[32];aotx_sha256_init(&sha);
        aotx_sha256_update(&sha,sources[i].data(),sources[i].size());aotx_sha256_final(&sha,digest);
        check(o.phase==AOTX_MEDIA_HASH && data==sources[i] && !memcmp(digest,o.digest,32),
            "mapped frames preserve exact source bytes, offsets and digest");
        check(o.slot==i && o.scope==AOTX_MEDIA_PRIVATE && o.principal[0]==i+1,
            "CUDA captures the current private owner binding");
    }
    auto cancel=[&](unsigned slot,unsigned source) {
        char id[33];for(unsigned j=0;j<16;++j)snprintf(id+j*2,3,"%02x",objects[source].transfer[j]);
        return f.command("image cancel "+std::to_string(slot)+" "+id);
    };
    auto before=f.applied();
    for(unsigned i=0;i<count;++i) {
        check(cancel((i+1)%AOTX_SLOTS,i)==1 && f.producer.pre->status==AOTX_MEDIA_UNAVAILABLE,
            "another slot cannot cancel the source");
    }
    check(f.applied()==before,"out-of-scope commands do not enter canonical state");
    aotx_media_transport_bind<<<1,1>>>(count,1);cu(cudaDeviceSynchronize());
    for(unsigned i=0;i<count;++i) check(cancel(i,i)==1 && f.producer.pre->status==AOTX_MEDIA_UNAVAILABLE,
        "a replacement private binding cannot cancel the previous user source");
    aotx_media_transport_bind<<<1,1>>>(count,0);cu(cudaDeviceSynchronize());
    for(unsigned i=0;i<count;++i) check(cancel(i,i)==1 && !f.producer.pre->status,"source owner cancellation is acknowledged");
    for(const auto &o:f.objects()) if(o.phase) check(o.phase==AOTX_MEDIA_REFUSED && o.status==AOTX_MEDIA_CANCELLED,
        "cancellation is terminal and releases the source reservation");
    std::string path=std::string(directory)+"/large";FILE *out=fopen(path.c_str(),"wb");if(!out)exit(1);
    check(!ftruncate(fileno(out),(off_t)f.state.profile.bytes+1),"capacity boundary file is made");fclose(out);
    before=f.applied();check(f.command("image load 0 private jpeg "+path)==1,"capacity refusal is handled");
    check(f.producer.pre->status==AOTX_MEDIA_LIMIT && f.applied()-before<=2,
        "begin refusal stops source bytes before the journal grows");
    unlink(path.c_str());rmdir(directory);
}
int main(void) {
    run(1);run(64);printf("checks=%u failures=%u\n",checks,failures);return failures?1:0;
}
