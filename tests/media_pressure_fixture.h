/* Purpose: Supply bounded source, feature and mapped service storage for media checks.
 * Owns: Distinct source identities and the device allocations used by each batch.
 * Launch shape: Ordered source batches and real media nodes at N=1 and N=64.
 * Lifetime: One case; all device and mapped allocations are released together. */
#ifndef AOTX_MEDIA_PRESSURE_FIXTURE_H
#define AOTX_MEDIA_PRESSURE_FIXTURE_H
#include "media/runtime.cuh"
#include "service/internal.cuh"
extern "C" {
#include "disk/modelfile/media_profile.h"
#include "disk/modelfile/audio_profile.h"
}
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static unsigned checks, failures;
static void check(bool good, const char *name)
{ ++checks; if (!good) { ++failures; fprintf(stderr,"FAIL %s\n",name); } }
static void cu(cudaError_t status)
{ if (status != cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(status)); exit(1); } }
template<class T> static void storage(T **p, size_t n)
{ cu(cudaMalloc(p,n*sizeof(T))); cu(cudaMemset(*p,0,n*sizeof(T))); }
template<class T> static std::vector<T> copy(const T *p, size_t n)
{
    std::vector<T> out(n); cu(cudaMemcpy(out.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return out;
}
static void aotx_pressure_sync(void) { cu(cudaGetLastError()); cu(cudaDeviceSynchronize()); }
__host__ __device__ static unsigned source_bytes(unsigned i) { return 64u+i; }
__host__ __device__ static unsigned audio_rows(unsigned i) { return 2u+i%3u; }
__host__ __device__ static void identity(unsigned char *p, unsigned i, unsigned epoch)
{ aotx_media_put(p,i+1,4); aotx_media_put(p+4,epoch,4); }
__host__ __device__ static void begin_body(unsigned char *p, unsigned i, unsigned epoch,
    unsigned long long bytes, bool audio)
{
    for (unsigned j=0;j<AOTX_BODY_BYTES;++j) p[j]=0;
    aotx_media_put(p,AOTX_MEDIA_SCHEMA,4); aotx_media_put(p+4,AOTX_MEDIA_BEGIN,4);
    identity(p+8,i,epoch); aotx_media_put(p+24,bytes,8);
    aotx_media_put(p+40,i,4); aotx_media_put(p+44,AOTX_MEDIA_PRIVATE,4);
    aotx_media_put(p+48,audio?AOTX_AUDIO_WAV:AOTX_IMAGE_JPEG,4);
    identity(p+64,i,7); identity(p+80,i,8);
    for (unsigned j=0;j<32;++j) p[96+j]=(unsigned char)(i*13u+epoch*5u+j);
}
struct aotx_pressure_result {
    aotx_media_object object;
    unsigned valid, found;
    unsigned long long accepted, refused;
};
struct fixture {
    unsigned count, rows=0;
    aotx_media_state m={}; aotx_audio_runtime_state a={};
    aotx_service_state s={}; aotx_seam_state seam={};
    aotx_service_mailbox *mailbox=nullptr;
    aotx_pressure_result *result=nullptr;
    static void profiles(const aotx_media_profile &image, const aotx_audio_profile &audio) {
        unsigned char bytes[AOTX_MEDIA_PROFILE_BYTES]; aotx_media_profile m; aotx_audio_profile a;
        aotx_media_profile_write(&image,bytes);
        check(!aotx_media_profile_read(bytes,sizeof bytes,&m) && aotx_media_profile_fits(&m),
            "image capacities pass the portable reader and fit this build");
        aotx_audio_profile_write(&audio,bytes);
        check(!aotx_audio_profile_read(bytes,AOTX_AUDIO_PROFILE_BYTES,&a) && aotx_audio_profile_fits(&a),
            "audio capacities pass the portable reader and fit this build");
    }
    explicit fixture(unsigned n, bool rpc=false, bool features=false):count(n) {
        for (unsigned i=0;i<n;++i) { m.profile.bytes+=source_bytes(i); rows+=audio_rows(i); }
        if (features) m.profile.bytes=2*n*128u;
        m.enabled=m.image_enabled=1; m.profile.objects=2*n;
        m.profile.feature_rows=64*n; m.profile.patches=256; m.profile.workers=1;
        m.profile.pixels=65536; m.profile.dimension=256; m.profile.horizontal=65536;
        storage(&m.objects,m.profile.objects); storage(&m.hash,m.profile.objects);
        storage(&m.source,m.profile.bytes); storage(&m.image,1); storage(&m.vision,1);
        storage(&m.workspace,(size_t)m.profile.pixels*3);
        storage(&m.owner,1); storage(&m.features,(size_t)m.profile.feature_rows*AOTX_VISION_OUTPUT);
        a.enabled=1; a.profile.workers=1; a.profile.feature_rows=rows; a.profile.source_frames=2560;
        profiles(m.profile,a.profile);
        storage(&a.jobs,1); storage(&a.owner,1); storage(&a.features,(size_t)rows*4096);
        storage(&result,n);
        if (rpc) {
            cu(cudaHostAlloc(&mailbox,AOTX_SERVICE_CHANNELS*sizeof(*mailbox),cudaHostAllocMapped));
            memset(mailbox,0,AOTX_SERVICE_CHANNELS*sizeof(*mailbox));
            cu(cudaHostGetDevicePointer(&s.mailbox,mailbox,0));
            storage(&s.frames,(size_t)AOTX_SERVICE_CHANNELS*AOTX_SERVICE_FRAME);
            storage(&s.ready,AOTX_SERVICE_CHANNELS); storage(&s.grants,n);
            storage(&s.uploads,m.profile.objects); s.media_count=m.profile.objects;
            s.grant_count=n; s.enabled=1; s.epoch=17;
            std::vector<aotx_service_grant> grants(n);
            for (unsigned i=0;i<n;++i) {
                identity(grants[i].principal,i,8); grants[i].revision=1;
                grants[i].actions=AOTX_SERVICE_UPLOAD; grants[i].media=2*n+2;
                grants[i].tokens=64; grants[i].requests=1;
                grants[i].media_bytes=4*m.profile.bytes;
            }
            cu(cudaMemcpy(s.grants,grants.data(),n*sizeof(grants[0]),cudaMemcpyHostToDevice));
        }
        seam.dev.slot_count=2048; seam.dev.mask=2047; seam.apply.state_hash=AOTX_FNV_BASIS;
        seam.replaying=rpc?0:1; storage(&seam.dev.base,2048*AOTX_SLOT_BYTES);
        cu(cudaMemcpyToSymbol(aotx_media,&m,sizeof m));
        cu(cudaMemcpyToSymbol(aotx_audio_runtime,&a,sizeof a));
        cu(cudaMemcpyToSymbol(aotx_service,&s,sizeof s));
        cu(cudaMemcpyToSymbol(aotx_seam,&seam,sizeof seam));
    }
    ~fixture() {
        cudaFree(m.objects); cudaFree(m.hash); cudaFree(m.source); cudaFree(m.image);
        cudaFree(m.vision); cudaFree(m.owner); cudaFree(m.features);
        cudaFree(m.workspace);
        cudaFree(a.jobs); cudaFree(a.owner); cudaFree(a.features); cudaFree(result);
        cudaFree(s.frames); cudaFree(s.ready); cudaFree(s.grants); cudaFree(s.uploads);
        if (mailbox) cudaFreeHost(mailbox);
        cudaFree(seam.dev.base); m={}; a={}; s={}; seam={};
        cu(cudaMemcpyToSymbol(aotx_media,&m,sizeof m));
        cu(cudaMemcpyToSymbol(aotx_audio_runtime,&a,sizeof a));
        cu(cudaMemcpyToSymbol(aotx_service,&s,sizeof s));
        cu(cudaMemcpyToSymbol(aotx_seam,&seam,sizeof seam));
    }
    void send(unsigned epoch, unsigned mode, bool audio) {
        for (unsigned i=0;i<count;++i) {
            auto &box=mailbox[i+1]; unsigned char *f=box.bytes;
            memset(f,0,AOTX_SERVICE_HEAD+112);
            memcpy(f,AOTX_SERVICE_MAGIC,8); aotx_media_put(f+8,AOTX_SERVICE_MEDIA,4);
            identity(f+16,i,8); aotx_media_put(f+32,1,8); identity(f+48,i,epoch);
            aotx_media_put(f+88,112,4); unsigned char *p=f+AOTX_SERVICE_HEAD;
            aotx_media_put(p,AOTX_MEDIA_SCHEMA,4); aotx_media_put(p+4,AOTX_MEDIA_BEGIN,4);
            identity(p+8,i,epoch); aotx_media_put(p+24,mode==1?m.profile.bytes+1:source_bytes(i),8);
            aotx_media_put(p+40,48,4); aotx_media_put(p+64,audio?AOTX_AUDIO_WAV:AOTX_IMAGE_JPEG,4);
            for (unsigned j=0;j<32;++j) p[80+j]=(unsigned char)(i*13u+epoch*5u+j);
            if (mode==2) p[44]=1;
            box.length=AOTX_SERVICE_HEAD+112; box.state=1;
        }
        aotx_service_copy<<<AOTX_SERVICE_CHANNELS,64>>>(); aotx_service_admit<<<1,1>>>(); aotx_pressure_sync();
    }
};
#endif
