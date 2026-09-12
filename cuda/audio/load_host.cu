/* Purpose: Load the audio component bound to the selected language model.
 * Owns: File-reader handles and a pinned transfer buffer during installation.
 * Launch shape: Host glue only; disk readers validate and supply weight bytes.
 * Lifetime: Runtime startup, before graph capture. */
#include "media/runtime.cuh"
#include "audio/runtime.cuh"
#include "model/roles.h"
#include <stdio.h>
#include <stddef.h>
#include <string.h>
extern "C" {
#include "disk/modelfile/audio.h"
#include "disk/modelfile/media_profile.h"
#include "disk/modelfile/audio_profile.h"
}
int aotx_audio_open(const char *store, const char *roles, int (*stopped)(void))
{
    aotx_manifest_entry pair[2], models[8];
    int count = aotx_audio_manifest(store, pair);
    if (!count) return 0;
    if (count != 2) { fprintf(stderr, "audio: component manifest is refused\n"); return 2; }
    count = aotx_manifest_read(store, models, 8);
    if (count <= 0 || aotx_audio_pair(pair, models, (unsigned)count) < 0) return 2;
    if (!aotx_role_wanted(roles, pair[0].role)) return 0;
    unsigned role = aotx_role_of(pair[0].role);
    aotx_audio_profile profile;
    if (aotx_audio_profile_store(store, &profile) || !aotx_audio_profile_fits(&profile) ||
        aotx_manifest_check(store, pair + 1)) return 2;
    aotx_media_state common;
    if(cudaMemcpyFromSymbol(&common,aotx_media,sizeof common)!=cudaSuccess)return 1;
    if(!common.enabled){
        aotx_media_profile source;unsigned char *unused=0;
        if(aotx_media_profile_store(store,&source) || !aotx_media_profile_fits(&source) ||
            aotx_media_allocate(&source,0,AOTX_MODEL_ROLES,&unused))return 2;
    }
    aotx_modelfile *file = 0;
    if (aotx_modelfile_open_entry(store, pair + 1, &file)) return 2;
    aotx_audio_desc desc;
    int rc = aotx_audio_file(file, &desc);
    unsigned char *weights = 0;
    if (!rc) rc = aotx_audio_allocate(&profile, &desc, role, &weights);
    unsigned char digest[32];
    if (!rc && (aotx_manifest_digest(pair[0].sha256, digest) ||
        cudaMemcpyToSymbol(aotx_audio_runtime, digest, 32, offsetof(aotx_audio_runtime_state, parent_digest)) != cudaSuccess)) rc = 1;
    void *buffer = 0;
    const unsigned long long capacity = 1048576;
    if (!rc && cudaMallocHost(&buffer, capacity) != cudaSuccess) rc = 1;
    for (unsigned long long at = 0; !rc && at < desc.bytes;) {
        if (stopped && stopped()) { rc = 1; break; }
        unsigned long long take = desc.bytes - at;
        if (take > capacity) take = capacity;
        if (aotx_modelfile_read(file, at, take, buffer) ||
            cudaMemcpy(weights + at, buffer, take, cudaMemcpyHostToDevice) != cudaSuccess) rc = 1;
        at += take;
    }
    if (buffer) cudaFreeHost(buffer);
    aotx_modelfile_close(file);
    if (rc) aotx_audio_close();
    return rc;
}
