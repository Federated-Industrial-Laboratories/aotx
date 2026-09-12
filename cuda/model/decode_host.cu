/* Purpose: Capture the decode of one tick and load the module of the memory bound product.
 * Owns: The child graph of the forward pass, the module handle and the module function.
 * Launch shape: Host glue only; the graphs hold the kernels.
 * Lifetime: From the first capture to the close at the end of the run. */
#include <cuda.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "model/decode_state.cuh"
#include "model/graph_host.h"
#include "model/decode_module_host.h"
#include "tool/tool_state.cuh"

static cudaGraph_t aotx_decode_pass[AOTX_MODEL_ROLES];
static unsigned int aotx_decode_role_now=AOTX_MODEL_ROLES;
static void aotx_decode_pass_close(void)
{
    for(unsigned role=0;role<AOTX_MODEL_ROLES;++role){
        if(aotx_decode_pass[role])cudaGraphDestroy(aotx_decode_pass[role]);
        aotx_decode_pass[role]=0;
    }
    aotx_decode_role_now=AOTX_MODEL_ROLES;
}

/* Capture the forward pass of one role as a graph of its own. The graph holds no copy node,
 * because the plan writes the call block on the device. */
static int aotx_decode_build(unsigned int role)
{
    aotx_model_hold *hold = aotx_model_hold_of(role);
    if (hold == 0 || hold->ready == 0u) {
        return 1;
    }
    aotx_decode_module_open();
    if(aotx_decode_pass[role])cudaGraphDestroy(aotx_decode_pass[role]);
    aotx_decode_pass[role]=0;
    hold->decode = 1u;
    hold->wave = AOTX_DECODE_WAVE;
    aotx_check_runtime(cudaStreamBeginCapture(hold->stream,
                                              cudaStreamCaptureModeThreadLocal),
                       "cudaStreamBeginCapture");
    aotx_model_open_rows<<<1, AOTX_SLOTS, 0, hold->stream>>>(role);
    aotx_model_gather<<<AOTX_DECODE_WAVE, AOTX_MODEL_ROW_THREADS, 0, hold->stream>>>(role);
    for (unsigned int l = 0u; l < hold->desc.layers; ++l) {
        aotx_model_capture_layer(hold, role, l);
    }
    aotx_model_capture_head(hold, role);
    aotx_model_shut_rows<<<1, AOTX_SLOTS, 0, hold->stream>>>(role);
    aotx_check_runtime(cudaStreamEndCapture(hold->stream, &aotx_decode_pass[role]),
                       "cudaStreamEndCapture");
    hold->decode = 0u;
    hold->wave = 0u;
    return (aotx_decode_pass[role] == 0) ? 1 : 0;
}

/* The language role of the run. The four bit role stands in when the eight bit role is
 * not loaded. A run of one language file therefore gives the decode its model. The pass of
 * a role that has no graph yet is captured here. */
static unsigned int aotx_decode_language(void)
{
    const unsigned int list[3] = { AOTX_MODEL_LANGUAGE, AOTX_MODEL_LANGUAGE_Q4, AOTX_MODEL_LANGUAGE_AUDIO };
    for (unsigned int i = 0u; i < 3u; ++i) {
        aotx_model_hold *hold = aotx_model_hold_of(list[i]);
        if (hold != 0 && hold->ready != 0u) {
            return list[i];
        }
    }
    for (unsigned int i = 0u; i < 3u; ++i) {
        unsigned int layers = 0u;
        aotx_check_runtime(cudaMemcpyFromSymbol(&layers, aotx_model, sizeof layers,
                                                (size_t)list[i] * sizeof(aotx_model_desc)
                                                + offsetof(aotx_model_desc, layers)),
                           "cudaMemcpyFromSymbol");
        if (layers != 0u && aotx_model_open(list[i], AOTX_MODEL_MAX_TOKENS) == 0) {
            return list[i];
        }
    }
    return AOTX_MODEL_ROLES;
}

int aotx_decode_open(void)
{
    unsigned int ready = 1u;
    unsigned int role = aotx_decode_language();
    if (role >= AOTX_MODEL_ROLES) {
        return 1;
    }
    unsigned mask=1u<<role;
    aotx_model_desc audio;
    aotx_check_runtime(cudaMemcpyFromSymbol(&audio,aotx_model,sizeof audio,
        AOTX_MODEL_LANGUAGE_AUDIO*sizeof audio),"cudaMemcpyFromSymbol");
    if(audio.layers){
        if(!aotx_model_hold_of(AOTX_MODEL_LANGUAGE_AUDIO)->ready &&
            aotx_model_open(AOTX_MODEL_LANGUAGE_AUDIO,AOTX_MODEL_MAX_TOKENS))return 1;
        mask|=1u<<AOTX_MODEL_LANGUAGE_AUDIO;
    }
    for(unsigned i=0;i<AOTX_MODEL_ROLES;++i)if(mask&(1u<<i)){
        aotx_model_batch_of(i);
        if(!aotx_decode_pass[i] && aotx_decode_build(i))return 1;
    }
    aotx_decode_role_now=role;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_decode,&role,sizeof role,
        offsetof(aotx_decode_state,default_role)),"cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_decode,&mask,sizeof mask,
        offsetof(aotx_decode_state,roles)),"cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_decode, &role, sizeof role,
                                          offsetof(aotx_decode_state, role)),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_decode, &ready, sizeof ready,
                                          offsetof(aotx_decode_state, ready)),
                       "cudaMemcpyToSymbol");
    return 0;
}

int aotx_decode_replace(unsigned int role)
{
    if (role >= AOTX_MODEL_ROLES || !aotx_model_is_language(role)) {
        return 1;
    }
    aotx_decode_pass_close();
    unsigned held = 1u << role;
    aotx_model_desc desc[AOTX_MODEL_ROLES];
    aotx_check_runtime(cudaMemcpyFromSymbol(desc, aotx_model, sizeof desc), "cudaMemcpyFromSymbol");
    for (unsigned i = 0; i < AOTX_MODEL_ROLES; ++i) {
        if (aotx_model_hold_of(i)->ready && desc[i].layers) held |= 1u << i;
        aotx_model_shut(i);
    }
    for (unsigned i = 0; i < AOTX_MODEL_ROLES; ++i)
        if ((held & (1u << i)) && aotx_model_open(i, AOTX_MODEL_MAX_TOKENS)) return 1;
    if ((held & (1u << AOTX_MODEL_EMBEDDING)) && aotx_tool_open()) return 1;
    return aotx_decode_open();
}

static int aotx_decode_child(cudaStream_t s,unsigned role)
{
    aotx_decode_select<<<1,1,0,s>>>(role);
    aotx_decode_plan<<<1,AOTX_SLOTS,0,s>>>(0ull);
    cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
    cudaGraph_t graph=0;const cudaGraphNode_t *depends=0;size_t held=0;
    aotx_check_runtime(cudaStreamGetCaptureInfo(s,&status,0,&graph,&depends,0,&held),"cudaStreamGetCaptureInfo");
    if(status!=cudaStreamCaptureStatusActive)return 1;
    cudaGraphNode_t child=0;
    aotx_check_runtime(cudaGraphAddChildGraphNode(&child,graph,depends,held,aotx_decode_pass[role]),"cudaGraphAddChildGraphNode");
    aotx_check_runtime(cudaStreamUpdateCaptureDependencies(s,&child,0,1u,cudaStreamSetCaptureDependencies),"cudaStreamUpdateCaptureDependencies");
    aotx_decode_commit<<<1,AOTX_SLOTS,0,s>>>(0ull);return 0;
}
int aotx_decode_capture(void *stream)
{
    cudaStream_t s=(cudaStream_t)stream;
    if(aotx_decode_role_now>=AOTX_MODEL_ROLES)return 1;
    aotx_decode_begin<<<1,1,0,s>>>();
    for(unsigned role=0;role<AOTX_MODEL_ROLES;++role)
        if(aotx_decode_pass[role] && aotx_decode_child(s,role))return 1;
    aotx_decode_select<<<1,1,0,s>>>(aotx_decode_role_now);return 0;
}
unsigned int aotx_decode_nodes(void)
{
    size_t total=0;
    for(unsigned role=0;role<AOTX_MODEL_ROLES;++role)if(aotx_decode_pass[role]){
        size_t n=0;aotx_check_runtime(cudaGraphGetNodes(aotx_decode_pass[role],0,&n),"cudaGraphGetNodes");total+=n;
    }
    return (unsigned)total;
}
void aotx_decode_close(void)
{
    aotx_decode_pass_close();aotx_decode_module_close();
}
