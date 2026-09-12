/* Purpose: Allocate bounded audio storage and capture resident encoder work.
 * Owns: Allocation handles, immutable coefficients and CUDA FFT plans.
 * Launch shape: Host allocation and graph glue; all audio data work runs on CUDA.
 * Lifetime: Validated component load through runtime close. */
#include "audio/runtime.cuh"
#include "audio/workspace.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static aotx_audio_runtime_state aotx_audio_host;
static aotx_audio_coefficients aotx_audio_coeff;
static unsigned char *aotx_audio_allocation,*aotx_audio_weights,*aotx_audio_tables;
static aotx_audio_desc *aotx_audio_descriptor;
static cufftHandle *aotx_audio_plans;
static float **aotx_audio_fft_input;
static cufftComplex **aotx_audio_spectra;
void aotx_audio_close(void)
{
    if(aotx_audio_plans)for(unsigned i=0;i<aotx_audio_host.profile.workers;++i)
        if(aotx_audio_plans[i])cufftDestroy(aotx_audio_plans[i]);
    free(aotx_audio_plans);free(aotx_audio_fft_input);free(aotx_audio_spectra);
    aotx_audio_plans=0;aotx_audio_fft_input=0;aotx_audio_spectra=0;
    cudaFree(aotx_audio_allocation);cudaFree(aotx_audio_weights);cudaFree(aotx_audio_tables);cudaFree(aotx_audio_descriptor);
    aotx_audio_allocation=aotx_audio_weights=aotx_audio_tables=0;aotx_audio_descriptor=0;
    memset(&aotx_audio_host,0,sizeof aotx_audio_host);memset(&aotx_audio_coeff,0,sizeof aotx_audio_coeff);
    cudaMemcpyToSymbol(aotx_audio_runtime,&aotx_audio_host,sizeof aotx_audio_host);
}
int aotx_audio_allocate(const aotx_audio_profile *p,const aotx_audio_desc *desc,unsigned role,unsigned char **weights)
{
    if(aotx_audio_allocation || !p || !desc || !p->workers || p->workers>65535u)return 1;
    aotx_audio_host.profile=*p;
    aotx_audio_plans=(cufftHandle*)calloc(p->workers,sizeof(cufftHandle));
    aotx_audio_fft_input=(float**)calloc(p->workers,sizeof(float*));
    aotx_audio_spectra=(cufftComplex**)calloc(p->workers,sizeof(cufftComplex*));
    if(!aotx_audio_plans || !aotx_audio_fft_input || !aotx_audio_spectra){aotx_audio_close();return 1;}
    size_t fft_bytes=0;
    for(unsigned i=0;i<p->workers;++i){
        size_t bytes=0;
        if(cufftCreate(aotx_audio_plans+i)!=CUFFT_SUCCESS ||
            cufftSetAutoAllocation(aotx_audio_plans[i],0)!=CUFFT_SUCCESS ||
            cufftMakePlan1d(aotx_audio_plans[i],400,CUFFT_R2C,3000,&bytes)!=CUFFT_SUCCESS){aotx_audio_close();return 1;}
        if(bytes>fft_bytes)fft_bytes=bytes;
    }
    unsigned long long work=aotx_audio_work_bytes(p->source_frames)+aotx_audio_round(fft_bytes);
    unsigned long long jobs=aotx_audio_round((unsigned long long)p->workers*sizeof(aotx_audio_job));
    unsigned long long owners=aotx_audio_round((unsigned long long)p->workers*sizeof(unsigned));
    unsigned long long features=aotx_audio_round((unsigned long long)p->feature_rows*16384u);
    unsigned long long bytes=jobs+owners+features+work*p->workers;
    unsigned long long tables=aotx_audio_round(160u*815u*4u)+aotx_audio_round(409u*4u)+aotx_audio_round(128u*201u*4u);
    unsigned long long total=bytes+desc->bytes+tables+sizeof(*desc);
    size_t free_bytes=0,device_bytes=0;
    if(cudaMemGetInfo(&free_bytes,&device_bytes)!=cudaSuccess || total>free_bytes){
        fprintf(stderr,"audio: storage needs %llu bytes; %zu bytes free\n",total,free_bytes);aotx_audio_close();return 2;
    }
    if(cudaMalloc(&aotx_audio_allocation,bytes)!=cudaSuccess || cudaMalloc(&aotx_audio_weights,desc->bytes)!=cudaSuccess ||
        cudaMalloc(&aotx_audio_tables,tables)!=cudaSuccess || cudaMalloc(&aotx_audio_descriptor,sizeof(*desc))!=cudaSuccess ||
        cudaMemset(aotx_audio_allocation,0,bytes)!=cudaSuccess ||
        cudaMemcpy(aotx_audio_descriptor,desc,sizeof(*desc),cudaMemcpyHostToDevice)!=cudaSuccess){aotx_audio_close();return 1;}
    unsigned char *at=aotx_audio_allocation;
    aotx_audio_runtime_state &s=aotx_audio_host;
    s.jobs=(aotx_audio_job*)aotx_audio_span(at,jobs);s.owner=(unsigned*)aotx_audio_span(at,owners);
    s.features=(float*)aotx_audio_span(at,features);s.workspace=at;s.workspace_each=work;
    s.allocated=total;s.role=role;s.enabled=1;
    at=aotx_audio_tables;aotx_audio_coeff.resample441=(float*)aotx_audio_span(at,160u*815u*4u);
    aotx_audio_coeff.resample48=(float*)aotx_audio_span(at,409u*4u);aotx_audio_coeff.mel=(float*)at;
    for(unsigned i=0;i<p->workers;++i){
        aotx_audio_job j={};unsigned char *base=s.workspace+i*work;aotx_audio_layout(j,base,p->source_frames);
        aotx_audio_fft_input[i]=j.fft_input;aotx_audio_spectra[i]=j.spectrum;
        if(fft_bytes && cufftSetWorkArea(aotx_audio_plans[i],base+aotx_audio_work_bytes(p->source_frames))!=CUFFT_SUCCESS){aotx_audio_close();return 1;}
    }
    if(cudaMemcpyToSymbol(aotx_audio_runtime,&s,sizeof s)!=cudaSuccess){aotx_audio_close();return 1;}
    aotx_audio_coefficients_make<<<128,256>>>(aotx_audio_coeff);
    aotx_audio_initialize<<<(p->workers+63u)/64u,64>>>();
    if(cudaDeviceSynchronize()!=cudaSuccess){aotx_audio_close();return 1;}
    *weights=aotx_audio_weights;
    printf("audio: %u feature rows, %u workspaces, %u source frames each, %llu device bytes\n",
        p->feature_rows,p->workers,p->source_frames,total);return 0;
}
void aotx_audio_runtime_capture(cudaStream_t on)
{
    if(!aotx_audio_host.enabled)return;
    aotx_audio_schedule<<<1,1,0,on>>>();
    aotx_audio_capture(on,aotx_audio_host.jobs,aotx_audio_host.profile.workers,aotx_audio_weights,aotx_audio_descriptor,
        &aotx_audio_coeff,aotx_audio_plans,aotx_audio_fft_input,aotx_audio_spectra,256);
    aotx_audio_complete<<<1,1,0,on>>>();
}
