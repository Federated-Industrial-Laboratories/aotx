/* Purpose: Check vocabulary-sized special token storage and longest whole-token matches.
 * Owns: Distinct large token tables, input batches and independently known token IDs.
 * Launch shape: Vocabulary construction and tokenization at N=1 and N=64.
 * Lifetime: One test process; each vocabulary allocation is released. */
#include "text/text.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
static unsigned checks, failures;
static void check(bool good, const char *name)
{
    ++checks; if (!good) { ++failures; fprintf(stderr, "FAIL %s\n", name); }
}
static void cu(cudaError_t rc)
{
    if (rc != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(rc)); exit(1); }
}
struct memory {
    std::vector<void *> held;
    template<class T> T *take(size_t n) {
        T *p; cu(cudaMallocManaged(&p, n * sizeof(T))); cu(cudaMemset(p, 0, n * sizeof(T)));
        held.push_back(p); return p;
    }
    ~memory() { for (void *p : held) cudaFree(p); }
};
static std::string marker(unsigned i) { return "<|s" + std::to_string(i) + "|>"; }
static void batch(unsigned specials, unsigned count)
{
    memory m;
    auto bytes=m.take<unsigned char>(count*64u);
    auto start=m.take<unsigned>(count), length=m.take<unsigned>(count);
    std::vector<unsigned> expected(count*4u);
    for (unsigned i=0; i<count; ++i) {
        unsigned a=(specials-1u+i*67u)%specials, b=(specials/2u+i*71u)%specials;
        std::string text=marker(a)+"ab<|s"+marker(b);
        start[i]=i*64u; length[i]=(unsigned)text.size();
        for (unsigned j=0; j<text.size(); ++j) bytes[start[i]+j]=(unsigned char)text[j];
        expected[4*i]=4u+a; expected[4*i+1]=2u; expected[4*i+2]=3u; expected[4*i+3]=4u+b;
    }
    aotx_text_batch input={bytes,start,length,count};
    aotx_text_pieces pieces={}; pieces.stride=64;
    pieces.start=m.take<unsigned>(count*64u); pieces.length=m.take<unsigned>(count*64u);
    pieces.token=m.take<unsigned>(count*64u); pieces.count=m.take<unsigned>(count);
    pieces.work=m.take<unsigned>(count*64u); pieces.works=m.take<unsigned>(1);
    aotx_text_tokens tokens={}; tokens.stride=64; tokens.warps=8;
    tokens.id=m.take<unsigned>(count*64u); tokens.count=m.take<unsigned>(count);
    tokens.chunk=m.take<unsigned>(count*64u); tokens.scratch=m.take<unsigned>(count*192u);
    tokens.merge=m.take<unsigned char>(8u*AOTX_TEXT_WARP_BYTES);
    aotx_text_pretok<<<1,64>>>(input,pieces);
    aotx_text_merge<<<4,64>>>(input,pieces,tokens);
    aotx_text_gather<<<1,64>>>(input,pieces,tokens);
    cu(cudaDeviceSynchronize());
    for (unsigned i=0; i<count; ++i) {
        check(tokens.count[i]==4,"complete special and ordinary token count");
        for (unsigned j=0; j<4; ++j) check(tokens.id[i*64+j]==expected[i*4+j],"exact token ID and longest special match");
    }
}
static void vocabulary(unsigned count)
{
    std::string bytes;
    std::vector<unsigned long long> offsets;
    std::vector<int> types;
    auto add=[&](const std::string &text,int type) { offsets.push_back(bytes.size()); bytes+=text; types.push_back(type); };
    add("a",1); add("b",1); add("ab",1); add("<|s",AOTX_TEXT_TYPE_CONTROL);
    for (unsigned i=0; i<count; ++i) add(marker(i),AOTX_TEXT_TYPE_USER);
    offsets.push_back(bytes.size());
    const unsigned long long merge_at[]={0,3};
    aotx_text_source source={};
    source.token_bytes=(const unsigned char *)bytes.data(); source.token_at=offsets.data();
    source.tokens=types.size(); source.token_type=types.data(); source.merges=1;
    source.merge_bytes=(const unsigned char *)"a b"; source.merge_at=merge_at;
    check(!aotx_text_family_find("qwen2",5,&source.family),"known token family");
    aotx_text_store store={}; int rc=aotx_text_vocab_build(&source,&store);
    check(!rc,"large vocabulary builds without a separate special-token limit");
    if (!rc) {
        aotx_text_vocab built; cu(cudaMemcpyFromSymbol(&built,aotx_text_vocab_table,sizeof built));
        check(built.specials==count+1u,"all special tokens are retained");
        check(store.blocks==7u,"special storage belongs to the vocabulary allocation");
        batch(count,1); batch(count,64);
    }
    aotx_text_vocab_release(&store);
}
int main()
{
    vocabulary(1025); vocabulary(4097);
    printf("text special checks=%u failures=%u\n",checks,failures);
    return failures?1:0;
}
