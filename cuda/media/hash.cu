/* Purpose: Compute source digests in bounded device steps.
 * Owns: Shared message schedules; each job retains its chaining state.
 * Launch shape: One thread per job, 64 threads per block.
 * Lifetime: One immutable byte extent. */
#include "media/hash.cuh"

static __device__ __constant__ unsigned aotx_media_hash_k[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};
__device__ __forceinline__ static unsigned rotate(unsigned x, unsigned n)
{
    return (x >> n) | (x << (32u-n));
}
__global__ void aotx_media_hash_step(aotx_media_hash *jobs, unsigned count, unsigned blocks)
{
    __shared__ unsigned words[16][64];
    unsigned id = blockIdx.x*64u + threadIdx.x;
    if (threadIdx.x >= 64u || id >= count) return;
    aotx_media_hash &j = jobs[id];
    if (!j.active || j.done) return;
    if ((j.bytes && !j.source) || j.bytes > (~0ull)/8u || j.cursor%64u) {
        j.status = 1; j.done = 1; return;
    }
    if (!j.cursor) {
        j.h[0]=0x6a09e667u; j.h[1]=0xbb67ae85u; j.h[2]=0x3c6ef372u; j.h[3]=0xa54ff53au;
        j.h[4]=0x510e527fu; j.h[5]=0x9b05688cu; j.h[6]=0x1f83d9abu; j.h[7]=0x5be0cd19u;
        j.status = 0;
    }
    unsigned long long padded = ((j.bytes+72u)/64u)*64u;
    if (j.cursor >= padded) { j.status = 1; j.done = 1; return; }
    for (unsigned block = 0; block < blocks && j.cursor < padded; ++block) {
        unsigned a=j.h[0], b=j.h[1], c=j.h[2], d=j.h[3], e=j.h[4], f=j.h[5], g=j.h[6], h=j.h[7];
        for (unsigned i=0; i<64; ++i) {
            unsigned word = 0;
            if (i < 16u) {
                for (unsigned byte = 0; byte < 4; ++byte) {
                    unsigned long long at = j.cursor + i*4u + byte;
                    unsigned value = at < j.bytes ? j.source[at] : at == j.bytes ? 128u :
                        at >= padded-8u ? (unsigned)((j.bytes*8u) >> ((padded-1u-at)*8u)) & 255u : 0u;
                    word = (word << 8u) | value;
                }
            } else {
                unsigned x = words[(i-15u)%16u][threadIdx.x], y = words[(i-2u)%16u][threadIdx.x];
                unsigned s0 = rotate(x,7)^rotate(x,18)^(x>>3);
                unsigned s1 = rotate(y,17)^rotate(y,19)^(y>>10);
                word = words[i%16u][threadIdx.x] + s0 + words[(i-7u)%16u][threadIdx.x] + s1;
            }
            words[i%16u][threadIdx.x] = word;
            unsigned s1 = rotate(e,6)^rotate(e,11)^rotate(e,25);
            unsigned t1 = h + s1 + ((e&f)^((~e)&g)) + aotx_media_hash_k[i] + word;
            unsigned s0 = rotate(a,2)^rotate(a,13)^rotate(a,22);
            unsigned t2 = s0 + ((a&b)^(a&c)^(b&c));
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        j.h[0]+=a; j.h[1]+=b; j.h[2]+=c; j.h[3]+=d;
        j.h[4]+=e; j.h[5]+=f; j.h[6]+=g; j.h[7]+=h; j.cursor+=64u;
    }
    if (j.cursor == padded) {
        for (unsigned i=0; i<32; ++i) j.digest[i] = (unsigned char)(j.h[i/4u] >> (24u-8u*(i%4u)));
        j.done = 1;
    }
}
