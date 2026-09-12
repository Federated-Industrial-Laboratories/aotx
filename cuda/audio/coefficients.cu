/* Purpose: Make the specified resampling and Slaney mel coefficients on CUDA.
 * Owns: Coefficient values in the caller's device spans.
 * Launch shape: Independent filter taps and mel bins across a fixed grid.
 * Lifetime: One immutable coefficient set per runtime. */
#include "audio/audio.cuh"
static __device__ double aotx_audio_i0(double x)
{
    double sum = 1.0, term = 1.0, square = x*x*0.25;
    for (unsigned i=1;i<=64;++i) { term *= square / ((double)i*i); sum += term; }
    return sum;
}
static __device__ double aotx_audio_hz(unsigned i)
{
    double upper = 15.0 + log(8.0) * 27.0 / log(6.4), mel = upper * i / 129.0;
    return mel < 15.0 ? mel * (200.0/3.0) : 1000.0 * exp((mel-15.0) * log(6.4) / 27.0);
}
__global__ void aotx_audio_coefficients_make(aotx_audio_coefficients c)
{
    unsigned tid = blockIdx.x*blockDim.x+threadIdx.x, stride = gridDim.x*blockDim.x;
    for (unsigned at=tid;at<160u*815u+409u;at+=stride) {
        bool low = at >= 160u*815u; unsigned index = low ? at-160u*815u : at;
        unsigned orig = low ? 3u : 441u, phases = low ? 1u : 160u, width = low ? 203u : 187u;
        unsigned taps = 2u*width+orig, phase = index/taps, tap = index%taps;
        double base = phases * 0.9475937167399596;
        double t = (-(double)phase/phases + ((double)tap-width)/orig) * base;
        t = fmin(64.0,fmax(-64.0,t));
        double window = aotx_audio_i0(14.769656459379492 * sqrt(fmax(0.0,1.0-(t/64.0)*(t/64.0)))) /
                        aotx_audio_i0(14.769656459379492);
        double angle = t * 3.14159265358979323846;
        double value = (t == 0.0 ? 1.0 : sin(angle)/angle) * window * base / orig;
        (low ? c.resample48 : c.resample441)[index] = (float)value;
    }
    for (unsigned at=tid;at<128u*201u;at+=stride) {
        unsigned mel=at/201u, bin=at%201u; double hz=bin*40.0;
        double left=aotx_audio_hz(mel), centre=aotx_audio_hz(mel+1u), right=aotx_audio_hz(mel+2u);
        c.mel[at]=(float)(fmax(0.0,fmin((hz-left)/(centre-left),(right-hz)/(right-centre))) * 2.0/(right-left));
    }
}
