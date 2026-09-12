/* Purpose: Validate complete WAV and canonical raw PCM sources on the device.
 * Owns: No source storage; accepted extents remain inside immutable bytes.
 * Launch shape: One source per job thread before batched sample access.
 * Lifetime: One admission step. */
#include "audio/audio.cuh"
#include "media/wire.h"
static __device__ bool aotx_audio_tag(const unsigned char *p, const char *s, unsigned n)
{
    for (unsigned i = 0; i < n; ++i) if (p[i] != (unsigned char)s[i]) return false;
    return true;
}
static __device__ bool aotx_audio_shape(aotx_audio_job &j)
{
    if ((j.encoding != AOTX_AUDIO_S16 && j.encoding != AOTX_AUDIO_F32) ||
        (j.rate != 16000 && j.rate != 44100 && j.rate != 48000) ||
        (j.channels != 1 && j.channels != 2)) return false;
    unsigned stride = j.channels * (j.encoding == AOTX_AUDIO_S16 ? 2u : 4u);
    if (!j.data_bytes || j.data_bytes % stride) return false;
    unsigned long long frames = j.data_bytes / stride;
    if (frames > j.source_capacity || frames > (unsigned long long)j.rate * 30u) {
        j.status = AOTX_AUDIO_LIMIT; return false;
    }
    j.source_frames = (unsigned)frames;
    j.samples = (unsigned)((frames * 16000u + j.rate - 1u) / j.rate);
    j.frames = (j.samples + 159u) / 160u; j.keys = (j.frames + 1u) / 2u; j.rows = j.keys / 2u;
    if (!j.rows || j.rows > j.feature_capacity) { j.status = AOTX_AUDIO_LIMIT; return false; }
    return true;
}
__device__ bool aotx_audio_header(aotx_audio_job &j)
{
    const unsigned char *p = j.source; unsigned long long n = j.source_bytes;
    if (!p) return false;
    if (j.format == AOTX_AUDIO_PCM) {
        if (n < AOTX_AUDIO_PCM_HEAD || !aotx_audio_tag(p, "AOTXPCM1", 8) || aotx_media_get(p+20,4)) return false;
        j.encoding = (unsigned)aotx_media_get(p+8,4); j.rate = (unsigned)aotx_media_get(p+12,4);
        j.channels = (unsigned)aotx_media_get(p+16,4); j.data_offset = AOTX_AUDIO_PCM_HEAD;
        j.data_bytes = n - AOTX_AUDIO_PCM_HEAD;
        return aotx_audio_shape(j) && aotx_media_get(p+24,8) == j.source_frames;
    }
    if (j.format != AOTX_AUDIO_WAV || n < 12 || !aotx_audio_tag(p,"RIFF",4) ||
        !aotx_audio_tag(p+8,"WAVE",4) || aotx_media_get(p+4,4) + 8u != n) return false;
    bool fmt = false, data = false, fact = false; unsigned fact_frames = 0;
    unsigned long long at = 12;
    while (at < n) {
        if (n-at < 8) return false;
        const unsigned char *h = p+at; unsigned long long bytes = aotx_media_get(h+4,4);
        at += 8; if (bytes > n-at || (bytes & 1u) > n-at-bytes) return false;
        const unsigned char *b = p+at;
        if (aotx_audio_tag(h,"fmt ",4)) {
            if (fmt || data || (bytes != 16 && bytes != 18 && bytes != 40)) return false;
            fmt = true; j.encoding = (unsigned)aotx_media_get(b,2);
            j.channels = (unsigned)aotx_media_get(b+2,2); j.rate = (unsigned)aotx_media_get(b+4,4);
            unsigned bits = (unsigned)aotx_media_get(b+14,2);
            if (j.encoding == 65534u) {
                if (bytes != 40 || aotx_media_get(b+16,2) != 22 || aotx_media_get(b+18,2) != bits) return false;
                unsigned mask = (unsigned)aotx_media_get(b+20,4);
                if (mask && mask != (j.channels == 1 ? 4u : 3u)) return false;
                const unsigned char tail[12] = {0,0,16,0,128,0,0,170,0,56,155,113};
                for (unsigned k=0;k<12;++k) if (b[28+k] != tail[k]) return false;
                j.encoding = (unsigned)aotx_media_get(b+24,4);
            } else if (bytes == 40 || (bytes == 18 && aotx_media_get(b+16,2))) return false;
            if ((j.encoding == AOTX_AUDIO_S16 && bits != 16) ||
                (j.encoding == AOTX_AUDIO_F32 && bits != 32)) return false;
            if (j.encoding != AOTX_AUDIO_S16 && j.encoding != AOTX_AUDIO_F32) return false;
            unsigned align = j.channels * (bits / 8u);
            if (aotx_media_get(b+12,2) != align ||
                aotx_media_get(b+8,4) != (unsigned long long)j.rate * align) return false;
        } else if (aotx_audio_tag(h,"data",4)) {
            if (!fmt || data) return false;
            data = true; j.data_offset = at; j.data_bytes = bytes;
        } else if (aotx_audio_tag(h,"fact",4)) {
            if (fact || bytes < 4) return false;
            fact = true; fact_frames = (unsigned)aotx_media_get(b,4);
        } else if (aotx_audio_tag(h,"plst",4) || aotx_audio_tag(h,"slnt",4) ||
            (aotx_audio_tag(h,"LIST",4) && (bytes < 4 || aotx_audio_tag(b,"wavl",4)))) return false;
        at += bytes + (bytes & 1u);
    }
    return fmt && data && aotx_audio_shape(j) && (!fact || fact_frames == j.source_frames);
}
