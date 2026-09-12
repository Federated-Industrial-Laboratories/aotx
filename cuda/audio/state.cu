/* Purpose: Advance finite sound work and refuse invalid or empty signal inputs.
 * Owns: Job progress, cancellation status and completion publication.
 * Launch shape: One thread per independent job.
 * Lifetime: One encoder request. */
#include "audio/audio.cuh"
__global__ void aotx_audio_step(aotx_audio_job *jobs, unsigned count)
{
    unsigned at = blockIdx.x * blockDim.x + threadIdx.x; if (at >= count) return;
    aotx_audio_job &j = jobs[at];
    if (j.phase >= AOTX_AUDIO_READY) return;
    if (j.cancel) j.status = AOTX_AUDIO_CANCELLED;
    if (j.status) { j.phase = AOTX_AUDIO_REFUSED; return; }
    if (j.phase != AOTX_AUDIO_NEW) return;
    j.peak = 0; j.log_peak = -10.0f; j.layer = j.query = 0;
    if (!aotx_audio_header(j)) {
        if (!j.status) j.status = AOTX_AUDIO_INVALID;
        j.phase = AOTX_AUDIO_REFUSED; return;
    }
    j.phase = AOTX_AUDIO_DECODE;
}
__global__ void aotx_audio_finish(aotx_audio_job *jobs, unsigned count, unsigned quantum)
{
    unsigned at = blockIdx.x * blockDim.x + threadIdx.x; if (at >= count) return;
    aotx_audio_job &j = jobs[at]; if (j.phase >= AOTX_AUDIO_READY) return;
    if (j.cancel) j.status = AOTX_AUDIO_CANCELLED;
    if (j.phase == AOTX_AUDIO_RESAMPLE && !j.peak && !j.status) j.status = AOTX_AUDIO_NO_SIGNAL;
    if (j.status) { j.phase = AOTX_AUDIO_REFUSED; return; }
    if (j.phase == AOTX_AUDIO_ATTEND) {
        j.query += quantum;
        if (j.query >= j.keys) j.phase = AOTX_AUDIO_BLOCK;
    } else if (j.phase == AOTX_AUDIO_BLOCK) {
        j.query = 0; ++j.layer;
        j.phase = j.layer == AOTX_AUDIO_LAYERS ? AOTX_AUDIO_POOL : AOTX_AUDIO_PREPARE;
    } else ++j.phase;
}
