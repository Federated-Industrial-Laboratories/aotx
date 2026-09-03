/* Purpose: Apply steer vectors, voice biases and page mass instruments.
 * Owns: The registered conduct tables and the page mass table.
 * Launch shape: One block for each row; one thread flushes all active pages.
 * Lifetime: From model load to model release. */
#include "model/conduct.cuh"

#include "seam/seam.cuh"
#include "text/text.cuh"
#ifdef AOTX_AFFECT
#include "affect/affect.cuh"
#endif

__device__ aotx_conduct_table aotx_conduct;
__device__ float aotx_page_mass[AOTX_SLOTS][AOTX_KV_PAGES_EACH];

__global__ void aotx_conduct_token(const unsigned char *text, unsigned int length,
                                   unsigned int *token)
{
    if (threadIdx.x == 0u && blockIdx.x == 0u) {
        *token = aotx_text_find_token(&aotx_text_vocab_table, text, length);
    }
}

static __device__ __forceinline__ int aotx_conduct_same(const char *a, const char *b,
                                                        unsigned int length)
{
    unsigned int i = 0u;
    while (i < length && i < AOTX_CONDUCT_NAME_BYTES && a[i] == b[i]) {
        i += 1u;
    }
    return i == length && i < AOTX_CONDUCT_NAME_BYTES && b[i] == '\0';
}

__device__ unsigned int aotx_conduct_vector(const char *name, unsigned int length)
{
    for (unsigned int i = 0u; i < aotx_conduct.vectors; ++i) {
        if (aotx_conduct_same(name, aotx_conduct.vector[i].name, length)) {
            return i;
        }
    }
    return AOTX_MODEL_CONDUCT_NONE;
}

__device__ unsigned int aotx_conduct_voice(const char *name, unsigned int length)
{
    for (unsigned int i = 0u; i < aotx_conduct.voices; ++i) {
        if (aotx_conduct_same(name, aotx_conduct.voice[i].name, length)) {
            return i;
        }
    }
    return AOTX_MODEL_CONDUCT_NONE;
}

__device__ float aotx_conduct_bias(unsigned int profile, unsigned int token)
{
    if (profile >= aotx_conduct.voices) {
        return 0.0f;
    }
    const aotx_voice_bias *voice = &aotx_conduct.voice[profile];
    for (unsigned int i = 0u; i < voice->count; ++i) {
        if (voice->token[i] == token) {
            return voice->bias[i];
        }
    }
    return 0.0f;
}

/* The largest bias magnitude of one profile. The pick shifts the plain distribution by it
 * and keeps each exponent of that distribution at or below zero. */
__device__ float aotx_conduct_bias_most(unsigned int profile)
{
    if (profile >= aotx_conduct.voices) {
        return 0.0f;
    }
    const aotx_voice_bias *voice = &aotx_conduct.voice[profile];
    float most = 0.0f;
    for (unsigned int i = 0u; i < voice->count; ++i) {
        most = fmaxf(most, fabsf(voice->bias[i]));
    }
    return most;
}

/* Count selected layers below one layer to find its compact vector row. */
static __device__ __forceinline__ unsigned int aotx_conduct_layer_at(unsigned long long mask,
                                                                    unsigned int layer)
{
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < layer; ++i) {
        at += (unsigned int)((mask >> i) & 1ull);
    }
    return at;
}

__global__ void aotx_model_conduct(unsigned int role, unsigned int layer)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    for (unsigned int row = blockIdx.x; row < run->tokens; row += gridDim.x) {
        unsigned int seq = aotx_model_which(run->offset, run->seqs, row);
        const aotx_model_how *how = (run->how != 0) ? &run->how[seq] : 0;
        for (unsigned int x = threadIdx.x; x < desc->hidden; x += blockDim.x) {
            float add = 0.0f;
            if (how != 0) {
                for (unsigned int i = 0u; i < AOTX_MODEL_STEERS; ++i) {
                    unsigned int id = how->steer[i];
#ifdef AOTX_AFFECT
                    if (i == AOTX_MODEL_CONDUCT_AFFECT) {
                        if (id == AOTX_MODEL_CONDUCT_AFFECT && run->agent[seq] < AOTX_SLOTS
                            && aotx_affect_composite_table.hidden == desc->hidden
                            && ((aotx_affect_composite_table.layers >> layer) & 1ull) != 0ull) {
                            unsigned int at = aotx_conduct_layer_at(
                                aotx_affect_composite_table.layers, layer);
                            unsigned long long cells = (unsigned long long)
                                aotx_affect_composite_table.layer_count * desc->hidden;
                            add += aotx_affect_steer[(unsigned long long)run->agent[seq] * cells
                                                    + (unsigned long long)at * desc->hidden + x];
                        }
                        continue;
                    }
#endif
                    if (id < aotx_conduct.vectors) {
                        const aotx_steer_vector *vector = &aotx_conduct.vector[id];
                        if (vector->hidden == desc->hidden
                            && ((vector->layers >> layer) & 1ull) != 0ull) {
                            unsigned int at = aotx_conduct_layer_at(vector->layers, layer);
                            const float *value = (const float *)vector->value;
                            add += how->steer_strength[i]
                                 * value[(unsigned long long)at * desc->hidden + x];
                        }
                    }
                }
            }
            if (add != 0.0f) {
                work->resid[(unsigned long long)row * desc->hidden + x] += add;
            }
            if (run->capture != 0 && row + 1u == run->offset[seq + 1u]) {
                for (unsigned int c = 0u; c < run->capture_count; ++c) {
                    if (run->capture_layer[c] == layer) {
                        run->capture[((unsigned long long)c * run->seqs + seq)
                                     * desc->hidden + x]
                            = work->resid[(unsigned long long)row * desc->hidden + x];
                    }
                }
            }
        }
#ifdef AOTX_AFFECT
        /* A pass with no how rows, a sequence with no affect mark, or a layer no probe
         * row reads takes no readout. The whole block takes the same branch. */
        if (how != 0 && how->affect != 0u
            && ((aotx_affect_rows.layers >> layer) & 1ull) != 0ull) {
            aotx_affect_readout(run, desc->hidden,
                                work->resid + (unsigned long long)row * desc->hidden,
                                row, seq, layer);
        }
#endif
    }
}

__device__ void aotx_page_flush(unsigned long long tick)
{
    if (tick == 0ull || tick % AOTX_PAGE_FLUSH_TICKS != 0ull) {
        return;
    }
    for (unsigned int agent = 0u; agent < AOTX_SLOTS; ++agent) {
        for (unsigned int page = 0u; page < AOTX_KV_PAGES_EACH; ++page) {
            float mass = atomicExch(&aotx_page_mass[agent][page], 0.0f);
            unsigned int resident = aotx_kv_page(agent, page) != 0ull ? 1u : 0u;
            if (mass == 0.0f && resident == 0u) {
                continue;
            }
            aotx_page_stats_body body;
            body.agent = agent;
            body.page = page;
            body.residency = resident;
            body.cadence = AOTX_PAGE_FLUSH_TICKS;
            body.mass = mass;
            body.slots = AOTX_KV_PAGES_EACH;
            aotx_seam_write(AOTX_WRITER_AGENT_BASE + agent, AOTX_CLASS_B,
                            AOTX_REC_PAGE_STATS, 0u, &body, (unsigned int)sizeof body);
        }
    }
}
