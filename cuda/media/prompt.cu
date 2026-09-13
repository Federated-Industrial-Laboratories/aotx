/* Purpose: Replace media source links with ordered trained feature rows.
 * Owns: Per-slot references, token scratch and model-specific rotary coordinates.
 * Launch shape: One thread per slot; immutable feature matrices are shared by readers.
 * Lifetime: One prompt and its installed sequence. */
#include "media/prompt.cuh"
#include "media/runtime.cuh"
#include "agent/agent_state.cuh"
#include "agent/transcript.cuh"
#include "cli/prompt.cuh"
#include "cognitive/intake.cuh"
#include "model/decode.cuh"
#include "service/service.cuh"
#include "shared/bridge.cuh"

__device__ aotx_media_prompt_state aotx_media_prompts[AOTX_SLOTS];
__device__ aotx_model_input aotx_media_input[AOTX_SLOTS][AOTX_SEQ_MAX_TOKENS];
static __device__ unsigned aotx_media_ids[AOTX_SLOTS][AOTX_SEQ_MAX_TOKENS];
static __device__ unsigned char aotx_media_raw[AOTX_SLOTS][AOTX_SAY_BYTES];
static __device__ bool aotx_media_word(const unsigned char *p, unsigned n, const char *word)
{
    unsigned i = 0;
    for (; word[i]; ++i) if (i >= n || p[i] != (unsigned char)word[i]) return false;
    return true;
}
static __device__ int aotx_media_hex(unsigned char c)
{
    return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
}
__device__ bool aotx_media_reserved(const unsigned char *p, unsigned n)
{
    return aotx_media_word(p, n, "<|vision_") || aotx_media_word(p, n, "<|image_pad|>") ||
        aotx_media_word(p, n, "<|video_pad|>") || aotx_media_word(p, n, "<|audio_") ||
        aotx_media_word(p, n, "<|AUDIO|>");
}
__global__ void aotx_media_prepare(void)
{
    unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= AOTX_SLOTS || !aotx_say.slot[slot].wanted) return;
    aotx_media_prompt_state &m = aotx_media_prompts[slot];
    if (m.stage == 1 || m.stage == 2) return;
    aotx_say_slot &s = aotx_say.slot[slot];
    unsigned char *p = aotx_say.prompt[slot];
    bool shared = aotx_shared_owns(slot);
    m.count = m.extra = m.turn_extra = 0;
    bool bad = s.length > AOTX_SAY_BYTES, pending = false;
    for (unsigned at = 0; !bad && at < s.length; ++at) {
        if (aotx_media_reserved(p + at, s.length - at)) { bad = true; break; }
        /* Recalled shared links are text. Only the current input supplies feature references. */
        if (shared && at < s.turn_at) continue;
        bool audio = aotx_media_word(p + at, s.length - at, "[audio:");
        if (!audio && !aotx_media_word(p + at, s.length - at, "[image:")) continue;
        if (s.length - at < 72u || p[at + 71u] != ']' || m.count >= AOTX_MEDIA_REFS) {
            bad = true; break;
        }
        for (unsigned j = 0; j < 32; ++j) {
            int a = aotx_media_hex(p[at + 7u + 2u*j]), b = aotx_media_hex(p[at + 8u + 2u*j]);
            if (a < 0 || b < 0) bad = true;
            m.digest[j] = a < 0 || b < 0 ? 0 : (unsigned char)((a << 4) | b);
        }
        unsigned role = aotx_prompt_role(slot);
        bool enabled = audio ? aotx_audio_runtime.enabled && aotx_audio_runtime.role == role
            : aotx_media.image_enabled && aotx_media.role == role;
        if (bad || !aotx_media.enabled || !enabled) { bad = true; break; }
        int index = aotx_media_find(m.digest, slot);
        if (index == -2) pending = true;
        else if (index < 0) bad = true;
        else {
            if (aotx_media_is_audio(aotx_media.objects[index].format) != audio ||
                !aotx_media.objects[index].rows) { bad = true; break; }
            m.reference[m.count].object = (unsigned)index;
            m.reference[m.count].generation = aotx_media.objects[index].generation;
            m.extra += aotx_media.objects[index].rows - 1u;
            if (at >= s.turn_at) m.turn_extra += aotx_media.objects[index].rows - 1u;
        }
        ++m.count; at += 71u;
    }
    if (bad) { m.stage = 2; return; }
    if (pending) { m.stage = 3; return; }
    m.raw_length = s.length; m.raw_turn = s.turn_at;
    m.raw_system = (aotx_service_owns(slot) || aotx_shared_owns(slot)) ? 0 : aotx_agent_gear[slot].system_bytes;
    if (m.count) for (unsigned i = 0; i < s.length; ++i) aotx_media_raw[slot][i] = p[i];
    /* The native marker span is shorter than the canonical source link. */
    unsigned out = 0, turn_at = s.turn_at, audio_number = 0;
    const char *span = "<|vision_start|><|image_pad|><|vision_end|>";
    for (unsigned at = 0; at < s.length;) {
        if (at == turn_at) s.turn_at = out;
        bool audio = aotx_media_word(p + at, s.length - at, "[audio:");
        if ((!shared || at >= turn_at) && (audio || aotx_media_word(p + at, s.length - at, "[image:"))) {
            if (turn_at > at && turn_at < at + 72u) s.turn_at = out;
            if (audio) {
                const char *head = "Audio ";
                for (unsigned j = 0; head[j]; ++j) p[out++] = (unsigned char)head[j];
                unsigned number = ++audio_number, divisor = 1;
                while (divisor <= number / 10u) divisor *= 10u;
                do {
                    p[out++] = (unsigned char)('0' + number / divisor);
                    number %= divisor; divisor /= 10u;
                } while (divisor);
                p[out++] = ':'; p[out++] = ' ';
            }
            const char *marker = audio ? "<|audio_bos|><|AUDIO|><|audio_eos|>\n" : span;
            for (unsigned j = 0; marker[j]; ++j) p[out++] = (unsigned char)marker[j];
            at += 72u;
        } else p[out++] = p[at++];
    }
    s.length = out; m.stage = 1;
}
__device__ unsigned aotx_media_expand(unsigned slot, unsigned count)
{
    aotx_media_prompt_state &m = aotx_media_prompts[slot];
    if (m.stage != 1 || count > AOTX_SAY_TOKENS || count > AOTX_SEQ_MAX_TOKENS ||
        m.extra > AOTX_SEQ_MAX_TOKENS - count) return 0;
    if (!m.count) return count;
    unsigned *ids = aotx_say_id + slot * AOTX_SAY_TOKENS;
    unsigned *out = aotx_media_ids[slot];
    aotx_model_input *input = aotx_media_input[slot];
    bool audio = aotx_prompt_role(slot) == AOTX_MODEL_LANGUAGE_AUDIO;
    unsigned start = audio ? AOTX_MEDIA_AUDIO_START : AOTX_MEDIA_VISION_START;
    unsigned pad = audio ? AOTX_MEDIA_AUDIO_PAD : AOTX_MEDIA_IMAGE_PAD;
    unsigned end = audio ? AOTX_MEDIA_AUDIO_END : AOTX_MEDIA_VISION_END;
    unsigned made = 0, ref = 0, position = 0;
    for (unsigned at = 0; at < count;) {
        if (ids[at] == pad || ids[at] == end) return 0;
        if (ids[at] != start) {
            out[made] = ids[at++]; input[made++] = {0, 0, {position, position, position}};
            ++position; continue;
        }
        if (ref >= m.count || at + 2u >= count || ids[at+1] != pad ||
            ids[at+2] != end) return 0;
        const aotx_media_reference &r = m.reference[ref++];
        if (r.object >= aotx_media.profile.objects) return 0;
        const aotx_media_object &o = aotx_media.objects[r.object];
        if (o.phase != AOTX_MEDIA_READY || o.generation != r.generation ||
            aotx_media_is_audio(o.format) != audio || !o.rows || !o.columns ||
            o.rows != o.columns * o.lines || o.rows > AOTX_SEQ_MAX_TOKENS - made - 2u) return 0;
        out[made] = start;
        input[made++] = {0, 0, {position, position, position}}; ++position;
        for (unsigned row = 0; row < o.rows; ++row) {
            out[made] = pad;
            unsigned width = audio ? 4096u : 1024u;
            float *features = audio ? aotx_audio_runtime.features : aotx_media.features;
            unsigned x = audio ? position + row : position;
            unsigned y = audio ? x : position + row / o.columns;
            unsigned z = audio ? x : position + row % o.columns;
            input[made++] = {features + ((unsigned long long)o.feature + row) * width,
                width, {x, y, z}, o.generation};
        }
        position += audio ? o.rows : max(o.columns, o.lines);
        out[made] = end;
        input[made++] = {0, 0, {position, position, position}}; ++position;
        at += 3u;
    }
    if (ref != m.count || made != count + m.extra || made > AOTX_SAY_TOKENS) return 0;
    for (unsigned j = 0; j < made; ++j) ids[j] = out[j];
    return made;
}
__device__ bool aotx_media_leased(unsigned object)
{
    const aotx_media_object &o = aotx_media.objects[object];
    if (aotx_service_media_leased(object, o.generation) || aotx_shared_media_leased(object, o.generation)) return true;
    bool audio = aotx_media_is_audio(o.format);
    unsigned width = audio ? 4096u : 1024u;
    const float *features = audio ? aotx_audio_runtime.features : aotx_media.features;
    unsigned long long first = (unsigned long long)(features + (unsigned long long)o.feature * width);
    unsigned long long end = first + (unsigned long long)o.span * width * sizeof(float);
    for (unsigned slot = 0; slot < AOTX_SLOTS; ++slot) {
        const aotx_media_prompt_state &m = aotx_media_prompts[slot];
        if ((aotx_say.slot[slot].wanted || aotx_intake_owns(slot)) && m.stage == 1)
            for (unsigned j = 0; j < m.count; ++j)
                if (m.reference[j].object == object && m.reference[j].generation == o.generation) return true;
        const aotx_seq &s = aotx_seqs.slot[slot];
        if ((s.state != AOTX_SEQ_STATE_PREFILL && s.state != AOTX_SEQ_STATE_DECODE) || !s.input_count) continue;
        for (unsigned j = 0; j < s.input_count; ++j) {
            const float *p = aotx_seq_input[slot][j].feature;
            unsigned long long address = (unsigned long long)p;
            if (p && address >= first && address < end) return true;
        }
    }
    return false;
}
/* A new plain turn can retire older hot turns after the exact expanded count is known.
 * The current turn and system text remain intact. Required cognitive context is not trimmed. */
__device__ bool aotx_media_retry(unsigned slot)
{
    aotx_media_prompt_state &m = aotx_media_prompts[slot];
    aotx_agent_work &g = aotx_agent_gear[slot];
    if (aotx_service_owns(slot) || aotx_live_bound(slot) || aotx_intake_owns(slot) || !g.wrote || g.result ||
        (g.kind != AOTX_AGENT_TURN_MESSAGE && g.kind != AOTX_AGENT_TURN_TASK) ||
        m.raw_system > m.raw_turn || m.raw_turn > m.raw_length || m.raw_length > AOTX_SAY_BYTES ||
        !aotx_transcript_give_hot(slot)) return false;
    unsigned char *out = aotx_say.prompt[slot];
    const unsigned char *raw = aotx_media_raw[slot];
    for (unsigned i = 0; i < m.raw_system; ++i) out[i] = raw[i];
    unsigned at = aotx_transcript_prompt(slot, out, m.raw_system);
    unsigned suffix = m.raw_length - m.raw_turn;
    if (at > AOTX_SAY_BYTES || suffix > AOTX_SAY_BYTES - at) return false;
    aotx_say_slot &s = aotx_say.slot[slot];
    s.turn_at = at;
    for (unsigned i = 0; i < suffix; ++i) out[at+i] = raw[m.raw_turn+i];
    s.length = at + suffix; s.ready = 0; s.prompt = 0;
    s.token_deadline = aotx_time_tick + AOTX_SAY_TOKEN_WAIT_TICKS;
    aotx_say_count[slot] = 0; m.stage = 0;
    g.prompt_len = s.length;
    g.input_hash = aotx_seam_fnv1a(AOTX_FNV_BASIS, out, s.length);
    return true;
}
