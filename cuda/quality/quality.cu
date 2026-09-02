/* Purpose: Measure the conversation quality figures of completed turns.
 * Owns: The quality state, the phrase table and the quality counts.
 * Launch shape: One block for each agent; one thread writes the record.
 * Lifetime: The whole run. */
#include "quality/quality.cuh"

#include "affect/affect.cuh"
#include "agent/agent_state.cuh"
#include "model/decode.cuh"
#include "seam/seam.cuh"
#include "tool/tool_state.cuh"

__device__ aotx_quality_slot aotx_quality_state[AOTX_SLOTS];
__device__ aotx_quality_phrase_table aotx_quality_phrases;
__device__ aotx_quality_counts aotx_quality_count;

__device__ void aotx_quality_open(unsigned int agent, aotx_model_how *how)
{
    if (agent >= AOTX_SLOTS || how == 0) return;
    aotx_quality_slot *state = &aotx_quality_state[agent];
    how->quality = (aotx_setting_count(AOTX_SET_QUALITY_ON) != 0u) ? 1u : 0u;
    state->active = how->quality;
    state->ended = 0u;
    state->last_count = 0u;
    state->total = 0u;
    state->distinct = 0u;
    state->active_guard[0] = 0.0f;
    state->active_guard[1] = 0.0f;
    state->active_guard_loaded = 0u;
    for (unsigned int i = 0u; i < AOTX_QUALITY_SET_WORDS; ++i) state->trigram[i] = 0u;
}

__device__ void aotx_quality_pick(unsigned int agent, const aotx_model_how *how,
                                  unsigned int token)
{
    if (agent >= AOTX_SLOTS || how == 0 || how->quality == 0u) return;
    aotx_quality_slot *state = &aotx_quality_state[agent];
    if (state->last_count == 2u) {
        unsigned int hash = 2166136261u;
        hash = (hash ^ state->last[0]) * 16777619u;
        hash = (hash ^ state->last[1]) * 16777619u;
        hash = (hash ^ token) * 16777619u;
        unsigned int bit = hash & 8191u;
        unsigned int mask = 1u << (bit & 31u);
        unsigned int *word = &state->trigram[bit >> 5];
        if ((*word & mask) == 0u) {
            *word |= mask;
            state->distinct += 1u;
        }
        state->total += 1u;
    } else {
        state->last_count += 1u;
    }
    state->last[0] = state->last[1];
    state->last[1] = token;
}

__device__ void aotx_quality_end(unsigned int agent)
{
    if (agent < AOTX_SLOTS && aotx_quality_state[agent].active != 0u) {
        aotx_quality_state[agent].ended = 1u;
    }
}

__device__ void aotx_quality_guard(unsigned int agent, float first, float second,
                                   unsigned int loaded)
{
    if (agent >= AOTX_SLOTS || aotx_quality_state[agent].active == 0u) return;
    aotx_quality_slot *state = &aotx_quality_state[agent];
    if (loaded != 0u) {
        state->active_guard[0] = isfinite(first) ? first : 0.0f;
        state->active_guard[1] = isfinite(second) ? second : 0.0f;
        state->active_guard_loaded = 1u;
    }
}

__device__ void aotx_quality_fill(unsigned int agent)
{
    if (agent >= AOTX_SLOTS) return;
    aotx_quality_slot *state = &aotx_quality_state[agent];
    unsigned int base = AOTX_SLOTS + 2u * agent;
    unsigned int text_at = AOTX_SLOTS * AOTX_TOOL_TEXT_BYTES + 2u * agent
                         * AOTX_QUALITY_BYTES;
    if (threadIdx.x == 0u) {
        aotx_tool_gear.start[base] = text_at;
        aotx_tool_gear.start[base + 1u] = text_at + AOTX_QUALITY_BYTES;
        aotx_tool_gear.length[base] = (state->row[0] == AOTX_QUALITY_ROW_WAIT)
                                   ? state->length[0] : 0u;
        aotx_tool_gear.length[base + 1u] = (state->row[1] == AOTX_QUALITY_ROW_WAIT)
                                        ? state->length[1] : 0u;
        if (aotx_tool_gear.length[base] != 0u || aotx_tool_gear.length[base + 1u] != 0u) {
            atomicAdd(&aotx_quality_count.filled, 1u);
        }
    }
    for (unsigned int i = threadIdx.x; i < AOTX_QUALITY_BYTES; i += blockDim.x) {
        if (i < state->length[0]) aotx_tool_gear.text[text_at + i] = state->text[0][i];
        if (i < state->length[1]) {
            aotx_tool_gear.text[text_at + AOTX_QUALITY_BYTES + i] = state->text[1][i];
        }
    }
}

static __device__ __forceinline__ unsigned char aotx_quality_fold(unsigned char c)
{
    return (c >= 'A' && c <= 'Z') ? (unsigned char)(c + ('a' - 'A')) : c;
}

static __device__ void aotx_quality_refusal(unsigned int agent, unsigned int *hit)
{
    if (threadIdx.x == 0u) *hit = 0u;
    __syncthreads();
    unsigned int phrase = threadIdx.x;
    if (phrase < aotx_quality_phrases.count && phrase < AOTX_QUALITY_PHRASES) {
        const aotx_agent_work *gear = &aotx_agent_gear[agent];
        unsigned int span = min(gear->reply_len, AOTX_QUALITY_BYTES);
        unsigned int length = aotx_quality_phrases.length[phrase];
        for (unsigned int at = 0u; length != 0u && at + length <= span; ++at) {
            unsigned int same = 1u;
            for (unsigned int i = 0u; i < length; ++i) {
                if (aotx_quality_fold(gear->reply[at + i])
                    != aotx_quality_fold(aotx_quality_phrases.text[phrase][i])) {
                    same = 0u;
                    break;
                }
            }
            if (same != 0u) atomicExch(hit, 1u);
        }
    }
    __syncthreads();
}

static __device__ void aotx_quality_write(unsigned int agent, aotx_quality_slot *state)
{
    aotx_quality_body body = {};
    body.agent = agent;
    body.turn = state->turn;
    body.coherence_prompt = ((state->flags & 1u) != 0u) ? state->coherence_prompt : 0.0f;
    body.coherence_turn = ((state->flags & 2u) != 0u) ? state->coherence_turn : 0.0f;
    body.repetition = (state->held_total != 0u)
                    ? 1.0f - (float)state->held_distinct / (float)state->held_total : 0.0f;
    body.repetition = fminf(1.0f, fmaxf(0.0f, body.repetition));
    body.tokens = state->tokens;
    body.limit = state->limit;
    body.refusal = state->refusal;
    body.guard[0] = state->guard[0];
    body.guard[1] = state->guard[1];
    body.flags = state->flags & 0x0fu;
    aotx_seam_write(AOTX_WRITER_AGENT_BASE + agent, AOTX_CLASS_B, AOTX_REC_QUALITY, 0u,
                    &body, (unsigned int)sizeof body);
    state->pending = 0u;
    state->row[0] = AOTX_QUALITY_ROW_NONE;
    state->row[1] = AOTX_QUALITY_ROW_NONE;
    atomicAdd(&aotx_quality_count.written, 1u);
}

static __device__ void aotx_quality_land(unsigned int agent, aotx_quality_slot *state)
{
    unsigned int width = min(aotx_tool_embed.width, AOTX_QUALITY_WIDTH);
    for (unsigned int row = 0u; row < 2u; ++row) {
        unsigned int place = state->place[row];
        if (state->row[row] != AOTX_QUALITY_ROW_RUN || place >= aotx_tool_embed.seqs
            || aotx_tool_embed.kind[place] != 2u * agent + row + 1u) continue;
        const float *vector = aotx_tool_embed.vector + (unsigned long long)place
                            * aotx_tool_embed.width;
        if (row == 0u) {
            for (unsigned int i = 0u; i < width; ++i) state->prompt[i] = vector[i];
            state->prompt_valid = (width != 0u) ? 1u : 0u;
        } else if (width != 0u) {
            float prompt = 0.0f, turn = 0.0f;
            for (unsigned int i = 0u; i < width; ++i) {
                prompt += vector[i] * state->prompt[i];
                turn += vector[i] * state->previous[i];
                state->previous[i] = vector[i];
            }
            if (state->prompt_valid != 0u) {
                state->coherence_prompt = fminf(1.0f, fmaxf(-1.0f, prompt));
                state->flags |= 1u;
            }
            if (state->previous_valid != 0u) {
                state->coherence_turn = fminf(1.0f, fmaxf(-1.0f, turn));
                state->flags |= 2u;
            }
            state->previous_valid = 1u;
        }
        state->row[row] = AOTX_QUALITY_ROW_DONE;
        state->asked[row] = 0u;
    }
}

__global__ void aotx_quality_turn(void)
{
    __shared__ unsigned int refused;
    unsigned int agent = blockIdx.x;
    if (agent >= AOTX_SLOTS) return;
    aotx_quality_slot *state = &aotx_quality_state[agent];
    aotx_quality_refusal(agent, &refused);
    if (threadIdx.x != 0u) return;
    if (aotx_seam.replaying != 0ull) {
        state->ended = 0u;
        state->active = 0u;
        state->pending = 0u;
        state->previous_valid = 0u;
        state->prompt_valid = 0u;
        state->row[0] = AOTX_QUALITY_ROW_NONE;
        state->row[1] = AOTX_QUALITY_ROW_NONE;
        return;
    }
    aotx_quality_land(agent, state);
    if (state->pending != 0u && state->row[0] == AOTX_QUALITY_ROW_DONE
        && state->row[1] == AOTX_QUALITY_ROW_DONE) aotx_quality_write(agent, state);
    if (state->ended == 0u) return;
    if (state->pending != 0u) {
        state->flags &= ~3u;
        state->coherence_prompt = 0.0f;
        state->coherence_turn = 0.0f;
        state->previous_valid = 0u;
        aotx_quality_write(agent, state);
        atomicAdd(&aotx_quality_count.late, 1u);
    }
    const aotx_agent_work *gear = &aotx_agent_gear[agent];
    state->turn = aotx_agents.agent[agent].turn;
    state->tokens = gear->out_tokens;
    state->limit = aotx_seqs.slot[agent].limit;
    state->refusal = refused;
    state->held_total = state->total;
    state->held_distinct = state->distinct;
    state->guard[0] = state->active_guard[0];
    state->guard[1] = state->active_guard[1];
    state->flags = (gear->limit_end != 0u ? 4u : 0u)
                 | (state->active_guard_loaded != 0u ? 8u : 0u);
    state->coherence_prompt = 0.0f;
    state->coherence_turn = 0.0f;
    state->length[0] = min(gear->message_len, AOTX_QUALITY_BYTES);
    state->length[1] = min(gear->reply_len, AOTX_QUALITY_BYTES);
    for (unsigned int i = 0u; i < state->length[0]; ++i) state->text[0][i] = gear->message[i];
    for (unsigned int i = 0u; i < state->length[1]; ++i) state->text[1][i] = gear->reply[i];
    state->pending = 1u;
    state->prompt_valid = 0u;
    state->ended = 0u;
    state->active = 0u;
    if (aotx_tool_embed.ready == 0u) {
        state->previous_valid = 0u;
        aotx_quality_write(agent, state);
        return;
    }
    for (unsigned int row = 0u; row < 2u; ++row) {
        state->row[row] = AOTX_QUALITY_ROW_WAIT;
        state->place[row] = AOTX_TOOL_BATCH_ROWS;
        state->asked[row] = 0u;
    }
}
