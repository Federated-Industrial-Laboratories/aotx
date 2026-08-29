/* Purpose: Keep ordered turns and build the hot, warm and summary memory blocks.
 * Owns: The transcript index, its text arena, its vectors and selection counters.
 * Launch shape: Device functions; the agent and tool steps call them by slot.
 * Lifetime: The whole run. */
#include "agent/agent_state.cuh"
#include "agent/transcript.cuh"
#include "bus/bus.cuh"
#include "cli/prompt.cuh"
#include "kvcache/kvcache.cuh"
#include "settings/settings.cuh"
#include "tool/tool_state.cuh"

__device__ aotx_transcript_agent aotx_transcript[AOTX_SLOTS];
__device__ unsigned char
    aotx_transcript_text[AOTX_SLOTS][AOTX_TRANSCRIPT_TEXT_BYTES];
__device__ float
    aotx_transcript_vector[AOTX_SLOTS][AOTX_MEMORY_TURNS][AOTX_TRANSCRIPT_VECTOR];
__device__ aotx_transcript_counts aotx_transcript_count;
__device__ unsigned long long aotx_transcript_source_seq;

static __device__ __forceinline__ unsigned int aotx_transcript_at(
    const aotx_transcript_agent *hold, unsigned int n)
{
    return (hold->first + n) % AOTX_MEMORY_TURNS;
}

static __device__ __forceinline__ unsigned int aotx_memory_put(
    unsigned char *out, unsigned int at, const char *text)
{
    for (unsigned int i = 0u; text[i] != '\0' && at < AOTX_SAY_BYTES; ++i) {
        out[at++] = (unsigned char)text[i];
    }
    return at;
}

static __device__ __forceinline__ unsigned int aotx_memory_run(
    unsigned char *out, unsigned int at, const unsigned char *text, unsigned int length)
{
    for (unsigned int i = 0u; i < length && at < AOTX_SAY_BYTES; ++i) {
        out[at++] = text[i];
    }
    return at;
}

static __device__ __forceinline__ unsigned int aotx_memory_number(
    unsigned char *out, unsigned int at, unsigned int value)
{
    char digit[16];
    unsigned int count = aotx_text_utoa((unsigned long long)value, digit,
                                        (unsigned int)sizeof digit);
    return aotx_memory_run(out, at, (const unsigned char *)digit, count);
}

__device__ void aotx_transcript_source(unsigned long long seq)
{
    aotx_transcript_source_seq = seq;
}

__device__ int aotx_transcript_pages(unsigned int agent, unsigned int pages)
{
    if (agent >= AOTX_SLOTS || (pages != AOTX_TRANSCRIPT_AUTO
        && (pages == 0u || pages > AOTX_KV_PAGES_EACH))) {
        return 1;
    }
    aotx_transcript[agent].pages = pages;
    return 0;
}

__device__ int aotx_transcript_compact(unsigned int agent)
{
    if (agent >= AOTX_SLOTS || aotx_agents.agent[agent].state == AOTX_AGENT_STATE_FREE) {
        return 1;
    }
    aotx_transcript[agent].compact = 1u;
    return 0;
}

__device__ unsigned int aotx_transcript_page_limit(unsigned int agent)
{
    if (agent >= AOTX_SLOTS) {
        return 0u;
    }
    aotx_transcript_agent *hold = &aotx_transcript[agent];
    unsigned int role = aotx_agents.agent[agent].role;
    unsigned int own = hold->pages;
    if (own == 0u && role < AOTX_MODULE_SLOTS) {
        own = aotx_catalog.entry[role].role.pages;
    }
    if (own == 0u) {
        own = aotx_setting_count(AOTX_SET_AGENT_PAGES);
    }
    if (own == 0u) {
        own = AOTX_KV_PAGES_EACH;
    }
    if (own != AOTX_TRANSCRIPT_AUTO) {
        return (own > AOTX_KV_PAGES_EACH) ? AOTX_KV_PAGES_EACH : own;
    }
    unsigned int least = (role < AOTX_MODULE_SLOTS)
                       ? aotx_catalog.entry[role].role.pages_least : 16u;
    if (least == 0u) {
        least = 16u;
    }
    unsigned int live = (aotx_agents.live == 0u) ? 1u : aotx_agents.live;
    unsigned int free = (aotx_kv.mapped_pages < AOTX_KV_PAGES)
                      ? (AOTX_KV_PAGES - aotx_kv.mapped_pages) : 0u;
    /* Pages already held by this slot can serve its next open. Share the available pool
     * among the live agents, so one automatic agent grows and a full table gives way. */
    unsigned int give = (free + aotx_kv.count[agent]) / live;
    if (give < least) {
        give = least;
    }
    return (give > AOTX_KV_PAGES_EACH) ? AOTX_KV_PAGES_EACH : give;
}

static __device__ int aotx_transcript_queue(unsigned int agent, unsigned int kind,
                                             unsigned int turn,
                                             const unsigned char *text,
                                             unsigned int length)
{
    if (aotx_tool_embed.ready == 0u) {
        return 0;
    }
    if (aotx_tool_embed.state[agent] != AOTX_TOOL_EMBED_NONE) {
        return 1;
    }
    unsigned int bytes = (length > AOTX_TOOL_TEXT_BYTES) ? AOTX_TOOL_TEXT_BYTES : length;
    unsigned char *to = aotx_tool_gear.text
                      + (unsigned long long)agent * AOTX_TOOL_TEXT_BYTES;
    for (unsigned int i = 0u; i < bytes; ++i) {
        to[i] = text[i];
    }
    aotx_tool_gear.bytes[agent] = bytes;
    aotx_tool_embed.asked[agent] = 0u;
    aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_WAIT;
    aotx_transcript[agent].embed_kind = kind;
    aotx_transcript[agent].embed_turn = turn;
    return 1;
}

static __device__ int aotx_transcript_find_seq(const aotx_transcript_agent *hold,
                                                unsigned long long seq)
{
    for (unsigned int i = 0u; i < hold->count; ++i) {
        unsigned int at = aotx_transcript_at(hold, i);
        if (hold->turn[at].seq == seq) {
            return (int)at;
        }
    }
    return -1;
}

static __device__ int aotx_transcript_tiers(unsigned int agent, int queue)
{
    aotx_transcript_agent *hold = &aotx_transcript[agent];
    unsigned int pages = 0u;
    unsigned int hot = 0u;
    for (unsigned int n = hold->count; n > 0u; --n) {
        unsigned int at = aotx_transcript_at(hold, n - 1u);
        aotx_transcript_turn *turn = &hold->turn[at];
        unsigned int add = (turn->pages == 0u) ? 1u : turn->pages;
        if (pages + add <= hold->limit) {
            pages += add;
            turn->tier = AOTX_MEMORY_HOT;
            hot += 1u;
        } else if (turn->tier == AOTX_MEMORY_HOT || turn->tier == 0u) {
            turn->tier = AOTX_MEMORY_WARM;
        }
    }
    hold->hot = hot;
    hold->warm = 0u;
    hold->folded = 0u;
    unsigned int embed = AOTX_MEMORY_TURNS;
    for (unsigned int i = 0u; i < hold->count; ++i) {
        unsigned int at = aotx_transcript_at(hold, i);
        aotx_transcript_turn *turn = &hold->turn[at];
        if (turn->tier == AOTX_MEMORY_WARM || turn->tier == AOTX_MEMORY_FOLDED) {
            hold->warm += (turn->tier == AOTX_MEMORY_WARM) ? 1u : 0u;
            hold->folded += (turn->tier == AOTX_MEMORY_FOLDED) ? 1u : 0u;
            if (turn->vector_ready == 0u && queue != 0
                && embed == AOTX_MEMORY_TURNS) {
                embed = at;
            }
        }
    }
    if (embed != AOTX_MEMORY_TURNS) {
        const aotx_transcript_turn *turn = &hold->turn[embed];
        const unsigned char *text = aotx_transcript_text[agent] + turn->text_at;
        return aotx_transcript_queue(agent, AOTX_MEMORY_EMBED_TURN, embed, text,
                                     turn->text_len) ? 0 : 1;
    }
    return 1;
}

__device__ int aotx_transcript_prepare(unsigned int agent, const unsigned char *text,
                                       unsigned int length, unsigned int turn)
{
    if (agent >= AOTX_SLOTS) {
        return 0;
    }
    aotx_transcript_agent *hold = &aotx_transcript[agent];
    hold->limit = aotx_transcript_page_limit(agent);
    int may_wait = (aotx_agents.agent[agent].state == AOTX_AGENT_STATE_IDLE
                    && aotx_seam.replaying == 0ull) ? 1 : 0;
    if (!aotx_transcript_tiers(agent, may_wait)) {
        return 0;
    }
    if (aotx_seam.replaying != 0ull) {
        hold->selected_count = 0u;
        if (hold->replay_ready == 0u) {
            /* A journal from before selection records can still open its first turn. It
             * has no memory choice to reproduce, so the empty choice is deterministic. */
            if (hold->count == 0u && hold->summary_len == 0u) {
                hold->choice.agent = agent;
                hold->choice.turn = turn;
                hold->choice.count = 0u;
                hold->choice.pages = hold->limit;
                hold->choice.summary_seq = 0ull;
                return 1;
            }
            return 0;
        }
        hold->limit = hold->replay_choice.pages;
        for (unsigned int i = 0u; i < hold->replay_choice.count
             && i < AOTX_SELECTION_MAX; ++i) {
            int at = aotx_transcript_find_seq(hold, hold->replay_choice.seq[i]);
            if (at >= 0) {
                hold->selected[hold->selected_count++] = (unsigned int)at;
            }
        }
        hold->choice = hold->replay_choice;
        hold->replay_ready = 0u;
        return 1;
    }
    if (aotx_agent_gear[agent].kind == AOTX_AGENT_TURN_COMPACT) {
        hold->selected_count = 0u;
        unsigned int take = hold->warm / 2u;
        for (unsigned int i = 0u; i < hold->count && take != 0u
             && hold->selected_count < AOTX_SELECTION_MAX; ++i) {
            unsigned int which = aotx_transcript_at(hold, i);
            if (hold->turn[which].tier == AOTX_MEMORY_WARM) {
                hold->selected[hold->selected_count++] = which;
                take -= 1u;
            }
        }
        aotx_selection_body *choice = &hold->choice;
        choice->agent = agent;
        choice->turn = turn;
        choice->pages = hold->limit;
        choice->summary_seq = hold->summary_seq;
        choice->count = hold->selected_count;
        for (unsigned int i = 0u; i < AOTX_SELECTION_MAX; ++i) {
            choice->seq[i] = (i < hold->selected_count)
                           ? hold->turn[hold->selected[i]].seq : 0ull;
        }
        hold->choice_pending = (aotx_seam.replaying == 0ull) ? 1u : 0u;
        return 1;
    }
    unsigned int recall = aotx_setting_count(AOTX_SET_RECALL_K);
    if (recall > AOTX_SELECTION_MAX) {
        recall = AOTX_SELECTION_MAX;
    }
    if (hold->query_ready == 0u) {
        hold->selected_count = 0u;
    }
    if (may_wait != 0 && hold->warm != 0u && recall != 0u
        && hold->query_ready == 0u) {
        return aotx_transcript_queue(agent, AOTX_MEMORY_EMBED_QUERY, turn, text, length)
             ? 0 : 1;
    }
    if (hold->query_ready != 0u) {
        hold->query_ready = 0u;
        for (unsigned int i = 1u; i < hold->selected_count; ++i) {
            unsigned int value = hold->selected[i];
            unsigned int at = i;
            while (at > 0u && hold->turn[hold->selected[at - 1u]].number
                              > hold->turn[value].number) {
                hold->selected[at] = hold->selected[at - 1u];
                at -= 1u;
            }
            hold->selected[at] = value;
        }
    }
    aotx_selection_body *choice = &hold->choice;
    choice->agent = agent;
    choice->turn = turn;
    choice->pages = hold->limit;
    choice->summary_seq = hold->summary_seq;
    choice->count = hold->selected_count;
    for (unsigned int i = 0u; i < AOTX_SELECTION_MAX; ++i) {
        choice->seq[i] = (i < hold->selected_count)
                       ? hold->turn[hold->selected[i]].seq : 0ull;
    }
    hold->choice_pending = 1u;
    return 1;
}

static __device__ unsigned int aotx_transcript_one(unsigned int agent,
                                                    unsigned int which,
                                                    unsigned char *out,
                                                    unsigned int at, int mark)
{
    const aotx_transcript_turn *turn = &aotx_transcript[agent].turn[which];
    at = aotx_memory_put(out, at, "<|im_start|>user\n");
    if (mark != 0) {
        at = aotx_memory_put(out, at, "[memory turn ");
        at = aotx_memory_number(out, at, turn->number);
        at = aotx_memory_put(out, at, "]\n");
    }
    at = aotx_memory_run(out, at, aotx_transcript_text[agent] + turn->text_at,
                         turn->text_len);
    at = aotx_memory_put(out, at, "<|im_end|>\n<|im_start|>assistant\n");
    at = aotx_memory_run(out, at, aotx_transcript_text[agent] + turn->reply_at,
                         turn->reply_len);
    return aotx_memory_put(out, at, "<|im_end|>\n");
}

__device__ unsigned int aotx_transcript_prompt(unsigned int agent, unsigned char *out,
                                               unsigned int at)
{
    if (agent >= AOTX_SLOTS) {
        return at;
    }
    aotx_transcript_agent *hold = &aotx_transcript[agent];
    if (aotx_agent_gear[agent].kind == AOTX_AGENT_TURN_COMPACT) {
        for (unsigned int i = 0u; i < hold->selected_count; ++i) {
            at = aotx_transcript_one(agent, hold->selected[i], out, at, 1);
        }
        return at;
    }
    if (hold->summary_len != 0u) {
        at = aotx_memory_put(out, at, "<|im_start|>user\n[memory summary]\n");
        at = aotx_memory_run(out, at, hold->summary, hold->summary_len);
        at = aotx_memory_put(out, at, "<|im_end|>\n");
    }
    for (unsigned int i = 0u; i < hold->selected_count; ++i) {
        at = aotx_transcript_one(agent, hold->selected[i], out, at, 1);
    }
    for (unsigned int i = 0u; i < hold->count; ++i) {
        unsigned int which = aotx_transcript_at(hold, i);
        if (hold->turn[which].tier == AOTX_MEMORY_HOT) {
            at = aotx_transcript_one(agent, which, out, at, 0);
        }
    }
    return at;
}

__device__ void aotx_transcript_finish(unsigned int agent, const unsigned char *text,
                                       unsigned int text_len, const unsigned char *reply,
                                       unsigned int reply_len, unsigned int pages,
                                       unsigned long long manifest_seq)
{
    if (agent >= AOTX_SLOTS || text == 0) {
        return;
    }
    aotx_transcript_agent *hold = &aotx_transcript[agent];
    if (hold->count >= AOTX_MEMORY_TURNS
        || hold->text_used + text_len + reply_len > AOTX_TRANSCRIPT_TEXT_BYTES) {
        atomicAdd(&aotx_transcript_count.text_refused, 1ull);
        return;
    }
    unsigned int which = aotx_transcript_at(hold, hold->count);
    aotx_transcript_turn *turn = &hold->turn[which];
    turn->seq = aotx_agent_gear[agent].source_seq;
    turn->manifest_seq = manifest_seq;
    turn->reply_seq = aotx_say.slot[agent].reply_first;
    turn->reply_records = aotx_say.slot[agent].reply_records;
    turn->token_first = aotx_say.slot[agent].prompt;
    turn->token_count = aotx_agent_gear[agent].out_tokens;
    turn->call_seq = aotx_requests.slot[agent].call_seq;
    turn->result_seq = 0ull;
    turn->answer_seq = 0ull;
    turn->number = aotx_agents.agent[agent].turn;
    turn->text_at = hold->text_used;
    turn->text_len = text_len;
    for (unsigned int i = 0u; i < text_len; ++i) {
        aotx_transcript_text[agent][hold->text_used + i] = text[i];
    }
    hold->text_used += text_len;
    turn->reply_at = hold->text_used;
    turn->reply_len = reply_len;
    for (unsigned int i = 0u; i < reply_len; ++i) {
        aotx_transcript_text[agent][hold->text_used + i] = reply[i];
    }
    hold->text_used += reply_len;
    turn->pages = (pages == 0u) ? 1u : pages;
    turn->tier = AOTX_MEMORY_HOT;
    turn->vector_ready = 0u;
    hold->count += 1u;
}

__device__ void aotx_transcript_result(unsigned int agent,
                                       unsigned long long result_seq,
                                       unsigned long long answer_seq)
{
    if (agent >= AOTX_SLOTS || aotx_transcript[agent].count == 0u) {
        return;
    }
    aotx_transcript_agent *hold = &aotx_transcript[agent];
    unsigned int which = aotx_transcript_at(hold, hold->count - 1u);
    hold->turn[which].result_seq = result_seq;
    hold->turn[which].answer_seq = answer_seq;
}

__device__ void aotx_transcript_embed_done(unsigned int agent, const float *vector,
                                           unsigned int width)
{
    if (agent >= AOTX_SLOTS || vector == 0 || width == 0u) {
        return;
    }
    if (width > AOTX_TRANSCRIPT_VECTOR) {
        width = AOTX_TRANSCRIPT_VECTOR;
    }
    aotx_transcript_agent *hold = &aotx_transcript[agent];
    if (hold->embed_kind == AOTX_MEMORY_EMBED_TURN) {
        unsigned int which = hold->embed_turn;
        for (unsigned int d = 0u; d < width; ++d) {
            aotx_transcript_vector[agent][which][d] = vector[d];
        }
        hold->turn[which].vector_ready = 1u;
        atomicAdd(&aotx_transcript_count.embedded, 1ull);
    } else if (hold->embed_kind == AOTX_MEMORY_EMBED_QUERY) {
        unsigned int recall = aotx_setting_count(AOTX_SET_RECALL_K);
        if (recall > AOTX_SELECTION_MAX) {
            recall = AOTX_SELECTION_MAX;
        }
        hold->selected_count = 0u;
        for (unsigned int i = 0u; i < hold->count; ++i) {
            unsigned int which = aotx_transcript_at(hold, i);
            const aotx_transcript_turn *turn = &hold->turn[which];
            if (turn->tier == AOTX_MEMORY_HOT || turn->vector_ready == 0u) {
                continue;
            }
            float score = 0.0f;
            for (unsigned int d = 0u; d < width; ++d) {
                score += vector[d] * aotx_transcript_vector[agent][which][d];
            }
            unsigned int at = 0u;
            while (at < hold->selected_count && score <= hold->selected_score[at]) {
                at += 1u;
            }
            if (at < recall) {
                unsigned int end = (hold->selected_count < recall)
                                 ? hold->selected_count : recall - 1u;
                while (end > at) {
                    hold->selected[end] = hold->selected[end - 1u];
                    hold->selected_score[end] = hold->selected_score[end - 1u];
                    end -= 1u;
                }
                hold->selected[at] = which;
                hold->selected_score[at] = score;
                if (hold->selected_count < recall) {
                    hold->selected_count += 1u;
                }
            }
        }
        hold->query_ready = 1u;
        atomicAdd(&aotx_transcript_count.searches, 1ull);
    }
    hold->embed_kind = AOTX_MEMORY_EMBED_NONE;
}

__device__ void aotx_transcript_selection_apply(const aotx_selection_body *body)
{
    if (body == 0 || body->agent >= AOTX_SLOTS || body->count > AOTX_SELECTION_MAX) {
        return;
    }
    aotx_transcript_agent *hold = &aotx_transcript[body->agent];
    hold->replay_choice = *body;
    hold->replay_ready = 1u;
    atomicAdd(&aotx_transcript_count.replay_selections, 1ull);
}

__device__ void aotx_transcript_commit(unsigned long long tick)
{
    (void)tick;
    for (unsigned int agent = 0u; agent < AOTX_SLOTS; ++agent) {
        aotx_transcript_agent *hold = &aotx_transcript[agent];
        if (hold->choice_pending == 0u || aotx_seam.replaying != 0ull) {
            hold->choice_pending = 0u;
            continue;
        }
        unsigned long long seq = aotx_seam_write(AOTX_WRITER_AGENT_BASE + agent,
                                                  AOTX_CLASS_A, AOTX_REC_SELECTION, 0u,
                                                  &hold->choice,
                                                  (unsigned int)sizeof hold->choice);
        aotx_seam.apply.state_hash = aotx_seam_fnv1a(
            aotx_seam.apply.state_hash, aotx_seam_body_of(seq),
            (unsigned int)sizeof hold->choice);
        aotx_seam.apply.applied_count += 1ull;
        hold->choice_pending = 0u;
    }
}

__device__ unsigned int aotx_transcript_needs_compact(unsigned int agent)
{
    if (agent >= AOTX_SLOTS) {
        return 0u;
    }
    aotx_transcript_agent *hold = &aotx_transcript[agent];
    unsigned int at = aotx_setting_count(AOTX_SET_COMPACT_AT);
    return (hold->compact != 0u || (hold->warm > at && at != 0u)) ? 1u : 0u;
}

__device__ int aotx_transcript_maintain(unsigned int agent)
{
    if (agent >= AOTX_SLOTS) {
        return 0;
    }
    aotx_transcript[agent].limit = aotx_transcript_page_limit(agent);
    return aotx_transcript_tiers(agent, 1);
}

__device__ void aotx_transcript_summary(unsigned int agent, const unsigned char *text,
                                        unsigned int length, unsigned long long tick)
{
    if (agent >= AOTX_SLOTS || text == 0 || length == 0u) {
        return;
    }
    aotx_transcript_agent *hold = &aotx_transcript[agent];
    unsigned int bytes = (length > AOTX_TRANSCRIPT_SUMMARY)
                       ? AOTX_TRANSCRIPT_SUMMARY : length;
    for (unsigned int i = 0u; i < bytes; ++i) {
        hold->summary[i] = text[i];
    }
    hold->summary_len = bytes;
    unsigned long long first = 0ull;
    unsigned int folded = 0u;
    for (unsigned int i = 0u; i < hold->selected_count; ++i) {
        unsigned int which = hold->selected[i];
        if (which < AOTX_MEMORY_TURNS
            && hold->turn[which].tier == AOTX_MEMORY_WARM) {
            if (first == 0ull) {
                first = hold->turn[which].seq;
            }
            hold->turn[which].tier = AOTX_MEMORY_FOLDED;
            folded += 1u;
        }
    }
    hold->summary_seq = aotx_bus_append(AOTX_WRITER_AGENT_BASE + agent,
                                         AOTX_BUS_FINDING, AOTX_PROV_COMPUTED,
                                         (const char *)hold->summary, hold->summary_len,
                                         first, hold->summary_seq, 0.0f, tick);
    hold->compact = 0u;
    hold->warm = (hold->warm > folded) ? (hold->warm - folded) : 0u;
    hold->folded += folded;
    atomicAdd(&aotx_transcript_count.compacted, 1ull);
}
