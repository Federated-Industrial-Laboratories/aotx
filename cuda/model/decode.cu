/* Purpose: Hold the sequence table and open, stop, replay and read a sequence.
 * Owns: The sequence table, the journal mark and the take mark of each slot.
 * Launch shape: Device functions; one call for each slot in one tick. The say path calls
 *   the open from one thread for each slot at once, so the counts of the table move by
 *   atomic add. Two calls for one slot in one tick are a defect of the caller.
 * Lifetime: The whole run. */
#include "model/decode_state.cuh"
#include "model/sampler.cuh"
#include "settings/settings.cuh"
#include "seam/seam.cuh"
#include "text/text.cuh"

__device__ aotx_seq_table aotx_seqs;
__device__ aotx_decode_state aotx_decode;
__device__ unsigned int aotx_seq_kept[AOTX_SLOTS];
__device__ unsigned int aotx_seq_shown[AOTX_SLOTS];
__device__ unsigned int aotx_seq_asked[AOTX_SLOTS];

/* Write one sequence event. The event is derived and a restore does not replay it. */
static __device__ __forceinline__ void aotx_seq_event(unsigned int slot, unsigned int event,
                                                      unsigned long long tick)
{
    const aotx_seq *seq = &aotx_seqs.slot[slot];
    aotx_sequence_body body;
    body.slot = slot;
    body.event = event;
    body.prompt_tokens = seq->prompt;
    body.sampled_tokens = seq->sampled;
    body.ticks = (tick > seq->opened) ? (tick - seq->opened) : 0ull;
    body.role = seq->role;
    body.reserved = 0u;
    aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_B, AOTX_REC_SEQUENCE, 0u,
                    &body, (unsigned int)sizeof body);
}

/* Put a slot in the state a new sequence starts from. The page count of the slot does not
 * change, because a slot that keeps its pages needs no new map. */
static __device__ __forceinline__ void aotx_seq_clear(unsigned int slot, unsigned int role,
                                                      const aotx_model_how *sample,
                                                      unsigned long long tick)
{
    aotx_seq *seq = &aotx_seqs.slot[slot];
    seq->role = role;
    seq->prompt = 0u;
    seq->held = 0u;
    seq->sampled = 0u;
    seq->limit = aotx_setting_count(AOTX_SET_REPLY_LIMIT);
    seq->page_limit = AOTX_KV_PAGES_EACH;
    seq->stop = AOTX_DECODE_STOP_END;
    seq->sample = *sample;
    seq->seed = sample->seed;
    seq->thinking = 0u;
    seq->think_tokens = 0u;
    seq->draw = 0ull;
    seq->opened = tick;
    seq->last = 0u;
    seq->flags = 0u;
    aotx_model_seen[slot] = 0u;
    aotx_model_draw[slot] = 0u;
    aotx_seq_kept[slot] = 0u;
    aotx_seq_shown[slot] = 0u;
    aotx_seq_asked[slot] = 0u;
    aotx_decode.rows[slot] = 0u;
    aotx_decode.first[slot] = 0u;
    aotx_decode.place[slot] = AOTX_SLOTS;
}

/* End a sequence and give its pages back. The slot is free at once, and the request stands
 * in front of the request of the sequence that takes the slot next. */
static __device__ __forceinline__ void aotx_seq_shut(unsigned int slot)
{
    aotx_kv_release(slot);
    aotx_seqs.slot[slot].state = AOTX_SEQ_STATE_FREE;
    aotx_seq_asked[slot] = 0u;
    if (aotx_seqs.live > 0u) {
        atomicSub(&aotx_seqs.live, 1u);
    }
}

/* Report whether a slot already holds this prompt, or the first part of it. A replay opens
 * a slot from its token records, and the command that made those records reaches the slot
 * after them.
 *
 * The apply takes AOTX_INBOUND_MAX_TICK records of a tick, so the records of one line may
 * cross more than one batch. The slot may therefore hold the first part of the prompt when
 * the open arrives. A part that agrees is not a refusal. The records that follow confirm
 * the rest at their own positions. The apply compares every prompt token with the token
 * that stands at its place.
 *
 * A slot that already made a sampled token holds a whole prompt, so a prompt which is
 * longer than that one belongs to another sequence. */
static __device__ __forceinline__ int aotx_seq_holds(unsigned int slot, const int *ids,
                                                     unsigned int count)
{
    const aotx_seq *seq = &aotx_seqs.slot[slot];
    unsigned int held = seq->prompt;
    if (held == 0u || held > count) {
        return 0;
    }
    if (held < count && seq->sampled != 0u) {
        return 0;
    }
    for (unsigned int i = 0u; i < held; ++i) {
        if (aotx_seqs.tokens[slot][i] != ids[i]) {
            return 0;
        }
    }
    return 1;
}

__device__ int aotx_seq_open(unsigned int slot, unsigned int role, const int *ids,
                             unsigned int count, unsigned int limit, unsigned int page_limit,
                             const aotx_model_how *sample,
                             unsigned long long tick)
{
    if (slot >= AOTX_SLOTS || aotx_model_is_language(role) == 0 || count == 0u
        || limit == 0u || page_limit == 0u || page_limit > AOTX_KV_PAGES_EACH
        || count + limit > AOTX_SEQ_MAX_TOKENS || sample == 0) {
        atomicAdd(&aotx_seqs.refused, 1u);
        return 1;
    }

    /* The token records of a replay reach the apply before the command that made them
     * reaches this call. The command opens its sequence in a node of the same tick, which
     * runs after the apply. A slot that already holds this prompt is therefore taken over
     * and not refused. The sequence keeps its tokens and its stream, and it takes the
     * sampling values and the reply limit of this call.
     *
     * A replay puts the records of many ticks in one tick. The whole reply of a turn may
     * therefore stand in the slot before the open of that turn arrives. Such a slot is
     * taken over in the done state as well, and the caller reads the reply the journal
     * holds. A live run takes over no slot that is done. A turn which ended gives its slot
     * to the turn that follows. */
    aotx_seq *hold = &aotx_seqs.slot[slot];
    int takes = (hold->state == AOTX_SEQ_STATE_PREFILL || hold->state == AOTX_SEQ_STATE_DECODE
                 || (hold->state == AOTX_SEQ_STATE_DONE && aotx_seam.replaying != 0ull))
              ? 1 : 0;
    if (takes != 0 && aotx_seq_kept[slot] != 0u && aotx_seq_holds(slot, ids, count) != 0) {
        hold->role = role;
        hold->limit = limit;
        hold->page_limit = page_limit;
        hold->sample = *sample;
        hold->seed = sample->seed;
        aotx_seq_pages(slot, role, count + limit);
        return 0;
    }

    /* A slot that holds a sequence which ended takes the new one at once. A journal with
     * two replies on one slot therefore opens both when a restore replays it. */
    if (hold->state == AOTX_SEQ_STATE_DONE) {
        aotx_seq_shut(slot);
    }
    if (aotx_seqs.slot[slot].state != AOTX_SEQ_STATE_FREE) {
        atomicAdd(&aotx_seqs.refused, 1u);
        return 1;
    }
    unsigned int need = aotx_kvl_pages(&aotx_model_space[role].shape, count + limit);
    if (need == 0u || need > page_limit) {
        atomicAdd(&aotx_seqs.refused, 1u);
        return 1;
    }
    aotx_seq_clear(slot, role, sample, tick);
    aotx_seq *seq = &aotx_seqs.slot[slot];
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_seqs.tokens[slot][i] = ids[i];
    }
    seq->prompt = count;
    seq->limit = limit;
    seq->page_limit = page_limit;
    seq->last = ids[count - 1u];
    seq->state = AOTX_SEQ_STATE_PREFILL;
    atomicAdd(&aotx_seqs.live, 1u);
    aotx_seq_pages(slot, role, count + limit);
    aotx_seq_event(slot, AOTX_SEQ_OPENED, tick);
    return 0;
}

__device__ void aotx_seq_stop(unsigned int slot)
{
    if (slot >= AOTX_SLOTS) {
        return;
    }
    aotx_seq *seq = &aotx_seqs.slot[slot];
    if (seq->state == AOTX_SEQ_STATE_PREFILL || seq->state == AOTX_SEQ_STATE_DECODE) {
        seq->flags |= AOTX_DECODE_MARK_STOP;
    }
}

__device__ int aotx_seq_apply(const aotx_token_body *body)
{
    unsigned int slot = body->slot;
    if (slot >= AOTX_SLOTS || aotx_model_is_language(body->role) == 0
        || body->position >= AOTX_SEQ_MAX_TOKENS) {
        atomicAdd(&aotx_seqs.refused, 1u);
        return 1;
    }
    aotx_seq *seq = &aotx_seqs.slot[slot];

    /* A record at position 0 starts a sequence. A slot that already took records for the
     * sequence before it gives its pages back and takes the new one in the same apply. */
    if (body->position == 0u && seq->state != AOTX_SEQ_STATE_FREE
        && aotx_seq_kept[slot] != 0u) {
        aotx_seq_shut(slot);
    }

    /* A line the replay derives opens the slot with the values it took when it was live.
     * A slot with no such open takes the values of the model card, and its reply is bound
     * by the standard limit. */
    if (seq->state == AOTX_SEQ_STATE_FREE) {
        if (body->position != 0u) {
            atomicAdd(&aotx_seqs.refused, 1u);
            return 1;
        }
        aotx_model_how sample = aotx_sampler.row[slot];
        sample.seed = body->seed;
        aotx_seq_clear(slot, body->role, &sample, aotx_time_tick);
        seq->state = AOTX_SEQ_STATE_PREFILL;
        atomicAdd(&aotx_seqs.live, 1u);
    }

    /* A prompt token of a slot the open filled must be the token that stands there. The
     * open put the prompt in the slot, so the record confirms it and adds nothing. */
    if ((body->flags & AOTX_TOKEN_PROMPT) != 0u && body->position < seq->prompt) {
        if (aotx_seqs.tokens[slot][body->position] != (int)body->token) {
            atomicAdd(&aotx_seqs.refused, 1u);
            return 1;
        }
        aotx_seq_kept[slot] = body->position + 1u;
        aotx_seq_pages(slot, seq->role, seq->prompt + seq->limit);
        return 0;
    }
    if (body->position != seq->prompt + seq->sampled) {
        atomicAdd(&aotx_seqs.refused, 1u);
        return 1;
    }

    /* The token joins the list at its position and no draw is taken. The pages of the slot
     * are empty, so the plan takes the whole list as a prompt and rebuilds them. */
    aotx_seqs.tokens[slot][body->position] = (int)body->token;
    seq->last = body->token;
    if ((body->flags & AOTX_TOKEN_SAMPLED) != 0u) {
        seq->sampled = body->position + 1u - seq->prompt;
        seq->seed = body->seed;
        seq->draw = body->draw;
        aotx_model_draw[slot] = (unsigned int)body->draw + 1u;
        if (body->token == AOTX_DECODE_THINK_OPEN) {
            seq->thinking = 1u;
            seq->think_tokens = 0u;
        } else if (seq->thinking != 0u && body->token == AOTX_DECODE_THINK_CLOSE) {
            seq->thinking = 0u;
        } else if (seq->thinking != 0u) {
            seq->think_tokens += 1u;
        }
    } else {
        seq->prompt = body->position + 1u;
    }
    aotx_seq_kept[slot] = body->position + 1u;
    aotx_seq_pages(slot, seq->role, seq->prompt + seq->limit);
    if ((body->flags & AOTX_TOKEN_LAST) != 0u) {
        seq->flags |= AOTX_TOKEN_LAST;
        seq->state = AOTX_SEQ_STATE_DONE;
    }
    return 0;
}

/* The bytes that one token gives. The count comes first. A token that does not fit in the
 * room that is left therefore leaves the mark of the slot where it stands. */
static __device__ __forceinline__ unsigned int aotx_seq_token_bytes(unsigned int token,
                                                                    unsigned char *out,
                                                                    unsigned int room,
                                                                    int write)
{
    const aotx_text_vocab *vocab = &aotx_text_vocab_table;
    if (token >= vocab->tokens) {
        return 0u;
    }
    /* A control token carries no text of the reply. Its bytes name the end of a turn or a
     * part of the chat template, and the console must not show them. The count pass and the
     * write pass take this one test, so the two passes agree. */
    if (aotx_text_is_control(vocab, token) != 0) {
        return 0u;
    }
    unsigned long long from = vocab->token_at[token];
    unsigned int span = (unsigned int)(vocab->token_at[token + 1u] - from);
    const unsigned char *text = vocab->token_bytes + from;
    unsigned int walk = 0u;
    unsigned int at = 0u;
    while (walk < span) {
        unsigned int point = 0u;
        walk += aotx_text_decode(text, span, walk, &point);
        unsigned int byte = aotx_text_point_byte(point);
        if (byte < 0x100u) {
            if (write != 0 && at < room) {
                out[at] = (unsigned char)byte;
            }
            at += 1u;
        } else {
            /* A code point the byte map does not hold is written as itself. The count
             * pass and the write pass take the same length, so the two passes agree and
             * no byte of the run is left unwritten. */
            unsigned int span = (point < 0x80u) ? 1u
                              : ((point < 0x800u) ? 2u
                                 : ((point < 0x10000u) ? 3u : 4u));
            if (write != 0 && at + span <= room) {
                aotx_text_encode(point, out + at);
            }
            at += span;
        }
    }
    return at;
}

__device__ unsigned int aotx_seq_token_text(unsigned int token, unsigned char *out,
                                            unsigned int room)
{
    unsigned int bytes = aotx_seq_token_bytes(token, out, room, 0);
    if (bytes > room) {
        return 0u;
    }
    aotx_seq_token_bytes(token, out, room, 1);
    return bytes;
}

__device__ unsigned int aotx_seq_take_text(unsigned int slot, unsigned char *out,
                                           unsigned int max)
{
    if (slot >= AOTX_SLOTS || out == 0 || max == 0u) {
        return 0u;
    }
    const aotx_seq *seq = &aotx_seqs.slot[slot];
    unsigned int from = aotx_seq_shown[slot];
    unsigned int at = 0u;
    while (from < seq->sampled) {
        unsigned int token = (unsigned int)aotx_seqs.tokens[slot][seq->prompt + from];
        unsigned int bytes = aotx_seq_token_bytes(token, out + at, max - at, 0);
        if (at + bytes > max) {
            break;
        }
        aotx_seq_token_bytes(token, out + at, max - at, 1);
        at += bytes;
        from += 1u;
    }
    aotx_seq_shown[slot] = from;
    return at;
}
