/* Purpose: Close the tick of every sequence: journal its tokens and move its state.
 * Owns: Nothing; the sequence table and the batch tables hold the state.
 * Launch shape: One block of one thread for each sequence slot.
 * Lifetime: One node of every tick.
 *
 * A token goes in the journal one time. A prompt token goes in when it enters the pages,
 * and the sampled token goes in when the draw makes it. The records of the tick take one
 * claimed run of sequences. The slots take their parts of that run in slot order. The
 * order of the token records of a tick is therefore the same in every run.
 *
 * That order is what lets the commit fold each token record into the state hash. The apply
 * of a restore folds the same bodies in the same order. A hash after a restore therefore
 * proves the token lists as well as the inputs.
 *
 * A slot that ends keeps its pages for the tick that ends it. The commit of the tick after
 * gives the pages back and the slot is free again. */
#include "model/decode_state.cuh"
#include "sched/sched.cuh"
#include "cognitive/intake.cuh"
#include "service/service.cuh"
#include "shared/bridge.cuh"

/* Write one token record into a sequence of the ring that the caller claimed. */
static __device__ __forceinline__ void aotx_commit_token(unsigned long long at,
                                                         unsigned int slot,
                                                         unsigned int token,
                                                         unsigned int position,
                                                         unsigned int flags,
                                                         unsigned long long seed,
                                                         unsigned long long draw,
                                                         unsigned int role)
{
    aotx_record_header *header = aotx_seam_slot(at);
    aotx_token_body *body = (aotx_token_body *)aotx_seam_body(header);
    body->slot = slot;
    body->token = token;
    body->position = position;
    body->flags = flags;
    body->seed = seed;
    body->draw = draw;
    body->role = role;
    body->text_len = ((flags & AOTX_TOKEN_SAMPLED) != 0u)
                   ? aotx_seq_token_text(token, (unsigned char *)body->text,
                                         (unsigned int)sizeof body->text, role) : 0u;
    for (unsigned int i = body->text_len; i < (unsigned int)sizeof body->text; ++i) {
        body->text[i] = '\0';
    }
    bool service = aotx_service_owns(slot) || aotx_shared_owns(slot);
    aotx_seam_publish(header, at, AOTX_WRITER_AGENT_BASE + slot, service ? AOTX_CLASS_B : AOTX_CLASS_A,
                      service ? AOTX_REC_SERVICE_TOKEN : AOTX_REC_TOKEN, 0u,
                      (unsigned int)sizeof *body);
}

/* Write one sequence event. The event is derived and a restore does not replay it. */
static __device__ __forceinline__ void aotx_commit_event(unsigned int slot,
                                                         unsigned int event,
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

/* An inclusive add over the slots of the block. Every thread of the block takes part. */
static __device__ __forceinline__ unsigned int aotx_commit_scan(unsigned int *cell,
                                                                unsigned int value)
{
    unsigned int at = threadIdx.x;
    __syncthreads();
    cell[at] = value;
    __syncthreads();
    for (unsigned int step = 1u; step < AOTX_SLOTS; step <<= 1) {
        unsigned int add = (at >= step) ? cell[at - step] : 0u;
        __syncthreads();
        cell[at] += add;
        __syncthreads();
    }
    return cell[at];
}

__global__ void aotx_decode_commit(unsigned long long tick)
{
    __shared__ unsigned int cell[AOTX_SLOTS];
    __shared__ unsigned long long claimed;
    __shared__ unsigned int total;

    unsigned int slot = threadIdx.x;
    if (slot >= AOTX_SLOTS) {
        return;
    }
    if (tick == 0ull) {
        tick = aotx_time_tick;
    }
    aotx_seq *seq = &aotx_seqs.slot[slot];
    unsigned int role = seq->role;
    unsigned int journal = 0u;
    unsigned int from = 0u;
    unsigned int extra = 0u;
    unsigned int token = 0u;
    unsigned int mark = 0u;
    unsigned int event = 0u;
    unsigned long long draw = 0ull;

    if (aotx_sched.held == 0ull && role == aotx_decode.role) {
        if (seq->state == AOTX_SEQ_STATE_DONE && !aotx_intake_owns(slot) && !aotx_shared_owns(slot)) {
            /* Shared results retain the terminal sequence until their completion record is applied. */
            aotx_kv_release(slot);
            seq->state = AOTX_SEQ_STATE_FREE;
            aotx_seq_asked[slot] = 0u;
            event = AOTX_SEQ_RELEASED;
            if (aotx_seqs.live > 0u) {
                atomicSub(&aotx_seqs.live, 1u);
            }
        } else if (seq->state == AOTX_SEQ_STATE_PREFILL || seq->state == AOTX_SEQ_STATE_DECODE) {
            /* Embedding can advance the shared cache cursor while this language slot is free. */
            unsigned int rows = aotx_decode.rows[slot];
            unsigned int start = aotx_decode.first[slot];
            unsigned int list = seq->prompt + seq->sampled;
            unsigned int end = start + rows;
            unsigned int kept = aotx_seq_kept[slot];
            journal = (end > kept) ? (end - kept) : 0u;
            from = kept;

            /* The head of the pass reads the last row of each sequence. The draw of that
             * row names the token that follows the list. It counts only when every token
             * of the list is in the pages. */
            if (rows > 0u && end == list && list < AOTX_SEQ_MAX_TOKENS) {
                unsigned int place = aotx_decode.place[slot];
                token = (unsigned int)aotx_decode.token[place];
                draw = (unsigned long long)aotx_decode.draw[place];
                extra = 1u;
            }
            if (rows > 0u) {
                seq->held = aotx_model_seen[slot];
                seq->state = (end == list) ? AOTX_SEQ_STATE_DECODE : AOTX_SEQ_STATE_PREFILL;
            }
            if (extra != 0u) {
                mark = AOTX_TOKEN_SAMPLED;
                aotx_seqs.tokens[slot][list] = (int)token;
                seq->sampled += 1u;
                seq->last = token;
                seq->draw = draw;
                if (aotx_wrap_think_open(seq->role, token)) {
                    seq->thinking = 1u;
                    seq->think_tokens = 0u;
                } else if (seq->thinking != 0u && aotx_wrap_think_close(seq->role, token)) {
                    seq->thinking = 0u;
                } else if (seq->thinking != 0u) {
                    seq->think_tokens += 1u;
                }
                if (token == seq->stop || aotx_wrap_end(seq->role, token)
                    || seq->sampled >= seq->limit
                    || (seq->flags & AOTX_DECODE_MARK_STOP) != 0u
                    || list + 1u >= AOTX_SEQ_MAX_TOKENS) {
                    mark |= AOTX_TOKEN_LAST;
                    seq->flags |= AOTX_TOKEN_LAST;
                    seq->state = AOTX_SEQ_STATE_DONE;
                    event = AOTX_SEQ_DONE;
                }
            }
            aotx_seq_kept[slot] = kept + journal + extra;

            /* The stop call ends the sequence at this commit, whatever it holds. */
            if ((seq->flags & AOTX_DECODE_MARK_STOP) != 0u
                && seq->state != AOTX_SEQ_STATE_FREE) {
                seq->state = AOTX_SEQ_STATE_DONE;
                event = AOTX_SEQ_STOPPED;
            }
        }
    }

    /* The records of the tick take one run of sequences, and the slots take their parts of
     * it in slot order. One thread claims the whole run, so the order never changes. */
    if (aotx_intake_owns(slot)) { journal = extra = event = 0; }
    unsigned int mine = journal + extra;
    unsigned int scan = aotx_commit_scan(cell, mine);
    unsigned int first = scan - mine;
    if (slot == AOTX_SLOTS - 1u) {
        total = scan;
        claimed = (scan > 0u) ? aotx_seam_claim(scan) : 0ull;
    }
    __syncthreads();

    for (unsigned int i = 0u; i < journal; ++i) {
        unsigned int position = from + i;
        aotx_commit_token(claimed + first + i, slot,
                          (unsigned int)aotx_seqs.tokens[slot][position], position,
                          AOTX_TOKEN_PROMPT, seq->seed, 0ull, role);
    }
    if (extra != 0u) {
        aotx_commit_token(claimed + first + journal, slot, token,
                          seq->prompt + seq->sampled - 1u, mark, seq->seed, draw, role);
    }
    __syncthreads();

    /* Only replayable token records enter the state hash. Ordinary service output is not restored as an agent turn. */
    if (slot == 0u && total != 0u) {
        unsigned long long hash = aotx_seam.apply.state_hash;
        unsigned int applied = 0;
        for (unsigned int r = 0u; r < total; ++r) {
            const unsigned char *body = aotx_seam_body_of(claimed + r);
            const aotx_record_header *header = (const aotx_record_header *)(body - AOTX_HEADER_BYTES);
            if (header->cls != AOTX_CLASS_A) continue;
            hash = aotx_seam_fnv1a(hash, body,
                                   (unsigned int)sizeof(aotx_token_body));
            ++applied;
        }
        aotx_seam.apply.state_hash = hash;
        aotx_seam.apply.applied_count += (unsigned long long)applied;
    }
    if (event != 0u) {
        aotx_commit_event(slot, event, tick);
    }
}
