/* Purpose: Apply the inbound records of one tick, echo each line, and feed the commands.
 * Owns: The inbound cursor, the state hash and the applied count.
 * Launch shape: AOTX_APPLY_BLOCKS blocks of AOTX_APPLY_THREADS; one thread for each input.
 * Lifetime: One node of every tick. */
#include "cognitive/live.cuh"
#include "media/runtime.cuh"
#include "catalog/catalog.cuh"
#include "agent/transcript.cuh"
#include "cli/cli.cuh"
#include "model/decode.cuh"
#include "model/load.cuh"
#include "seam/seam.cuh"
#include "settings/settings.cuh"
#include "tool/tool.cuh"
#ifdef AOTX_AFFECT
#include "affect/affect.cuh"
#endif

/* The blocks that reached the end of the apply. The last one stores the inbound cursor. */
__device__ unsigned int aotx_seam_apply_done = 0u;
__device__ unsigned char aotx_seam_line[AOTX_SAY_BYTES];
__device__ unsigned int aotx_seam_line_length = 0u;
__device__ unsigned int aotx_seam_line_parts = 0u;
__device__ unsigned int aotx_seam_line_replayed = 0u;
__device__ unsigned long long aotx_seam_line_seq = 0ull;
__device__ unsigned long long aotx_seam_line_holds = 0ull;
__device__ unsigned long long aotx_seam_line_orphans = 0ull;

/* The header fields of one inbound slot, read once. Inbound memory is host memory and
 * another process writes it, so every field is taken in one read and the copy is used. */
typedef struct aotx_apply_view {
    unsigned int magic;
    unsigned int layout;
    unsigned int cls;
    unsigned int type;
    unsigned int flags;
    unsigned int body_len;
} aotx_apply_view;

static __device__ __forceinline__ aotx_apply_view aotx_apply_read(
    const volatile aotx_record_header *header)
{
    aotx_apply_view view;
    view.magic = header->magic;
    view.layout = header->layout;
    view.cls = header->cls;
    view.type = header->type;
    view.flags = header->flags;
    view.body_len = header->body_len;
    return view;
}

/* The device takes an input line, a key event, a token, a tool reply and a setting. It
 * takes an import, a remove, a tick start marker and a restore report. In a build with the
 * affect substrate it takes an affect state record.
 * The device makes its own boot and commit markers, so it refuses those and counts them.
 * File bytes are not trusted, so the length is checked against the slot size. */
static __device__ __forceinline__ int aotx_apply_takes(const aotx_apply_view *view)
{
    if (view->magic != AOTX_WIRE_MAGIC || view->layout != (unsigned int)AOTX_WIRE_LAYOUT) {
        return 0;
    }
    if (view->body_len > AOTX_BODY_BYTES) {
        return 0;
    }
    if (view->cls == (unsigned int)AOTX_CLASS_A) {
        if (view->type == AOTX_REC_MEDIA) return view->body_len >= 24u;
        if (view->type == AOTX_REC_COGNITIVE) return view->body_len > AOTX_LIVE_PART;
        if (view->type == (unsigned int)AOTX_REC_KEY) {
            return view->body_len >= (unsigned int)sizeof(aotx_key_body);
        }
        if (view->type == (unsigned int)AOTX_REC_TOKEN) {
            return view->body_len >= (unsigned int)sizeof(aotx_token_body);
        }
        if (view->type == (unsigned int)AOTX_REC_TOOL_REPLY) {
            return view->body_len >= (unsigned int)sizeof(aotx_tool_reply_body);
        }
        if (view->type == (unsigned int)AOTX_REC_SETTING) {
            return view->body_len >= (unsigned int)sizeof(aotx_setting_body);
        }
        if (view->type == (unsigned int)AOTX_REC_IMPORT) {
            /* A head is longer than a part, so the shorter of the two is the bound the
             * take reads. The catalog reads the part field and checks the length again. */
            return view->body_len >= (unsigned int)sizeof(aotx_import_head)
                || view->body_len >= (unsigned int)sizeof(aotx_import_part);
        }
        if (view->type == (unsigned int)AOTX_REC_REMOVE) {
            return view->body_len >= (unsigned int)sizeof(aotx_remove_body);
        }
        if (view->type == (unsigned int)AOTX_REC_SELECTION) {
            return view->body_len >= (unsigned int)sizeof(aotx_selection_body);
        }
        if (view->type == (unsigned int)AOTX_REC_MODEL) {
            return view->body_len >= (unsigned int)sizeof(aotx_model_body);
        }
#ifdef AOTX_AFFECT
        if (view->type == (unsigned int)AOTX_REC_AFFECT) {
            return view->body_len >= (unsigned int)sizeof(aotx_affect_body);
        }
#endif
        return (view->type == (unsigned int)AOTX_REC_INPUT_LINE
                || view->type == (unsigned int)AOTX_REC_TICK_START);
    }
    if (view->cls == (unsigned int)AOTX_CLASS_B
        && view->type == (unsigned int)AOTX_REC_RESTORE) {
        return view->body_len >= (unsigned int)sizeof(aotx_restore_body);
    }
    return 0;
}

/* The device applies every class A record. A record of another class takes the path of the
 * restore report. */
static __device__ __forceinline__ int aotx_apply_folds(const aotx_apply_view *view)
{
    return view->cls == (unsigned int)AOTX_CLASS_A;
}

static __device__ __forceinline__ int aotx_apply_selection_takes(
    const volatile unsigned char *bytes)
{
    const volatile aotx_selection_body *body =
        (const volatile aotx_selection_body *)bytes;
    if (body->agent >= AOTX_SLOTS || body->count > AOTX_SELECTION_MAX
        || body->pages == 0u || body->pages > AOTX_KV_PAGES_EACH
        || body->current_seq == 0ull) {
        return 0;
    }
    for (unsigned int i = 0u; i < body->count; ++i) {
        if (body->seq[i] == 0ull || body->seq[i] >= body->current_seq) {
            return 0;
        }
    }
    return body->summary_seq == 0ull || body->summary_seq < body->current_seq;
}

static __device__ __forceinline__ const volatile aotx_record_header *aotx_apply_slot(
    unsigned long long index)
{
    unsigned long long at = (index & aotx_seam.in.mask) * (unsigned long long)AOTX_SLOT_BYTES;
    return (const volatile aotx_record_header *)(aotx_seam.in.slots + at);
}

/* Copy a body into a record that the device ring holds. */
static __device__ __forceinline__ void aotx_apply_copy(unsigned char *to,
                                                       const volatile unsigned char *from,
                                                       unsigned int length)
{
    for (unsigned int b = 0u; b < length; ++b) {
        to[b] = from[b];
    }
}

/* Put one complete input line in console records. The records carry the same fragment
 * rule as a streamed reply, so a reader joins them into one line. */
static __device__ __forceinline__ void aotx_apply_echo_line(const unsigned char *text,
                                                            unsigned int length)
{
    unsigned int at = 0u;
    unsigned int part = 0u;
    while (at < length || part == 0u) {
        unsigned long long seq = aotx_seam_claim(1u);
        aotx_record_header *header = aotx_seam_slot(seq);
        unsigned char *body = aotx_seam_body(header);
        unsigned int used = 0u;
        if (part == 0u) {
            body[used++] = (unsigned char)'>';
            body[used++] = (unsigned char)' ';
        }
        while (at < length && used < AOTX_BODY_BYTES) {
            body[used++] = text[at++];
        }
        aotx_seam_publish(header, seq, AOTX_WRITER_CONSOLE, AOTX_CLASS_B,
                          AOTX_REC_CONSOLE,
                          (part == 0u) ? 0u : AOTX_FLAG_FRAGMENT, used);
        aotx_console_put(body, used);
        part += 1u;
    }
}

/* A feeder-generated import refusal is already a console answer. Do not echo its internal
 * command form before the parser replaces it with the one stated refusal line. */
static __device__ __forceinline__ int aotx_apply_import_refusal(const unsigned char *text,
                                                                unsigned int length)
{
    static const char head[] = "import ";
    static const char mark[] = " refused: ";
    if (length < sizeof(head) + sizeof(mark) - 2u) {
        return 0;
    }
    for (unsigned int i = 0u; i < (unsigned int)sizeof(head) - 1u; ++i) {
        if (text[i] != (unsigned char)head[i]) return 0;
    }
    for (unsigned int i = (unsigned int)sizeof(head) - 1u;
         i + (unsigned int)sizeof(mark) - 1u <= length; ++i) {
        unsigned int same = 1u;
        for (unsigned int k = 0u; k < (unsigned int)sizeof(mark) - 1u; ++k) {
            same &= (text[i + k] == (unsigned char)mark[k]) ? 1u : 0u;
        }
        if (same != 0u) {
            return 1;
        }
    }
    return 0;
}

static __device__ __forceinline__ void aotx_apply_finish_line(unsigned long long tick)
{
    if (aotx_seam_line_parts == 0u) {
        return;
    }
    aotx_transcript_source(aotx_seam_line_seq);
    if (aotx_seam_line_replayed == 0u
        && aotx_apply_import_refusal(aotx_seam_line, aotx_seam_line_length) == 0) {
        aotx_apply_echo_line(aotx_seam_line, aotx_seam_line_length);
    }
    aotx_cli_line(aotx_seam_line, aotx_seam_line_length, tick);
    aotx_seam_line_length = 0u;
    aotx_seam_line_parts = 0u;
    aotx_seam_line_replayed = 0u;
    aotx_seam_line_seq = 0ull;
}

/* The clock of a replay, in the ticks of the run that wrote the journal. Zero while no
 * replay runs. */
__device__ unsigned long long aotx_seam_replay_clock = 0ull;
__device__ unsigned long long aotx_seam_replay_holds = 0ull;

__device__ unsigned int aotx_seam_replay_take(unsigned long long base, unsigned int ready)
{
    if (aotx_seam.replaying == 0ull) {
        aotx_seam_replay_clock = 0ull;
        return ready;
    }
    if (ready == 0u) {
        return 0u;
    }
    /* A record carries the tick of the run that wrote it. The apply takes the records of
     * one such tick in one tick of this run. The inputs of the operator then reach the
     * device at the place in the flow of the agents they had before.
     *
     * A replay that took every record it found would give a line to an agent which was
     * still in the turn before it. The command layer refuses such a line.
     *
     * The clock starts at the tick of the first record. The clock moves only when the take
     * saw a record of a later tick, because that record proves the tick of the clock is
     * complete. A take that ends at the cap of the apply, or at the end of what the ring
     * holds, keeps the clock. The rest of that tick of the journal then comes in the tick
     * after it. It never comes with the records of the tick that follows it.
     *
     * A tick of the journal therefore spills into more ticks of this run and never merges
     * with the next one. The last tick of the journal keeps the clock until the record of
     * the restore program ends the replay. */
    if (aotx_seam_replay_clock == 0ull) {
        aotx_seam_replay_clock = aotx_apply_slot(base)->tick;
    }
    unsigned int at = 0u;
    while (at < ready && aotx_apply_slot(base + at)->tick <= aotx_seam_replay_clock) {
        at += 1u;
    }
    if (at < ready) {
        aotx_seam_replay_clock += 1ull;
    }
    if (at == 0u) {
        aotx_seam_replay_holds += 1ull;
    }
    return at;
}

/* Each input takes a sequence that comes from its position and not from an atomic add.
 * The journal keeps the order of the inputs, so a replay gives the same state hash. The
 * tick start reserves that run of sequences. A record that the command layer writes takes
 * a sequence after the run, and never one inside it. */
__global__ void aotx_seam_apply_inbound(void)
{
    __shared__ unsigned char aotx_apply_body[AOTX_BODY_BYTES];

    const unsigned long long count = aotx_seam.apply.this_tick;
    const unsigned long long base = aotx_seam.in.consumed;
    const unsigned long long first = aotx_seam.apply.first_seq;

    /* The hash is a fold in order, so one thread makes it while the rest of the work runs.
     * The same thread writes the restore record, because that record carries the hash as it
     * stands at its own place in the order. */
    if (blockIdx.x == 0u && threadIdx.x == 0u && count > 0ull) {
        unsigned long long hash = aotx_seam.apply.state_hash;
        unsigned long long applied = aotx_seam.apply.applied_count;
        unsigned long long wall = aotx_seam.apply.wall_ns;
        unsigned long long rejected = aotx_seam.apply.rejected;
        /* A selection is written after the line that opened its prompt. A restore must
         * install every valid choice before it applies the lines of the same journal tick.
         * The hash still folds the records in their journal order in the loop below. */
        for (unsigned long long i = 0ull; i < count; ++i) {
            const volatile aotx_record_header *header = aotx_apply_slot(base + i);
            aotx_apply_view view = aotx_apply_read(header);
            const volatile unsigned char *body = (const volatile unsigned char *)header
                                               + AOTX_HEADER_BYTES;
            if (view.type != (unsigned int)AOTX_REC_SELECTION
                || !aotx_apply_takes(&view) || !aotx_apply_selection_takes(body)) {
                continue;
            }
            for (unsigned int b = 0u; b < (unsigned int)sizeof(aotx_selection_body); ++b) {
                aotx_apply_body[b] = body[b];
            }
            aotx_transcript_selection_apply((const aotx_selection_body *)aotx_apply_body);
        }
        for (unsigned long long i = 0ull; i < count; ++i) {
            const volatile aotx_record_header *header = aotx_apply_slot(base + i);
            aotx_apply_view view = aotx_apply_read(header);
            if (!aotx_apply_takes(&view)) {
                /* A bad continuation makes the line bad. Do not parse the valid head as
                 * a shorter command when the next head arrives. */
                if (view.type == (unsigned int)AOTX_REC_INPUT_LINE
                    && (view.flags & AOTX_FLAG_FRAGMENT) != 0u
                    && aotx_seam_line_parts != 0u) {
                    aotx_seam_line_length = 0u;
                    aotx_seam_line_parts = 0u;
                    aotx_seam_line_replayed = 0u;
                    aotx_seam_line_seq = 0ull;
                }
                rejected += 1ull;
                continue;
            }
            const volatile unsigned char *body = (const volatile unsigned char *)header
                                               + AOTX_HEADER_BYTES;
            if (view.type == (unsigned int)AOTX_REC_SELECTION
                && !aotx_apply_selection_takes(body)) {
                rejected += 1ull;
                continue;
            }
            if (!aotx_apply_folds(&view)) {
                /* The restore report states what the device holds, so the device puts its
                 * own hash in the body before the record goes in the journal. */
                aotx_record_header *again = aotx_seam_slot(first + i);
                unsigned char *to = aotx_seam_body(again);
                aotx_apply_copy(to, body, view.body_len);
                const aotx_restore_body *expected = (const aotx_restore_body *)to;
                if (aotx_runtime_enabled && (expected->state_hash != hash ||
                    expected->replayed_count != applied)) ++rejected;
                ((aotx_restore_body *)to)->state_hash = hash;
                aotx_seam_publish(again, first + i, AOTX_WRITER_RESTORE, AOTX_CLASS_B,
                                  AOTX_REC_RESTORE, view.flags, view.body_len);
                aotx_seam_pad(first + count + i);
                /* The replay ends with this record. An import whose last part is not in
                 * the journal never lands, and the number of an import is unique while
                 * that import arrives. Every such import therefore goes out here. */
                aotx_catalog_restore_end(aotx_time_tick);
                aotx_media_restore_end();
                if (!aotx_live_restore_end()) ++rejected;
                continue;
            }
            for (unsigned int b = 0u; b < view.body_len; ++b) {
                hash ^= (unsigned long long)body[b];
                hash *= AOTX_FNV_PRIME;
            }
            if (view.type == (unsigned int)AOTX_REC_TICK_START
                && view.body_len >= sizeof(aotx_clock_body)) {
                wall = ((const volatile aotx_clock_body *)body)->wall_ns;
            }
            applied += 1ull;

            /* The command layer sees each key and each line in slot order, whether the
             * feeder sent it or a restore sent it again. The device makes the command from
             * the keys, so the journal holds the keys and not the command. */
            if (view.type == AOTX_REC_MEDIA) {
                for (unsigned b = 0; b < view.body_len; ++b) aotx_apply_body[b] = body[b];
                unsigned long long source = (view.flags & AOTX_FLAG_REPLAYED) != 0u
                    ? (unsigned long long)header->source_seq[0] | ((unsigned long long)header->source_seq[1] << 32)
                    : first + i;
                if (!aotx_media_part(aotx_apply_body, view.body_len, source)) ++rejected;
            } else if (view.type == AOTX_REC_COGNITIVE) {
                for (unsigned int b = 0; b < view.body_len; ++b) aotx_apply_body[b] = body[b];
                unsigned long long source = (view.flags & AOTX_FLAG_REPLAYED) != 0u
                    ? (unsigned long long)header->source_seq[0] | ((unsigned long long)header->source_seq[1] << 32)
                    : first + i;
                aotx_live_part(aotx_apply_body, view.body_len, source,
                    aotx_live_record_flags(body, view.body_len, view.flags));
            } else if (view.type == (unsigned int)AOTX_REC_KEY) {
                for (unsigned int b = 0u; b < (unsigned int)sizeof(aotx_key_body); ++b) {
                    aotx_apply_body[b] = body[b];
                }
                aotx_cli_key((const aotx_key_body *)aotx_apply_body, aotx_time_tick);
            } else if (view.type == (unsigned int)AOTX_REC_TOKEN) {
                /* A replayed token joins its sequence and no draw is taken. The pages of
                 * the slot are rebuilt by the prefill of the ticks that follow. */
                for (unsigned int b = 0u; b < (unsigned int)sizeof(aotx_token_body); ++b) {
                    aotx_apply_body[b] = body[b];
                }
                const aotx_token_body *token = (const aotx_token_body *)aotx_apply_body;
                if (token->slot < AOTX_SLOTS
                    && (token->flags & AOTX_TOKEN_LAST) != 0u) {
                    aotx_transcript_replay_tick[token->slot] = header->tick;
                }
                if (aotx_seq_apply(token) != 0) rejected += 1ull;
            } else if (view.type == (unsigned int)AOTX_REC_TOOL_REPLY) {
                /* The answer of the feeder to a host tool. The record is class A, so a
                 * restore applies the recorded answer and the feeder runs nothing again. */
                for (unsigned int b = 0u;
                     b < (unsigned int)sizeof(aotx_tool_reply_body); ++b) {
                    aotx_apply_body[b] = body[b];
                }
                aotx_tool_reply_apply((const aotx_tool_reply_body *)aotx_apply_body,
                                      first + i);
            } else if (view.type == (unsigned int)AOTX_REC_SETTING) {
                /* The settings table takes the value. The record is class A, so the fold
                 * above put it in the state hash and a restore applies it again. */
                for (unsigned int b = 0u;
                     b < (unsigned int)sizeof(aotx_setting_body); ++b) {
                    aotx_apply_body[b] = body[b];
                }
                aotx_settings_apply((const aotx_setting_body *)aotx_apply_body,
                                    aotx_time_tick);
#ifdef AOTX_AFFECT
            } else if (view.type == (unsigned int)AOTX_REC_AFFECT) {
                /* The state table takes the recorded state. The record is class A, so the
                 * fold above put it in the state hash. Its place is the place the turn node
                 * gave it in the run that wrote it. */
                for (unsigned int b = 0u; b < (unsigned int)sizeof(aotx_affect_body); ++b) {
                    aotx_apply_body[b] = body[b];
                }
                aotx_affect_apply((const aotx_affect_body *)aotx_apply_body);
#endif
            } else if (view.type == (unsigned int)AOTX_REC_MODEL) {
                for (unsigned int b = 0u; b < (unsigned int)sizeof(aotx_model_body); ++b) {
                    aotx_apply_body[b] = body[b];
                }
                if (aotx_model_load_apply((const aotx_model_body *)aotx_apply_body) != 0) {
                    rejected += 1ull;
                }
            } else if (view.type == (unsigned int)AOTX_REC_IMPORT
                       || view.type == (unsigned int)AOTX_REC_REMOVE) {
                /* One module goes in as a head and the parts of its files, and one module
                 * goes out by name. Both records are class A, so the fold above put each
                 * one in the state hash. A restore builds the catalog from the journal
                 * and opens no file. */
                for (unsigned int b = 0u; b < view.body_len; ++b) {
                    aotx_apply_body[b] = body[b];
                }
                aotx_catalog_apply(view.type, aotx_apply_body, view.body_len, first + i);
            } else if (view.type == (unsigned int)AOTX_REC_INPUT_LINE) {
                unsigned int fragment = (view.flags & AOTX_FLAG_FRAGMENT) != 0u;
                if (fragment == 0u && aotx_seam_line_parts != 0u) {
                    aotx_apply_finish_line(aotx_time_tick);
                }
                if (fragment != 0u && aotx_seam_line_parts == 0u) {
                    rejected += 1ull;
                    aotx_seam_line_orphans += 1ull;
                } else {
                    if (fragment == 0u) {
                        aotx_seam_line_seq = ((view.flags & AOTX_FLAG_REPLAYED) != 0u)
                                           ? ((unsigned long long)header->source_seq[1] << 32)
                                             | (unsigned long long)header->source_seq[0]
                                           : first + i;
                        aotx_seam_line_replayed =
                            ((view.flags & AOTX_FLAG_REPLAYED) != 0u) ? 1u : 0u;
                    }
                    if (aotx_seam_line_parts >= AOTX_LINE_PARTS_MAX
                        || aotx_seam_line_length + view.body_len > AOTX_SAY_BYTES) {
                        rejected += 1ull;
                        aotx_seam_line_length = 0u;
                        aotx_seam_line_parts = 0u;
                    } else {
                        for (unsigned int b = 0u; b < view.body_len; ++b) {
                            aotx_seam_line[aotx_seam_line_length + b] = body[b];
                        }
                        aotx_seam_line_length += view.body_len;
                        aotx_seam_line_parts += 1u;
                    }
                }
                int next_fragment = 0;
                if (aotx_seam_line_parts != 0u) {
                    unsigned long long ready = aotx_seam.apply.available;
                    if (i + 1ull < count) {
                        next_fragment = (aotx_apply_slot(base + i + 1ull)->flags
                                         & AOTX_FLAG_FRAGMENT) != 0u;
                    } else if (count < ready) {
                        next_fragment = (aotx_apply_slot(base + count)->flags
                                         & AOTX_FLAG_FRAGMENT) != 0u;
                    }
                    if (next_fragment == 0) {
                        aotx_apply_finish_line(aotx_time_tick);
                    } else if (i + 1ull == count) {
                        aotx_seam_line_holds += 1ull;
                    }
                }
            }
        }
        aotx_seam.apply.state_hash = hash;
        aotx_seam.apply.applied_count = applied;
        aotx_seam.apply.wall_ns = wall;
        aotx_seam.apply.rejected = rejected;
    }

    /* Each input takes two sequences: the journal record and the echo. A sequence that
     * carries nothing takes a pad record, so the run of sequences stays whole. */
    const unsigned long long stride = (unsigned long long)(gridDim.x * blockDim.x);
    for (unsigned long long i = (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x);
         i < count; i += stride) {
        const volatile aotx_record_header *header = aotx_apply_slot(base + i);
        aotx_apply_view view = aotx_apply_read(header);
        unsigned long long journal = first + i;
        unsigned long long echoed = first + count + i;
        if (!aotx_apply_takes(&view)) {
            aotx_seam_pad(journal);
            aotx_seam_pad(echoed);
            continue;
        }
        const volatile unsigned char *body = (const volatile unsigned char *)header
                                           + AOTX_HEADER_BYTES;
        if (view.type == (unsigned int)AOTX_REC_SELECTION
            && !aotx_apply_selection_takes(body)) {
            aotx_seam_pad(journal);
            aotx_seam_pad(echoed);
            continue;
        }
        if (!aotx_apply_folds(&view)) {
            continue;   /* the restore record is written in order by the one thread above */
        }
        int replayed = (view.flags & AOTX_FLAG_REPLAYED) != 0u;
        if (view.type == AOTX_REC_COGNITIVE) view.flags = aotx_live_record_flags(body, view.body_len, view.flags);
        /* Long line echoes claim their records after assembly. The reserved echo place is
         * a pad for every input part. A key event has no echo either. */

        aotx_record_header *again = aotx_seam_slot(journal);
        aotx_apply_copy(aotx_seam_body(again), body, view.body_len);
        aotx_seam_publish_at(again, journal,
                             replayed ? AOTX_WRITER_RESTORE : AOTX_WRITER_FEEDER,
                             AOTX_CLASS_A, view.type, view.flags, view.body_len,
                             replayed ? header->tick : aotx_time_tick,
                             replayed ? (unsigned long long)header->source_seq[0] |
                                 ((unsigned long long)header->source_seq[1] << 32) : 0ull);
        /* The thread that keeps the order of the inputs writes the echo. This thread fills
         * the sequence of an input that has no echo, so the run of sequences stays whole. */
        aotx_seam_pad(echoed);
    }

    /* The last block to arrive tells the feeder which slots are free again. The tick start
     * already moved the ring tail past the records that the apply owns. */
    __syncthreads();
    if (threadIdx.x == 0u) {
        __threadfence();
        unsigned int done = atomicAdd(&aotx_seam_apply_done, 1u) + 1u;
        if (done == gridDim.x) {
            aotx_seam_apply_done = 0u;
            unsigned long long taken = base + count;
            aotx_seam.in.consumed = taken;
            aotx_inbound_preamble *preamble =
                (aotx_inbound_preamble *)aotx_seam.in.preamble;
            aotx_seam_release_sys(&preamble->consumed, taken);
        }
    }
}
