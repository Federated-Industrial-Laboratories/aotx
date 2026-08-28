/* Purpose: Apply the inbound records of one tick, echo each line, and feed the commands.
 * Owns: The inbound cursor, the state hash and the applied count.
 * Launch shape: AOTX_APPLY_BLOCKS blocks of AOTX_APPLY_THREADS; one thread for each input.
 * Lifetime: One node of every tick. */
#include "cli/cli.cuh"
#include "model/decode.cuh"
#include "seam/seam.cuh"
#include "tool/tool.cuh"

/* The blocks that reached the end of the apply. The last one stores the inbound cursor. */
__device__ unsigned int aotx_seam_apply_done = 0u;

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

/* The device takes an input line, a key event, a token, a tool reply, a tick start marker
 * and a restore report.
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
        if (view->type == (unsigned int)AOTX_REC_KEY) {
            return view->body_len >= (unsigned int)sizeof(aotx_key_body);
        }
        if (view->type == (unsigned int)AOTX_REC_TOKEN) {
            return view->body_len >= (unsigned int)sizeof(aotx_token_body);
        }
        if (view->type == (unsigned int)AOTX_REC_TOOL_REPLY) {
            return view->body_len >= (unsigned int)sizeof(aotx_tool_reply_body);
        }
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
     * The clock starts at the tick of the first record. A tick of the journal with more
     * records than the apply takes in one tick spills into the tick after it. That gives
     * the device more ticks and never fewer. */
    if (aotx_seam_replay_clock == 0ull) {
        aotx_seam_replay_clock = aotx_apply_slot(base)->tick;
    }
    unsigned int at = 0u;
    while (at < ready && aotx_apply_slot(base + at)->tick <= aotx_seam_replay_clock) {
        at += 1u;
    }
    aotx_seam_replay_clock += 1ull;
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
    __shared__ unsigned char aotx_apply_line[AOTX_BODY_BYTES];

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
        for (unsigned long long i = 0ull; i < count; ++i) {
            const volatile aotx_record_header *header = aotx_apply_slot(base + i);
            aotx_apply_view view = aotx_apply_read(header);
            if (!aotx_apply_takes(&view)) {
                rejected += 1ull;
                continue;
            }
            const volatile unsigned char *body = (const volatile unsigned char *)header
                                               + AOTX_HEADER_BYTES;
            if (!aotx_apply_folds(&view)) {
                /* The restore report states what the device holds, so the device puts its
                 * own hash in the body before the record goes in the journal. */
                aotx_record_header *again = aotx_seam_slot(first + i);
                unsigned char *to = aotx_seam_body(again);
                aotx_apply_copy(to, body, view.body_len);
                ((aotx_restore_body *)to)->state_hash = hash;
                aotx_seam_publish(again, first + i, AOTX_WRITER_RESTORE, AOTX_CLASS_B,
                                  AOTX_REC_RESTORE, view.flags, view.body_len);
                aotx_seam_pad(first + count + i);
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
            if (view.type == (unsigned int)AOTX_REC_KEY) {
                for (unsigned int b = 0u; b < (unsigned int)sizeof(aotx_key_body); ++b) {
                    aotx_apply_line[b] = body[b];
                }
                aotx_cli_key((const aotx_key_body *)aotx_apply_line, aotx_time_tick);
            } else if (view.type == (unsigned int)AOTX_REC_TOKEN) {
                /* A replayed token joins its sequence and no draw is taken. The pages of
                 * the slot are rebuilt by the prefill of the ticks that follow. */
                for (unsigned int b = 0u; b < (unsigned int)sizeof(aotx_token_body); ++b) {
                    aotx_apply_line[b] = body[b];
                }
                aotx_seq_apply((const aotx_token_body *)aotx_apply_line);
            } else if (view.type == (unsigned int)AOTX_REC_TOOL_REPLY) {
                /* The answer of the feeder to a host tool. The record is class A, so a
                 * restore applies the recorded answer and the feeder runs nothing again. */
                for (unsigned int b = 0u;
                     b < (unsigned int)sizeof(aotx_tool_reply_body); ++b) {
                    aotx_apply_line[b] = body[b];
                }
                aotx_tool_reply_apply((const aotx_tool_reply_body *)aotx_apply_line);
            } else if (view.type == (unsigned int)AOTX_REC_INPUT_LINE) {
                for (unsigned int b = 0u; b < view.body_len; ++b) {
                    aotx_apply_line[b] = body[b];
                }
                /* The echo takes the second sequence of this input. This thread writes it,
                 * because the console buffer must hold the echo above the answer, and only
                 * this thread keeps the order of the inputs. A replayed line has no echo. */
                if ((view.flags & AOTX_FLAG_REPLAYED) == 0u) {
                    aotx_cli_echo(first + count + i, aotx_apply_line, view.body_len);
                }
                aotx_cli_line(aotx_apply_line, view.body_len, aotx_time_tick);
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
        if (!aotx_apply_folds(&view)) {
            continue;   /* the restore record is written in order by the one thread above */
        }
        const volatile unsigned char *body = (const volatile unsigned char *)header
                                           + AOTX_HEADER_BYTES;
        int replayed = (view.flags & AOTX_FLAG_REPLAYED) != 0u;
        int echo = (view.type == (unsigned int)AOTX_REC_INPUT_LINE) && !replayed;
        /* A key event has no echo. The line editor shows the line it builds. */

        aotx_record_header *again = aotx_seam_slot(journal);
        aotx_apply_copy(aotx_seam_body(again), body, view.body_len);
        aotx_seam_publish(again, journal,
                          replayed ? AOTX_WRITER_RESTORE : AOTX_WRITER_FEEDER,
                          AOTX_CLASS_A, view.type, view.flags, view.body_len);
        /* The thread that keeps the order of the inputs writes the echo. This thread fills
         * the sequence of an input that has no echo, so the run of sequences stays whole. */
        if (!echo) {
            aotx_seam_pad(echoed);
        }
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
