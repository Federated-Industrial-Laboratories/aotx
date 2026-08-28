/* Purpose: Run the device tools of a tick over one batch of the embedding model.
 * Owns: The batch of the tick and the tokenizer memory of the tool path.
 * Launch shape: One block of one thread for each request slot.
 * Lifetime: The whole run.
 *
 * memory_write puts one finding on the bus and keeps its vector beside it. memory_recall
 * searches those vectors and gives the nearest notes back. Both need the vector of a text.
 * The texts of a tick therefore go through the tokenizer of this path. They then go
 * through the pass of the embedding role as one batch of the tick graph. */
#include "agent/agent.cuh"
#include "bus/bus.cuh"
#include "sched/sched.cuh"
#include "tool/tool_state.cuh"

__device__ aotx_tool_work aotx_tool_gear;
__device__ aotx_tool_batch aotx_tool_embed;

/* An inclusive add over the slots of the block. Every thread of the block takes part. */
__device__ __forceinline__ static unsigned int aotx_tool_scan(unsigned int *cell,
                                                              unsigned int value)
{
    unsigned int at = threadIdx.x;
    __syncthreads();
    cell[at] = value;
    __syncthreads();
    for (unsigned int step = 1u; step < AOTX_REQUEST_SLOTS; step <<= 1) {
        unsigned int add = (at >= step) ? cell[at - step] : 0u;
        __syncthreads();
        cell[at] += add;
        __syncthreads();
    }
    return cell[at];
}

__global__ void aotx_tool_fill(void)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= AOTX_REQUEST_SLOTS) {
        return;
    }
    /* Every slot is in the batch of every tick, so the shape of the graph never changes. A
     * slot with no text gives a byte run of no length and no piece. */
    aotx_tool_gear.start[slot] = slot * AOTX_TOOL_TEXT_BYTES;
    aotx_tool_gear.length[slot] =
        (aotx_tool_embed.state[slot] == AOTX_TOOL_EMBED_WAIT) ? aotx_tool_gear.bytes[slot]
                                                              : 0u;
    if (slot == 0u) {
        aotx_tool_gear.works = 0u;
        aotx_tool_embed.seqs = 0u;
        aotx_tool_embed.tokens = 0u;
    }
}

__global__ void aotx_tool_plan(unsigned long long tick)
{
    __shared__ unsigned int cell[AOTX_REQUEST_SLOTS];
    __shared__ unsigned int total_rows;
    __shared__ unsigned int total_seqs;

    (void)tick;
    unsigned int slot = threadIdx.x;
    unsigned int role = aotx_tool_embed.role;
    /* A run with no embedding role writes no call block, because that block belongs to a
     * role which holds no model. */
    if (slot >= AOTX_REQUEST_SLOTS || aotx_tool_embed.ready == 0u
        || role >= AOTX_MODEL_ROLES) {
        return;
    }
    /* The pass runs while a replay runs, because a device tool is derived work and a
     * replay derives it again. A held tick advances nothing. */
    int runs = (aotx_tool_embed.ready != 0u) && (role < AOTX_MODEL_ROLES)
             && (aotx_sched.held == 0ull);

    /* A slot joins the batch when its text has tokens and its pages are in hand. A slot
     * that waits for a page asks for it and joins a later tick. */
    unsigned int want = 0u;
    if (runs && aotx_tool_embed.state[slot] == AOTX_TOOL_EMBED_WAIT) {
        unsigned int tokens = aotx_tool_gear.count[slot];
        if (tokens > AOTX_TOOL_TOKENS) {
            tokens = AOTX_TOOL_TOKENS;
        }
        if (tokens > 0u) {
            unsigned int need = aotx_kvl_pages(&aotx_model_space[role].shape, tokens);
            unsigned int held = aotx_kv.count[slot];
            if (need <= held) {
                aotx_tool_embed.asked[slot] = held;
                want = tokens;
            } else {
                unsigned int asked = (aotx_tool_embed.asked[slot] > held)
                                   ? aotx_tool_embed.asked[slot] : held;
                if (need > asked) {
                    aotx_tool_embed.asked[slot] =
                        (aotx_kv_request(slot, need - asked) != 0) ? need : held;
                }
                atomicAdd(&aotx_tool_embed.short_of, 1u);
            }
        }
    }

    /* The rows of the tick take the budget of one pass in slot order. A text that crosses
     * the end of the budget waits for the tick after. */
    unsigned int want_scan = aotx_tool_scan(cell, want);
    unsigned int before = want_scan - want;
    unsigned int give = 0u;
    if (want > 0u) {
        if (before + want <= AOTX_MODEL_MAX_TOKENS) {
            give = want;
        } else {
            atomicAdd(&aotx_tool_embed.waited, 1u);
        }
    }
    unsigned int give_scan = aotx_tool_scan(cell, give);
    unsigned int start = give_scan - give;
    unsigned int mark = (give > 0u) ? 1u : 0u;
    unsigned int mark_scan = aotx_tool_scan(cell, mark);
    if (slot == AOTX_REQUEST_SLOTS - 1u) {
        total_seqs = mark_scan;
        total_rows = give_scan;
    }
    __syncthreads();

    unsigned int place = AOTX_REQUEST_SLOTS;
    if (give > 0u) {
        place = mark_scan - mark;
        aotx_tool_embed.agent[place] = slot;
        aotx_tool_embed.who[place] = slot;
        aotx_tool_embed.offset[place] = start;
        aotx_tool_embed.live[place] =
            (aotx_requests.slot[slot].tool == AOTX_TOOL_MEMORY_RECALL) ? 1u : 0u;
        const unsigned int *ids = aotx_tool_gear.id
                                + (unsigned long long)slot * AOTX_TOOL_TOKENS;
        for (unsigned int i = 0u; i < give; ++i) {
            aotx_tool_embed.ids[start + i] = (int)ids[i];
        }
        /* The pass writes the key and the value of every token from position zero. The
         * text of this tick therefore stands alone in the pages of the slot. */
        aotx_model_seen[slot] = 0u;
        aotx_tool_embed.state[slot] = AOTX_TOOL_EMBED_RUN;
    }
    aotx_tool_embed.place[slot] = place;
    __syncthreads();

    if (slot == 0u) {
        aotx_model_run *run = &aotx_model_call[role];
        aotx_tool_embed.offset[total_seqs] = total_rows;
        aotx_tool_embed.seqs = total_seqs;
        aotx_tool_embed.tokens = total_rows;
        run->ids = aotx_tool_embed.ids;
        run->offset = aotx_tool_embed.offset;
        run->agent = aotx_tool_embed.agent;
        run->logits = 0;
        run->pooled = aotx_tool_embed.vector;
        run->score = 0;
        run->token = 0;
        run->draw = 0;
        run->how = 0;
        run->seed = 0ull;
        run->seqs = total_seqs;
        run->tokens = total_rows;
        run->rows = total_seqs;
        run->select = AOTX_MODEL_ROWS_LAST;
        run->top_k = 0u;
        run->top_p = 1.0f;
        run->temperature = 0.0f;
    }
}

/* Write the result of a memory_write: the finding goes on the bus and its vector goes in
 * the note store beside it. */
__device__ __forceinline__ static void aotx_tool_write_note(unsigned int slot,
                                                            unsigned long long tick)
{
    aotx_request *hold = &aotx_requests.slot[slot];
    unsigned int place = aotx_tool_embed.place[slot];
    unsigned int width = aotx_tool_embed.width;
    unsigned long long seq = aotx_bus_append(AOTX_WRITER_AGENT_BASE + hold->agent,
                                             AOTX_BUS_FINDING, aotx_tool_embed.prov[slot],
                                             hold->result, hold->result_len, 0ull, 0ull,
                                             0.0f, tick);
    unsigned int at = 0u;
    if (seq == 0ull) {
        hold->status = AOTX_TOOL_ERROR;
        at = aotx_tool_put(hold->result, 0u, "the note has no source and did not go in");
        hold->result_len = at;
        return;
    }
    unsigned int where = aotx_embed_keep(aotx_tool_embed.vector
                                         + (unsigned long long)place * width,
                                         width, hold->result, hold->result_len, seq);
    if (where >= AOTX_EMBED_NOTES) {
        hold->status = AOTX_TOOL_ERROR;
        at = aotx_tool_put(hold->result, 0u, "memory is full and the note has no vector");
        hold->result_len = at;
        return;
    }
    at = aotx_tool_put(hold->result, 0u, "note ");
    at += aotx_text_utoa(seq, hold->result + at, AOTX_TOOL_RESULT_BYTES - at);
    at = aotx_tool_put(hold->result, at, " is in memory");
    hold->result_len = at;
    hold->status = AOTX_TOOL_OK;
    atomicAdd(&aotx_tool_count.written, 1u);
}

/* Write the result of a memory_recall: the text of each note the search found. */
__device__ __forceinline__ static void aotx_tool_recall_notes(unsigned int slot)
{
    aotx_request *hold = &aotx_requests.slot[slot];
    unsigned int place = aotx_tool_embed.place[slot];
    unsigned int at = 0u;
    unsigned int given = 0u;
    for (unsigned int h = 0u; h < AOTX_EMBED_HITS; ++h) {
        unsigned int which = aotx_tool_embed.hit[place * AOTX_EMBED_HITS + h];
        if (which >= AOTX_EMBED_NOTES || which >= aotx_embed_notes.count) {
            continue;
        }
        if (given != 0u) {
            at = aotx_tool_put(hold->result, at, "\n");
        }
        at = aotx_tool_put_run(hold->result, at, aotx_embed_notes.text[which],
                               aotx_embed_notes.len[which]);
        given += 1u;
    }
    if (given == 0u) {
        at = aotx_tool_put(hold->result, 0u, "memory holds no note");
    }
    hold->result_len = at;
    hold->status = AOTX_TOOL_OK;
    atomicAdd(&aotx_tool_count.recalled, 1u);
}

__global__ void aotx_tool_step(unsigned long long parameter)
{
    /* The node of the tick graph carries the parameter of its capture. The step therefore
     * takes the tick from the device clock, as the commit of the decode does. */
    const unsigned long long tick = (parameter != 0ull) ? parameter : aotx_time_tick;
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int replaying = (aotx_seam.replaying != 0ull) ? 1u : 0u;

    /* A replay applies the recorded reply of a request by its number, and no deadline
     * passes while it runs. A request that still waits at the end of a replay takes a new
     * deadline from that tick, so the operator sees it again. Every thread of the block
     * reads the mark before the first thread writes the new one. */
    unsigned int was = aotx_tool_embed.replayed;
    __syncthreads();
    if (threadIdx.x == 0u) {
        aotx_tool_embed.replayed = replaying;
    }
    if (slot >= AOTX_REQUEST_SLOTS) {
        return;
    }
    aotx_request *hold = &aotx_requests.slot[slot];
    if (hold->request == 0u || aotx_tool_done[slot] != 0u) {
        return;
    }
    if (was != 0u && replaying == 0u) {
        hold->deadline = tick + (unsigned long long)AOTX_TOOL_DEADLINE;
        /* A restore presents again every host request that waited at the crash. The
         * record goes in the journal a second time with the same number. The drain then
         * puts it in the requests file and the operator sees the one that waits. */
        if (hold->tool == AOTX_TOOL_FS_READ) {
            aotx_tool_note_request(hold, aotx_agents.agent[hold->agent].turn);
        }
    }

    /* A device tool whose text went through the pass this tick takes its vector now. */
    if (aotx_tool_embed.state[slot] == AOTX_TOOL_EMBED_RUN) {
        if (hold->tool == AOTX_TOOL_MEMORY_WRITE) {
            aotx_tool_write_note(slot, tick);
        } else {
            aotx_tool_recall_notes(slot);
        }
        aotx_tool_embed.state[slot] = AOTX_TOOL_EMBED_NONE;
        aotx_tool_done[slot] = 1u;
        atomicAdd(&aotx_tool_count.device_done, 1u);
        return;
    }

    /* The operator refused the tool. The call fails with the reason. */
    if (hold->auth == AOTX_AUTH_REFUSED) {
        hold->status = AOTX_TOOL_REFUSED;
        hold->result_len = aotx_tool_put(hold->result, 0u,
                                         "the operator refused this tool");
        aotx_tool_done[slot] = 1u;
        return;
    }

    /* The deadline of a request that no answer reached. The call fails with the reason,
     * and a reply that comes after it finds no request and is refused. */
    if (tick > hold->deadline && replaying == 0u) {
        /* A request that waited for the operator no longer waits. The count the panel
         * shows and the answer command reads must lose it. */
        if (hold->auth == AOTX_AUTH_PENDING && aotx_requests.pending_auth > 0u) {
            atomicSub(&aotx_requests.pending_auth, 1u);
        }
        hold->status = AOTX_TOOL_LATE;
        hold->result_len = aotx_tool_put(hold->result, 0u,
                                         "the tool gave no answer before its deadline");
        aotx_tool_embed.state[slot] = AOTX_TOOL_EMBED_NONE;
        aotx_tool_done[slot] = 1u;
        atomicAdd(&aotx_tool_count.late, 1u);
    }
}
