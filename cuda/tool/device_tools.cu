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
#include "catalog/catalog.cuh"
#include "sched/sched.cuh"
#include "tool/module.cuh"
#include "tool/tool_state.cuh"

__device__ aotx_tool_work aotx_tool_gear;
__device__ aotx_tool_embed_batch aotx_tool_embed;

/* An inclusive add over the slots of the block. Every thread of the block takes part. */
__device__ __forceinline__ static unsigned int aotx_tool_scan(unsigned int *cell,
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

__global__ void aotx_tool_fill(void)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= AOTX_SLOTS) {
        return;
    }
    /* Every slot is in the batch of every tick, so the shape of the graph never changes. A
     * slot with no text gives a byte run of no length and no piece. */
    aotx_tool_gear.start[slot] = slot * AOTX_TOOL_TEXT_BYTES;
    aotx_tool_gear.length[slot] =
        (aotx_tool_embed.state[slot] == AOTX_TOOL_EMBED_WAIT) ? aotx_tool_gear.bytes[slot]
                                                              : 0u;
    /* The rows of every module node come from the same step, so a module node reads a
     * batch of this tick and of no other. */
    aotx_tool_module_fill(slot);
    if (slot == 0u) {
        aotx_tool_gear.works = 0u;
        aotx_tool_embed.seqs = 0u;
        aotx_tool_embed.tokens = 0u;
    }
}

__global__ void aotx_tool_plan(unsigned long long tick)
{
    __shared__ unsigned int cell[AOTX_SLOTS];
    __shared__ unsigned int total_rows;
    __shared__ unsigned int total_seqs;

    (void)tick;
    unsigned int slot = threadIdx.x;
    unsigned int role = aotx_tool_embed.role;
    /* A run with no embedding role writes no call block, because that block belongs to a
     * role which holds no model. */
    if (slot >= AOTX_SLOTS || aotx_tool_embed.ready == 0u
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
    if (slot == AOTX_SLOTS - 1u) {
        total_seqs = mark_scan;
        total_rows = give_scan;
    }
    __syncthreads();

    unsigned int place = AOTX_SLOTS;
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

/* The step runs as one block with one thread for each request slot. The claim of the late
 * records takes a scan over the whole block, so the shape is a condition of this file. */
typedef char aotx_tool_step_check[(AOTX_TOOL_SLOT_BLOCKS == 1u
                                   && AOTX_TOOL_SLOT_THREADS == AOTX_SLOTS) ? 1 : -1];

/* Fill the reply body that ends a request which reached its deadline. The body is the body
 * of a tool reply of the late status, in one part, and it carries the reason. */
__device__ __forceinline__ static void aotx_tool_late_body(const aotx_request *hold,
                                                           aotx_tool_reply_body *body)
{
    const char *why = "the tool gave no answer before its deadline";
    body->agent = hold->agent;
    body->request = hold->request;
    body->status = AOTX_TOOL_LATE;
    body->part = 0u;
    body->parts = 1u;
    unsigned int at = 0u;
    while (why[at] != '\0' && at < AOTX_TOOL_REPLY_BYTES) {
        body->bytes[at] = why[at];
        at += 1u;
    }
    body->len = at;
    for (unsigned int i = at; i < AOTX_TOOL_REPLY_BYTES; ++i) {
        body->bytes[i] = '\0';
    }
}

/* skill_use gives the body of one skill of the catalog. The body is bytes of the catalog
 * arena and the result is bytes of the request, so nothing of a skill use reaches the
 * host. The mark keeps the body of the copy out of the frame of the tool step, which the
 * spill gate holds to a figure. */
__device__ __noinline__ static void aotx_tool_skill_body(aotx_request *hold)
{
    unsigned int which = aotx_catalog_find(hold->result, hold->result_len,
                                           AOTX_MODULE_SKILL);
    if (which >= AOTX_MODULE_SLOTS) {
        /* The name of the call stands at the front of the result, so the reason names the
         * skill the model asked for. */
        hold->result_len = aotx_tool_put(hold->result, hold->result_len,
                                         " is not a skill of the catalog");
        hold->status = AOTX_TOOL_ERROR;
        atomicAdd(&aotx_catalog.count.skill_lost, 1u);
        return;
    }
    aotx_catalog_run body = aotx_catalog.entry[which].body;
    unsigned int bytes = (body.length > AOTX_TOOL_RESULT_BYTES) ? AOTX_TOOL_RESULT_BYTES
                                                                : body.length;
    for (unsigned int i = 0u; i < bytes; ++i) {
        hold->result[i] = (char)aotx_catalog_arena[body.at + i];
    }
    hold->result_len = bytes;
    hold->status = AOTX_TOOL_OK;
    atomicAdd(&aotx_catalog.count.skill_used, 1u);
}

__global__ void aotx_tool_step(unsigned long long parameter)
{
    __shared__ unsigned int cell[AOTX_SLOTS];
    __shared__ unsigned int lates;
    __shared__ unsigned long long claimed;

    /* The node of the tick graph carries the parameter of its capture. The step therefore
     * takes the tick from the device clock, as the commit of the decode does. */
    const unsigned long long tick = (parameter != 0ull) ? parameter : aotx_time_tick;
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int replaying = (aotx_seam.replaying != 0ull) ? 1u : 0u;

    /* A replay applies the recorded reply of a request by its number, and no deadline
     * passes while it runs. A request that still waits at the end of a replay takes a new
     * deadline from that tick, so the operator sees it again. A request that waits for the
     * operator keeps no deadline, and the record presents it again. Every thread of the
     * block reads the mark before the first thread writes the new one. */
    unsigned int was = aotx_tool_embed.replayed;
    __syncthreads();
    if (threadIdx.x == 0u) {
        aotx_tool_embed.replayed = replaying;
    }

    /* Every thread of the block stays to the end of the claim, because the scan of the late
     * requests takes the whole block. A thread that holds no request gives a zero to it. */
    aotx_request *hold = (slot < AOTX_SLOTS) ? &aotx_requests.slot[slot] : 0;
    unsigned int live = (hold != 0 && hold->request != 0u && aotx_tool_done[slot] == 0u)
                      ? 1u : 0u;
    if (live != 0u && was != 0u && replaying == 0u) {
        hold->deadline = (hold->auth == AOTX_AUTH_PENDING)
                       ? AOTX_TOOL_NO_DEADLINE
                       : tick + aotx_setting_deadline();
        /* A restore presents again every host request that waited at the crash. The
         * record goes in the journal a second time with the same number. The drain then
         * puts it in the requests file and the operator sees the one that waits. */
        if (aotx_catalog_on_disk(hold->entry) != 0) {
            aotx_tool_note_request(hold, aotx_agents.agent[hold->agent].turn);
        }
    }

    /* The deadline of a request that no answer reached. A request that waits for the
     * operator has no deadline, so it is not late while it waits. A request the operator
     * refused ends by its own branch below. A device tool whose vector came in this tick
     * ends by its own branch too. Neither takes a late verdict here. */
    unsigned int late = (live != 0u && hold->auth != AOTX_AUTH_PENDING
                         && hold->auth != AOTX_AUTH_REFUSED
                         && aotx_tool_embed.state[slot] != AOTX_TOOL_EMBED_RUN
                         && hold->status == AOTX_TOOL_OK
                         && tick > hold->deadline && replaying == 0u) ? 1u : 0u;

    /* The late verdict is a decision which this device makes on its own. It therefore goes
     * in the ring as a class A record and it folds into the state hash. A replay applies
     * that record where it stands. The decision is then made at the same place in the
     * order. The slots claim one run of sequences in slot order, so two runs of the same
     * inputs put the same records in the same places. */
    unsigned int rank = aotx_tool_scan(cell, late);
    if (threadIdx.x == AOTX_TOOL_SLOT_THREADS - 1u) {
        lates = rank;
        claimed = (rank > 0u) ? aotx_seam_claim(rank) : 0ull;
    }
    __syncthreads();
    if (late != 0u) {
        /* The body is built in the slot of the ring and not in a frame of this kernel, so
         * the step keeps its stack frame. */
        unsigned long long seq = claimed + (unsigned long long)(rank - 1u);
        aotx_record_header *header = aotx_seam_slot(seq);
        aotx_tool_reply_body *body = (aotx_tool_reply_body *)aotx_seam_body(header);
        aotx_tool_late_body(hold, body);
        aotx_seam_publish(header, seq, AOTX_WRITER_AGENT_BASE + hold->agent, AOTX_CLASS_A,
                          AOTX_REC_TOOL_REPLY, 0u, (unsigned int)sizeof *body);
        /* The apply of the record gives the result to the request. The live run and the
         * replay therefore take one path, and the reason lands in the same bytes. */
        aotx_tool_reply_apply(body);
        atomicAdd(&aotx_tool_count.late, 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0u && lates != 0u) {
        unsigned long long hash = aotx_seam.apply.state_hash;
        for (unsigned int r = 0u; r < lates; ++r) {
            hash = aotx_seam_fnv1a(hash, aotx_seam_body_of(claimed + (unsigned long long)r),
                                   (unsigned int)sizeof(aotx_tool_reply_body));
        }
        aotx_seam.apply.state_hash = hash;
        aotx_seam.apply.applied_count += (unsigned long long)lates;
    }
    if (live == 0u || late != 0u) {
        return;
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
    if (hold->auth == AOTX_AUTH_PENDING) {
        return;
    }

    /* skill_use gives the body of a skill of the catalog, on the device. */
    if (hold->tool == AOTX_TOOL_SKILL_USE) {
        aotx_tool_skill_body(hold);
        aotx_tool_done[slot] = 1u;
        atomicAdd(&aotx_tool_count.device_done, 1u);
        return;
    }
    /* A device tool that came in as a module runs in a node of its own. The node wrote its
     * output before this step, because the graph holds the node before this one. */
    if (aotx_catalog_is_module(hold->entry) != 0) {
        if (aotx_tool_module_reap(slot, hold) != 0) {
            aotx_tool_done[slot] = 1u;
            atomicAdd(&aotx_tool_count.device_done, 1u);
        }
        return;
    }

    /* A host tool waits for the answer of the feeder. A call to a tool the catalog does
     * not hold ends with the reason. */
    if (hold->tool == AOTX_TOOL_NONE) {
        hold->status = AOTX_TOOL_ERROR;
        hold->result_len = aotx_tool_put(hold->result, 0u,
                                         "this tool does not run in this version");
        aotx_tool_done[slot] = 1u;
        atomicAdd(&aotx_tool_count.device_done, 1u);
    }
}
