/* Purpose: Parse one command line and write the records the command asks for.
 * Owns: The command table, the help text and the counters of the parser.
 * Launch shape: One thread; the apply step calls the parser in slot order.
 * Lifetime: The whole run. */
#include "bus/bus.cuh"
#include "agent/transcript.cuh"
#include "catalog/console.cuh"
#include "cli/help.cuh"
#include "cli/prompt.cuh"
#include "kvcache/kvcache.cuh"
#include "mem/mem.cuh"
#include "model/model.cuh"
#include "model/load.cuh"
#include "sched/sched.cuh"
#include "settings/console.cuh"

/* One word of a command line: where it starts and how long it is. */
typedef struct aotx_cli_word {
    const unsigned char *at;
    unsigned int length;
} aotx_cli_word;

/* Move past the spaces from a position. */
static __device__ __forceinline__ unsigned int aotx_cli_space(const unsigned char *text,
                                                              unsigned int length,
                                                              unsigned int at)
{
    while (at < length && text[at] == ' ') {
        at += 1u;
    }
    return at;
}

/* Take the next word and move the position past it. */
static __device__ __forceinline__ aotx_cli_word aotx_cli_take(const unsigned char *text,
                                                              unsigned int length,
                                                              unsigned int *at)
{
    aotx_cli_word word;
    unsigned int start = aotx_cli_space(text, length, *at);
    unsigned int end = start;
    while (end < length && text[end] != ' ') {
        end += 1u;
    }
    word.at = text + start;
    word.length = end - start;
    *at = end;
    return word;
}

/* Report whether a word is the name. The match is exact, so a shorter name does not pass. */
static __device__ __forceinline__ int aotx_cli_is(aotx_cli_word word, const char *name)
{
    unsigned int length = aotx_cli_length(name);
    if (word.length != length) {
        return 0;
    }
    for (unsigned int i = 0u; i < length; ++i) {
        if (word.at[i] != (unsigned char)name[i]) {
            return 0;
        }
    }
    return 1;
}

/* Give the provenance value of a source word, or zero when the word is not a source. */
static __device__ __forceinline__ unsigned int aotx_cli_source(aotx_cli_word word)
{
    if (aotx_cli_is(word, "computed")) {
        return AOTX_PROV_COMPUTED;
    }
    if (aotx_cli_is(word, "fetched")) {
        return AOTX_PROV_FETCHED;
    }
    if (aotx_cli_is(word, "recalled")) {
        return AOTX_PROV_RECALLED;
    }
    if (aotx_cli_is(word, "testimony")) {
        return AOTX_PROV_TESTIMONY;
    }
    return 0u;
}

/* Give the name of a provenance value. A message that is not a finding gives a dash. */
__device__ const char *aotx_cli_source_name(unsigned int provenance)
{
    switch (provenance) {
    case AOTX_PROV_COMPUTED:  return "computed";
    case AOTX_PROV_FETCHED:   return "fetched";
    case AOTX_PROV_RECALLED:  return "recalled";
    case AOTX_PROV_TESTIMONY: return "testimony";
    default:                  return "-";
    }
}

/* Give the bus kind of a word, or zero when the word is not a kind. */
static __device__ __forceinline__ unsigned int aotx_cli_kind(aotx_cli_word word)
{
    if (aotx_cli_is(word, "finding")) {
        return AOTX_BUS_FINDING;
    }
    if (aotx_cli_is(word, "rank")) {
        return AOTX_BUS_RANK;
    }
    if (aotx_cli_is(word, "question")) {
        return AOTX_BUS_QUESTION;
    }
    if (aotx_cli_is(word, "answer")) {
        return AOTX_BUS_ANSWER;
    }
    if (aotx_cli_is(word, "handoff")) {
        return AOTX_BUS_HANDOFF;
    }
    if (aotx_cli_is(word, "cost")) {
        return AOTX_BUS_COST;
    }
    if (aotx_cli_is(word, "note")) {
        return AOTX_BUS_NOTE;
    }
    return 0u;
}

__device__ const char *aotx_cli_kind_name(unsigned int kind)
{
    switch (kind) {
    case AOTX_BUS_FINDING:  return "finding";
    case AOTX_BUS_RANK:     return "rank";
    case AOTX_BUS_QUESTION: return "question";
    case AOTX_BUS_ANSWER:   return "answer";
    case AOTX_BUS_HANDOFF:  return "handoff";
    case AOTX_BUS_COST:     return "cost";
    case AOTX_BUS_NOTE:     return "note";
    default:                return "-";
    }
}

/* Give the catalog entry of the role of a word, or AOTX_MODULE_SLOTS when the catalog
 * holds no installed role of that name. */
static __device__ __forceinline__ unsigned int aotx_cli_role_of(aotx_cli_word word)
{
    return aotx_catalog_find((const char *)word.at, word.length, AOTX_MODULE_ROLE);
}

/* Give the module kind of a word, or zero when the word is not a kind. */
static __device__ __forceinline__ unsigned int aotx_cli_module_kind(aotx_cli_word word)
{
    if (aotx_cli_is(word, "skill")) {
        return AOTX_MODULE_SKILL;
    }
    if (aotx_cli_is(word, "role")) {
        return AOTX_MODULE_ROLE;
    }
    if (aotx_cli_is(word, "tool")) {
        return AOTX_MODULE_TOOL;
    }
    return 0u;
}

/* Read a word as a decimal count. The return is 1 when every byte is a digit and the value
 * is under a million, which is above every table of this version. */
static __device__ __forceinline__ int aotx_cli_count_of(aotx_cli_word word,
                                                        unsigned int *value)
{
    unsigned int got = 0u;
    if (word.length == 0u || word.length > 7u) {
        return 0;
    }
    for (unsigned int i = 0u; i < word.length; ++i) {
        if (word.at[i] < (unsigned char)'0' || word.at[i] > (unsigned char)'9') {
            return 0;
        }
        got = got * 10u + (unsigned int)(word.at[i] - (unsigned char)'0');
    }
    *value = got;
    return 1;
}

static __device__ __noinline__ void aotx_cli_help(aotx_cli_out *out)
{
    for (unsigned int i = 0u; i < AOTX_CLI_HELP; ++i) {
        aotx_cli_say(out, aotx_cli_help_line(i));
        aotx_cli_console(out);
    }
}

/* Show one bus message: the writer, the kind, the source and the text. */
static __device__ __noinline__ void aotx_cli_message(aotx_cli_out *out,
                                                     unsigned long long seq)
{
    const aotx_bus_body *body = aotx_bus_body_of(seq);
    if (body == 0) {
        return;
    }
    /* The body sits one header past the start of its slot, so the header of the record is
     * at that distance below it. The header carries the writer that was stamped. */
    const aotx_record_header *header =
        (const aotx_record_header *)((const unsigned char *)body - AOTX_HEADER_BYTES);
    unsigned int length = body->text_len;
    if (length > AOTX_BUS_TEXT_BYTES) {
        length = AOTX_BUS_TEXT_BYTES;
    }
    aotx_cli_num(out, (unsigned long long)header->writer);
    aotx_cli_say(out, " ");
    aotx_cli_say(out, aotx_cli_kind_name(body->kind));
    aotx_cli_say(out, " ");
    aotx_cli_say(out, aotx_cli_source_name(body->provenance));
    aotx_cli_say(out, " ");
    aotx_cli_add(out, body->text, length);
    /* The ring can write over the record while the line is built. The second read of the
     * sequence states whether the bytes that went in the line are still the record's. */
    if (!aotx_cli_holds((const volatile aotx_record_header *)header, seq, AOTX_REC_BUS)) {
        aotx_cli_clear(out);
        return;
    }
    aotx_cli_console(out);
}

/* Show the most recent bus messages of the kinds in the mask, the newest first. */
static __device__ __noinline__ void aotx_cli_show_bus(aotx_cli_out *out,
                                                      unsigned int kind_mask)
{
    unsigned int count = aotx_bus_recent(kind_mask, AOTX_CLI_LIST, aotx_cli.recent);
    if (count == 0u) {
        aotx_cli_say(out, "bus: no messages");
        aotx_cli_console(out);
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_cli_message(out, aotx_cli.recent[i]);
    }
}

/* Show the memory regions and what they take of the reservation. */
static __device__ __noinline__ void aotx_cli_show_mem(aotx_cli_out *out)
{
    unsigned long long mapped = 0ull;
    aotx_cli_say(out, "mem: kind base bytes");
    aotx_cli_console(out);
    for (unsigned int i = 0u; i < aotx_mem_region_table.count; ++i) {
        const aotx_mem_region *region = &aotx_mem_region_table.region[i];
        mapped += region->bytes;
        aotx_cli_say(out, "  ");
        aotx_cli_num(out, (unsigned long long)region->kind);
        aotx_cli_say(out, " ");
        aotx_cli_num(out, region->base);
        aotx_cli_say(out, " ");
        aotx_cli_num(out, region->bytes);
        aotx_cli_console(out);
    }
    aotx_cli_say(out, "budget: total MB ");
    aotx_cli_num(out, aotx_mem_budget_table.total >> 20);
    aotx_cli_say(out, " free at boot ");
    aotx_cli_num(out, aotx_mem_budget_table.free_boot >> 20);
    aotx_cli_say(out, " free now ");
    aotx_cli_num(out, aotx_mem_budget_table.free_now >> 20);
    aotx_cli_console(out);
    aotx_cli_say(out, "budget: held MB ");
    aotx_cli_num(out, aotx_mem_budget_table.reserved >> 20);
    aotx_cli_say(out, " mapped ");
    aotx_cli_num(out, mapped >> 20);
    aotx_cli_say(out, " regions ");
    aotx_cli_num(out, (unsigned long long)aotx_mem_region_table.count);
    aotx_cli_console(out);
}

/* Show the page pool and the current limit of each live agent. */
static __device__ __noinline__ void aotx_cli_show_memory(aotx_cli_out *out)
{
    aotx_cli_say(out, "memory: pages free ");
    aotx_cli_num(out, (unsigned long long)(AOTX_KV_PAGES - aotx_kv.mapped_pages));
    aotx_cli_say(out, " of ");
    aotx_cli_num(out, (unsigned long long)AOTX_KV_PAGES);
    aotx_cli_console(out);
    for (unsigned int agent = 0u; agent < AOTX_SLOTS; ++agent) {
        if (aotx_agents.agent[agent].state == AOTX_AGENT_STATE_FREE) {
            continue;
        }
        aotx_cli_say(out, "  agent ");
        aotx_cli_num(out, (unsigned long long)agent);
        aotx_cli_say(out, " pages ");
        if (aotx_transcript[agent].pages == AOTX_TRANSCRIPT_AUTO) {
            aotx_cli_say(out, "auto ");
        }
        aotx_cli_num(out, (unsigned long long)aotx_transcript_page_limit(agent));
        aotx_cli_console(out);
    }
}

/* Show or change one transcript. A change takes effect when the next turn opens. */
static __device__ __noinline__ void aotx_cli_agent(aotx_cli_out *out,
                                                   const unsigned char *text,
                                                   unsigned int length,
                                                   unsigned int *position)
{
    aotx_cli_word number = aotx_cli_take(text, length, position);
    unsigned int agent = AOTX_SLOTS;
    if (!aotx_cli_count_of(number, &agent) || agent >= AOTX_SLOTS
        || aotx_agents.agent[agent].state == AOTX_AGENT_STATE_FREE) {
        aotx_cli_say(out, "agent: give the number of a live agent");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    aotx_cli_word action = aotx_cli_take(text, length, position);
    if (action.length == 0u) {
        const aotx_transcript_agent *hold = &aotx_transcript[agent];
        aotx_cli_say(out, "agent ");
        aotx_cli_num(out, (unsigned long long)agent);
        aotx_cli_say(out, ": turns ");
        aotx_cli_num(out, (unsigned long long)hold->count);
        aotx_cli_say(out, " hot ");
        aotx_cli_num(out, (unsigned long long)hold->hot);
        aotx_cli_say(out, " warm ");
        aotx_cli_num(out, (unsigned long long)hold->warm);
        aotx_cli_say(out, " summary ");
        aotx_cli_num(out, hold->summary_seq);
        aotx_cli_console(out);
        return;
    }
    if (aotx_cli_is(action, "compact")) {
        if (aotx_transcript_compact(agent) != 0) {
            aotx_cli_say(out, "agent: compaction was refused");
            aotx_cli_count.refused += 1u;
        } else {
            aotx_cli_say(out, "agent: compaction waits for its turn");
        }
        aotx_cli_console(out);
        return;
    }
    if (aotx_cli_is(action, "pages")) {
        aotx_cli_word value = aotx_cli_take(text, length, position);
        unsigned int pages = 0u;
        if (aotx_cli_is(value, "auto")) {
            pages = AOTX_TRANSCRIPT_AUTO;
        } else if (!aotx_cli_count_of(value, &pages)) {
            pages = 0u;
        }
        if (aotx_transcript_pages(agent, pages) != 0) {
            aotx_cli_say(out, "agent: give pages from 1 to ");
            aotx_cli_num(out, (unsigned long long)AOTX_KV_PAGES_EACH);
            aotx_cli_say(out, ", or auto");
            aotx_cli_count.refused += 1u;
        } else {
            aotx_cli_say(out, "agent: pages change at the next turn");
        }
        aotx_cli_console(out);
        return;
    }
    aotx_cli_say(out, "agent: give pages or compact");
    aotx_cli_console(out);
    aotx_cli_count.refused += 1u;
}

/* Make agents of a role and state the slots they took. The count is from 1 to 8. */
static __device__ __noinline__ void aotx_cli_spawn(aotx_cli_out *out, unsigned int role,
                                                   unsigned int count,
                                                   unsigned long long tick)
{
    unsigned int made = 0u;
    aotx_cli_say(out, "spawn: ");
    aotx_cli_say(out, aotx_cli_role_name(role));
    aotx_cli_say(out, " on slots");
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int slot = aotx_agent_spawn(role, 0u, tick);
        if (slot == ~0u) {
            aotx_cli_clear(out);
            aotx_cli_say(out, "spawn: the agent table is full");
            aotx_cli_console(out);
            aotx_cli_count.refused += 1u;
            return;
        }
        aotx_cli_say(out, " ");
        aotx_cli_num(out, (unsigned long long)slot);
        made += 1u;
    }
    if (made != 0u) {
        aotx_cli_console(out);
    }
}

/* Open a task for an agent or for a role. The text is every word after the name, and the
 * word verify at the end of the text asks a verifier to judge the result. */
static __device__ __noinline__ void aotx_cli_task(aotx_cli_out *out, aotx_cli_word name,
                                                  const unsigned char *text,
                                                  unsigned int length,
                                                  unsigned long long tick)
{
    unsigned int agent = ~0u;
    unsigned int role = aotx_cli_role_of(name);
    unsigned int slot = 0u;
    unsigned int verify = AOTX_VERIFY_NONE;

    if (role >= AOTX_MODULE_SLOTS) {
        if (!aotx_cli_count_of(name, &slot) || slot >= AOTX_SLOTS) {
            aotx_cli_say(out, "task: the agent or the role is not known; give a slot "
                              "below ");
            aotx_cli_num(out, (unsigned long long)AOTX_SLOTS);
            aotx_cli_say(out, ", or the name of a role of the catalog");
            aotx_cli_console(out);
            aotx_cli_count.refused += 1u;
            return;
        }
        if (aotx_agents.agent[slot].state == AOTX_AGENT_STATE_FREE) {
            aotx_cli_say(out, "task: no agent runs on that slot");
            aotx_cli_console(out);
            aotx_cli_count.refused += 1u;
            return;
        }
        if (aotx_agents.agent[slot].state != AOTX_AGENT_STATE_IDLE) {
            aotx_cli_say(out, "task: the agent is busy");
            aotx_cli_console(out);
            aotx_cli_count.refused += 1u;
            return;
        }
        agent = slot;
        role = aotx_agents.agent[slot].role;
    }
    /* The word verify at the end of the text asks for the check of a sibling. */
    static const char mark[] = "verify";
    unsigned int end = length;
    while (end > 0u && text[end - 1u] == (unsigned char)' ') {
        end -= 1u;
    }
    if (end >= 6u && (end == 6u || text[end - 7u] == (unsigned char)' ')) {
        unsigned int same = 1u;
        for (unsigned int i = 0u; i < 6u; ++i) {
            same &= (text[end - 6u + i] == (unsigned char)mark[i]) ? 1u : 0u;
        }
        if (same != 0u) {
            verify = AOTX_VERIFY_SIBLING;
            end -= 6u;
            while (end > 0u && text[end - 1u] == (unsigned char)' ') {
                end -= 1u;
            }
        }
    }
    length = end;
    if (length == 0u) {
        aotx_cli_say(out, "task: the text is missing");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    /* A task holds AOTX_SAY_BYTES of text. A text over that bound is cut, so the
     * parser refuses it and names the bound. The task entry refuses it as well. */
    if (length > AOTX_SAY_BYTES) {
        aotx_cli_say(out, "task: the text is too long; give ");
        aotx_cli_num(out, (unsigned long long)AOTX_SAY_BYTES);
        aotx_cli_say(out, " bytes at most");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    unsigned int task = aotx_task_open(agent, role, text, length, verify, tick);
    if (task == ~0u) {
        aotx_cli_say(out, "task: the task table is full");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    aotx_cli_say(out, "task: ");
    aotx_cli_num(out, (unsigned long long)task);
    aotx_cli_say(out, " for ");
    if (agent == ~0u) {
        aotx_cli_say(out, "the role ");
        aotx_cli_say(out, aotx_cli_role_name(role));
    } else {
        aotx_cli_say(out, "agent ");
        aotx_cli_num(out, (unsigned long long)agent);
    }
    if (verify != AOTX_VERIFY_NONE) {
        aotx_cli_say(out, " with a check");
    }
    aotx_cli_console(out);
}

/* Send a text to the language model on the slot of the conductor. The parser cannot launch
 * a kernel, so the wrapped bytes wait in the prompt table. Nodes of this tick tokenize them
 * and open the sequence. The line this command writes is the line the reply grows into. */
static __device__ __noinline__ void aotx_cli_say_text(aotx_cli_out *out,
                                                      const unsigned char *text,
                                                      unsigned int length,
                                                      unsigned long long tick)
{
    /* A replay of the journal sends every line again. This command then opens the
     * sequence again with the same values it took when the line was live. The token
     * records of the reply give that sequence its tokens and no draw is taken. */
    if (aotx_model[AOTX_MODEL_LANGUAGE].layers == 0u
        && aotx_model[AOTX_MODEL_LANGUAGE_Q4].layers == 0u) {
        aotx_cli_say(out, "say: no language model is loaded");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        aotx_say.refused += 1u;
        return;
    }
    if (aotx_agents.agent[AOTX_SAY_SLOT].state == AOTX_AGENT_STATE_FREE) {
        aotx_cli_say(out, "say: no conductor agent runs; give the spawn command");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        aotx_say.refused += 1u;
        return;
    }
    /* One reply runs at a time. The conductor holds the message until its next turn. The
     * console therefore looks at the message the agent holds. It also looks at the prompt
     * that waits for the tokenize step and at the reply that grows a line. */
    const aotx_say_slot *state = &aotx_say.slot[AOTX_SAY_SLOT];
    if (aotx_agent_gear[AOTX_SAY_SLOT].has_message != 0u || state->wanted != 0u
        || state->live != 0u
        || aotx_agents.agent[AOTX_SAY_SLOT].state != AOTX_AGENT_STATE_IDLE) {
        aotx_cli_say(out, "say: a reply runs; give the stop command to end it");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        aotx_say.refused += 1u;
        return;
    }
    /* The mailbox of an agent holds AOTX_SAY_BYTES. A text over that bound is cut,
     * so the parser refuses it and names the bound. The agent entry refuses it as well. */
    if (length > AOTX_SAY_BYTES) {
        aotx_cli_say(out, "say: the text is too long; give ");
        aotx_cli_num(out, (unsigned long long)AOTX_SAY_BYTES);
        aotx_cli_say(out, " bytes at most");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        aotx_say.refused += 1u;
        return;
    }
    /* The message goes to the conductor. That agent builds its next prompt from the
     * message and answers, and the reply of its slot grows the line below. */
    if (aotx_agent_message(AOTX_SAY_SLOT, text, length, tick) != 0) {
        aotx_cli_say(out, "say: a reply runs; give the stop command to end it");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        aotx_say.refused += 1u;
        return;
    }
    aotx_say.said += 1u;
    if (!aotx_cli_allow()) {
        return;
    }
    aotx_cli_say(out, "conductor: ");
    aotx_say.slot[AOTX_SAY_SLOT].at = aotx_console_start(out->text, out->at);
    aotx_say.slot[AOTX_SAY_SLOT].column = 1u;
    aotx_cli_clear(out);
}

/* End the reply of the conductor. A prompt that waits for the tokenize step is dropped; a
 * sequence that runs is stopped at the next tick. */
static __device__ __noinline__ void aotx_cli_stop(aotx_cli_out *out)
{
    aotx_say_slot *state = &aotx_say.slot[AOTX_SAY_SLOT];
    unsigned int seq = aotx_seqs.slot[AOTX_SAY_SLOT].state;
    if (state->wanted == 0u && (seq == AOTX_SEQ_STATE_FREE || state->live == 0u)) {
        aotx_cli_say(out, "stop: no reply runs");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    if (state->wanted != 0u) {
        state->wanted = 0u;
    } else {
        aotx_seq_stop(AOTX_SAY_SLOT);
    }
    aotx_say.stopped += 1u;
    aotx_cli_say(out, "stop: the reply ends");
    aotx_cli_console(out);
}

/* Show the counts of the last tick. The statistics record gives them, and the tick gives
 * the time the line was written. */
static __device__ __noinline__ void aotx_cli_show_stats(aotx_cli_out *out,
                                                           unsigned long long tick)
{
    unsigned long long seq = aotx_cli_last(AOTX_REC_STATS);
    aotx_cli_say(out, "stats: tick ");
    aotx_cli_num(out, tick);
    if (seq == 0ull) {
        aotx_cli_say(out, " no statistics record");
        aotx_cli_console(out);
        return;
    }
    const volatile aotx_record_header *header = aotx_cli_slot(seq);
    const volatile aotx_stats_body *body =
        (const volatile aotx_stats_body *)((const volatile unsigned char *)header
                                           + AOTX_HEADER_BYTES);
    unsigned long long tick_ns = body->tick_ns;
    unsigned long long records = body->records;
    unsigned long long inbound = body->inbound;
    if (!aotx_cli_holds(header, seq, AOTX_REC_STATS)) {
        aotx_cli_say(out, " no statistics record");
        aotx_cli_console(out);
        return;
    }
    aotx_cli_say(out, " time ");
    aotx_cli_num(out, tick_ns);
    aotx_cli_say(out, " ns records ");
    aotx_cli_num(out, records);
    aotx_cli_say(out, " inbound ");
    aotx_cli_num(out, inbound);
    aotx_cli_console(out);
}

/* Append a bus message and report the sequence, or report the refusal. */
static __device__ __noinline__ void aotx_cli_append(aotx_cli_out *out, unsigned int kind,
                                                       unsigned int provenance,
                                                       const unsigned char *text,
                                                       unsigned int length,
                                                       unsigned long long tick)
{
    /* The message is a record of the line, so it takes the allowance the same way. */
    if (!aotx_cli_allow()) {
        return;
    }
    unsigned long long seq = aotx_bus_append(AOTX_WRITER_CONSOLE, kind, provenance,
                                             (const char *)text, length, 0ull, 0ull, 0.0f,
                                             tick);
    if (seq == 0ull) {
        aotx_cli.written -= 1u;
        aotx_cli_say(out, "bus: the message was refused");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    aotx_cli_count.appended += 1u;
    aotx_cli_say(out, aotx_cli_kind_name(kind));
    aotx_cli_say(out, " ");
    aotx_cli_num(out, seq);
    aotx_cli_console(out);
}

/* Act on one line. Every exit of this function goes through aotx_cli_line, which states a
 * cut when the allowance stopped a record. */
static __device__ __noinline__ void aotx_cli_act(aotx_cli_out *out,
                                                 const unsigned char *text,
                                                 unsigned int length,
                                                 unsigned long long tick)
{
    unsigned int at = 0u;
    aotx_cli_word first = aotx_cli_take(text, length, &at);
    if (first.length == 0u) {
        return;
    }

    if (aotx_cli_is(first, "help")) {
        aotx_cli_help(out);
        return;
    }
    if (aotx_cli_is(first, "bus")) {
        aotx_cli_word kind = aotx_cli_take(text, length, &at);
        unsigned int mask = 0xfeu;
        if (kind.length != 0u) {
            unsigned int one = aotx_cli_kind(kind);
            if (one == 0u) {
                aotx_cli_say(out, "bus: the kind is not known");
                aotx_cli_console(out);
                aotx_cli_count.refused += 1u;
                return;
            }
            mask = 1u << one;
        }
        aotx_cli_show_bus(out, mask);
        return;
    }
    if (aotx_cli_is(first, "note")) {
        unsigned int start = aotx_cli_space(text, length, at);
        if (start >= length) {
            aotx_cli_say(out, "note: the text is missing");
            aotx_cli_console(out);
            aotx_cli_count.refused += 1u;
            return;
        }
        aotx_cli_append(out, AOTX_BUS_NOTE, 0u, text + start, length - start, tick);
        return;
    }
    if (aotx_cli_is(first, "finding")) {
        aotx_cli_word source = aotx_cli_take(text, length, &at);
        unsigned int provenance = aotx_cli_source(source);
        unsigned int start = aotx_cli_space(text, length, at);
        if (provenance == 0u || start >= length) {
            aotx_cli_say(out, "finding: give a source and a text; a source is computed, "
                                "fetched, recalled or testimony");
            aotx_cli_console(out);
            aotx_cli_count.refused += 1u;
            return;
        }
        aotx_cli_append(out, AOTX_BUS_FINDING, provenance, text + start, length - start,
                        tick);
        return;
    }
    if (aotx_cli_is(first, "mem")) {
        aotx_cli_show_mem(out);
        return;
    }
    if (aotx_cli_is(first, "memory")) {
        aotx_cli_show_memory(out);
        return;
    }
    if (aotx_cli_is(first, "say")) {
        unsigned int start = aotx_cli_space(text, length, at);
        if (start >= length) {
            aotx_cli_say(out, "say: the text is missing");
            aotx_cli_console(out);
            aotx_cli_count.refused += 1u;
            aotx_say.refused += 1u;
            return;
        }
        aotx_cli_say_text(out, text + start, length - start, tick);
        return;
    }
    if (aotx_cli_is(first, "stop")) {
        aotx_cli_stop(out);
        return;
    }
    if (aotx_cli_is(first, "spawn")) {
        aotx_cli_word name = aotx_cli_take(text, length, &at);
        aotx_cli_word number = aotx_cli_take(text, length, &at);
        unsigned int role = aotx_cli_role_of(name);
        unsigned int count = 1u;
        if (role >= AOTX_MODULE_SLOTS) {
            aotx_cli_say(out, "spawn: the role is not known; give the roles command for "
                              "the names");
            aotx_cli_console(out);
            aotx_cli_count.refused += 1u;
            return;
        }
        if (number.length != 0u
            && (!aotx_cli_count_of(number, &count) || count == 0u || count > 8u)) {
            aotx_cli_say(out, "spawn: give a count from 1 to 8");
            aotx_cli_console(out);
            aotx_cli_count.refused += 1u;
            return;
        }
        /* Slot 0 belongs to the role of the console and holds one agent. A second agent
         * of that role has no slot. */
        if (role == aotx_catalog.conductor
            && aotx_agents.agent[0].state != AOTX_AGENT_STATE_FREE) {
            aotx_cli_say(out, "spawn: an agent of that role runs already");
            aotx_cli_console(out);
            aotx_cli_count.refused += 1u;
            return;
        }
        aotx_cli_spawn(out, role, count, tick);
        return;
    }
    if (aotx_cli_is(first, "task")) {
        aotx_cli_word name = aotx_cli_take(text, length, &at);
        unsigned int start = aotx_cli_space(text, length, at);
        if (name.length == 0u) {
            aotx_cli_say(out, "task: give an agent or a role, and a text");
            aotx_cli_console(out);
            aotx_cli_count.refused += 1u;
            return;
        }
        aotx_cli_task(out, name, text + start, length - start, tick);
        return;
    }
    if (aotx_cli_is(first, "authorize") || aotx_cli_is(first, "refuse")) {
        aotx_cli_word number = aotx_cli_take(text, length, &at);
        unsigned int request = 0u;
        unsigned int granted = aotx_cli_is(first, "authorize") ? AOTX_CLI_GRANT
                                                               : AOTX_CLI_REFUSE;
        if (!aotx_cli_count_of(number, &request)) {
            request = 0u;
        }
        aotx_cli_answer(out, request, granted, tick);
        return;
    }
    if (aotx_cli_is(first, "agents")) {
        aotx_cli_show_agents(out);
        return;
    }
    if (aotx_cli_is(first, "agent")) {
        aotx_cli_agent(out, text, length, &at);
        return;
    }
    if (aotx_cli_is(first, "stats")) {
        aotx_cli_show_stats(out, tick);
        return;
    }
    if (aotx_cli_is(first, "settings")) {
        aotx_settings_show_command(out);
        return;
    }
    if (aotx_cli_is(first, "models")) {
        aotx_model_show_command(out);
        return;
    }
    if (aotx_cli_is(first, "model")) {
        aotx_cli_word action = aotx_cli_take(text, length, &at);
        aotx_cli_word role = aotx_cli_take(text, length, &at);
        aotx_cli_word name = aotx_cli_take(text, length, &at);
        aotx_cli_word extra = aotx_cli_take(text, length, &at);
        if (aotx_cli_is(action, "load")) {
            aotx_model_load_command(out, (const char *)role.at, role.length,
                                    (const char *)name.at, name.length, extra.length, tick);
            return;
        }
        if (aotx_cli_is(action, "fetch")) {
            aotx_cli_say(out, "model fetch: give this line at the terminal, the feeder takes it");
            aotx_cli_console(out);
            aotx_cli_count.refused += 1u;
            return;
        }
        aotx_cli_say(out, "model: give load or fetch");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    if (aotx_cli_is(first, "set")) {
        aotx_cli_word key = aotx_cli_take(text, length, &at);
        aotx_cli_word value = aotx_cli_take(text, length, &at);
        aotx_settings_set_command(out, (const char *)key.at, key.length,
                                  (const char *)value.at, value.length, tick);
        return;
    }
    if (aotx_cli_is(first, "modules") || aotx_cli_is(first, "skills")
        || aotx_cli_is(first, "roles") || aotx_cli_is(first, "tools")) {
        unsigned int kind = 0u;
        if (aotx_cli_is(first, "skills")) {
            kind = AOTX_MODULE_SKILL;
        } else if (aotx_cli_is(first, "roles")) {
            kind = AOTX_MODULE_ROLE;
        } else if (aotx_cli_is(first, "tools")) {
            kind = AOTX_MODULE_TOOL;
        } else {
            aotx_cli_word word = aotx_cli_take(text, length, &at);
            if (word.length != 0u) {
                kind = aotx_cli_module_kind(word);
                if (kind == 0u) {
                    aotx_cli_say(out, "modules: the kind is not known; give skill, role "
                                      "or tool");
                    aotx_cli_console(out);
                    aotx_cli_count.refused += 1u;
                    return;
                }
            }
        }
        aotx_catalog_modules_command(out, kind);
        return;
    }
    if (aotx_cli_is(first, "module")) {
        aotx_cli_word name = aotx_cli_take(text, length, &at);
        if (name.length == 0u) {
            aotx_cli_say(out, "module: give the name of one module");
            aotx_cli_console(out);
            aotx_cli_count.refused += 1u;
            return;
        }
        aotx_catalog_module_command(out, (const char *)name.at, name.length);
        return;
    }
    if (aotx_cli_is(first, "remove")) {
        aotx_cli_word name = aotx_cli_take(text, length, &at);
        aotx_catalog_remove_command(out, (const char *)name.at, name.length, tick);
        return;
    }
    if (aotx_cli_is(first, "import")) {
        aotx_cli_word path = aotx_cli_take(text, length, &at);
        aotx_cli_word mark = aotx_cli_take(text, length, &at);
        /* The feeder takes an import line of its own standard input and reads the
         * directory. A line that reaches the device came from a surface the feeder does
         * not read, such as the window. It may also be the report of an import the feeder
         * refused, which carries one word after the path. */
        if (aotx_cli_is(mark, "refused:")) {
            aotx_catalog_import_said(out, (const char *)text, length, tick);
            return;
        }
        aotx_catalog_import_command(out, (const char *)path.at, path.length, tick);
        return;
    }
    if (aotx_cli_is(first, "quit")) {
        /* A replay of the journal sends every key again. The run must not close on a quit
         * that a past run typed, so the flag stands only when no replay runs. */
        if (aotx_seam.replaying != 0ull) {
            aotx_cli_say(out, "quit: the run holds until the replay ends");
            aotx_cli_console(out);
            return;
        }
        aotx_cli_quit = 1u;
        aotx_cli_say(out, "quit: the run stops");
        aotx_cli_console(out);
        return;
    }

    aotx_cli_say(out, "unknown command: ");
    aotx_cli_add(out, (const char *)first.at, first.length);
    aotx_cli_say(out, "; type help");
    aotx_cli_console(out);
    aotx_cli_count.unknown += 1u;
}

__device__ void aotx_cli_line(const unsigned char *text, unsigned int length,
                              unsigned long long tick)
{
    aotx_cli_out *out = &aotx_cli.out;
    /* The input records carry the complete line. The derived command record remains one
     * record, as it was before lines gained parts, and names the start of that line. */
    unsigned int shown = (length > AOTX_BODY_BYTES) ? AOTX_BODY_BYTES : length;
    aotx_cli.written = 1u;
    aotx_cli.cut = 0u;
    aotx_seam_write(AOTX_WRITER_CONSOLE, AOTX_CLASS_B, AOTX_REC_COMMAND, 0u, text, shown);
    aotx_cli_count.commands += 1u;
    aotx_cli_clear(out);

    aotx_cli_act(out, text, length, tick);

    /* The last sequence of the allowance states what the allowance stopped. */
    if (aotx_cli.cut != 0u) {
        aotx_cli_say(out, "output cut at ");
        aotx_cli_num(out, (unsigned long long)aotx_cli.written);
        aotx_cli_say(out, " lines");
        aotx_console_write(out->text, out->at);
        aotx_cli_clear(out);
        aotx_cli.written += 1u;
    }
}
