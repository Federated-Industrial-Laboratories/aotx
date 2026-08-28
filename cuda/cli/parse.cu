/* Purpose: Parse one command line and write the records the command asks for.
 * Owns: The command table, the help text and the counters of the parser.
 * Launch shape: One thread; the apply step calls the parser in slot order.
 * Lifetime: The whole run. */
#include "bus/bus.cuh"
#include "cli/cli.cuh"
#include "mem/mem.cuh"
#include "model/model.cuh"
#include "sched/sched.cuh"

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

/* One line of the help text. The lines are in the register the documentation uses. */
static __device__ __forceinline__ const char *aotx_cli_help_line(unsigned int index)
{
    switch (index) {
    case 0u: return "commands:";
    case 1u: return "  help                     show these lines";
    case 2u: return "  bus [kind]               show the last bus messages of a kind";
    case 3u: return "  note <text>              put a note on the bus";
    case 4u: return "  finding <source> <text>  put a finding on the bus";
    case 5u: return "      a source is computed, fetched, recalled or testimony";
    case 6u: return "  say <text>               send a message to the language model";
    case 7u: return "  stop                     end the reply that runs";
    case 8u: return "  mem                      show the memory regions and the budget";
    case 9u: return "  agents                   show the agents";
    case 10u: return "  stats                    show the counts of the last tick";
    default: return "  quit                     stop the run";
    }
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

/* Show one row for each sequence slot that is not free. A sequence is the decode half of an
 * agent. A row holds the slot, the role and the state. It then holds the tokens the key
 * value cache holds, the reply tokens, and the reply tokens each second. The rate comes
 * from two samples of the window, which carry the device clock. */
static __device__ __noinline__ void aotx_cli_show_agents(aotx_cli_out *out)
{
    aotx_cli_say(out, "agents: slot role state position reply rate");
    aotx_cli_console(out);
    unsigned int live = 0u;
    for (unsigned int slot = 0u; slot < AOTX_SEQ_SLOTS; ++slot) {
        const aotx_seq *seq = &aotx_seqs.slot[slot];
        if (seq->state == AOTX_SEQ_STATE_FREE) {
            continue;
        }
        live += 1u;
        aotx_cli_say(out, "  ");
        aotx_cli_num(out, (unsigned long long)slot);
        aotx_cli_say(out, " ");
        aotx_cli_say(out, aotx_say_role_name(seq->role));
        aotx_cli_say(out, " ");
        aotx_cli_say(out, aotx_say_state_name(seq->state));
        aotx_cli_say(out, " ");
        aotx_cli_num(out, (unsigned long long)seq->held);
        aotx_cli_say(out, " ");
        aotx_cli_num(out, (unsigned long long)seq->sampled);
        aotx_cli_say(out, " ");
        aotx_cli_num(out, aotx_say_rate(slot));
        aotx_cli_console(out);
    }
    if (live == 0u) {
        aotx_cli_say(out, "  no agents");
        aotx_cli_console(out);
    }
}

/* Send a text to the language model on the slot of the conductor. The parser cannot launch
 * a kernel, so the wrapped bytes wait in the prompt table. Nodes of this tick tokenize them
 * and open the sequence. The line this command writes is the line the reply grows into. */
static __device__ __noinline__ void aotx_cli_say_text(aotx_cli_out *out,
                                                      const unsigned char *text,
                                                      unsigned int length)
{
    const aotx_say_slot *state = &aotx_say.slot[AOTX_SAY_SLOT];
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
    if (aotx_seqs.slot[AOTX_SAY_SLOT].state != AOTX_SEQ_STATE_FREE || state->wanted != 0u
        || state->live != 0u) {
        aotx_cli_say(out, "say: a reply runs; give the stop command to end it");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        aotx_say.refused += 1u;
        return;
    }
    if (aotx_say_ask(AOTX_SAY_SLOT, text, length) != 0) {
        aotx_cli_say(out, "say: the text is too long");
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
    if (aotx_cli_is(first, "say")) {
        unsigned int start = aotx_cli_space(text, length, at);
        if (start >= length) {
            aotx_cli_say(out, "say: the text is missing");
            aotx_cli_console(out);
            aotx_cli_count.refused += 1u;
            aotx_say.refused += 1u;
            return;
        }
        aotx_cli_say_text(out, text + start, length - start);
        return;
    }
    if (aotx_cli_is(first, "stop")) {
        aotx_cli_stop(out);
        return;
    }
    if (aotx_cli_is(first, "agents")) {
        aotx_cli_show_agents(out);
        return;
    }
    if (aotx_cli_is(first, "stats")) {
        aotx_cli_show_stats(out, tick);
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
    if (length > AOTX_BODY_BYTES) {
        length = AOTX_BODY_BYTES;
    }
    /* The command record is the first record of the allowance of this line. */
    aotx_cli.written = 1u;
    aotx_cli.cut = 0u;
    aotx_seam_write(AOTX_WRITER_CONSOLE, AOTX_CLASS_B, AOTX_REC_COMMAND, 0u, text, length);
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
