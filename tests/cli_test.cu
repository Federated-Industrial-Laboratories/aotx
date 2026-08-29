/* Purpose: Check the line editor and the command parser at one line and at AOTX_SLOTS.
 * Owns: The key fixtures, the record probe and the counts of the cases.
 * Launch shape: One thread feeds the editor; a grid gathers the records.
 * Lifetime: One run of the test program. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cuda.h>

#include "boot/check.h"
#include "bus/bus.cuh"
#include "cli/agents.cuh"
#include "cli/prompt.cuh"
#include "mem/mem.cuh"
#include "sched/sched.cuh"
#include "seam/seam.cuh"
#include "settings/settings.cuh"

#include "catalog_feed.h"

#define AOTX_TEST_KEYS    4096u
#define AOTX_TEST_FOUND   4096u
/* The batch of every case set. The profile gives the count, so a case runs at N=1 and at
 * N=AOTX_SLOTS on every profile. */
#define AOTX_TEST_BATCH   AOTX_SLOTS

/* One record the probe found, with the fields the cases compare. */
typedef struct aotx_test_record {
    unsigned long long seq;
    unsigned int writer;
    unsigned int flags;
    unsigned int length;
    unsigned char body[AOTX_BODY_BYTES];
} aotx_test_record;

static unsigned int aotx_test_applied;
static unsigned int aotx_test_failed;

/* The side that reads one number setting, on the host side. The list comes from keys.h. */
static unsigned int aotx_setting_side_of(unsigned int index)
{
    switch (index) {
#define AOTX_SETTING_ONE(symbol, name, side, effect, ...) \
    case symbol: return AOTX_SETTING_SIDE_##side;
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    default: return 0u;
    }
}

static void aotx_test_check(int ok, const char *what)
{
    aotx_test_applied += 1u;
    if (!ok) {
        aotx_test_failed += 1u;
        printf("cli: FAILED %s\n", what);
    }
}

/* Feed a run of key events to the editor, one after the other, as the apply does. */
__global__ void aotx_test_type(const aotx_key_body *keys, unsigned int count)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_cli_key(&keys[i], aotx_time_tick);
    }
}

/* The most records that one line wrote. The allowance bounds this figure. */
__device__ unsigned int aotx_test_worst;

/* Parse a run of lines, one after the other, as the apply does. */
__global__ void aotx_test_parse(const unsigned char *lines, const unsigned int *lengths,
                                unsigned int count)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_cli_line(lines + i * AOTX_BODY_BYTES, lengths[i], aotx_time_tick);
        if (aotx_cli.written > aotx_test_worst) {
            aotx_test_worst = aotx_cli.written;
        }
    }
}

/* Append bus messages so that a list command has more of them than one line may show. */
/* The tick commit node writes the records of the set lines; this stands in for it. */
__global__ void aotx_test_commit_settings(void)
{
    if (threadIdx.x == 0u && blockIdx.x == 0u) {
        aotx_settings_commit(aotx_time_tick);
    }
}

__global__ void aotx_test_messages(unsigned int count)
{
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = lane; i < count; i += stride) {
        char text[64];
        unsigned int at = 0u;
        const char *head = "bus fill ";
        for (unsigned int b = 0u; head[b] != '\0'; ++b) {
            text[at++] = head[b];
        }
        at += aotx_text_utoa(i, text + at, 64u - at);
        aotx_bus_append(AOTX_WRITER_CONSOLE, AOTX_BUS_NOTE, 0u, text, at, 0ull, 0ull, 0.0f,
                        aotx_time_tick);
    }
}

/* Gather every record of a type from the device ring. One thread takes one sequence. */
__global__ void aotx_test_gather(unsigned int type, aotx_test_record *out, unsigned int max,
                                 unsigned int *count)
{
    unsigned long long tail = aotx_seam.dev.tail;
    unsigned long long stride = (unsigned long long)(gridDim.x * blockDim.x);
    for (unsigned long long seq = 1ull + (unsigned long long)(blockIdx.x * blockDim.x
                                                              + threadIdx.x);
         seq <= tail; seq += stride) {
        const volatile aotx_record_header *header = aotx_cli_slot(seq);
        if (!aotx_cli_holds(header, seq, type)) {
            continue;
        }
        unsigned int at = atomicAdd(count, 1u);
        if (at >= max) {
            continue;
        }
        unsigned int length = header->body_len;
        if (length > AOTX_BODY_BYTES) {
            length = AOTX_BODY_BYTES;
        }
        out[at].seq = seq;
        out[at].writer = header->writer;
        out[at].flags = header->flags;
        out[at].length = length;
        const volatile unsigned char *body = (const volatile unsigned char *)header
                                           + AOTX_HEADER_BYTES;
        for (unsigned int b = 0u; b < length; ++b) {
            out[at].body[b] = body[b];
        }
    }
}

static int aotx_test_order(const void *left, const void *right)
{
    const aotx_test_record *a = (const aotx_test_record *)left;
    const aotx_test_record *b = (const aotx_test_record *)right;
    return (a->seq < b->seq) ? -1 : ((a->seq > b->seq) ? 1 : 0);
}

/* Copy every record of a type out of the device ring, in sequence order. */
static unsigned int aotx_test_records(unsigned int type, aotx_test_record *out,
                                      unsigned int max)
{
    aotx_test_record *device = NULL;
    unsigned int *counter = NULL;
    unsigned int count = 0u;
    aotx_check_runtime(cudaMalloc(&device, (size_t)max * sizeof *device), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&counter, sizeof *counter), "cudaMalloc");
    aotx_check_runtime(cudaMemset(counter, 0, sizeof *counter), "cudaMemset");
    aotx_test_gather<<<64, 128>>>(type, device, max, counter);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&count, counter, sizeof count, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    if (count > max) {
        count = max;
    }
    aotx_check_runtime(cudaMemcpy(out, device, (size_t)count * sizeof *out,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    cudaFree(device);
    cudaFree(counter);
    qsort(out, count, sizeof *out, aotx_test_order);
    return count;
}

/* Report whether a record body is the text. */
static int aotx_test_same(const aotx_test_record *record, const char *text)
{
    size_t length = strlen(text);
    return record->length == (unsigned int)length
        && memcmp(record->body, text, length) == 0;
}

/* Build key events. A code point above zero is a character event; a key code is a key
 * event. Both carry the press action, which is what the window sends. */
static unsigned int aotx_test_text(aotx_key_body *keys, unsigned int at, const char *text)
{
    for (unsigned int i = 0u; text[i] != '\0'; ++i) {
        keys[at].key = 0u;
        keys[at].codepoint = (unsigned int)(unsigned char)text[i];
        keys[at].action = AOTX_CLI_PRESS;
        keys[at].mods = 0u;
        at += 1u;
    }
    return at;
}

static unsigned int aotx_test_key(aotx_key_body *keys, unsigned int at, unsigned int code)
{
    keys[at].key = code;
    keys[at].codepoint = 0u;
    keys[at].action = AOTX_CLI_PRESS;
    keys[at].mods = 0u;
    return at + 1u;
}

/* A release event changes nothing, so the editor must drop it. */
static unsigned int aotx_test_release(aotx_key_body *keys, unsigned int at, unsigned int code)
{
    keys[at].key = code;
    keys[at].codepoint = 0u;
    keys[at].action = AOTX_CLI_RELEASE;
    keys[at].mods = 0u;
    return at + 1u;
}

static void aotx_test_send(const aotx_key_body *keys, unsigned int count)
{
    aotx_key_body *device = NULL;
    aotx_check_runtime(cudaMalloc(&device, (size_t)count * sizeof *device), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device, keys, (size_t)count * sizeof *device,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_test_type<<<1, 1>>>(device, count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaFree(device);
}

static void aotx_test_lines(const char lines[][AOTX_BODY_BYTES], const unsigned int *lengths,
                            unsigned int count)
{
    unsigned char *device = NULL;
    unsigned int *sizes = NULL;
    aotx_check_runtime(cudaMalloc(&device, (size_t)count * AOTX_BODY_BYTES), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&sizes, (size_t)count * sizeof *sizes), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device, lines, (size_t)count * AOTX_BODY_BYTES,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(sizes, lengths, (size_t)count * sizeof *sizes,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_test_parse<<<1, 1>>>(device, sizes, count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaFree(device);
    cudaFree(sizes);
}

static unsigned int aotx_test_written(void)
{
    aotx_cli_state *state = (aotx_cli_state *)malloc(sizeof *state);
    unsigned int written = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(state, aotx_cli, sizeof *state),
                       "cudaMemcpyFromSymbol");
    written = state->written;
    free(state);
    return written;
}

static aotx_cli_counts aotx_test_counts(void)
{
    aotx_cli_counts counts;
    aotx_check_runtime(cudaMemcpyFromSymbol(&counts, aotx_cli_count, sizeof counts),
                       "cudaMemcpyFromSymbol");
    return counts;
}

static unsigned int aotx_test_quit(void)
{
    unsigned int quit = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&quit, aotx_cli_quit, sizeof quit),
                       "cudaMemcpyFromSymbol");
    return quit;
}

/* Build one line of a batch with an edit that ends at the same text. The edit kind comes
 * from the position, so the batch covers the backspace, the cursor moves and the delete. */
static unsigned int aotx_test_edited(aotx_key_body *keys, unsigned int at, const char *text,
                                     unsigned int kind)
{
    char rest[AOTX_BODY_BYTES];
    char first[2];
    switch (kind) {
    case 0u:
        at = aotx_test_text(keys, at, text);
        break;
    case 1u:
        at = aotx_test_text(keys, at, text);
        at = aotx_test_text(keys, at, "X");
        at = aotx_test_key(keys, at, AOTX_CLI_KEY_BACKSPACE);
        break;
    case 2u:
        /* Type every byte but the first, go to the start, then type the first byte. */
        snprintf(rest, sizeof rest, "%s", text + 1);
        first[0] = text[0];
        first[1] = '\0';
        at = aotx_test_text(keys, at, rest);
        at = aotx_test_key(keys, at, AOTX_CLI_KEY_HOME);
        at = aotx_test_text(keys, at, first);
        at = aotx_test_key(keys, at, AOTX_CLI_KEY_END);
        break;
    default:
        /* Type a byte that does not belong, go back to it, and delete it. */
        at = aotx_test_text(keys, at, "Z");
        at = aotx_test_text(keys, at, text);
        at = aotx_test_key(keys, at, AOTX_CLI_KEY_HOME);
        at = aotx_test_key(keys, at, AOTX_CLI_KEY_DELETE);
        at = aotx_test_key(keys, at, AOTX_CLI_KEY_END);
        break;
    }
    return aotx_test_key(keys, at, AOTX_CLI_KEY_ENTER);
}

/* Type a batch of lines with distinct text, then compare the command records. */
static void aotx_test_batch(unsigned int count)
{
    aotx_key_body *keys = (aotx_key_body *)malloc(AOTX_TEST_KEYS * 16u * sizeof *keys);
    aotx_test_record *found = (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);
    char expect[AOTX_TEST_BATCH][64];
    unsigned int at = 0u;
    unsigned int before = 0u;
    unsigned int matched = 0u;
    unsigned int commands = 0u;

    before = aotx_test_records(AOTX_REC_COMMAND, found, AOTX_TEST_FOUND);
    for (unsigned int i = 0u; i < count; ++i) {
        snprintf(expect[i], sizeof expect[i], "note item %u of %u", i, count);
        at = aotx_test_edited(keys, at, expect[i], i % 4u);
    }
    aotx_test_send(keys, at);
    commands = aotx_test_records(AOTX_REC_COMMAND, found, AOTX_TEST_FOUND);

    for (unsigned int i = 0u; i < count; ++i) {
        if (before + i < commands && aotx_test_same(&found[before + i], expect[i])) {
            matched += 1u;
        } else if (before + i < commands) {
            printf("cli: line %u is '%.*s', not '%s'\n", i,
                   (int)found[before + i].length, found[before + i].body, expect[i]);
        }
    }
    aotx_test_check(commands == before + count, "the batch made one command for each line");
    aotx_test_check(matched == count, "every line of the batch has its text");
    printf("cli: batch of %u lines, %u keys, %u lines matched\n", count, at, matched);
    free(keys);
    free(found);
}

/* The history holds the lines, and up and down move through them. */
static void aotx_test_history(void)
{
    aotx_key_body keys[512];
    aotx_test_record *found = (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);
    unsigned int at = 0u;
    unsigned int before = aotx_test_records(AOTX_REC_COMMAND, found, AOTX_TEST_FOUND);
    unsigned int after = 0u;

    at = aotx_test_text(keys, at, "note alpha");
    at = aotx_test_key(keys, at, AOTX_CLI_KEY_ENTER);
    at = aotx_test_text(keys, at, "note beta");
    at = aotx_test_key(keys, at, AOTX_CLI_KEY_ENTER);
    /* Two steps back is the first line, which the enter sends again. */
    at = aotx_test_key(keys, at, AOTX_CLI_KEY_UP);
    at = aotx_test_key(keys, at, AOTX_CLI_KEY_UP);
    at = aotx_test_key(keys, at, AOTX_CLI_KEY_ENTER);
    /* One step back and one step forward gives an empty line, which sends nothing. */
    at = aotx_test_key(keys, at, AOTX_CLI_KEY_UP);
    at = aotx_test_key(keys, at, AOTX_CLI_KEY_DOWN);
    at = aotx_test_release(keys, at, AOTX_CLI_KEY_UP);
    at = aotx_test_key(keys, at, AOTX_CLI_KEY_ENTER);
    aotx_test_send(keys, at);

    after = aotx_test_records(AOTX_REC_COMMAND, found, AOTX_TEST_FOUND);
    aotx_test_check(after == before + 3u, "the history sends three lines and no more");
    aotx_test_check(after > before + 2u && aotx_test_same(&found[before], "note alpha"),
                    "the first line is the first command");
    aotx_test_check(after > before + 2u && aotx_test_same(&found[before + 1u], "note beta"),
                    "the second line is the second command");
    aotx_test_check(after > before + 2u && aotx_test_same(&found[before + 2u], "note alpha"),
                    "the recalled line is the third command");
    free(found);
}

/* Every command, and the refusals each one gives. */
static void aotx_test_commands(void)
{
    static const char *cases[] = {
        "help", "bus", "note hello there", "note", "finding computed a fact",
        "finding banana a fact", "finding", "mem", "agents", "stats",
        "bus note", "bus wibble", "wibble", "quit"
    };
    const unsigned int count = (unsigned int)(sizeof cases / sizeof cases[0]);
    char (*lines)[AOTX_BODY_BYTES] = (char (*)[AOTX_BODY_BYTES])
        malloc((size_t)count * AOTX_BODY_BYTES);
    unsigned int *lengths = (unsigned int *)malloc((size_t)count * sizeof *lengths);
    aotx_test_record *found = (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);
    aotx_cli_counts before = aotx_test_counts();
    aotx_cli_counts after;
    unsigned int console = 0u;
    unsigned int bus = 0u;
    unsigned int findings = 0u;
    unsigned int notes = 0u;
    unsigned int unknown = 0u;

    memset(lines, 0, (size_t)count * AOTX_BODY_BYTES);
    for (unsigned int i = 0u; i < count; ++i) {
        lengths[i] = (unsigned int)strlen(cases[i]);
        memcpy(lines[i], cases[i], lengths[i]);
    }
    unsigned int console_before = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    unsigned int bus_before = aotx_test_records(AOTX_REC_BUS, found, AOTX_TEST_FOUND);
    aotx_test_lines(lines, lengths, count);
    console = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    for (unsigned int i = console_before; i < console; ++i) {
        if (found[i].length > 16u && memcmp(found[i].body, "unknown command:", 16u) == 0) {
            unknown += 1u;
        }
    }
    bus = aotx_test_records(AOTX_REC_BUS, found, AOTX_TEST_FOUND);
    for (unsigned int i = bus_before; i < bus; ++i) {
        const aotx_bus_body *body = (const aotx_bus_body *)found[i].body;
        if (body->kind == AOTX_BUS_FINDING) {
            findings += 1u;
        }
        if (body->kind == AOTX_BUS_NOTE) {
            notes += 1u;
        }
        aotx_test_check(found[i].writer == AOTX_WRITER_CONSOLE,
                        "the append stamps the console writer");
    }
    after = aotx_test_counts();

    aotx_test_check(after.commands == before.commands + count,
                    "the parser counts every line");
    aotx_test_check(console > console_before + AOTX_CLI_HELP,
                    "the help writes its lines to the console");
    aotx_test_check(unknown == 1u, "an unknown word gives one refusal line");
    aotx_test_check(after.unknown == before.unknown + 1u, "the parser counts the unknown word");
    aotx_test_check(notes == 1u, "one note reaches the bus");
    aotx_test_check(findings == 1u, "one finding reaches the bus");
    aotx_test_check(bus == bus_before + 2u, "a refused message appends nothing");
    aotx_test_check(after.refused == before.refused + 4u,
                    "the missing text, the bad source, the missing source and the bad kind "
                    "are refused");
    aotx_test_check(after.appended == before.appended + 2u, "two messages were appended");
    aotx_test_check(aotx_test_quit() == 1u, "the quit command sets the flag");
    printf("cli: %u command lines, %u console lines, %u bus messages\n",
           count, console - console_before, bus - bus_before);
    free(lines);
    free(lengths);
    free(found);
}

/* Put one record in the inbound ring, as the feeder does, and publish the head. */
static void aotx_test_put(const aotx_seam_rings *rings, unsigned long long boot_id,
                          unsigned long long index, unsigned int writer, unsigned int flags,
                          const char *line, unsigned int length)
{
    aotx_inbound_preamble *preamble = (aotx_inbound_preamble *)rings->inbound_map;
    unsigned char *at = rings->inbound_map + sizeof(aotx_inbound_preamble)
                      + (index & (AOTX_INBOUND_SLOTS - 1ull)) * AOTX_SLOT_BYTES;
    aotx_record_header *record = (aotx_record_header *)at;
    memset(at, 0, AOTX_SLOT_BYTES);
    record->magic = AOTX_WIRE_MAGIC;
    record->layout = (unsigned short)AOTX_WIRE_LAYOUT;
    record->header_bytes = (unsigned short)AOTX_HEADER_BYTES;
    record->boot_id = boot_id;
    record->writer = writer;
    record->cls = (unsigned char)AOTX_CLASS_A;
    record->type = (unsigned char)AOTX_REC_INPUT_LINE;
    record->flags = (unsigned short)flags;
    record->body_len = length;
    memcpy(at + AOTX_HEADER_BYTES, line, length);
    __atomic_store_n(&record->seq, index + 1ull, __ATOMIC_RELEASE);
    __atomic_store_n(&preamble->head, index + 1ull, __ATOMIC_RELEASE);
}

/* Read the console buffer of the device. */
static void aotx_test_console_state(aotx_console_state *state)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(state, aotx_console, sizeof *state),
                       "cudaMemcpyFromSymbol");
}

/* Give the line of a line number from a copy of the console buffer. */
static const aotx_console_line *aotx_test_at(const aotx_console_state *state,
                                             unsigned long long at)
{
    const aotx_console_line *line = &state->line[(at - 1ull) & (AOTX_CONSOLE_LINES - 1u)];
    return (line->seq == at) ? line : NULL;
}

static int aotx_test_says(const aotx_console_line *line, const char *text)
{
    unsigned int length = (unsigned int)strlen(text);
    return line != NULL && line->length == length
        && memcmp(line->text, text, length) == 0;
}

/* One line through the inbound ring and the apply, then one replayed line. The apply writes
 * the echo of a line that comes in. The echo goes in the console buffer before the answer of
 * the parser. A replayed line has no echo. The console buffer of a run that restores holds
 * the answers the parser makes again, and no echo. */
static void aotx_test_echo(const aotx_seam_rings *rings, unsigned long long boot_id)
{
    aotx_console_state *state = (aotx_console_state *)malloc(sizeof *state);
    aotx_test_record *found = (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);
    unsigned long long before = 0ull;
    unsigned long long after = 0ull;
    unsigned long long replayed = 0ull;
    unsigned int echoes = 0u;
    unsigned int records = 0u;

    aotx_test_console_state(state);
    before = state->count;
    aotx_test_put(rings, boot_id, 0ull, AOTX_WRITER_FEEDER, 0u, "note alpha", 10u);
    aotx_sched_tick_start<<<1, 1>>>(0ull);
    aotx_seam_apply_inbound<<<AOTX_APPLY_BLOCKS, AOTX_APPLY_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_console_state(state);
    after = state->count;

    aotx_test_check(after >= before + 2ull, "the line gives an echo and an answer");
    aotx_test_check(aotx_test_says(aotx_test_at(state, before + 1ull), "> note alpha"),
                    "the echo of the line is the first line of the buffer");
    aotx_test_check(aotx_test_at(state, before + 2ull) != NULL
                    && !aotx_test_says(aotx_test_at(state, before + 2ull), "> note alpha"),
                    "the answer of the parser comes after the echo");

    /* The same line again, with the mark of a replay. */
    aotx_test_put(rings, boot_id, 1ull, AOTX_WRITER_RESTORE, AOTX_FLAG_REPLAYED,
                  "note beta", 9u);
    aotx_sched_tick_start<<<1, 1>>>(0ull);
    aotx_seam_apply_inbound<<<AOTX_APPLY_BLOCKS, AOTX_APPLY_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_console_state(state);
    replayed = state->count;
    for (unsigned long long at = after + 1ull; at <= replayed; ++at) {
        const aotx_console_line *line = aotx_test_at(state, at);
        if (line != NULL && line->length >= 2u && line->text[0] == (unsigned char)'>') {
            echoes += 1u;
        }
    }
    aotx_test_check(replayed >= after + 1ull, "the replayed line gives an answer");
    aotx_test_check(echoes == 0u, "the replayed line gives no echo");

    records = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    for (unsigned int i = 0u; i < records; ++i) {
        if (found[i].length == 12u && memcmp(found[i].body, "> note alpha", 12u) == 0) {
            echoes += 1u;
        }
        if (found[i].length == 11u && memcmp(found[i].body, "> note beta", 11u) == 0) {
            echoes += 8u;
        }
    }
    aotx_test_check(echoes == 1u,
                    "the ring holds the echo record of the line and none of the replay");
    printf("cli: the apply gave %llu console lines for a line and %llu for a replayed line\n",
           after - before, replayed - after);
    free(state);
    free(found);
}

/* A line writes at most the records the tick start reserves for it. A list of more bus
 * messages than one line may show is cut, and the last record says so. */
static void aotx_test_allowance(void)
{
    aotx_test_record *found = (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);
    char (*lines)[AOTX_BODY_BYTES] = (char (*)[AOTX_BODY_BYTES]) malloc(AOTX_BODY_BYTES);
    unsigned int length = 3u;
    unsigned int console = 0u;
    unsigned int written = 0u;
    unsigned int worst = 0u;

    aotx_test_messages<<<4, 32>>>(AOTX_CLI_LIST + 8u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    memset(lines, 0, AOTX_BODY_BYTES);
    memcpy(lines[0], "bus", length);
    aotx_test_lines(lines, &length, 1u);
    written = aotx_test_written();
    console = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    aotx_check_runtime(cudaMemcpyFromSymbol(&worst, aotx_test_worst, sizeof worst),
                       "cudaMemcpyFromSymbol");

    aotx_test_check(written == (unsigned int)AOTX_CLI_RECORDS_EACH,
                    "the line writes the records the reservation covers and no more");
    aotx_test_check(console > 0u && found[console - 1u].length > 14u
                    && memcmp(found[console - 1u].body, "output cut at ", 14u) == 0,
                    "the last record of a cut line says that the output was cut");
    aotx_test_check(worst <= (unsigned int)AOTX_CLI_RECORDS_EACH,
                    "no command line writes more records than the allowance");
    printf("cli: the cut line wrote %u records, the allowance is %u, the worst line is %u\n",
           written, (unsigned int)AOTX_CLI_RECORDS_EACH, worst);
    free(found);
    free(lines);
}

/* The set command and the settings command. A whole number, a number with decimals, a key
 * that is not known and a value outside its range. The console lines state each result. */
static void aotx_test_settings(void)
{
    aotx_test_record *found = (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);
    char (*lines)[AOTX_BODY_BYTES] = (char (*)[AOTX_BODY_BYTES]) malloc(5u * AOTX_BODY_BYTES);
    unsigned int lengths[5];
    const char *given[5] = { "set tick.period_ms 25",
                             "set sample.temperature 0.35",
                             "set tick.period_nope 3",
                             "set tick.period_ms 0",
                             "settings" };
    unsigned int console = 0u;
    unsigned int at = 0u;
    unsigned int keys = 0u;

    aotx_settings_state table;
    memset(lines, 0, 5u * AOTX_BODY_BYTES);
    for (unsigned int i = 0u; i < 5u; ++i) {
        lengths[i] = (unsigned int)strlen(given[i]);
        memcpy(lines[i], given[i], lengths[i]);
    }
    console = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    at = console;
    /* The four set lines, then the commit that writes their records, then the list. */
    aotx_test_lines(lines, lengths, 4u);
    aotx_test_commit_settings<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_lines(lines + 4, lengths + 4, 1u);
    console = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_setting_table, sizeof table),
                       "cudaMemcpyFromSymbol");

    aotx_test_check(console > at + 4u, "the five lines wrote console records");
    aotx_test_check(aotx_test_same(&found[at], "set: tick.period_ms 25 tick"),
                    "a whole number lands and the line states the key and the effect");
    aotx_test_check(aotx_test_same(&found[at + 1u],
                                   "set: sample.temperature 0.35 sequence"),
                    "a number with decimals lands and the line states it again");
    aotx_test_check(aotx_test_same(&found[at + 2u],
                                   "set: the key tick.period_nope is not known"),
                    "a key that is not known is refused with the key");
    aotx_test_check(aotx_test_same(&found[at + 3u], "set: tick.period_ms takes 1 to 1000"),
                    "a value outside the range is refused with the range");
    aotx_test_check(aotx_test_same(&found[at + 4u], "settings: key value effect"),
                    "the settings command writes the head of the list");
    aotx_test_check(table.row[AOTX_SET_TICK_PERIOD_MS].value == 25
                    && table.row[AOTX_SET_TEMPERATURE].value == 3500,
                    "the two lines that landed hold the rows");

    for (unsigned int i = 0u; i < (unsigned int)AOTX_SETTING_NUMBER_COUNT; ++i) {
        keys += (aotx_setting_side_of(i) == AOTX_SETTING_SIDE_DEVICE) ? 1u : 0u;
    }
    aotx_test_check(console >= at + 5u + keys,
                    "the settings command writes one line for each key of the device");
    aotx_test_check(aotx_test_same(&found[at + 5u], "  tick.period_ms 25 tick"),
                    "the first line of the list holds the key, the value and the effect");
    printf("cli: the settings list wrote %u lines for %u keys\n", console - at - 5u, keys);
    free(found);
    free(lines);
}

#include "cli_say.h"
#include "cli_control.h"
#include "cli_agents.h"
#include "cli_modules.h"

int main(int argc, char **argv)
{
    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    unsigned long long boot_id = 0x0c11beefull;
    const char *fixtures = (argc > 1) ? argv[1] : "tests/fixtures/tokenizer";
    const char *models = (argc > 2) ? argv[2] : "models";

    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("cli: the memory map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    if (aotx_test_catalog_setup() != 0) {
        printf("cli: the catalog did not take the built-in tools and the three roles\n");
        return 1;
    }

    aotx_test_batch(1u);
    aotx_test_batch(AOTX_TEST_BATCH);
    aotx_test_history();
    aotx_test_commands();
    aotx_test_allowance();
    aotx_test_echo(&rings, boot_id);
    aotx_test_say();
    aotx_test_grow();
    aotx_test_stream(1u);
    aotx_test_stream(AOTX_TEST_BATCH);
    aotx_test_stream_batch(1u);
    aotx_test_stream_batch(AOTX_TEST_BATCH);
    aotx_test_say_tokens(fixtures, models, &rings, boot_id);
    aotx_test_control_bytes(fixtures, models, 1u);
    aotx_test_control_bytes(fixtures, models, AOTX_TEST_BATCH);
    aotx_test_spawn(1u);
    aotx_test_spawn(AOTX_TEST_BATCH);
    aotx_test_spawn_refusals();
    aotx_test_task(1u);
    aotx_test_task(AOTX_TEST_BATCH);
    aotx_test_task_refusals();
    aotx_test_authorise(1u);
    aotx_test_authorise(AOTX_TEST_BATCH);
    aotx_test_focus_keys();
    aotx_test_say_conductor();
    aotx_test_text_bound();
    aotx_test_agents_list(1u);
    aotx_test_agents_list(AOTX_TEST_BATCH);
    aotx_test_settings();
    aotx_test_modules_commands();

    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    printf("cli: %u cases applied, %u passed, %u failed\n", aotx_test_applied,
           aotx_test_applied - aotx_test_failed, aotx_test_failed);
    if (aotx_test_applied == 0u) {
        printf("cli: no case ran\n");
        return 1;
    }
    return (aotx_test_failed == 0u) ? 0 : 1;
}
