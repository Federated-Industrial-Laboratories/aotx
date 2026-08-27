/* Purpose: Check the line editor and the command parser at one line and at 64 lines.
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
#include "cli/cli.cuh"
#include "mem/mem.cuh"
#include "seam/seam.cuh"

#define AOTX_TEST_KEYS    4096u
#define AOTX_TEST_FOUND   4096u
#define AOTX_TEST_BATCH   64u

/* One record the probe found, with the fields the cases compare. */
typedef struct aotx_test_record {
    unsigned long long seq;
    unsigned int writer;
    unsigned int length;
    unsigned char body[AOTX_BODY_BYTES];
} aotx_test_record;

static unsigned int aotx_test_applied;
static unsigned int aotx_test_failed;

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
        at += aotx_cli_utoa(i, text + at, 64u - at);
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

int main(void)
{
    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    unsigned long long boot_id = 0x0c11beefull;

    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("cli: the memory map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);

    aotx_test_batch(1u);
    aotx_test_batch(AOTX_TEST_BATCH);
    aotx_test_history();
    aotx_test_commands();
    aotx_test_allowance();

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
