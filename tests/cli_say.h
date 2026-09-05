/* Purpose: Check the say command, the reply it streams, and the tokens of its wrapped prompt.
 * Owns: The text fixtures of the say path and the buffers of its cases.
 * Threading: One thread; the command line check calls these one at a time.
 * Lifetime: The program.
 *
 * The file is a part of the command line check. It reads the helpers of that check, so it
 * comes after them in the same translation unit. */
#ifndef AOTX_TEST_CLI_SAY_H
#define AOTX_TEST_CLI_SAY_H

extern "C" {
#include "disk/modelfile/modelfile.h"
}

#include "model/forward.cuh"
#include "model/model.cuh"

#define AOTX_TEST_TEXTS   8u
#define AOTX_TEST_STREAM  (64u * 1024u)

/* ---- The say command, the reply it streams, and the tokens of its wrapped prompt ---- */

/* The wrapped prompt of a text, built on the host from the same two pieces. The check
 * compares the bytes the command layer made with these. */
static unsigned int aotx_test_wrap(const char *text, unsigned char *out)
{
    static const char head[] = "<|im_start|>user\n";
    static const char tail[] = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    unsigned int at = (unsigned int)strlen(head);
    memcpy(out, head, at);
    memcpy(out + at, text, strlen(text));
    at += (unsigned int)strlen(text);
    memcpy(out + at, tail, strlen(tail));
    return at + (unsigned int)strlen(tail);
}

/* Put a text in the prompt table of a slot, as the say command does. The check fills a
 * batch of slots this way, which is the path an agent takes in a later version. */
__global__ void aotx_test_ask(const unsigned char *text, const unsigned int *start,
                              const unsigned int *length, unsigned int count,
                              unsigned int *bad)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) {
        return;
    }
    if (aotx_say_ask(slot, text + start[slot], length[slot]) != 0) {
        atomicAdd(bad, 1u);
    }
}

/* Free a run of sequence slots and the say state of each one, so the next case starts from
 * a table with nothing in it. The agent table starts again as well, with one conductor on
 * slot 0. The say command sends its text to that agent. */
__global__ void aotx_test_free(unsigned int count)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) {
        return;
    }
    aotx_seqs.slot[slot].state = AOTX_SEQ_STATE_FREE;
    aotx_say.slot[slot].wanted = 0u;
    aotx_say.slot[slot].live = 0u;
    aotx_say.slot[slot].ready = 0u;
    aotx_say.slot[slot].at = 0ull;
    aotx_say.slot[slot].column = 0u;
    if (slot < AOTX_SLOTS) {
        aotx_agents.agent[slot].state = AOTX_AGENT_STATE_FREE;
    }
    __syncthreads();
    if (slot == 0u) {
        aotx_seqs.live = 0u;
        aotx_agents.live = 0u;
        aotx_agent_spawn(aotx_catalog.conductor, 0u, aotx_time_tick);
    }
}

/* Put one run of reply bytes on the console line of each slot, as the reply node does with
 * the bytes a take gave it. */
__global__ void aotx_test_show(const unsigned char *text, const unsigned int *start,
                               const unsigned int *length, unsigned int count)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) {
        return;
    }
    aotx_say_show(slot, text + start[slot], length[slot]);
}

/* Set the layer count of the language model and the shape its key value cache takes. The
 * say command refuses a text when no language model is resident, and the open of a sequence
 * refuses a shape that gives no page. The figures are the ones of the 4B file: 36 layers,
 * 8 key value heads and a head of 128. */
__global__ void aotx_test_model(unsigned int layers)
{
    aotx_model[AOTX_MODEL_LANGUAGE].layers = layers;
    aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, layers, 8u, 128u);
}

/* Run one agent step. That node of the tick turns the message of an agent into a prompt in
 * the table of the say path. */
static void aotx_test_agent_tick(void)
{
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_agent_step<<<1, AOTX_SLOTS>>>(tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

/* Report whether the prompt of a slot holds a text. The agent puts an overlay of its role
 * before the message, so the check looks for the message inside the prompt. */
static int aotx_test_prompt_holds(const aotx_say_state *state, unsigned int slot,
                                  const char *text)
{
    unsigned int length = (unsigned int)strlen(text);
    if (state->slot[slot].length < length) {
        return 0;
    }
    for (unsigned int at = 0u; at + length <= state->slot[slot].length; ++at) {
        if (memcmp(state->prompt[slot] + at, text, length) == 0) {
            return 1;
        }
    }
    return 0;
}

static aotx_say_state *aotx_test_say_state(void)
{
    static aotx_say_state state;
    aotx_check_runtime(cudaMemcpyFromSymbol(&state, aotx_say, sizeof state),
                       "cudaMemcpyFromSymbol");
    return &state;
}

/* Send one command line and give the console lines it made back. */
static void aotx_test_one(const char *line)
{
    char (*lines)[AOTX_BODY_BYTES] = (char (*)[AOTX_BODY_BYTES]) malloc(AOTX_BODY_BYTES);
    unsigned int length = (unsigned int)strlen(line);
    memset(lines, 0, AOTX_BODY_BYTES);
    memcpy(lines[0], line, length);
    aotx_test_lines(lines, &length, 1u);
    free(lines);
}

/* The say command wraps the text in the chat template and keeps the bytes for the tokenize
 * step. A text with no language model resident is refused. A second say while the first
 * waits is refused, and stop drops the one that waits. */
static void aotx_test_say(void)
{
    unsigned char want[AOTX_SAY_BYTES];
    aotx_console_state *console = (aotx_console_state *)malloc(sizeof *console);
    unsigned int wanted = aotx_test_wrap("hello there", want);
    aotx_cli_counts before;
    aotx_cli_counts after;
    const aotx_say_state *state = NULL;
    unsigned int said = 0u;
    unsigned int refused = 0u;
    unsigned int stopped = 0u;

    aotx_test_free<<<1, AOTX_SLOTS>>>(AOTX_SLOTS);
    aotx_test_model<<<1, 1>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    state = aotx_test_say_state();
    said = state->said;
    refused = state->refused;
    stopped = state->stopped;

    /* No language model is resident, so the command is refused and says why. */
    aotx_test_one("say hello there");
    aotx_test_console_state(console);
    state = aotx_test_say_state();
    aotx_test_check(state->slot[0].wanted == 0u && state->refused == refused + 1u,
                    "a say with no language model resident is refused");
    aotx_test_check(aotx_test_says(aotx_test_at(console, console->count),
                                   "say: no language model is loaded"),
                    "the refusal names the model that is missing");

    /* The language model is resident from here. */
    aotx_test_model<<<1, 1>>>(36u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    /* A loaded model with a failed wrap check cannot receive a conductor message. */
    aotx_wrap wrap;
    aotx_agent_work work_before, work_after;
    aotx_agent agent_before, agent_after;
    aotx_check_runtime(cudaMemcpyFromSymbol(&wrap, aotx_model_wrap, sizeof wrap,
                        AOTX_MODEL_LANGUAGE * sizeof wrap), "cudaMemcpyFromSymbol");
    unsigned int usable = wrap.usable;
    wrap.usable = 0u;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_wrap, &wrap, sizeof wrap,
                        AOTX_MODEL_LANGUAGE * sizeof wrap), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&work_before, aotx_agent_gear, sizeof work_before), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&agent_before, aotx_agents, sizeof agent_before), "cudaMemcpyFromSymbol");
    aotx_say_slot slot_before = aotx_test_say_state()->slot[0];
    before = aotx_test_counts();
    aotx_test_one("say this message must not enter the agent");
    after = aotx_test_counts();
    state = aotx_test_say_state();
    aotx_check_runtime(cudaMemcpyFromSymbol(&work_after, aotx_agent_gear, sizeof work_after), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&agent_after, aotx_agents, sizeof agent_after), "cudaMemcpyFromSymbol");
    aotx_test_check(after.refused == before.refused + 1u && state->refused == refused + 2u
                    && state->said == said, "a failed wrap check refuses the say command at admission");
    aotx_test_check(memcmp(&work_before, &work_after, sizeof work_before) == 0
                    && memcmp(&agent_before, &agent_after, sizeof agent_before) == 0
                    && memcmp(&slot_before, &state->slot[0], sizeof slot_before) == 0,
                    "a refused wrap leaves the agent, its message and the say slot unchanged");
    aotx_test_console_state(console);
    aotx_test_check(aotx_test_says(aotx_test_at(console, console->count),
                                   "say: the model wrap did not pass its load check"),
                    "the say refusal states the failed load check");
    wrap.usable = usable;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_wrap, &wrap, sizeof wrap,
                        AOTX_MODEL_LANGUAGE * sizeof wrap), "cudaMemcpyToSymbol");
    before = aotx_test_counts();
    aotx_test_one("say hello there");
    state = aotx_test_say_state();
    aotx_test_check(state->slot[0].at != 0ull, "the say command opens a console line");
    /* The command gives the text to the conductor. The agent step of the tick then builds
     * the prompt of the turn in the table of the say path. */
    aotx_test_agent_tick();
    state = aotx_test_say_state();
    aotx_test_check(state->slot[0].wanted == 1u,
                    "the turn of the conductor leaves a prompt to open");
    aotx_test_check(aotx_test_prompt_holds(state, 0u, "hello there"),
                    "the prompt holds the text of the message");
    aotx_test_check(state->slot[0].length > wanted,
                    "the prompt of an agent is longer than the chat wrap of the text, "
                    "because the overlay of the role stands before the message");
    aotx_test_console_state(console);
    aotx_test_check(aotx_test_says(aotx_test_at(console, state->slot[0].at), "conductor: "),
                    "the line the reply grows into names the conductor");

    /* A second say while the first prompt waits is refused, and the parser counts it. */
    aotx_test_one("say again");
    after = aotx_test_counts();
    state = aotx_test_say_state();
    aotx_test_check(after.refused == before.refused + 1u,
                    "a second say while a reply runs is refused");
    aotx_test_check(state->said == said + 1u && state->refused == refused + 3u,
                    "the say path counts one text taken and three refused");

    /* The stop command drops the prompt that waits, and a stop with nothing to end is
     * refused. */
    aotx_test_one("stop");
    state = aotx_test_say_state();
    aotx_test_check(state->slot[0].wanted == 0u, "stop drops the prompt that waits");
    aotx_test_check(state->stopped == stopped + 1u, "the say path counts the stop");
    after = aotx_test_counts();
    aotx_test_one("stop");
    aotx_test_check(aotx_test_counts().refused == after.refused + 1u,
                    "a stop with no reply is refused");
    printf("cli: the wrap of a text of 11 bytes is %u bytes\n", wanted);
    free(console);
}

/* The reply of a slot grows one console line. A newline byte in the reply ends that line.
 * The check runs at one take and at 64 takes, and at 64 slots that take together. */
static void aotx_test_stream(unsigned int takes)
{
    aotx_console_state *console = (aotx_console_state *)malloc(sizeof *console);
    unsigned char *run = (unsigned char *)malloc(AOTX_TEST_STREAM);
    aotx_test_record *found = (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);
    unsigned int *starts = (unsigned int *)malloc(AOTX_TEST_BATCH * sizeof *starts);
    unsigned int *sizes = (unsigned int *)malloc(AOTX_TEST_BATCH * sizeof *sizes);
    unsigned char *device = NULL;
    unsigned int *at_device = NULL;
    unsigned int *size_device = NULL;
    unsigned long long before = 0ull;
    unsigned int matched = 0u;
    char want[64];

    aotx_check_runtime(cudaMalloc(&device, AOTX_TEST_STREAM), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&at_device, AOTX_TEST_BATCH * sizeof *at_device),
                       "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&size_device, AOTX_TEST_BATCH * sizeof *size_device),
                       "cudaMalloc");
    aotx_test_free<<<1, AOTX_SLOTS>>>(AOTX_SLOTS);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_console_state(console);
    before = console->count;

    /* Each take holds one distinct byte run and the newline that ends its line. */
    for (unsigned int i = 0u; i < takes; ++i) {
        unsigned int length = (unsigned int)snprintf(want, sizeof want, "reply %u of %u\n",
                                                     i, takes);
        starts[0] = 0u;
        sizes[0] = length;
        memcpy(run, want, length);
        aotx_check_runtime(cudaMemcpy(device, run, length, cudaMemcpyHostToDevice),
                           "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(at_device, starts, sizeof *starts,
                                      cudaMemcpyHostToDevice), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(size_device, sizes, sizeof *sizes,
                                      cudaMemcpyHostToDevice), "cudaMemcpy");
        aotx_test_show<<<1, 1>>>(device, at_device, size_device, 1u);
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_console_state(console);

    for (unsigned int i = 0u; i < takes; ++i) {
        snprintf(want, sizeof want, "reply %u of %u", i, takes);
        if (aotx_test_says(aotx_test_at(console, before + 1ull + (unsigned long long)i),
                           want)) {
            matched += 1u;
        }
    }
    unsigned int records = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    unsigned int marked = 0u;
    for (unsigned int i = 0u; i < records; ++i) {
        snprintf(want, sizeof want, "reply %u of %u\n", 0u, takes);
        if (found[i].length > 6u && memcmp(found[i].body, "reply ", 6u) == 0
            && (found[i].flags & AOTX_FLAG_FRAGMENT) != 0u) {
            marked += 1u;
        }
    }
    aotx_test_check(marked == 0u, "a take that starts a line carries no fragment flag");
    aotx_test_check(console->count == before + (unsigned long long)takes,
                    "each take with a newline gives one console line");
    aotx_test_check(matched == takes, "every console line holds the text of its take");
    printf("cli: %u takes gave %llu console lines, %u matched\n", takes,
           console->count - before, matched);

    free(console);
    free(run);
    free(found);
    free(starts);
    free(sizes);
    cudaFree(device);
    cudaFree(at_device);
    cudaFree(size_device);
}

/* A take with no newline grows the line that is open, so a reply of many takes is one line. */
static void aotx_test_grow(void)
{
    aotx_console_state *console = (aotx_console_state *)malloc(sizeof *console);
    aotx_test_record *found = (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);
    unsigned int records = 0u;
    unsigned int joined = 0u;
    unsigned int flagged = 0u;
    unsigned int opened = 0u;
    unsigned char *device = NULL;
    unsigned int *at_device = NULL;
    unsigned int *size_device = NULL;
    unsigned int starts[1] = { 0u };
    unsigned int sizes[1] = { 3u };
    unsigned long long line = 0ull;
    const aotx_say_state *state = NULL;

    aotx_check_runtime(cudaMalloc(&device, 16u), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&at_device, sizeof starts), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&size_device, sizeof sizes), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(at_device, starts, sizeof starts, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(size_device, sizes, sizeof sizes, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_test_free<<<1, AOTX_SLOTS>>>(AOTX_SLOTS);
    aotx_test_model<<<1, 1>>>(36u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_one("say hello there");
    state = aotx_test_say_state();
    line = state->slot[0].at;

    aotx_check_runtime(cudaMemcpy(device, "abc", 3u, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_test_show<<<1, 1>>>(device, at_device, size_device, 1u);
    aotx_check_runtime(cudaMemcpy(device, "def", 3u, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_test_show<<<1, 1>>>(device, at_device, size_device, 1u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_console_state(console);

    aotx_test_check(aotx_test_says(aotx_test_at(console, line), "conductor: abcdef"),
                    "two takes with no newline grow the line the say command started");

    /* The two takes go at the end of a line that is open. Each record therefore carries
     * the fragment flag, and a reader joins the three records into one line. */
    records = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    for (unsigned int i = 0u; i < records; ++i) {
        if (aotx_test_same(&found[i], "abc") || aotx_test_same(&found[i], "def")) {
            joined += 1u;
            if ((found[i].flags & AOTX_FLAG_FRAGMENT) != 0u) {
                flagged += 1u;
            }
        }
        if (aotx_test_same(&found[i], "conductor: ")
            && (found[i].flags & AOTX_FLAG_FRAGMENT) == 0u) {
            opened += 1u;
        }
    }
    aotx_test_check(joined == 2u, "each take of the reply wrote one console record");
    aotx_test_check(flagged == 2u, "a take that grows an open line carries the fragment "
                                   "flag");
    aotx_test_check(opened >= 1u, "the line the say command started carries no flag");
    printf("cli: %u records of a grown line, %u of them marked as a fragment\n", joined,
           flagged);
    free(found);
    free(console);
    cudaFree(device);
    cudaFree(at_device);
    cudaFree(size_device);
}

/* Every slot takes its own byte run in one launch, which is the shape of the reply node. */
static void aotx_test_stream_batch(unsigned int count)
{
    aotx_console_state *console = (aotx_console_state *)malloc(sizeof *console);
    unsigned char *run = (unsigned char *)malloc(AOTX_TEST_STREAM);
    unsigned int *starts = (unsigned int *)malloc(count * sizeof *starts);
    unsigned int *sizes = (unsigned int *)malloc(count * sizeof *sizes);
    unsigned char *device = NULL;
    unsigned int *at_device = NULL;
    unsigned int *size_device = NULL;
    unsigned long long before = 0ull;
    unsigned int matched = 0u;
    unsigned int at = 0u;
    char want[64];

    aotx_check_runtime(cudaMalloc(&device, AOTX_TEST_STREAM), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&at_device, count * sizeof *at_device), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&size_device, count * sizeof *size_device), "cudaMalloc");
    aotx_test_free<<<1, AOTX_SLOTS>>>(AOTX_SLOTS);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_console_state(console);
    before = console->count;

    for (unsigned int slot = 0u; slot < count; ++slot) {
        unsigned int length = (unsigned int)snprintf(want, sizeof want, "slot %u says %u\n",
                                                     slot, count);
        starts[slot] = at;
        sizes[slot] = length;
        memcpy(run + at, want, length);
        at += length;
    }
    aotx_check_runtime(cudaMemcpy(device, run, at, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(at_device, starts, count * sizeof *starts,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(size_device, sizes, count * sizeof *sizes,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_test_show<<<1, AOTX_SLOTS>>>(device, at_device, size_device, count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_console_state(console);

    for (unsigned int slot = 0u; slot < count; ++slot) {
        snprintf(want, sizeof want, "slot %u says %u", slot, count);
        for (unsigned long long i = before + 1ull; i <= console->count; ++i) {
            if (aotx_test_says(aotx_test_at(console, i), want)) {
                matched += 1u;
                break;
            }
        }
    }
    aotx_test_check(console->count >= before + (unsigned long long)count,
                    "every slot of the batch gave a console line");
    aotx_test_check(matched == count, "every slot of the batch holds its own text");
    printf("cli: %u slots gave %llu console lines, %u matched\n", count,
           console->count - before, matched);

    free(console);
    free(run);
    free(starts);
    free(sizes);
    cudaFree(device);
    cudaFree(at_device);
    cudaFree(size_device);
}

/* Read the tokenizer arrays and the family of a model file and build the vocabulary of
 * the device. */
static int aotx_test_vocab(const char *path, aotx_text_store *store)
{
    aotx_modelfile *file = NULL;
    if (aotx_modelfile_open(path, &file) != 0) {
        return 1;
    }
    aotx_string_array tokens;
    aotx_string_array merges;
    const int32_t *types = NULL;
    uint64_t type_count = 0ull;
    const char *pre = NULL;
    size_t pre_length = 0u;
    aotx_text_source source;
    int bad = 1;
    if (aotx_modelfile_strings(file, "tokenizer.ggml.tokens", &tokens) == 0
        && aotx_modelfile_strings(file, "tokenizer.ggml.merges", &merges) == 0
        && aotx_modelfile_i32s(file, "tokenizer.ggml.token_type", &types, &type_count) == 0
        && aotx_modelfile_string(file, "tokenizer.ggml.pre", &pre, &pre_length) == 0) {
        memset(&source, 0, sizeof source);
        source.token_bytes = tokens.bytes;
        source.token_at = (const unsigned long long *)tokens.offsets;
        source.tokens = tokens.count;
        source.merge_bytes = merges.bytes;
        source.merge_at = (const unsigned long long *)merges.offsets;
        source.merges = merges.count;
        source.token_type = (const int *)types;
        bad = aotx_text_family_find(pre, pre_length, &source.family);
        if (bad == 0) {
            bad = aotx_text_vocab_build(&source, store);
        }
    }
    aotx_modelfile_close(file);
    return bad;
}

/* Read the texts of the fixture, one for each line. The newline is not part of a text. */
static unsigned int aotx_test_texts(const char *path, char text[][AOTX_BODY_BYTES],
                                    unsigned int max)
{
    FILE *file = fopen(path, "rb");
    unsigned int count = 0u;
    char line[512];
    if (file == NULL) {
        return 0u;
    }
    while (count < max && fgets(line, sizeof line, file) != NULL) {
        size_t length = strlen(line);
        while (length > 0u && (line[length - 1u] == '\n' || line[length - 1u] == '\r')) {
            length -= 1u;
        }
        if (length == 0u || length >= AOTX_BODY_BYTES) {
            continue;
        }
        memcpy(text[count], line, length);
        text[count][length] = '\0';
        count += 1u;
    }
    fclose(file);
    return count;
}

/* Read one golden list. A line that starts with a number sign is a header line. */
static unsigned int aotx_test_golden(const char *path, unsigned int *ids,
                                     unsigned int *count, unsigned int rows)
{
    FILE *file = fopen(path, "rb");
    char line[8192];
    unsigned int row = 0u;
    if (file == NULL) {
        return 0u;
    }
    while (row < rows && fgets(line, sizeof line, file) != NULL) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }
        unsigned int held = 0u;
        const char *at = line;
        while (*at != '\0' && *at != '\n') {
            if (*at >= '0' && *at <= '9') {
                unsigned int value = 0u;
                while (*at >= '0' && *at <= '9') {
                    value = value * 10u + (unsigned int)(*at - '0');
                    at += 1u;
                }
                if (held < AOTX_SAY_TOKENS) {
                    ids[row * AOTX_SAY_TOKENS + held] = value;
                    held += 1u;
                }
            } else {
                at += 1u;
            }
        }
        count[row] = held;
        row += 1u;
    }
    fclose(file);
    return row;
}

/* Put the nodes of the say path in a graph and run it once. The graph holds them in the
 * shape and the order the tick graph holds them in. */
static void aotx_test_pipeline_run(void)
{
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    aotx_check_runtime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags");
    aotx_check_runtime(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                       "cudaStreamBeginCapture");
    aotx_test_check(aotx_cli_say_capture(stream) == 0, "the say nodes go in the capture");
    aotx_test_check(aotx_cli_reply_capture(stream) == 0,
                    "the reply node goes in the capture");
    aotx_check_runtime(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    aotx_check_runtime(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
    aotx_check_runtime(cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
    aotx_check_runtime(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

/* Put a batch of texts in the prompt table, one for each slot, and run the nodes of the say
 * path. The nodes go in a graph, which is how the tick graph holds them. */
static void aotx_test_pipeline(const char text[][AOTX_BODY_BYTES], unsigned int texts,
                               unsigned int count, unsigned int *refused)
{
    unsigned char *run = (unsigned char *)malloc(AOTX_TEST_STREAM);
    unsigned int *starts = (unsigned int *)malloc(count * sizeof *starts);
    unsigned int *sizes = (unsigned int *)malloc(count * sizeof *sizes);
    unsigned char *device = NULL;
    unsigned int *at_device = NULL;
    unsigned int *size_device = NULL;
    unsigned int *bad = NULL;
    unsigned int at = 0u;

    aotx_check_runtime(cudaMalloc(&device, AOTX_TEST_STREAM), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&at_device, count * sizeof *at_device), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&size_device, count * sizeof *size_device), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&bad, sizeof *bad), "cudaMalloc");
    aotx_check_runtime(cudaMemset(bad, 0, sizeof *bad), "cudaMemset");
    for (unsigned int slot = 0u; slot < count; ++slot) {
        unsigned int length = (unsigned int)strlen(text[slot % texts]);
        starts[slot] = at;
        sizes[slot] = length;
        memcpy(run + at, text[slot % texts], length);
        at += length;
    }
    aotx_check_runtime(cudaMemcpy(device, run, at, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(at_device, starts, count * sizeof *starts,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(size_device, sizes, count * sizeof *sizes,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_test_free<<<1, AOTX_SLOTS>>>(AOTX_SLOTS);
    aotx_test_ask<<<1, AOTX_SLOTS>>>(device, at_device, size_device, count, bad);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    aotx_test_pipeline_run();

    aotx_check_runtime(cudaMemcpy(refused, bad, sizeof *bad, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    free(run);
    free(starts);
    free(sizes);
    cudaFree(device);
    cudaFree(at_device);
    cudaFree(size_device);
    cudaFree(bad);
}


/* Apply a run of token records to their slots, as the restore does. */
__global__ void aotx_test_tokens(const aotx_token_body *body, unsigned int count,
                                 unsigned int *bad)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        if (aotx_seq_apply(&body[i]) != 0) {
            *bad += 1u;
        }
    }
}

/* Give the sequence state and the reply tokens of a slot. */
static void aotx_test_slot(unsigned int slot, aotx_seq *out)
{
    static aotx_seq_table *table = NULL;
    if (table == NULL) {
        table = (aotx_seq_table *)malloc(sizeof *table);
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_seqs, sizeof *table),
                       "cudaMemcpyFromSymbol");
    *out = table->slot[slot];
}

/* Put one line in the inbound ring at the slot the apply reads next, then run one tick of
 * the apply. The flags carry the mark of a replay when the caller gives it. */
static void aotx_test_feed(const aotx_seam_rings *rings, unsigned long long boot_id,
                           const char *line, unsigned int flags)
{
    aotx_seam_state *seam = (aotx_seam_state *)malloc(sizeof *seam);
    aotx_check_runtime(cudaMemcpyFromSymbol(seam, aotx_seam, sizeof *seam),
                       "cudaMemcpyFromSymbol");
    unsigned int writer = (flags & AOTX_FLAG_REPLAYED) ? AOTX_WRITER_RESTORE
                                                       : AOTX_WRITER_FEEDER;
    aotx_test_put(rings, boot_id, seam->in.consumed, writer, flags, line,
                  (unsigned int)strlen(line));
    aotx_sched_tick_start<<<1, 1>>>(0ull);
    aotx_seam_apply_inbound<<<AOTX_APPLY_BLOCKS, AOTX_APPLY_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    free(seam);
}

/* A replay of the journal sends the say line again. The line gives the conductor its
 * message again, the agent step takes the turn, and the say path opens the slot. The
 * replayed token records then land on that slot and the replayed stop ends the reply. This
 * replayed say opens the slot the same way with an agent in the path. */
static void aotx_test_replay_say(const aotx_seam_rings *rings, unsigned long long boot_id)
{
    aotx_console_state *console = (aotx_console_state *)malloc(sizeof *console);
    aotx_token_body *body = (aotx_token_body *)calloc(2u, sizeof *body);
    aotx_agent_work *gear = (aotx_agent_work *)malloc(sizeof *gear);
    aotx_token_body *device = NULL;
    unsigned int *bad = NULL;
    unsigned int refused = 0u;
    unsigned int stopped = 0u;
    unsigned int said = 0u;
    unsigned int prompt = 0u;
    unsigned int wrong = 0u;
    aotx_seq seq;
    const aotx_say_state *state = NULL;

    aotx_check_runtime(cudaMalloc(&device, 2u * sizeof *device), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&bad, sizeof *bad), "cudaMalloc");
    aotx_check_runtime(cudaMemset(bad, 0, sizeof *bad), "cudaMemset");
    aotx_test_free<<<1, AOTX_SLOTS>>>(AOTX_SLOTS);
    aotx_test_model<<<1, 1>>>(36u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    state = aotx_test_say_state();
    stopped = state->stopped;
    said = state->said;
    refused = aotx_test_counts().refused;

    /* From here the run is replaying a journal, as a restore does. */
    aotx_seam_set_replaying(1);
    aotx_test_feed(rings, boot_id, "say hello there", AOTX_FLAG_REPLAYED);
    state = aotx_test_say_state();
    aotx_check_runtime(cudaMemcpyFromSymbol(gear, aotx_agent_gear, sizeof *gear),
                       "cudaMemcpyFromSymbol");
    aotx_test_check(state->said == said + 1u && aotx_test_counts().refused == refused,
                    "a replayed say is not refused");
    aotx_test_check(gear->has_message != 0u && gear->message_len == 11u,
                    "a replayed say gives the conductor its message again");

    /* The agent step takes the turn while the replay runs, so the prompt is there for the
     * tokenize step of the same tick. */
    aotx_test_agent_tick();
    state = aotx_test_say_state();
    aotx_check_runtime(cudaMemcpyFromSymbol(gear, aotx_agent_gear, sizeof *gear),
                       "cudaMemcpyFromSymbol");
    aotx_test_check(gear->has_message == 0u && state->slot[0].wanted == 1u,
                    "the turn of the replayed message leaves a prompt to open");
    aotx_test_pipeline_run();
    state = aotx_test_say_state();
    aotx_test_slot(0u, &seq);
    prompt = seq.prompt;
    aotx_test_check(state->slot[0].live == 1u && seq.state != AOTX_SEQ_STATE_FREE,
                    "the say path opens the slot of the conductor while the replay runs");
    aotx_test_check(seq.sample.top_k == aotx_settings_default(AOTX_SET_TOP_K)
                    && seq.limit == aotx_settings_default(AOTX_SET_REPLY_LIMIT),
                    "the open takes the sampling and the limit of the console");

    /* Two reply tokens come back from the journal. The apply gives them to the slot and no
     * draw is taken. */
    for (unsigned int i = 0u; i < 2u; ++i) {
        body[i].slot = 0u;
        body[i].token = 1000u + i;
        body[i].position = prompt + i;
        body[i].flags = AOTX_TOKEN_SAMPLED;
        body[i].seed = seq.seed;
        body[i].draw = i;
        body[i].role = seq.role;
    }
    aotx_check_runtime(cudaMemcpy(device, body, 2u * sizeof *body, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_test_tokens<<<1, 1>>>(device, 2u, bad);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&wrong, bad, sizeof wrong, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_test_slot(0u, &seq);
    aotx_test_check(wrong == 0u, "the apply takes the reply tokens of the journal");
    aotx_test_check(seq.sampled == 2u, "the slot holds the two reply tokens again");

    /* The stop line of the journal must land, because the say state was rebuilt. */
    refused = aotx_test_counts().refused;
    aotx_test_feed(rings, boot_id, "stop", AOTX_FLAG_REPLAYED);
    state = aotx_test_say_state();
    aotx_test_console_state(console);
    aotx_test_check(state->stopped == stopped + 1u, "a replayed stop ends the reply");
    aotx_test_check(aotx_test_counts().refused == refused,
                    "a replayed stop is not refused");
    aotx_test_check(aotx_test_says(aotx_test_at(console, console->count),
                                   "stop: the reply ends"),
                    "the replayed stop says that the reply ends");

    aotx_seam_set_replaying(0);
    printf("cli: a replayed say opened %u prompt tokens while the replay ran\n", prompt);
    aotx_test_free<<<1, AOTX_SLOTS>>>(AOTX_SLOTS);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    free(console);
    free(body);
    free(gear);
    cudaFree(device);
    cudaFree(bad);
}

/* The tokens of a wrapped prompt, at one slot and at 64, against the golden list of the
 * tokenizer tool. The check needs the vocabulary of the language model file. */
static void aotx_test_say_tokens(const char *fixtures, const char *models,
                                 const aotx_seam_rings *rings,
                                 unsigned long long boot_id)
{
    char path[1024];
    char (*text)[AOTX_BODY_BYTES] = (char (*)[AOTX_BODY_BYTES])
        calloc(AOTX_TEST_TEXTS, AOTX_BODY_BYTES);
    unsigned int *golden = (unsigned int *)calloc(AOTX_TEST_TEXTS * AOTX_SAY_TOKENS,
                                                  sizeof(unsigned int));
    unsigned int *golden_count = (unsigned int *)calloc(AOTX_TEST_TEXTS,
                                                        sizeof(unsigned int));
    unsigned int *ids = (unsigned int *)malloc(AOTX_SLOTS * AOTX_SAY_TOKENS
                                               * sizeof(unsigned int));
    unsigned int *counts = (unsigned int *)malloc(AOTX_SLOTS * sizeof(unsigned int));
    aotx_text_store store;
    unsigned int texts = 0u;
    unsigned int rows = 0u;
    static const unsigned int batches[2] = { 1u, AOTX_TEST_BATCH };

    snprintf(path, sizeof path, "%s/say-texts.dat", fixtures);
    texts = aotx_test_texts(path, text, AOTX_TEST_TEXTS);
    snprintf(path, sizeof path, "%s/golden-say.ids", fixtures);
    rows = aotx_test_golden(path, golden, golden_count, AOTX_TEST_TEXTS);
    if (texts == 0u || rows != texts) {
        printf("cli: the say fixture holds %u texts and %u golden rows; the check is skipped\n",
               texts, rows);
        goto done;
    }
    snprintf(path, sizeof path, "%s/Qwen3-4B-Q8_0.gguf", models);
    memset(&store, 0, sizeof store);
    if (aotx_test_vocab(path, &store) != 0) {
        printf("cli: the language model file is not at %s; the token check is skipped\n",
               path);
        goto done;
    }

    for (unsigned int b = 0u; b < 2u; ++b) {
        unsigned int count = batches[b];
        unsigned int refused = 0u;
        unsigned int wrong = 0u;
        aotx_test_pipeline(text, texts, count, &refused);
        aotx_check_runtime(cudaMemcpyFromSymbol(counts, aotx_say_count,
                                                AOTX_SLOTS * sizeof(unsigned int)),
                           "cudaMemcpyFromSymbol");
        aotx_check_runtime(cudaMemcpyFromSymbol(ids, aotx_say_id,
                                                (size_t)AOTX_SLOTS * AOTX_SAY_TOKENS
                                                * sizeof(unsigned int)),
                           "cudaMemcpyFromSymbol");
        for (unsigned int slot = 0u; slot < count; ++slot) {
            unsigned int row = slot % texts;
            const unsigned int *want = golden + row * AOTX_SAY_TOKENS;
            const unsigned int *got = ids + slot * AOTX_SAY_TOKENS;
            int same = (counts[slot] == golden_count[row]);
            for (unsigned int i = 0u; same && i < counts[slot]; ++i) {
                same = (got[i] == want[i]);
            }
            if (!same) {
                if (wrong < 2u) {
                    printf("cli: slot %u gave %u tokens and the golden row holds %u\n", slot,
                           counts[slot], golden_count[row]);
                    for (unsigned int i = 0u; i < counts[slot] && i < 8u; ++i) {
                        printf("cli:  token %u is %u and %u is asked for\n", i, got[i],
                               want[i]);
                    }
                }
                wrong += 1u;
            }
        }
        aotx_test_check(refused == 0u, "every slot of the batch took its text");
        aotx_test_check(wrong == 0u, "every slot gives the tokens of the golden list");
        /* The control names of the wrap are one token each, which is the rule the
         * tokenizer holds for a special token. */
        aotx_test_check(counts[0] > 4u && ids[0] == 151644u && ids[2] == 198u,
                        "the wrap starts with one token for the control name and one for "
                        "the newline");
        aotx_test_check(counts[0] > 4u && ids[counts[0] - 4u] == 151667u
                        && ids[counts[0] - 2u] == 151668u,
                        "the think block of the wrap is two control tokens");
        printf("cli: %u slots tokenized, %u rows differ, slot 0 gives %u tokens\n", count,
               wrong, counts[0]);
    }
    aotx_test_replay_say(rings, boot_id);
    aotx_text_vocab_release(&store);

done:
    free(text);
    free(golden);
    free(golden_count);
    free(ids);
    free(counts);
}

#endif
