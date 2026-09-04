/* Purpose: Check the tokenizer against the golden lists, and the parts it is made of.
 * Owns: The test buffers and the counts of the cases.
 * Launch shape: The kernels of the text module, at one sequence and at the slots of the profile.
 * Lifetime: The program. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "profile/profile.cuh"
#include "boot/check.h"
#include "text/text.cuh"

extern "C" {
#include "disk/modelfile/modelfile.h"
}

/* The fixture holds up to 300 lines. The last row of a golden list is the whole file as one
 * sequence, which is the row that holds newline characters inside a sequence. */
#define AOTX_TEST_MAX       320u
#define AOTX_TEST_BATCH     AOTX_SLOTS
#define AOTX_TEST_RUN       (64u * 1024u)
#define AOTX_TEST_CLEAN     16384u
/* Token slots each sequence holds. The whole file row gives 8,129 tokens on the Qwen files.
 * The bound is above that count, because a row the bound cuts is not read in full. */
#define AOTX_TEST_STRIDE    16384u
#define AOTX_TEST_CASES     32u
#define AOTX_TEST_REPEATS   20u
#define AOTX_TEST_BLOCKS    128u
#define AOTX_TEST_BAD       8u

typedef struct aotx_test_gear {
    unsigned char *bytes;
    unsigned int *start;
    unsigned int *length;
    unsigned char *clean;
    unsigned int *clean_start;
    unsigned int *clean_length;
    aotx_text_pieces pieces;
    aotx_text_tokens tokens;
    unsigned char *out;
    unsigned int *out_length;
    unsigned int *points;
    unsigned int *point_count;
    unsigned int *point_run;
} aotx_test_gear;

static void *aotx_test_take(unsigned long long bytes)
{
    void *block = 0;
    aotx_check_runtime(cudaMalloc(&block, (size_t)bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemset(block, 0, (size_t)bytes), "cudaMemset");
    return block;
}

static void aotx_test_open(aotx_test_gear *gear)
{
    unsigned long long slots = (unsigned long long)AOTX_TEST_MAX * AOTX_TEST_STRIDE;
    unsigned long long clean = (unsigned long long)AOTX_TEST_MAX * AOTX_TEST_CLEAN;
    unsigned int warps = AOTX_TEST_BLOCKS * AOTX_TEXT_WARPS;
    memset(gear, 0, sizeof *gear);
    gear->bytes = (unsigned char *)aotx_test_take(AOTX_TEST_RUN);
    gear->start = (unsigned int *)aotx_test_take(AOTX_TEST_MAX * sizeof(unsigned int));
    gear->length = (unsigned int *)aotx_test_take(AOTX_TEST_MAX * sizeof(unsigned int));
    gear->clean = (unsigned char *)aotx_test_take(clean);
    gear->clean_start = (unsigned int *)aotx_test_take(AOTX_TEST_MAX * sizeof(unsigned int));
    gear->clean_length = (unsigned int *)aotx_test_take(AOTX_TEST_MAX * sizeof(unsigned int));
    gear->pieces.start = (unsigned int *)aotx_test_take(slots * sizeof(unsigned int));
    gear->pieces.length = (unsigned int *)aotx_test_take(slots * sizeof(unsigned int));
    gear->pieces.token = (unsigned int *)aotx_test_take(slots * sizeof(unsigned int));
    gear->pieces.count = (unsigned int *)aotx_test_take(AOTX_TEST_MAX * sizeof(unsigned int));
    gear->pieces.work = (unsigned int *)aotx_test_take(slots * sizeof(unsigned int));
    gear->pieces.works = (unsigned int *)aotx_test_take(sizeof(unsigned int));
    gear->pieces.stride = AOTX_TEST_STRIDE;
    gear->tokens.id = (unsigned int *)aotx_test_take(slots * sizeof(unsigned int));
    gear->tokens.count = (unsigned int *)aotx_test_take(AOTX_TEST_MAX * sizeof(unsigned int));
    gear->tokens.chunk = (unsigned int *)aotx_test_take(slots * sizeof(unsigned int));
    gear->tokens.scratch = (unsigned int *)aotx_test_take(clean * sizeof(unsigned int));
    gear->tokens.merge = (unsigned char *)aotx_test_take((unsigned long long)warps
                                                         * AOTX_TEXT_WARP_BYTES);
    gear->tokens.warps = warps;
    gear->tokens.stride = AOTX_TEST_STRIDE;
    gear->out = (unsigned char *)aotx_test_take(clean);
    gear->out_length = (unsigned int *)aotx_test_take(AOTX_TEST_MAX * sizeof(unsigned int));
    gear->points = (unsigned int *)aotx_test_take(slots * sizeof(unsigned int));
    gear->point_count = (unsigned int *)aotx_test_take(AOTX_TEST_MAX * sizeof(unsigned int));
    gear->point_run = (unsigned int *)aotx_test_take(AOTX_TEST_RUN * sizeof(unsigned int));
}

/* One tokenize of a batch. The clean step gives the byte run that the pattern reads. A
 * byte run which is not a character is the replacement character before the split. */
static void aotx_test_tokenize(aotx_test_gear *gear, unsigned int first, unsigned int count)
{
    aotx_text_batch raw;
    raw.bytes = gear->bytes;
    raw.start = gear->start + first;
    raw.length = gear->length + first;
    raw.count = count;
    unsigned int blocks = (count + 63u) / 64u;
    aotx_text_clean<<<blocks, 64u>>>(raw, gear->clean, gear->clean_start,
                                     gear->clean_length, AOTX_TEST_CLEAN);
    aotx_text_batch batch;
    batch.bytes = gear->clean;
    batch.start = gear->clean_start;
    batch.length = gear->clean_length;
    batch.count = count;
    unsigned int zero = 0u;
    aotx_check_runtime(cudaMemcpy(gear->pieces.works, &zero, sizeof zero,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_text_pretok<<<blocks, 64u>>>(batch, gear->pieces);
    aotx_text_merge<<<AOTX_TEST_BLOCKS, 32u * AOTX_TEXT_WARPS>>>(batch, gear->pieces,
                                                                 gear->tokens);
    aotx_text_gather<<<blocks, 64u>>>(batch, gear->pieces, gear->tokens);
}

/* Read the fixture lines. Each sequence is one line with the newline byte that ends it,
 * because the tool that made the golden lists read the file that way. The last sequence is
 * the whole file, which holds newline characters inside it. */
static int aotx_test_lines(const char *dir, unsigned char *run, unsigned int *start,
                           unsigned int *length, unsigned int *rows)
{
    char path[512];
    snprintf(path, sizeof path, "%s/lines.bin", dir);
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    unsigned int held = (unsigned int)fread(run, 1u, AOTX_TEST_RUN / 2u, file);
    fclose(file);
    unsigned int row = 0u;
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < held; ++i) {
        if (run[i] == '\n') {
            if (row + 1u >= AOTX_TEST_MAX) {
                return 1;
            }
            start[row] = at;
            length[row] = i + 1u - at;
            row += 1u;
            at = i + 1u;
        }
    }
    /* The whole file is one more sequence, at the end of the same run. */
    memcpy(run + held, run, held);
    start[row] = held;
    length[row] = held;
    row += 1u;
    *rows = row;
    return (row > 1u && held > 0u) ? 0 : 1;
}

/* Read one golden list. A line that starts with a number sign is a header line. */
static int aotx_test_golden(const char *path, unsigned int *ids, unsigned int *count,
                            unsigned int rows)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    char line[65536];
    unsigned int row = 0u;
    while (fgets(line, sizeof line, file) != NULL) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }
        if (row >= rows) {
            fclose(file);
            return 1;
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
                if (held < AOTX_TEST_STRIDE) {
                    ids[row * AOTX_TEST_STRIDE + held] = value;
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
    return (row == rows) ? 0 : 1;
}

/* Read the three tokenizer arrays and the family of a model file. */
static int aotx_test_arrays(const aotx_modelfile *file, aotx_text_source *source)
{
    aotx_string_array tokens;
    aotx_string_array merges;
    const int32_t *types = NULL;
    uint64_t type_count = 0ull;
    const char *pre = NULL;
    size_t pre_length = 0u;
    if (aotx_modelfile_strings(file, "tokenizer.ggml.tokens", &tokens) != 0
        || aotx_modelfile_strings(file, "tokenizer.ggml.merges", &merges) != 0
        || aotx_modelfile_i32s(file, "tokenizer.ggml.token_type", &types, &type_count) != 0
        || aotx_modelfile_string(file, "tokenizer.ggml.pre", &pre, &pre_length) != 0) {
        return 1;
    }
    memset(source, 0, sizeof *source);
    source->token_bytes = tokens.bytes;
    source->token_at = (const unsigned long long *)tokens.offsets;
    source->tokens = tokens.count;
    source->merge_bytes = merges.bytes;
    source->merge_at = (const unsigned long long *)merges.offsets;
    source->merges = merges.count;
    source->token_type = (const int *)types;
    return aotx_text_family_find(pre, pre_length, &source->family);
}

/* Build the vocabulary from a model file. */
static int aotx_test_vocab(const char *path, aotx_text_store *store)
{
    aotx_modelfile *file = NULL;
    if (aotx_modelfile_open(path, &file) != 0) {
        return 1;
    }
    aotx_text_source source;
    int bad = aotx_test_arrays(file, &source);
    if (bad == 0) {
        bad = aotx_text_vocab_build(&source, store);
    }
    aotx_modelfile_close(file);
    return bad;
}

/* Compare the tokens of the rows of a batch with the golden list. The return is the count
 * of rows that differ. */
static unsigned int aotx_test_compare(const aotx_test_gear *gear, unsigned int first,
                                      unsigned int count, const unsigned int *ids,
                                      const unsigned int *held, int say)
{
    static unsigned int got[AOTX_TEST_STRIDE];
    unsigned int counts[AOTX_TEST_MAX];
    unsigned int wrong = 0u;
    aotx_check_runtime(cudaMemcpy(counts, gear->tokens.count, count * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_check_runtime(cudaMemcpy(got, gear->tokens.id + i * AOTX_TEST_STRIDE,
                                      AOTX_TEST_STRIDE * sizeof(unsigned int),
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        unsigned int row = first + i;
        int same = (counts[i] == held[row]);
        for (unsigned int k = 0u; same && k < counts[i]; ++k) {
            same = (got[k] == ids[row * AOTX_TEST_STRIDE + k]);
        }
        if (!same) {
            wrong += 1u;
            if (say) {
                printf("text: row %u gives %u tokens and the golden list holds %u\n",
                       row, counts[i], held[row]);
                for (unsigned int k = 0u; k < counts[i] && k < 8u; ++k) {
                    printf("text:  %u against %u\n", got[k],
                           ids[row * AOTX_TEST_STRIDE + k]);
                }
            }
        }
    }
    return wrong;
}

/* Put every sequence of the batch on the device. */
static void aotx_test_send(aotx_test_gear *gear, const unsigned char *run, unsigned int bytes,
                           const unsigned int *start, const unsigned int *length,
                           unsigned int rows)
{
    aotx_check_runtime(cudaMemcpy(gear->bytes, run, bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->start, start, rows * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->length, length, rows * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
}

static double aotx_test_now(void)
{
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (double)at.tv_sec + (double)at.tv_nsec * 1e-9;
}

/* One case of the round trip: the bytes that go in, and the bytes that must come out. The
 * reader takes the same byte runs as the reference tokenizer. A form which is longer than
 * the code point needs gives that code point and not the replacement. */
typedef struct aotx_test_pair {
    const char *in;
    unsigned int in_bytes;
    const char *out;
    unsigned int out_bytes;
} aotx_test_pair;

static const aotx_test_pair aotx_test_broken[AOTX_TEST_BAD] = {
    { "\xC3\x28", 2u, "\xEF\xBF\xBD(", 4u },
    { "\xE2\x82", 2u, "\xEF\xBF\xBD\xEF\xBF\xBD", 6u },
    { "\x80", 1u, "\xEF\xBF\xBD", 3u },
    { "\xC0\xAF", 2u, "/", 1u },
    { "\xED\xA0\x80", 3u, "\xED\xA0\x80", 3u },
    { "\xF5\x80\x80\x80", 4u, "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD", 12u },
    { "a\xFF" "b", 3u, "a\xEF\xBF\xBD" "b", 5u },
    { "ok\xC2", 3u, "ok\xEF\xBF\xBD", 5u },
};

/* Run the round trip over a batch and give the count of sequences that differ. */
static unsigned int aotx_test_trip(aotx_test_gear *gear, const aotx_test_pair *cases,
                                   unsigned int count)
{
    static unsigned char run[AOTX_TEST_RUN];
    static unsigned char back[AOTX_TEST_CLEAN];
    unsigned int start[AOTX_TEST_MAX];
    unsigned int length[AOTX_TEST_MAX];
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        memcpy(run + at, cases[i].in, cases[i].in_bytes);
        start[i] = at;
        length[i] = cases[i].in_bytes;
        at += cases[i].in_bytes;
    }
    aotx_test_send(gear, run, at, start, length, count);
    aotx_text_batch batch;
    batch.bytes = gear->bytes;
    batch.start = gear->start;
    batch.length = gear->length;
    batch.count = count;
    unsigned int blocks = (count + 63u) / 64u;
    dim3 grid(4u, count, 1u);
    aotx_text_decode_run<<<grid, 64u>>>(batch, gear->point_run);
    aotx_text_gather_points<<<blocks, 64u>>>(batch, gear->point_run, gear->points,
                                             gear->point_count, AOTX_TEST_STRIDE);
    aotx_text_encode_run<<<blocks, 64u>>>(gear->points, gear->point_count, count,
                                          AOTX_TEST_STRIDE, gear->out, gear->out_length,
                                          AOTX_TEST_CLEAN);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int held[AOTX_TEST_MAX];
    aotx_check_runtime(cudaMemcpy(held, gear->out_length, count * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int wrong = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        int same = (held[i] == cases[i].out_bytes);
        if (same) {
            aotx_check_runtime(cudaMemcpy(back, gear->out + (unsigned long long)i
                                          * AOTX_TEST_CLEAN, held[i],
                                          cudaMemcpyDeviceToHost), "cudaMemcpy");
            same = (memcmp(back, cases[i].out, held[i]) == 0);
        }
        if (!same) {
            printf("text: the round trip of case %u gave %u bytes of %u\n", i, held[i],
                   cases[i].out_bytes);
            wrong += 1u;
        }
    }
    return wrong;
}

/* One case of the piece boundaries: the text, and the bytes of each piece it must give. */
typedef struct aotx_test_cut {
    const char *text;
    unsigned int bytes;
    unsigned int pieces;
    unsigned int length[8];
} aotx_test_cut;

/* The 32 cases cover every alternative of the pattern and the lookahead of the sixth one.
 * A case with two spaces in the middle proves the lookahead. The run gives back its last
 * character when a character which is not a space follows it. */
static const aotx_test_cut aotx_test_cuts[AOTX_TEST_CASES] = {
    { "'s", 2u, 1u, { 2u } },
    { "'S", 2u, 1u, { 2u } },
    { "'re", 3u, 1u, { 3u } },
    { "'VE", 3u, 1u, { 3u } },
    { "'ll", 3u, 1u, { 3u } },
    { "'D", 2u, 1u, { 2u } },
    { "it's", 4u, 2u, { 2u, 2u } },
    { "IT'S", 4u, 2u, { 2u, 2u } },
    { "hello", 5u, 1u, { 5u } },
    { " hello", 6u, 1u, { 6u } },
    { "!hello", 6u, 1u, { 6u } },
    { "\nhello", 6u, 2u, { 1u, 5u } },
    { "5", 1u, 1u, { 1u } },
    { "42", 2u, 2u, { 1u, 1u } },
    { "1234", 4u, 4u, { 1u, 1u, 1u, 1u } },
    { " 42", 3u, 3u, { 1u, 1u, 1u } },
    { "!!!", 3u, 1u, { 3u } },
    { " !!!", 4u, 1u, { 4u } },
    { "...\n", 4u, 1u, { 4u } },
    { "a\n\n", 3u, 2u, { 1u, 2u } },
    { "a  b", 4u, 3u, { 1u, 1u, 2u } },
    { "a   b", 5u, 3u, { 1u, 2u, 2u } },
    { "a ", 2u, 2u, { 1u, 1u } },
    { "a  ", 3u, 2u, { 1u, 2u } },
    { "\t\tx", 3u, 2u, { 1u, 2u } },
    { "  \n  ", 5u, 2u, { 3u, 2u } },
    { "x\r\ny", 4u, 3u, { 1u, 2u, 1u } },
    { "  ", 2u, 1u, { 2u } },
    { "3.14", 4u, 4u, { 1u, 1u, 1u, 1u } },
    { "caf\xC3\xA9", 5u, 1u, { 5u } },
    { "\xE4\xBD\xA0\xE5\xA5\xBD", 6u, 1u, { 6u } },
    { "a<|im_end|>b", 12u, 3u, { 1u, 10u, 1u } },
};

/* Run one boundary case and give one when it differs. */
static unsigned int aotx_test_bounds(aotx_test_gear *gear, const aotx_test_cut *cut)
{
    static unsigned int got[AOTX_TEST_STRIDE];
    unsigned int start = 0u;
    unsigned int length = cut->bytes;
    aotx_test_send(gear, (const unsigned char *)cut->text, cut->bytes, &start, &length, 1u);
    aotx_test_tokenize(gear, 0u, 1u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int count = 0u;
    aotx_check_runtime(cudaMemcpy(&count, gear->pieces.count, sizeof count,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(got, gear->pieces.length,
                                  AOTX_TEST_STRIDE * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    int same = (count == cut->pieces);
    for (unsigned int i = 0u; same && i < count; ++i) {
        same = (got[i] == cut->length[i]);
    }
    if (!same) {
        printf("text: the pattern gave %u pieces of %s and %u are asked for\n", count,
               cut->text, cut->pieces);
        for (unsigned int i = 0u; i < count && i < 8u; ++i) {
            printf("text:  piece %u holds %u bytes\n", i, got[i]);
        }
        return 1u;
    }
    return 0u;
}

/* The gpt2 pattern has no model file in the set, so its split is checked on boundary cases
 * alone. The cases cover its six alternatives: small letter contractions, an optional space
 * before letters, numbers and marks, whole runs of numbers, and no newline rule. */
#define AOTX_TEST_GPT2_CASES  12u

static const aotx_test_cut aotx_test_gpt2_cuts[AOTX_TEST_GPT2_CASES] = {
    { "it's", 4u, 2u, { 2u, 2u } },
    { "IT'S", 4u, 3u, { 2u, 1u, 1u } },
    { " hello", 6u, 1u, { 6u } },
    { "!hello", 6u, 2u, { 1u, 5u } },
    { "1234", 4u, 1u, { 4u } },
    { " 1234 5", 7u, 2u, { 5u, 2u } },
    { " !!!", 4u, 1u, { 4u } },
    { "...\n", 4u, 2u, { 3u, 1u } },
    { "a\n\nb", 4u, 4u, { 1u, 1u, 1u, 1u } },
    { "a  b", 4u, 3u, { 1u, 1u, 2u } },
    { "\nhello", 6u, 2u, { 1u, 5u } },
    { "x\r\ny", 4u, 4u, { 1u, 1u, 1u, 1u } },
};

/* Set the pattern row of the table on the device and give the row that was there. */
static unsigned int aotx_test_pattern(unsigned int pattern)
{
    aotx_text_vocab table;
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_text_vocab_table, sizeof table),
                       "cudaMemcpyFromSymbol");
    unsigned int held = table.pattern;
    table.pattern = pattern;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_text_vocab_table, &table, sizeof table),
                       "cudaMemcpyToSymbol");
    return held;
}

/* The model files and the golden list of each one. The first three files of the set hold
 * the same tokenizer, so their lists must hold the same rows. The fourth file is of the
 * llama3 family. That family cuts numbers in groups of three and takes a whole piece which
 * is a token as that token. The set check reads the first three only. */
#define AOTX_TEST_MODELS   4u
#define AOTX_TEST_SET      3u

static const char *aotx_test_files[AOTX_TEST_MODELS][2] = {
    { "Qwen3-Embedding-0.6B-Q8_0.gguf", "golden-embedding.ids" },
    { "qwen3-reranker-0.6b-q8_0.gguf", "golden-reranker.ids" },
    { "Qwen3-4B-Q8_0.gguf", "golden-language.ids" },
    { "Llama-3.2-1B-Instruct-Q8_0.gguf", "golden-llama.ids" },
};

/* Turn the ranks of the pair table around and give the table that was there. A rank which
 * merges first then merges last, so the token lists must change. The check that finds no
 * difference here is a check which cannot fail. */
static unsigned int *aotx_test_flip(unsigned int *pairs)
{
    aotx_text_vocab table;
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_text_vocab_table, sizeof table),
                       "cudaMemcpyFromSymbol");
    unsigned int count = table.pairs;
    unsigned int *kept = (unsigned int *)malloc(count * sizeof(unsigned int));
    unsigned int *made = (unsigned int *)malloc(count * sizeof(unsigned int));
    if (kept == NULL || made == NULL) {
        free(kept);
        free(made);
        return NULL;
    }
    aotx_check_runtime(cudaMemcpy(kept, table.pair_rank, count * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    for (unsigned int i = 0u; i < count; ++i) {
        made[i] = (kept[i] == AOTX_TEXT_NONE) ? AOTX_TEXT_NONE : (0x00FFFFFFu - kept[i]);
    }
    aotx_check_runtime(cudaMemcpy((void *)table.pair_rank, made,
                                  count * sizeof(unsigned int), cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    free(made);
    *pairs = count;
    return kept;
}

static void aotx_test_restore(unsigned int *kept, unsigned int pairs)
{
    aotx_text_vocab table;
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_text_vocab_table, sizeof table),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpy((void *)table.pair_rank, kept,
                                  pairs * sizeof(unsigned int), cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    free(kept);
}

/* Write the bytes of the token lists and compare them with the byte run of the clean step,
 * which is the run the tokenizer read. */
static unsigned int aotx_test_detok(aotx_test_gear *gear, unsigned int count)
{
    static unsigned char back[AOTX_TEST_CLEAN];
    static unsigned char want[AOTX_TEST_CLEAN];
    unsigned int blocks = (count + 63u) / 64u;
    aotx_text_detok<<<blocks, 64u>>>(gear->tokens.id, gear->tokens.count, count,
                                     AOTX_TEST_STRIDE, gear->out, gear->out_length,
                                     AOTX_TEST_CLEAN);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int held[AOTX_TEST_MAX];
    unsigned int asked[AOTX_TEST_MAX];
    aotx_check_runtime(cudaMemcpy(held, gear->out_length, count * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(asked, gear->clean_length, count * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int wrong = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        int same = (held[i] == asked[i]);
        if (same) {
            unsigned long long place = (unsigned long long)i * AOTX_TEST_CLEAN;
            aotx_check_runtime(cudaMemcpy(back, gear->out + place, held[i],
                                          cudaMemcpyDeviceToHost), "cudaMemcpy");
            aotx_check_runtime(cudaMemcpy(want, gear->clean + place, asked[i],
                                          cudaMemcpyDeviceToHost), "cudaMemcpy");
            same = (memcmp(back, want, held[i]) == 0);
        }
        if (!same) {
            printf("text: the bytes of row %u came back as %u of %u\n", i, held[i],
                   asked[i]);
            wrong += 1u;
        }
    }
    return wrong;
}

/* One table serves the set. The table comes from the file with the most tokens, and every
 * other file must hold the same token for the same id. */
static unsigned int aotx_test_prefix(const char *models, unsigned int *applied)
{
    unsigned long long most = 0ull;
    unsigned int largest = 0u;
    for (unsigned int m = 0u; m < AOTX_TEST_SET; ++m) {
        char path[512];
        snprintf(path, sizeof path, "%s/%s", models, aotx_test_files[m][0]);
        aotx_modelfile *file = NULL;
        if (aotx_modelfile_open(path, &file) != 0) {
            return 0u;
        }
        aotx_text_source source;
        if (aotx_test_arrays(file, &source) == 0 && source.tokens > most) {
            most = source.tokens;
            largest = m;
        }
        aotx_modelfile_close(file);
    }
    char path[512];
    snprintf(path, sizeof path, "%s/%s", models, aotx_test_files[largest][0]);
    aotx_text_store store;
    memset(&store, 0, sizeof store);
    if (aotx_test_vocab(path, &store) != 0) {
        printf("text: the largest vocabulary did not build\n");
        *applied += 1u;
        return 1u;
    }
    printf("text: the largest vocabulary is %s with %llu tokens\n",
           aotx_test_files[largest][0], most);
    unsigned int wrong = 0u;
    for (unsigned int m = 0u; m < AOTX_TEST_SET; ++m) {
        if (m == largest) {
            continue;
        }
        snprintf(path, sizeof path, "%s/%s", models, aotx_test_files[m][0]);
        aotx_modelfile *file = NULL;
        if (aotx_modelfile_open(path, &file) != 0) {
            continue;
        }
        aotx_text_source source;
        unsigned int differ = 0u;
        if (aotx_test_arrays(file, &source) != 0
            || aotx_text_vocab_prefix(source.token_bytes, source.token_at, source.tokens,
                                      &differ) != 0) {
            differ = 1u;
        }
        *applied += 1u;
        if (differ != 0u) {
            printf("text: %u tokens of %s are not the tokens of the table\n", differ,
                   aotx_test_files[m][0]);
            wrong += 1u;
        }
        aotx_modelfile_close(file);
    }
    /* A vocabulary that is not of the set must be refused, and the check is shown to
     * fail here. The merge strings are not tokens, so every one of them must differ. */
    aotx_modelfile *file = NULL;
    snprintf(path, sizeof path, "%s/%s", models, aotx_test_files[largest][0]);
    if (aotx_modelfile_open(path, &file) == 0) {
        aotx_text_source source;
        unsigned int differ = 0u;
        if (aotx_test_arrays(file, &source) == 0) {
            aotx_text_vocab_prefix(source.merge_bytes, source.merge_at, 1024ull, &differ);
            *applied += 1u;
            printf("text: a vocabulary of other strings differs at %u tokens of 1024\n",
                   differ);
            if (differ != 1024u) {
                wrong += 1u;
            }
        }
        aotx_modelfile_close(file);
    }
    aotx_text_vocab_release(&store);
    return wrong;
}

int main(int argc, char **argv)
{
    static unsigned char run[AOTX_TEST_RUN];
    unsigned int applied = 0u;
    unsigned int failed = 0u;
    unsigned int skipped = 0u;
    unsigned int skipped_set = 0u;
    if (argc < 3) {
        printf("text: give the fixture directory and the model directory\n");
        return 2;
    }
    const char *fixtures = argv[1];
    const char *models = argv[2];

    unsigned int start[AOTX_TEST_MAX];
    unsigned int length[AOTX_TEST_MAX];
    unsigned int rows = 0u;
    memset(run, 0, sizeof run);
    if (aotx_test_lines(fixtures, run, start, length, &rows) != 0) {
        printf("text: the fixture lines did not read\n");
        return 1;
    }
    unsigned int lines = rows - 1u;
    unsigned int bytes = start[rows - 1u] * 2u;
    printf("text: %u lines, %u rows, %u bytes of text\n", lines, rows, bytes);

    aotx_test_gear gear;
    aotx_check_runtime(cudaFree(0), "cudaFree");
    aotx_test_open(&gear);
    unsigned int *ids = (unsigned int *)malloc((unsigned long long)rows * AOTX_TEST_STRIDE
                                               * sizeof(unsigned int));
    unsigned int *counts = (unsigned int *)malloc(rows * sizeof(unsigned int));
    if (ids == NULL || counts == NULL) {
        printf("text: the golden list did not fit\n");
        return 1;
    }

    for (unsigned int m = 0u; m < AOTX_TEST_MODELS; ++m) {
        char path[512];
        snprintf(path, sizeof path, "%s/%s", models, aotx_test_files[m][0]);
        aotx_text_store store;
        memset(&store, 0, sizeof store);
        if (aotx_test_vocab(path, &store) != 0) {
            printf("text: skipped the model file %s\n", aotx_test_files[m][0]);
            skipped += 1u;
            if (m < AOTX_TEST_SET) {
                skipped_set += 1u;
            }
            continue;
        }
        snprintf(path, sizeof path, "%s/%s", fixtures, aotx_test_files[m][1]);
        if (aotx_test_golden(path, ids, counts, rows) != 0) {
            printf("text: the golden list %s did not read\n", aotx_test_files[m][1]);
            failed += 1u;
            applied += 1u;
            aotx_text_vocab_release(&store);
            continue;
        }

        /* One launch of 64 sequences, then one launch of every line of the fixture. */
        aotx_test_send(&gear, run, bytes, start, length, rows);
        aotx_test_tokenize(&gear, 0u, AOTX_TEST_BATCH);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        unsigned int wrong = aotx_test_compare(&gear, 0u, AOTX_TEST_BATCH, ids, counts, 1);
        applied += AOTX_TEST_BATCH;
        failed += wrong;
        printf("text: %s at 64 sequences gives %u rows of %u\n", aotx_test_files[m][0],
               AOTX_TEST_BATCH - wrong, AOTX_TEST_BATCH);

        aotx_test_tokenize(&gear, 0u, lines);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        wrong = aotx_test_compare(&gear, 0u, lines, ids, counts, 1);
        applied += lines;
        failed += wrong;
        printf("text: %s at %u sequences gives %u rows of %u\n", aotx_test_files[m][0],
               lines, lines - wrong, lines);

        /* The bytes must come back from the tokens without a change. */
        failed += aotx_test_detok(&gear, lines);
        applied += lines;

        /* One sequence at a time, over every row, which holds the whole file as well. */
        unsigned int alone = 0u;
        for (unsigned int i = 0u; i < rows; ++i) {
            aotx_test_tokenize(&gear, i, 1u);
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            alone += aotx_test_compare(&gear, i, 1u, ids, counts, 1);
        }
        applied += rows;
        failed += alone;
        printf("text: %s at one sequence gives %u rows of %u\n", aotx_test_files[m][0],
               rows - alone, rows);

        if (m == 0u) {
            /* The check must find a difference when the merge order changes. */
            unsigned int pairs = 0u;
            unsigned int *kept = aotx_test_flip(&pairs);
            applied += 1u;
            if (kept == NULL) {
                printf("text: the pair table did not copy\n");
                failed += 1u;
            } else {
                aotx_test_tokenize(&gear, 0u, AOTX_TEST_BATCH);
                aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                unsigned int moved = aotx_test_compare(&gear, 0u, AOTX_TEST_BATCH, ids,
                                                       counts, 0);
                printf("text: the turned pair table changes %u rows of %u\n", moved,
                       AOTX_TEST_BATCH);
                if (moved == 0u) {
                    printf("text: the check did not see the turned pair table\n");
                    failed += 1u;
                }
                aotx_test_restore(kept, pairs);
                aotx_test_tokenize(&gear, 0u, AOTX_TEST_BATCH);
                aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                applied += 1u;
                if (aotx_test_compare(&gear, 0u, AOTX_TEST_BATCH, ids, counts, 0) != 0u) {
                    printf("text: the pair table did not come back\n");
                    failed += 1u;
                }
            }

            /* The pattern cases, which cover the seven alternatives. */
            unsigned int cut = 0u;
            for (unsigned int i = 0u; i < AOTX_TEST_CASES; ++i) {
                cut += aotx_test_bounds(&gear, &aotx_test_cuts[i]);
            }
            applied += AOTX_TEST_CASES;
            failed += cut;
            printf("text: the pattern gives %u boundary cases of %u\n",
                   AOTX_TEST_CASES - cut, AOTX_TEST_CASES);

            /* The gpt2 pattern, on the same table with its pattern row changed. The row
             * comes back before the next check. */
            unsigned int held = aotx_test_pattern(AOTX_TEXT_PATTERN_GPT2);
            unsigned int gpt2 = 0u;
            for (unsigned int i = 0u; i < AOTX_TEST_GPT2_CASES; ++i) {
                gpt2 += aotx_test_bounds(&gear, &aotx_test_gpt2_cuts[i]);
            }
            aotx_test_pattern(held);
            applied += AOTX_TEST_GPT2_CASES;
            failed += gpt2;
            printf("text: the gpt2 pattern gives %u boundary cases of %u\n",
                   AOTX_TEST_GPT2_CASES - gpt2, AOTX_TEST_GPT2_CASES);

            /* The round trip, at 64 sequences and at one. */
            aotx_test_pair trip[AOTX_TEST_BATCH];
            for (unsigned int i = 0u; i < AOTX_TEST_BATCH - AOTX_TEST_BAD; ++i) {
                trip[i].in = (const char *)run + start[i];
                trip[i].in_bytes = length[i];
                trip[i].out = trip[i].in;
                trip[i].out_bytes = length[i];
            }
            for (unsigned int i = 0u; i < AOTX_TEST_BAD; ++i) {
                trip[AOTX_TEST_BATCH - AOTX_TEST_BAD + i] = aotx_test_broken[i];
            }
            unsigned int bad = aotx_test_trip(&gear, trip, AOTX_TEST_BATCH);
            applied += AOTX_TEST_BATCH;
            failed += bad;
            printf("text: the round trip at 64 sequences gives %u of %u\n",
                   AOTX_TEST_BATCH - bad, AOTX_TEST_BATCH);
            unsigned int one = 0u;
            for (unsigned int i = 0u; i < AOTX_TEST_BATCH; ++i) {
                one += aotx_test_trip(&gear, &trip[i], 1u);
            }
            applied += AOTX_TEST_BATCH;
            failed += one;
            printf("text: the round trip at one sequence gives %u of %u\n",
                   AOTX_TEST_BATCH - one, AOTX_TEST_BATCH);

            /* The rate of the batch, at the first 64 lines and at every line. The last
             * two lines hold one run of 600 letters and one run of 1,508 letters, which
             * are one piece each. One warp takes one piece, and the cost of a piece grows
             * with the square of its symbols. Those two lines hold the whole batch. */
            aotx_test_send(&gear, run, bytes, start, length, rows);
            unsigned int spans[2] = { AOTX_TEST_BATCH, lines };
            for (unsigned int s = 0u; s < 2u; ++s) {
                unsigned int over = 0u;
                for (unsigned int i = 0u; i < spans[s]; ++i) {
                    over += length[i];
                }
                aotx_test_tokenize(&gear, 0u, spans[s]);
                aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                double from = aotx_test_now();
                for (unsigned int i = 0u; i < AOTX_TEST_REPEATS; ++i) {
                    aotx_test_tokenize(&gear, 0u, spans[s]);
                }
                aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                double spent = aotx_test_now() - from;
                double moved = (double)over * (double)AOTX_TEST_REPEATS;
                printf("text: %u launches of %u sequences, %u bytes, %.0f us a launch, "
                       "%.2f MB a second\n", AOTX_TEST_REPEATS, spans[s], over,
                       spent * 1e6 / (double)AOTX_TEST_REPEATS,
                       moved / spent / (1024.0 * 1024.0));
            }
        }
        aotx_text_vocab_release(&store);
    }

    /* One table for the set: the largest vocabulary, with every other one compared. */
    if (skipped_set == 0u) {
        failed += aotx_test_prefix(models, &applied);
    }

    free(ids);
    free(counts);
    if (skipped != 0u) {
        printf("text: skipped %u model files that are not there\n", skipped);
    }
    printf("text: %u cases applied, %u failed\n", applied, failed);
    return (failed == 0u && applied > 0u) ? 0 : 1;
}
