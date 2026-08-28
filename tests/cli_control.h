/* Purpose: Check that the console holds no bytes of a control token.
 * Owns: The buffers of that case.
 * Threading: One host thread; the command line check calls the case one at a time.
 * Lifetime: The program.
 *
 * The file is a part of the command line check. It reads the helpers of the say check, so
 * it comes after them in the same translation unit. */
#ifndef AOTX_TEST_CLI_CONTROL_H
#define AOTX_TEST_CLI_CONTROL_H

/* The token that ends a turn of this model family. The wrap of the say path holds it, so
 * the golden list of a text names the tokens of the text in front of it. */
#define AOTX_TEST_END_TOKEN  151645u

/* Report whether a run of bytes holds a text. */
static int aotx_test_holds(const unsigned char *bytes, unsigned int length,
                           const char *text)
{
    unsigned int span = (unsigned int)strlen(text);
    if (span == 0u || length < span) {
        return 0;
    }
    for (unsigned int at = 0u; at + span <= length; ++at) {
        if (memcmp(bytes + at, text, span) == 0) {
            return 1;
        }
    }
    return 0;
}

/* The reply of a slot ends with the token that ends a turn. That token is of the control
 * type, so the console shows no byte of it and the console record carries none.
 *
 * The reply tokens are the tokens of the text of the slot. The golden list holds them
 * between the head of the wrap and the token that ends the turn. The console line and the
 * console record must therefore hold that text and nothing else. */
static void aotx_test_control_bytes(const char *fixtures, const char *models,
                                    unsigned int count)
{
    char path[1024];
    char (*text)[AOTX_BODY_BYTES] = (char (*)[AOTX_BODY_BYTES])
        calloc(AOTX_TEST_TEXTS, AOTX_BODY_BYTES);
    unsigned int *golden = (unsigned int *)calloc(AOTX_TEST_TEXTS * AOTX_SAY_TOKENS,
                                                  sizeof(unsigned int));
    unsigned int *golden_count = (unsigned int *)calloc(AOTX_TEST_TEXTS,
                                                        sizeof(unsigned int));
    aotx_console_state *console = (aotx_console_state *)malloc(sizeof *console);
    aotx_test_record *found = (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);
    aotx_token_body *body = NULL;
    aotx_token_body *on = NULL;
    unsigned int *bad = NULL;
    aotx_text_store store;
    unsigned int texts = 0u;
    unsigned int rows = 0u;
    unsigned int refused = 0u;
    unsigned int wrong = 0u;
    unsigned int made = 0u;
    unsigned int shown = 0u;
    unsigned int clean = 0u;
    unsigned int carried = 0u;
    unsigned int before = 0u;
    unsigned long long lines = 0ull;
    aotx_seq seq;

    snprintf(path, sizeof path, "%s/say-texts.dat", fixtures);
    texts = aotx_test_texts(path, text, AOTX_TEST_TEXTS);
    snprintf(path, sizeof path, "%s/golden-say.ids", fixtures);
    rows = aotx_test_golden(path, golden, golden_count, AOTX_TEST_TEXTS);
    if (texts == 0u || rows != texts) {
        printf("cli: the say fixture holds %u texts and %u golden rows; the control check "
               "is skipped\n", texts, rows);
        goto done;
    }
    snprintf(path, sizeof path, "%s/Qwen3-4B-Q8_0.gguf", models);
    memset(&store, 0, sizeof store);
    if (aotx_test_vocab(path, &store) != 0) {
        printf("cli: the language model file is not at %s; the control check is skipped\n",
               path);
        goto done;
    }

    aotx_check_runtime(cudaMalloc(&on, (size_t)count * AOTX_SAY_TOKENS * sizeof *on),
                       "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&bad, sizeof *bad), "cudaMalloc");
    aotx_check_runtime(cudaMemset(bad, 0, sizeof *bad), "cudaMemset");
    body = (aotx_token_body *)calloc((size_t)count * AOTX_SAY_TOKENS, sizeof *body);

    /* The slots take the wrapped texts and the say path opens each one. */
    aotx_test_pipeline(text, texts, count, &refused);
    before = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    aotx_test_console_state(console);
    lines = console->count;

    /* The reply of each slot is the tokens of its text and then the token that ends the
     * turn. The last of them carries the mark that ends the sequence. */
    for (unsigned int slot = 0u; slot < count; ++slot) {
        unsigned int row = slot % texts;
        const unsigned int *ids = golden + row * AOTX_SAY_TOKENS;
        unsigned int end = 0u;
        while (end < golden_count[row] && ids[end] != AOTX_TEST_END_TOKEN) {
            end += 1u;
        }
        aotx_test_slot(slot, &seq);
        for (unsigned int i = 3u; i <= end; ++i) {
            aotx_token_body *one = &body[made];
            one->slot = slot;
            one->token = (i < end) ? ids[i] : AOTX_TEST_END_TOKEN;
            one->position = seq.prompt + (i - 3u);
            one->flags = AOTX_TOKEN_SAMPLED | ((i == end) ? AOTX_TOKEN_LAST : 0u);
            one->seed = seq.seed;
            one->draw = i - 3u;
            one->role = seq.role;
            made += 1u;
        }
    }
    aotx_check_runtime(cudaMemcpy(on, body, (size_t)made * sizeof *body,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_test_tokens<<<1, 1>>>(on, made, bad);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&wrong, bad, sizeof wrong, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");

    /* The reply node of the tick takes the text of the new tokens and shows it. */
    aotx_say_reply<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    shown = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    aotx_test_console_state(console);

    /* No console record of this case holds the bytes of the token that ends the turn.
     * One record of each slot holds the text of that slot and nothing more. */
    for (unsigned int i = before; i < shown; ++i) {
        if (!aotx_test_holds(found[i].body, found[i].length, "<|im_end|>")) {
            clean += 1u;
        }
    }
    for (unsigned int slot = 0u; slot < count; ++slot) {
        const char *want = text[slot % texts];
        unsigned int span = (unsigned int)strlen(want);
        for (unsigned int i = before; i < shown; ++i) {
            if (found[i].length == span && memcmp(found[i].body, want, span) == 0) {
                carried += 1u;
                break;
            }
        }
    }
    aotx_test_check(refused == 0u && wrong == 0u,
                    "every slot took its text and its reply tokens");
    aotx_test_check(shown >= before + count, "each slot of the batch wrote a console record");
    aotx_test_check(clean == shown - before,
                    "no console record holds the bytes of a control token");
    aotx_test_check(carried == count,
                    "one console record of each slot holds the text of its reply");

    /* The console buffer holds the same bytes. No line of this case names the control
     * token, and the line of each slot is the text of that slot. */
    clean = 0u;
    carried = 0u;
    for (unsigned long long i = lines + 1ull; i <= console->count; ++i) {
        const aotx_console_line *line = aotx_test_at(console, i);
        if (line != NULL && !aotx_test_holds(line->text, line->length, "<|im_end|>")) {
            clean += 1u;
        }
    }
    for (unsigned int slot = 0u; slot < count; ++slot) {
        for (unsigned long long i = lines + 1ull; i <= console->count; ++i) {
            if (aotx_test_says(aotx_test_at(console, i), text[slot % texts])) {
                carried += 1u;
                break;
            }
        }
    }
    aotx_test_check(console->count >= lines + (unsigned long long)count,
                    "each slot of the batch gave a console line");
    aotx_test_check(clean == (unsigned int)(console->count - lines),
                    "no console line holds the bytes of a control token");
    aotx_test_check(carried == count, "the console line of each slot is the text of its "
                                      "reply");
    printf("cli: %u slots replied with %u tokens that end with the control token, %u "
           "console records and %llu console lines hold no control bytes\n", count, made,
           shown - before, console->count - lines);
    aotx_text_vocab_release(&store);

done:
    free(text);
    free(golden);
    free(golden_count);
    free(console);
    free(found);
    free(body);
    cudaFree(on);
    cudaFree(bad);
}

#endif
