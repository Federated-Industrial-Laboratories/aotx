/* Purpose: Score the two sides of each conversation of a pairs file on a rubric.
 * Owns: The pairs and their turns, the rubric, the queries and the output files of a run.
 * Launch shape: Host glue only; the answer read, the means, the tallies and the blind order run in kernels.
 * Lifetime: One program run. */
#include "tools/quality_score.h"

#define AOTX_PAIR_MAX      256u
#define AOTX_PAIR_TOKENS   400u
#define AOTX_PAIR_MARGIN   0.01f
#define AOTX_PAIR_SEED     7u
#define AOTX_PAIR_Z        1.644853627f /* the normal quantile of the two-sided 90 percent interval */

__global__ void aotx_quality_score_answer(const float *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float *);
__global__ void aotx_quality_score_pairs(const float *, unsigned int, unsigned int, float, float *, float *, float *);
__global__ void aotx_quality_score_tally(const float *, const float *, unsigned int, unsigned int, float, double *);
__global__ void aotx_quality_score_shuffle(unsigned int, unsigned int, unsigned int *);

/* Read the pairs file and the rubric. A malformed line of either is refused by its number. */
static int read_inputs(aotx_pair_set *set, const char *pairs_path, const char *rubric_path)
{
    char *line = (char *)malloc(AOTX_PAIR_LINE), *value = (char *)malloc(AOTX_PAIR_LINE); unsigned int number = 0u;
    set->pair = (aotx_pair *)calloc(AOTX_PAIR_MAX, sizeof *set->pair);
    FILE *in = fopen(pairs_path, "r");
    if (line == 0 || value == 0 || set->pair == 0) { fprintf(stderr, "the pairs do not fit in memory\n"); return 1; }
    if (in == 0) { fprintf(stderr, "the pairs file %s does not open\n", pairs_path); return 1; }
    while (fgets(line, (int)AOTX_PAIR_LINE, in) != 0) {
        number += 1u;
        if (strchr(line, '\n') == 0 && !feof(in)) { fprintf(stderr, "the pairs file %s line %u is longer than %u bytes\n", pairs_path, number, AOTX_PAIR_LINE); return 1; }
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') continue;
        if (set->pairs >= AOTX_PAIR_MAX) { fprintf(stderr, "the pairs file %s holds more than %u pairs\n", pairs_path, AOTX_PAIR_MAX); return 1; }
        if (read_pair_line(line, &set->pair[set->pairs], value)) {
            fprintf(stderr, "the pairs file %s line %u is malformed: a pair has a name and 1 to %u turns, each with the keys user, a and b\n", pairs_path, number, AOTX_PAIR_TURNS);
            return 1;
        }
        set->pairs += 1u;
    }
    fclose(in); free(line); free(value);
    if (set->pairs == 0u) { fprintf(stderr, "the pairs file %s holds no pair\n", pairs_path); return 1; }
    char row[AOTX_PAIR_QUESTION + AOTX_PAIR_ID]; in = fopen(rubric_path, "r"); number = 0u;
    if (in == 0) { fprintf(stderr, "the rubric %s does not open\n", rubric_path); return 1; }
    while (fgets(row, sizeof row, in) != 0) {
        number += 1u;
        row[strcspn(row, "\r\n")] = '\0';
        if (row[0] == '\0') continue;
        char *tab = strchr(row, '\t');
        if (tab == 0 || tab == row || tab[1] == '\0' || (size_t)(tab - row) >= AOTX_PAIR_ID || strlen(tab + 1) >= AOTX_PAIR_QUESTION || strchr(tab + 1, '\t') != 0
            || strspn(row, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != (size_t)(tab - row)) {
            fprintf(stderr, "the rubric %s line %u is malformed: an item is a name of letters, digits, underscore and dash under %u bytes, one tab and a question under %u bytes\n", rubric_path, number, AOTX_PAIR_ID, AOTX_PAIR_QUESTION);
            return 1;
        }
        if (set->items >= AOTX_PAIR_ITEMS) { fprintf(stderr, "the rubric %s holds more than %u items\n", rubric_path, AOTX_PAIR_ITEMS); return 1; }
        *tab = '\0';
        snprintf(set->id[set->items], AOTX_PAIR_ID, "%.31s", row);
        snprintf(set->question[set->items], AOTX_PAIR_QUESTION, "%.511s", tab + 1);
        set->items += 1u;
    }
    fclose(in);
    if (set->items == 0u) { fprintf(stderr, "the rubric %s holds no item\n", rubric_path); return 1; }
    return 0;
}

/* The tokens of yes and of no, found once. A word that is not one token stops the tool. */
static int answers_of(aotx_steer_run *run, unsigned int *yes, unsigned int *no)
{
    unsigned int ids[2u * AOTX_STEER_STRIDE], counts[2]; char *text[2] = { (char *)"yes", (char *)"no" };
    if (aotx_steer_tokenize(&run->tokenizer, text, 2u, ids, counts) != 0) { fprintf(stderr, "the answer words do not fit the tokenizer batch\n"); return 1; }
    for (unsigned int w = 0u; w < 2u; ++w) {
        if (counts[w] != 1u) { fprintf(stderr, "the word %s is %u tokens, not one; the tool does not run\n", text[w], counts[w]); return 1; }
        printf("answer %s: token %u\n", text[w], ids[w * AOTX_STEER_STRIDE]);
    }
    *yes = ids[0]; *no = ids[AOTX_STEER_STRIDE];
    return 0;
}

static char *join(const char *a, const char *b, const char *c)
{
    size_t n = strlen(a) + strlen(b) + strlen(c);
    char *out = (char *)malloc(n + 1u);
    if (out == 0 || n >= AOTX_STEER_CLEAN) { free(out); return 0; }
    snprintf(out, n + 1u, "%s%s%s", a, b, c);
    return out;
}

/* Build the queries of one side of a pair: the conversation with the replies of the side,
 * then each question with the answer line. A conversation over the token bound is cut to
 * its last turns inside the bound, with the first user line kept. When the first user
 * line and the last turn do not fit together, the reply of the last turn stands alone.
 * It is cut from its front to the bound. A cut pair is marked. The token counts of the
 * blocks come from the tokenizer batch. */
static int build_side(aotx_steer_run *run, aotx_pair *pair, int side, const aotx_pair_set *set, aotx_steer_set *queries)
{
    char *block[2u * AOTX_PAIR_TURNS]; unsigned int counts[2u * AOTX_PAIR_TURNS], keep = 0u, total = 0u, last = pair->turns - 1u;
    const char *reply = side ? pair->turn[last].b : pair->turn[last].a;
    for (unsigned int t = 0u; t < pair->turns; ++t) {
        block[2u * t] = join(AOTX_SCORE_USER_HEAD, pair->turn[t].user, AOTX_SCORE_BLOCK_END);
        block[2u * t + 1u] = join(AOTX_SCORE_ASSISTANT_HEAD, side ? pair->turn[t].b : pair->turn[t].a, AOTX_SCORE_BLOCK_END);
        if (block[2u * t] == 0 || block[2u * t + 1u] == 0) { fprintf(stderr, "the pair %s turn %u does not fit one block of %u bytes\n", pair->name, t + 1u, AOTX_STEER_CLEAN); return 1; }
    }
    if (aotx_steer_tokenize(&run->tokenizer, block, 2u * pair->turns, 0, counts) != 0) { fprintf(stderr, "the pair %s does not fit the tokenizer batch\n", pair->name); return 1; }
    for (unsigned int i = 0u; i < 2u * pair->turns; ++i) total += counts[i];
    int alone = 0;
    if (total > AOTX_PAIR_TOKENS) {
        unsigned int room = (counts[0] < AOTX_PAIR_TOKENS) ? AOTX_PAIR_TOKENS - counts[0] : 0u, tail = 0u;
        keep = last;
        while (keep > 0u && tail + counts[2u * keep] + counts[2u * keep + 1u] <= room) { tail += counts[2u * keep] + counts[2u * keep + 1u]; keep -= 1u; }
        if (tail != 0u) keep += 1u; else alone = 1;
        pair->cut = 1u;
    }
    char *text = (char *)calloc(AOTX_STEER_CLEAN * 2u, 1u); size_t at = 0u;
    if (text == 0) return 1;
    if (alone) {
        /* The reply of the last turn alone. Each round cuts it from its front by the byte
         * share of the tokens over the bound, until it fits. The user block of the turn
         * goes with it when both fit. */
        size_t skip = 0u, length = strlen(reply); unsigned int count = counts[2u * last + 1u];
        for (unsigned int round = 0u; round < 8u && count > AOTX_PAIR_TOKENS; ++round) {
            skip += (length - skip) * (count - AOTX_PAIR_TOKENS) / count + 1u;
            char *piece = join(AOTX_SCORE_ASSISTANT_HEAD, reply + skip, AOTX_SCORE_BLOCK_END);
            if (piece == 0 || aotx_steer_tokenize(&run->tokenizer, &piece, 1u, 0, &count) != 0) { free(piece); free(text); return 1; }
            free(piece);
        }
        if (counts[2u * last] + count <= AOTX_PAIR_TOKENS) at += (size_t)snprintf(text, AOTX_STEER_CLEAN * 2u, "%s", block[2u * last]);
        at += (size_t)snprintf(text + at, AOTX_STEER_CLEAN * 2u - at, "%s%s%s", AOTX_SCORE_ASSISTANT_HEAD, reply + skip, AOTX_SCORE_BLOCK_END);
    } else {
        /* The kept blocks: the first user line alone before a cut, then the tail turns. */
        if (keep != 0u) at += (size_t)snprintf(text, AOTX_STEER_CLEAN * 2u, "%s", block[0]);
        for (unsigned int i = (keep == 0u) ? 0u : 2u * keep; i < 2u * pair->turns; ++i) {
            if (at < AOTX_STEER_CLEAN * 2u) at += (size_t)snprintf(text + at, AOTX_STEER_CLEAN * 2u - at, "%s", block[i]);
        }
    }
    for (unsigned int i = 0u; i < 2u * pair->turns; ++i) free(block[i]);
    for (unsigned int item = 0u; item < set->items; ++item) {
        char question[AOTX_PAIR_QUESTION + 128u];
        snprintf(question, sizeof question, "%s%s\nAnswer yes or no.%s%s%s", AOTX_SCORE_USER_HEAD, set->question[item], AOTX_SCORE_BLOCK_END, AOTX_SCORE_ASSISTANT_HEAD, AOTX_SCORE_THINK_OFF);
        queries->text[queries->texts] = join(text, question, "");
        if (queries->text[queries->texts] == 0) { fprintf(stderr, "the pair %s does not fit one query of %u bytes\n", pair->name, AOTX_STEER_CLEAN); return 1; }
        queries->texts += 1u;
    }
    free(text);
    return 0;
}

/* Write a text as the content of a JSON string. The quote and the backslash take their
 * escape, and a control byte takes its unicode escape. */
static void print_json_text(FILE *out, const char *text)
{
    for (; *text; ++text) {
        if (*text == '"' || *text == '\\') fprintf(out, "\\%c", *text);
        else if ((unsigned char)*text < 0x20u) fprintf(out, "\\u%04x", (unsigned char)*text);
        else fputc(*text, out);
    }
}

/* Write the blinded transcripts and their key. Side b prints first as X where the order
 * of the pair says so, else side a does. */
static int write_blind(const char *out_dir, const aotx_pair_set *set, const unsigned int *swap)
{
    char path[AOTX_STEER_PATH];
    snprintf(path, sizeof path, "%s/pairs-blind.md", out_dir); FILE *md = fopen(path, "w");
    snprintf(path, sizeof path, "%s/pairs-key.tsv", out_dir); FILE *key = fopen(path, "w");
    if (md == 0 || key == 0 || fprintf(key, "pair\tname\tX\tY\n") < 0) { fprintf(stderr, "the blind files of %s do not write\n", out_dir); return 1; }
    for (unsigned int i = 0u; i < set->pairs; ++i) {
        const aotx_pair *pair = &set->pair[i];
        fprintf(md, "## pair %u: %s\n\n", i + 1u, pair->name);
        fprintf(key, "%u\t%s\t%s\t%s\n", i + 1u, pair->name, swap[i] ? "b" : "a", swap[i] ? "a" : "b");
        for (unsigned int t = 0u; t < pair->turns; ++t) {
            const char *x = swap[i] ? pair->turn[t].b : pair->turn[t].a, *y = swap[i] ? pair->turn[t].a : pair->turn[t].b;
            fprintf(md, "**user:** %s\n\n**X:** %s\n\n**Y:** %s\n\n", pair->turn[t].user, x, y);
        }
    }
    return (fclose(md) != 0) | (fclose(key) != 0);
}

int aotx_quality_pair(const char *models, const char *role, const char *pairs_path, const char *rubric_path, const char *out_dir, int blind)
{
    aotx_pair_set set; aotx_steer_run run; unsigned int yes, no; char path[AOTX_STEER_PATH];
    memset(&set, 0, sizeof set);
    setvbuf(stdout, 0, _IOLBF, 0);
    if (read_inputs(&set, pairs_path, rubric_path)) return 2;
    if (mkdir(out_dir, 0755) != 0 && errno != EEXIST) { fprintf(stderr, "the directory %s does not open\n", out_dir); return 2; }
    if (aotx_steer_run_open(&run, models, role)) return 1;
    printf("pairs %s: role %s, %u pairs, %u items, blind %d\n", pairs_path, role, set.pairs, set.items, blind ? 1 : 0);
    unsigned int chances = set.pairs * 2u * set.items;
    float *p = (float *)aotx_steer_run_take(&run, chances * sizeof(float));
    float *side = (float *)aotx_steer_run_take(&run, set.pairs * 2u * sizeof(float));
    float *result = (float *)aotx_steer_run_take(&run, set.pairs * sizeof(float));
    float *mark = (float *)aotx_steer_run_take(&run, set.pairs * set.items * sizeof(float));
    double *figures = (double *)aotx_steer_run_take(&run, (5u + set.items) * sizeof(double));
    unsigned int *swap = (unsigned int *)aotx_steer_run_take(&run, set.pairs * sizeof(unsigned int));
    float *logits = (float *)aotx_steer_run_take(&run, (size_t)AOTX_STEER_PASS_TEXTS * run.desc.vocab * sizeof(float));
    if (answers_of(&run, &yes, &no)) return 1;
    for (unsigned int i = 0u; i < set.pairs; ++i) {
        aotx_steer_set queries; memset(&queries, 0, sizeof queries); queries.name = set.pair[i].name;
        if (build_side(&run, &set.pair[i], 0, &set, &queries) || build_side(&run, &set.pair[i], 1, &set, &queries)) return 1;
        if (aotx_steer_set_count(&run, &queries)) return 1;
        aotx_steer_set_plan(&queries);
        for (unsigned int pass = 0u; pass < queries.passes; ++pass) {
            unsigned int seqs = aotx_steer_set_seqs(&queries, pass);
            if (aotx_steer_run_pass(&run, &queries, pass, 0, logits, AOTX_MODEL_ROWS_LAST, 0, 0, 0u) != 0) return 1;
            aotx_quality_score_answer<<<(seqs + 255u) / 256u, 256u>>>(logits, run.desc.vocab, yes, no, i * 2u * set.items + queries.first[pass], seqs, p);
        }
        for (unsigned int q = 0u; q < queries.texts; ++q) free(queries.text[q]);
    }
    aotx_quality_score_pairs<<<(set.pairs + 255u) / 256u, 256u>>>(p, set.pairs, set.items, AOTX_PAIR_MARGIN, side, result, mark);
    aotx_quality_score_tally<<<1, 256u>>>(result, mark, set.pairs, set.items, AOTX_PAIR_Z, figures);
    aotx_quality_score_shuffle<<<(set.pairs + 255u) / 256u, 256u>>>(set.pairs, AOTX_PAIR_SEED, swap);
    /* The figures come to the host for the printed lines and the files. */
    float *host_p = (float *)calloc(chances + set.pairs * 3u, sizeof(float)), *host_side = host_p + chances, *host_result = host_side + set.pairs * 2u;
    double *host_figures = (double *)calloc(5u + set.items, sizeof(double)); unsigned int *host_swap = (unsigned int *)calloc(set.pairs, sizeof(unsigned int));
    if (!host_p || !host_figures || !host_swap) { fprintf(stderr, "the figures do not fit in memory\n"); return 1; }
    aotx_check_runtime(cudaMemcpy(host_p, p, chances * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(host_side, side, set.pairs * 2u * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(host_result, result, set.pairs * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(host_figures, figures, (5u + set.items) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(host_swap, swap, set.pairs * sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy");
    snprintf(path, sizeof path, "%s/pairs.jsonl", out_dir);
    FILE *out = fopen(path, "w"); unsigned int cuts = 0u;
    if (out == 0) { fprintf(stderr, "the pairs file %s does not write\n", path); return 1; }
    for (unsigned int i = 0u; i < set.pairs; ++i) {
        const char *word = (host_result[i] == 1.0f) ? "win" : ((host_result[i] == 0.0f) ? "loss" : "tie");
        cuts += set.pair[i].cut;
        fprintf(out, "{\"name\":\""); print_json_text(out, set.pair[i].name); fprintf(out, "\",\"cut\":%u", set.pair[i].cut);
        for (unsigned int s = 0u; s < 2u; ++s) {
            fprintf(out, ",\"%c\":{", s ? 'b' : 'a');
            for (unsigned int item = 0u; item < set.items; ++item) fprintf(out, "%s\"%s\":%.9g", item ? "," : "", set.id[item], (double)host_p[(i * 2u + s) * set.items + item]);
            fprintf(out, "}");
        }
        fprintf(out, ",\"a_score\":%.9g,\"b_score\":%.9g,\"result\":\"%s\"}\n", (double)host_side[i * 2u], (double)host_side[i * 2u + 1u], word);
        printf("pair %u %s: a %.9g b %.9g %s%s\n", i + 1u, set.pair[i].name, (double)host_side[i * 2u], (double)host_side[i * 2u + 1u], word, set.pair[i].cut ? ", cut" : "");
    }
    char line[2048]; int n = snprintf(line, sizeof line, "{\"summary\":1,\"pairs\":%u,\"wins\":%u,\"ties\":%u,\"win_rate\":%.9g,\"wilson_low\":%.9g,\"wilson_high\":%.9g,\"items\":{",
                                      set.pairs, (unsigned int)host_figures[0], (unsigned int)host_figures[1], host_figures[2], host_figures[3], host_figures[4]);
    for (unsigned int item = 0u; item < set.items && n > 0 && (size_t)n < sizeof line; ++item) n += snprintf(line + n, sizeof line - (size_t)n, "%s\"%s\":%.9g", item ? "," : "", set.id[item], host_figures[5u + item]);
    if (n > 0 && (size_t)n < sizeof line) n += snprintf(line + n, sizeof line - (size_t)n, "},\"cut\":%u}", cuts);
    if (n < 0 || (size_t)n >= sizeof line || fprintf(out, "%s\n", line) < 0 || fclose(out) != 0) { fprintf(stderr, "the pairs file %s does not write\n", path); return 1; }
    printf("%s\n", line);
    int state = blind ? write_blind(out_dir, &set, host_swap) : 0;
    free(host_p); free(host_figures); free(host_swap);
    for (unsigned int i = 0u; i < set.pairs; ++i) for (unsigned int t = 0u; t < set.pair[i].turns; ++t) { free(set.pair[i].turn[t].user); free(set.pair[i].turn[t].a); free(set.pair[i].turn[t].b); }
    free(set.pair); aotx_steer_run_close(&run);
    return state;
}
