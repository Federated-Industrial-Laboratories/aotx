/* Purpose: Score a four-choice task set under a steer vector at each named dose.
 * Owns: The capability file and its lines; the task queries and the buffers of a run.
 * Launch shape: Host glue only; the letter read, the argmax and the mean run in kernels.
 * Lifetime: One program run. The pair mode has its own glue in quality_pair_host.cu. */
#include "tools/quality_score.h"

#define AOTX_SCORE_LETTERS 4u
#define AOTX_SCORE_DOSES   8u
#define AOTX_SCORE_ID      32u
#define AOTX_SCORE_QUERY   (AOTX_STEER_SET_LINE + 256u)

__global__ void aotx_quality_score_letter(const float *, unsigned int, const unsigned int *,
                                          unsigned int, const unsigned int *, unsigned int,
                                          unsigned int, unsigned int *, float *, float *);
__global__ void aotx_quality_score_mean(const float *, unsigned int, double *);

/* The chat wrap of one user message with a generation prompt and thinking off: the bytes
 * the say path puts around a text (cli/prompt.cuh). The answer of the model starts after
 * the wrap. The last row of the query therefore holds the logits of the first letter. */
static const char aotx_score_head[] = AOTX_SCORE_USER_HEAD;
static const char aotx_score_tail[] = AOTX_SCORE_BLOCK_END AOTX_SCORE_ASSISTANT_HEAD AOTX_SCORE_THINK_OFF;
static const char aotx_score_letter_text[AOTX_SCORE_LETTERS][2] = { "A", "B", "C", "D" };

/* The task set: one query text for each item. It also holds the answer of each item as a
 * letter number and its name for the printed lines. */
typedef struct aotx_score_tasks {
    aotx_steer_set set;
    unsigned int answer[AOTX_STEER_SET_TEXTS];
    char id[AOTX_STEER_SET_TEXTS][AOTX_SCORE_ID];
} aotx_score_tasks;

/* Read the task file. Each line is one item of seven fields with a tab between them: the
 * name, the question, the four choices and the answer letter. A line that does not have
 * that form is refused by its number, and the tool does not run. An empty line holds no
 * item. The query of an item is the question, the four choices and the instruction line,
 * in the chat wrap. */
static int read_tasks(aotx_score_tasks *tasks, const char *path)
{
    FILE *in = fopen(path, "r");
    char line[AOTX_STEER_SET_LINE], query[AOTX_SCORE_QUERY];
    unsigned int number = 0u;
    memset(tasks, 0, sizeof *tasks);
    tasks->set.name = path;
    if (in == 0) { fprintf(stderr, "the tasks file %s does not open\n", path); return 1; }
    while (fgets(line, sizeof line, in) != 0) {
        char *field[7]; unsigned int fields = 0u, letter;
        number += 1u;
        if (strchr(line, '\n') == 0 && !feof(in)) {
            fprintf(stderr, "the tasks file %s line %u is longer than %u bytes\n", path, number, AOTX_STEER_SET_LINE);
            fclose(in); return 1;
        }
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') continue;
        char *at = line;
        while (at != 0 && fields < 7u) {
            char *tab = strchr(at, '\t');
            field[fields++] = at;
            if (tab != 0) *tab++ = '\0';
            at = tab;
        }
        int malformed = fields != 7u || at != 0 || strlen(field[0]) >= AOTX_SCORE_ID;
        for (unsigned int f = 0u; f < fields; ++f) if (field[f][0] == '\0') malformed = 1;
        letter = malformed ? 0u : (unsigned int)(field[6][0] - 'A');
        if (!malformed && (field[6][1] != '\0' || letter >= AOTX_SCORE_LETTERS)) malformed = 1;
        if (malformed) {
            fprintf(stderr, "the tasks file %s line %u is malformed: an item has seven fields, none empty, "
                            "a name under %u bytes and an answer of one letter A to D\n", path, number, AOTX_SCORE_ID);
            fclose(in); return 1;
        }
        if (tasks->set.texts >= AOTX_STEER_SET_TEXTS) {
            fprintf(stderr, "the tasks file %s holds more than %u items\n", path, AOTX_STEER_SET_TEXTS);
            fclose(in); return 1;
        }
        int n = snprintf(query, sizeof query, "%s%s\nA. %s\nB. %s\nC. %s\nD. %s\nAnswer with one letter.%s",
                         aotx_score_head, field[1], field[2], field[3], field[4], field[5], aotx_score_tail);
        if (n < 0 || (size_t)n >= sizeof query) {
            fprintf(stderr, "the tasks file %s line %u does not fit one query\n", path, number);
            fclose(in); return 1;
        }
        unsigned int i = tasks->set.texts;
        tasks->set.text[i] = strdup(query);
        if (tasks->set.text[i] == 0) { fprintf(stderr, "the tasks file %s does not fit in memory\n", path); fclose(in); return 1; }
        snprintf(tasks->id[i], AOTX_SCORE_ID, "%.31s", field[0]);
        tasks->answer[i] = letter;
        tasks->set.texts += 1u;
    }
    fclose(in);
    if (tasks->set.texts == 0u) { fprintf(stderr, "the tasks file %s holds no item\n", path); return 1; }
    return 0;
}

/* The token of each letter, found once by the tokenizer batch. A letter that is not one
 * token stops the tool, because the score reads one logit for each letter. */
static int letters_of(aotx_steer_run *run, unsigned int *device_letter)
{
    unsigned int ids[AOTX_SCORE_LETTERS * AOTX_STEER_STRIDE], counts[AOTX_SCORE_LETTERS], token[AOTX_SCORE_LETTERS];
    char *text[AOTX_SCORE_LETTERS];
    for (unsigned int l = 0u; l < AOTX_SCORE_LETTERS; ++l) text[l] = (char *)aotx_score_letter_text[l];
    if (aotx_steer_tokenize(&run->tokenizer, text, AOTX_SCORE_LETTERS, ids, counts) != 0) {
        fprintf(stderr, "the letters do not fit the tokenizer batch\n");
        return 1;
    }
    for (unsigned int l = 0u; l < AOTX_SCORE_LETTERS; ++l) {
        if (counts[l] != 1u) {
            fprintf(stderr, "the letter %s is %u tokens, not one; the tool does not run\n", text[l], counts[l]);
            return 1;
        }
        token[l] = ids[l * AOTX_STEER_STRIDE];
        printf("letter %s: token %u\n", text[l], token[l]);
    }
    aotx_check_runtime(cudaMemcpy(device_letter, token, sizeof token, cudaMemcpyHostToDevice), "cudaMemcpy");
    return 0;
}

/* The doses of the list: each a finite figure at or above zero. */
static int doses_of(const char *text, float *dose)
{
    unsigned int count = 0u;
    while (*text && count < AOTX_SCORE_DOSES) {
        char *end = 0; float value = strtof(text, &end);
        if (end == text || !isfinite(value) || value < 0.0f) return -1;
        dose[count++] = value;
        if (*end == ',') text = end + 1; else if (*end == '\0') text = end; else return -1;
    }
    return (count && *text == '\0') ? (int)count : -1;
}

/* One pass set at one dose. Every query of a pass takes the how row with the vector in the
 * first steer slot at the dose. Dose zero runs the passes plain. The score of each item and
 * the mean over the set come back from the kernels. */
static int score_dose(aotx_steer_run *run, const aotx_steer_set *set, unsigned int vector_id, float dose,
                      float *logits, const unsigned int *letter, const unsigned int *answer,
                      unsigned int *largest, float *top, float *right, double *mean, double *score)
{
    aotx_model_how one; const aotx_model_how *how = 0;
    if (dose > 0.0f) {
        aotx_steer_how_plain(&one);
        one.steer[0] = vector_id; one.steer_strength[0] = dose;
        how = aotx_steer_run_how(run, &one);
    }
    for (unsigned int p = 0u; p < set->passes; ++p) {
        unsigned int seqs = aotx_steer_set_seqs(set, p), first = set->first[p];
        if (aotx_steer_run_pass(run, set, p, how, logits, AOTX_MODEL_ROWS_LAST, 0, 0, 0u) != 0) return 1;
        aotx_quality_score_letter<<<(seqs + 255u) / 256u, 256u>>>(logits, run->desc.vocab, letter, AOTX_SCORE_LETTERS,
                                                                  answer, first, seqs, largest, top, right);
    }
    aotx_quality_score_mean<<<1, 256u>>>(right, set->texts, mean);
    aotx_check_runtime(cudaMemcpy(score, mean, sizeof *score, cudaMemcpyDeviceToHost), "cudaMemcpy");
    return 0;
}

static int usage(void)
{
    fprintf(stderr, "usage: aotx_quality_score --models DIR --tasks FILE --axis NAME --doses LIST --out DIR\n"
                    "                          [--role NAME] [--print-items]\n"
                    "       aotx_quality_score --models DIR --pairs FILE --rubric FILE --out DIR [--blind 1]\n"
                    "                          [--role NAME]\n");
    return 2;
}

int main(int argc, char **argv)
{
    const char *models = 0, *tasks_path = 0, *axis = 0, *dose_text = 0, *out_dir = 0, *role = "language";
    const char *pairs_path = 0, *rubric_path = 0;
    int print_items = 0, blind = 0; float dose[AOTX_SCORE_DOSES];
    aotx_score_tasks tasks; aotx_steer_run run; aotx_steer_vector vector; unsigned int vector_id;
    char path[AOTX_STEER_PATH];
    /* The printed lines are the evidence of a run, so they leave the program as they come. */
    setvbuf(stdout, 0, _IOLBF, 0);
    for (int i = 1; i < argc; ) {
        if (!strcmp(argv[i], "--print-items")) { print_items = 1; i += 1; continue; }
        /* Every other option takes one value. A last option with no value is refused by name. */
        if (i + 1 >= argc) { fprintf(stderr, "the option %s has no value\n", argv[i]); return 2; }
        const char *value = argv[i + 1];
        if (!strcmp(argv[i], "--models")) models = value;
        else if (!strcmp(argv[i], "--tasks")) tasks_path = value;
        else if (!strcmp(argv[i], "--axis")) axis = value;
        else if (!strcmp(argv[i], "--doses")) dose_text = value;
        else if (!strcmp(argv[i], "--out")) out_dir = value;
        else if (!strcmp(argv[i], "--role")) role = value;
        else if (!strcmp(argv[i], "--pairs")) pairs_path = value;
        else if (!strcmp(argv[i], "--rubric")) rubric_path = value;
        else if (!strcmp(argv[i], "--blind")) blind = atoi(value);
        else { fprintf(stderr, "the option %s is not known\n", argv[i]); return 2; }
        i += 2;
    }
    if (pairs_path || rubric_path) {
        if (!models || !pairs_path || !rubric_path || !out_dir || tasks_path || axis || dose_text) return usage();
        return aotx_quality_pair(models, role, pairs_path, rubric_path, out_dir, blind);
    }
    if (!models || !tasks_path || !axis || !dose_text || !out_dir) return usage();
    int doses = doses_of(dose_text, dose);
    if (doses < 0) { fprintf(stderr, "the dose list %s holds 1 to %u figures at or above zero\n", dose_text, AOTX_SCORE_DOSES); return 2; }
    if (read_tasks(&tasks, tasks_path)) return 2;
    if (mkdir(out_dir, 0755) != 0 && errno != EEXIST) { fprintf(stderr, "the directory %s does not open\n", out_dir); return 2; }
    if (aotx_steer_run_open(&run, models, role)) return 1;
    if (aotx_steer_vector_of(axis, &vector_id, &vector)) return 1;
    printf("tasks %s: role %s, axis %s (vector %u, %u layers), %u items, %d doses\n", tasks_path, role, axis,
           vector_id, vector.layer_count, tasks.set.texts, doses);
    unsigned int *letter = (unsigned int *)aotx_steer_run_take(&run, AOTX_SCORE_LETTERS * sizeof(unsigned int));
    unsigned int *answer = (unsigned int *)aotx_steer_run_take(&run, tasks.set.texts * sizeof(unsigned int));
    unsigned int *largest = (unsigned int *)aotx_steer_run_take(&run, tasks.set.texts * sizeof(unsigned int));
    float *top = (float *)aotx_steer_run_take(&run, tasks.set.texts * sizeof(float));
    float *right = (float *)aotx_steer_run_take(&run, tasks.set.texts * sizeof(float));
    float *logits = (float *)aotx_steer_run_take(&run, (size_t)AOTX_STEER_PASS_TEXTS * run.desc.vocab * sizeof(float));
    double *mean = (double *)aotx_steer_run_take(&run, sizeof(double));
    if (letters_of(&run, letter)) return 1;
    aotx_check_runtime(cudaMemcpy(answer, tasks.answer, tasks.set.texts * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");
    if (aotx_steer_set_count(&run, &tasks.set)) return 1;
    aotx_steer_set_plan(&tasks.set);
    snprintf(path, sizeof path, "%s/capability.jsonl", out_dir);
    FILE *out = fopen(path, "a");
    if (out == 0) { fprintf(stderr, "the capability file %s does not write\n", path); return 1; }
    unsigned int *host_largest = (unsigned int *)malloc(tasks.set.texts * sizeof(unsigned int));
    float *host_top = (float *)malloc(tasks.set.texts * sizeof(float));
    for (int d = 0; d < doses; ++d) {
        double score = 0.0; char line[256];
        if (score_dose(&run, &tasks.set, vector_id, dose[d], logits, letter, answer, largest, top, right, mean, &score)) return 1;
        /* The item line carries the logit of the largest letter, so a check sees a dose
         * that reaches the model before it changes a letter. */
        if (print_items) {
            aotx_check_runtime(cudaMemcpy(host_largest, largest, tasks.set.texts * sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy");
            aotx_check_runtime(cudaMemcpy(host_top, top, tasks.set.texts * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
            for (unsigned int i = 0u; i < tasks.set.texts; ++i) {
                printf("item %s at dose %.9g: answer %c, largest %c, %s, logit %.9g\n", tasks.id[i], (double)dose[d], (char)('A' + tasks.answer[i]),
                       (char)('A' + host_largest[i]), (host_largest[i] == tasks.answer[i]) ? "right" : "wrong", (double)host_top[i]);
            }
        }
        snprintf(line, sizeof line, "{\"axis\":\"%s\",\"dose\":%.9g,\"score\":%.9g,\"items\":%u}", axis, (double)dose[d], score, tasks.set.texts);
        if (fprintf(out, "%s\n", line) < 0 || fflush(out) != 0) { fprintf(stderr, "the capability file %s does not write\n", path); return 1; }
        printf("%s\n", line);
    }
    int state = fclose(out) != 0;
    free(host_largest); free(host_top); aotx_steer_run_close(&run);
    return state;
}
