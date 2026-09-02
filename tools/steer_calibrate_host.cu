/* Purpose: Measure the closed-loop response and the potency matrix of the affect axes.
 * Owns: The calibration line, the composite vector files and the buffers of the passes.
 * Launch shape: Host glue only; every figure comes from a kernel over rows, texts or width.
 * Lifetime: One program run. */
#include "tools/steer_set.h"

#define AOTX_CALIBRATE_ROWS  6u   /* probe rows at the most: the axes and the guards */
#define AOTX_CALIBRATE_AXES  2u   /* steered axes at the most: the how row holds two slots */
#define AOTX_CALIBRATE_RUNS  6u   /* variants at two axes: plain, two singles, one pair, two doubles */
#define AOTX_CALIBRATE_NAME  32u

/* One probe row of the calibration: its catalog figures and its place in the capture. */
typedef struct aotx_calibrate_row {
    char name[AOTX_CALIBRATE_NAME];
    unsigned int axis, layer, at;
    float accuracy, agreement, mean, scale;
} aotx_calibrate_row;

/* Split a list with commas between the names. The count comes back, or -1 for too many. */
static int names_of(const char *text, char names[][AOTX_CALIBRATE_NAME], unsigned int max)
{
    unsigned int count = 0u;
    if (text == 0) return 0;
    while (*text) {
        const char *end = strchr(text, ','); size_t n = end ? (size_t)(end - text) : strlen(text);
        if (n == 0u || n >= AOTX_CALIBRATE_NAME || count >= max) return -1;
        memcpy(names[count], text, n); names[count][n] = '\0'; count += 1u;
        text = end ? end + 1 : text + n;
    }
    return (int)count;
}

/* Find one probe row of the catalog by name and read its head and its direction. */
static int probe_of(const char *models, aotx_calibrate_row *row, unsigned int hidden,
                    unsigned int layers, float *direction)
{
    char path[AOTX_STEER_PATH], line[AOTX_STEER_PATH], file[256]; int found = 0;
    snprintf(path, sizeof path, "%s/probes.jsonl", models);
    FILE *in = fopen(path, "r");
    while (in != 0 && !found && fgets(line, sizeof line, in) != 0) {
        char got[AOTX_CALIBRATE_NAME]; unsigned int axis, layer; float accuracy;
        if (sscanf(line, "{\"name\":\"%31[a-z0-9_-]\",\"file\":\"%255[^\"]\",\"axis\":%u,\"layer\":%u,\"accuracy\":%f}",
                   got, file, &axis, &layer, &accuracy) == 5 && strcmp(got, row->name) == 0) found = 1;
    }
    if (in != 0) fclose(in);
    if (!found) { fprintf(stderr, "the probe row %s is not in the model store\n", row->name); return 1; }
    snprintf(path, sizeof path, "%s/%s", models, file);
    aotx_probe_head head; in = fopen(path, "rb");
    int bad = in == 0 || fread(&head, sizeof head, 1u, in) != 1u || memcmp(head.magic, "AOTXPRB1", 8u) != 0
           || head.hidden != hidden || head.layer >= layers || !(head.scale > 0.0f) || !isfinite(head.scale)
           || !isfinite(head.mean) || fread(direction, sizeof(float), hidden, in) != hidden;
    if (in != 0) fclose(in);
    if (bad) { fprintf(stderr, "the probe file of %s does not read at width %u under %u layers\n", row->name, hidden, layers); return 1; }
    row->axis = head.axis; row->layer = head.layer; row->accuracy = head.accuracy;
    row->agreement = head.agreement; row->mean = head.mean; row->scale = head.scale;
    return 0;
}

/* Find one steer vector of the placed store by name. */
static int vector_of(const char *name, unsigned int *id, aotx_steer_vector *row)
{
    aotx_conduct_table table;
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_conduct, sizeof table), "cudaMemcpyFromSymbol");
    for (unsigned int i = 0u; i < table.vectors; ++i) {
        if (strcmp(name, table.vector[i].name) == 0) { *id = i; *row = table.vector[i]; return 0; }
    }
    fprintf(stderr, "the steer vector %s is not in the model store\n", name);
    return 1;
}

/* The compact row of a layer in a vector, or the layer count when the vector has no row. */
static unsigned int row_at(const aotx_steer_vector *v, unsigned int layer)
{
    if (((v->layers >> layer) & 1ull) == 0ull) return v->layer_count;
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < layer; ++i) at += (unsigned int)((v->layers >> i) & 1ull);
    return at;
}

/* Write the composite of one axis. Its rows are orthogonalized against the other axis at
 * every layer both hold, and copied at every layer it holds alone. */
static int compose_write(aotx_steer_run *run, const char *models, const char *name,
                         const aotx_steer_vector *own, const aotx_steer_vector *other,
                         float potency, float *wa, float *wb, double *gram)
{
    unsigned int hidden = run->desc.hidden, layers[AOTX_CONDUCT_LAYERS], count = 0u; char path[AOTX_STEER_PATH];
    float *host = (float *)malloc((size_t)own->layer_count * hidden * sizeof(float));
    for (unsigned int layer = 0u; layer < AOTX_CONDUCT_LAYERS && count < own->layer_count; ++layer) {
        unsigned int at = row_at(own, layer), across = other ? row_at(other, layer) : 0u;
        if (at == own->layer_count) continue;
        const float *a = (const float *)own->value + (size_t)at * hidden;
        if (other != 0 && across < other->layer_count) {
            const float *b = (const float *)other->value + (size_t)across * hidden;
            aotx_steer_gram<<<1, 256u>>>(a, b, hidden, gram);
            aotx_steer_compose<<<(hidden + 255u) / 256u, 256u>>>(a, b, gram, hidden, wa, wb);
            a = wa;
        }
        aotx_check_runtime(cudaMemcpy(host + (size_t)count * hidden, a, hidden * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
        layers[count++] = layer;
    }
    snprintf(path, sizeof path, "%s/affect", models);
    if (mkdir(path, 0755) != 0 && errno != EEXIST) { fprintf(stderr, "the directory %s does not open\n", path); free(host); return 1; }
    snprintf(path, sizeof path, "%s/affect/composite-%s.aotxvec", models, name);
    int state = aotx_steer_write_values(path, layers, count, hidden, host, potency);
    free(host);
    return state;
}

static void print_list(FILE *out, const float *values, unsigned int count)
{
    fputc('[', out);
    for (unsigned int i = 0u; i < count; ++i) fprintf(out, "%s%.9g", i ? "," : "", (double)values[i]);
    fputc(']', out);
}

int aotx_steer_calibrate(const char *models, const char *role_name, const char *axes_text,
                         const char *guards_text, const char *neutral_path, float dose, float surgical)
{
    char names[AOTX_CALIBRATE_ROWS][AOTX_CALIBRATE_NAME]; aotx_calibrate_row row[AOTX_CALIBRATE_ROWS];
    aotx_steer_vector vector[AOTX_CALIBRATE_AXES]; unsigned int id[AOTX_CALIBRATE_AXES];
    aotx_steer_set neutral; aotx_steer_run run; char path[AOTX_STEER_PATH];
    int k = names_of(axes_text, names, AOTX_CALIBRATE_AXES);
    if (k <= 0) { fprintf(stderr, "the axes list holds one or two names, the how row has two steer slots\n"); return 2; }
    int g = names_of(guards_text, names + k, AOTX_CALIBRATE_ROWS - (unsigned int)k);
    if (g < 0) { fprintf(stderr, "the guard list holds at most %u names\n", AOTX_CALIBRATE_ROWS - (unsigned int)k); return 2; }
    unsigned int rows = (unsigned int)(k + g), axes = (unsigned int)k;
    if (aotx_steer_set_read(&neutral, neutral_path, 0)) return 2;
    if (aotx_steer_run_open(&run, models, role_name)) return 1;
    unsigned int hidden = run.desc.hidden, vocab = run.desc.vocab, N = neutral.texts;
    for (unsigned int j = 0u; j < axes; ++j) if (vector_of(names[j], &id[j], &vector[j])) return 1;
    float *host_probe = (float *)malloc((size_t)rows * hidden * sizeof(float));
    for (unsigned int r = 0u; r < rows; ++r) {
        memset(&row[r], 0, sizeof row[r]); memcpy(row[r].name, names[r], AOTX_CALIBRATE_NAME);
        if (probe_of(models, &row[r], hidden, run.desc.layers, host_probe + (size_t)r * hidden)) return 1;
    }
    /* The capture takes each distinct probe layer once, in ascending order. */
    unsigned int capture_layer[AOTX_CALIBRATE_ROWS], captures = 0u;
    for (unsigned int layer = 0u; layer < run.desc.layers; ++layer) {
        int used = 0;
        for (unsigned int r = 0u; r < rows; ++r) if (row[r].layer == layer) { row[r].at = captures; used = 1; }
        if (used) capture_layer[captures++] = layer;
    }
    if (aotx_steer_set_count(&run, &neutral)) return 1;
    aotx_steer_set_plan(&neutral);
    /* The variants: the strength of each axis in each pass. */
    float strength[AOTX_CALIBRATE_RUNS][AOTX_CALIBRATE_AXES] = { { 0.0f, 0.0f }, { dose, 0.0f }, { 0.0f, dose }, { dose, dose }, { 2.0f * dose, 0.0f }, { 0.0f, 2.0f * dose } };
    unsigned int variants = (axes == 2u) ? 6u : 3u, single[2] = { 1u, 2u }, twice[2] = { 4u, 5u }, dual = 3u;
    if (axes == 1u) { strength[2][0] = 2.0f * dose; twice[0] = 2u; }
    printf("calibrate: role %s, %u layers, hidden %u, dose %.9g, %u axes, %u guards, %u variants\n",
           role_name, run.desc.layers, hidden, (double)dose, axes, rows - axes, variants);
    unsigned int *device_layers = (unsigned int *)aotx_steer_run_take(&run, captures * sizeof(unsigned int));
    float *probe = (float *)aotx_steer_run_take(&run, (size_t)rows * hidden * sizeof(float));
    float *plain_all = (float *)aotx_steer_run_take(&run, (size_t)AOTX_MODEL_MAX_TOKENS * vocab * sizeof(float));
    float *steered_all = (float *)aotx_steer_run_take(&run, (size_t)AOTX_MODEL_MAX_TOKENS * vocab * sizeof(float));
    float *plain_last = (float *)aotx_steer_run_take(&run, (size_t)AOTX_STEER_PASS_TEXTS * vocab * sizeof(float));
    float *steered_last = (float *)aotx_steer_run_take(&run, (size_t)AOTX_STEER_PASS_TEXTS * vocab * sizeof(float));
    float *capture_pass = (float *)aotx_steer_run_take(&run, (size_t)captures * AOTX_STEER_PASS_TEXTS * hidden * sizeof(float));
    float *readout = (float *)aotx_steer_run_take(&run, (size_t)variants * rows * N * sizeof(float));
    float *sum = (float *)aotx_steer_run_take(&run, sizeof(float));
    double *nll_sum = (double *)aotx_steer_run_take(&run, variants * sizeof(double));
    unsigned int *nll_count = (unsigned int *)aotx_steer_run_take(&run, variants * sizeof(unsigned int));
    float *shift = (float *)aotx_steer_run_take(&run, (size_t)rows * axes * sizeof(float));
    float *wa = (float *)aotx_steer_run_take(&run, hidden * sizeof(float));
    float *wb = (float *)aotx_steer_run_take(&run, hidden * sizeof(float));
    double *gram = (double *)aotx_steer_run_take(&run, 3u * sizeof(double));
    aotx_check_runtime(cudaMemcpy(device_layers, capture_layer, captures * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(probe, host_probe, (size_t)rows * hidden * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
    printf("calibrate: the all-row logits take two buffers of %u rows by %u, %.1f MB each\n", AOTX_MODEL_MAX_TOKENS, vocab,
           (double)AOTX_MODEL_MAX_TOKENS * vocab * sizeof(float) / (1024.0 * 1024.0));
    float kl[AOTX_CALIBRATE_RUNS] = { 0.0f };
    for (unsigned int p = 0u; p < neutral.passes; ++p) {
        unsigned int seqs = aotx_steer_set_seqs(&neutral, p), first = neutral.first[p], pass_rows = aotx_steer_set_rows(&neutral, p);
        for (unsigned int v = 0u; v < variants; ++v) {
            aotx_model_how one; aotx_steer_how_plain(&one);
            for (unsigned int j = 0u; j < axes; ++j) { one.steer[j] = id[j]; one.steer_strength[j] = strength[v][j]; }
            float *logits = (v == 0u) ? plain_all : steered_all, *last = (v == 0u) ? plain_last : steered_last;
            if (aotx_steer_run_pass(&run, &neutral, p, (v == 0u) ? 0 : aotx_steer_run_how(&run, &one), logits,
                                    AOTX_MODEL_ROWS_ALL, capture_pass, device_layers, captures)) return 1;
            aotx_steer_nll<<<pass_rows, 256u>>>(logits, run.ids, run.offset, seqs, pass_rows, vocab, nll_sum + v, nll_count + v);
            aotx_steer_last<<<seqs, 256u>>>(logits, run.offset, vocab, last);
            if (v != 0u) kl[v] += aotx_steer_run_kl(&run, seqs, plain_last, steered_last, sum) * (float)seqs / (float)N;
            for (unsigned int r = 0u; r < rows; ++r) {
                aotx_probe_read<<<dim3(seqs, 1u), 256u>>>(capture_pass + (size_t)row[r].at * seqs * hidden, probe + (size_t)r * hidden,
                                                          seqs, hidden, readout + ((size_t)v * rows + r) * N + first);
            }
        }
    }
    /* The figures: M over the rows and the steered axes, K, the ratios, the perplexities. */
    float M[AOTX_CALIBRATE_ROWS][AOTX_CALIBRATE_AXES], K[AOTX_CALIBRATE_AXES][AOTX_CALIBRATE_AXES];
    float ratio[AOTX_CALIBRATE_AXES], perplexity[AOTX_CALIBRATE_AXES], perplexity_twice[AOTX_CALIBRATE_AXES];
    double host_nll[AOTX_CALIBRATE_RUNS]; unsigned int host_count[AOTX_CALIBRATE_RUNS];
    aotx_check_runtime(cudaMemcpy(host_nll, nll_sum, variants * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(host_count, nll_count, variants * sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy");
    for (unsigned int r = 0u; r < rows; ++r) for (unsigned int j = 0u; j < axes; ++j) {
        aotx_probe_shift<<<1, 256u>>>(readout + ((size_t)0u * rows + r) * N, readout + ((size_t)single[j] * rows + r) * N, N, row[r].scale, dose, shift + r * axes + j);
    }
    aotx_check_runtime(cudaMemcpy(M, shift, (size_t)rows * axes * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    int dominant = 1, orthogonal = 1, finite = 1;
    for (unsigned int j = 0u; j < axes; ++j) {
        K[j][j] = 2.0f * kl[single[j]] / (dose * dose);
        ratio[j] = kl[twice[j]] / kl[single[j]];
        perplexity[j] = (float)exp(host_nll[single[j]] / host_count[single[j]] - host_nll[0] / host_count[0]);
        perplexity_twice[j] = (float)exp(host_nll[twice[j]] / host_count[twice[j]] - host_nll[0] / host_count[0]);
        float others = 0.0f;
        for (unsigned int i = 0u; i < axes; ++i) if (i != j) others += fabsf(M[i][j]);
        if (!(M[j][j] > 0.0f && M[j][j] > others)) dominant = 0;
        if (!isfinite(K[j][j]) || !isfinite(ratio[j]) || !isfinite(perplexity[j]) || !isfinite(perplexity_twice[j])) finite = 0;
        for (unsigned int r = 0u; r < rows; ++r) if (!isfinite(M[r][j])) finite = 0;
    }
    float normalized = 0.0f;
    if (axes == 2u) {
        K[0][1] = K[1][0] = (kl[dual] - kl[single[0]] - kl[single[1]]) / (dose * dose);
        normalized = fabsf(K[0][1]) / sqrtf(K[0][0] * K[1][1]);
        orthogonal = isfinite(normalized) && normalized < 0.3f;
        if (!isfinite(K[0][1])) finite = 0;
    }
    for (unsigned int r = 0u; r < rows; ++r) {
        for (unsigned int i = 0u; i < rows; ++i) if (i != r && row[i].axis == row[r].axis) { fprintf(stderr, "the rows %s and %s name one axis\n", row[i].name, row[r].name); return 1; }
        printf("probe %s: axis %u, layer %u, accuracy %.9g %s, agreement %.9g %s, mean %.9g, scale %.9g\n", row[r].name, row[r].axis, row[r].layer,
               (double)row[r].accuracy, (r >= axes) ? "monitor" : (row[r].accuracy >= 0.8f) ? "pass" : "fail",
               (double)row[r].agreement, (r >= axes) ? "monitor" : (row[r].agreement >= 0.9f) ? "pass" : "fail", (double)row[r].mean, (double)row[r].scale);
        for (unsigned int j = 0u; j < axes; ++j) printf("M %s under %s: %.9g\n", row[r].name, row[j].name, (double)M[r][j]);
    }
    printf("M dominant: %s\n", dominant ? "pass" : "fail");
    for (unsigned int j = 0u; j < axes; ++j) {
        printf("K %s: %.9g nats per unit dose squared %s\n", row[j].name, (double)K[j][j], isfinite(K[j][j]) ? "pass" : "fail");
        printf("dose-response %s: %.9g %s\n", row[j].name, (double)ratio[j], (ratio[j] >= 3.0f && ratio[j] <= 5.0f) ? "pass" : "fail");
        printf("perplexity %s: %.9g at the dose %s (bound %.9g), %.9g at twice the dose\n", row[j].name, (double)perplexity[j],
               (perplexity[j] < surgical) ? "pass" : "fail", (double)surgical, (double)perplexity_twice[j]);
    }
    if (axes == 2u) printf("K off-diagonal: %.9g, normalized %.9g %s\n", (double)K[0][1], (double)normalized, orthogonal ? "pass" : "fail");
    for (unsigned int j = 0u; j < axes; ++j) {
        if (compose_write(&run, models, row[j].name, &vector[j], (axes == 2u) ? &vector[1u - j] : 0, 0.5f * K[j][j], wa, wb, gram)) return 1;
        printf("composite %s: affect/composite-%s.aotxvec, %u layers\n", row[j].name, row[j].name, vector[j].layer_count);
    }
    snprintf(path, sizeof path, "%s/affect/calibration.jsonl", models);
    FILE *out = fopen(path, "a");
    if (out == 0) { fprintf(stderr, "the calibration file %s does not write\n", path); return 1; }
    fprintf(out, "{\"role\":\"%s\",\"axes\":[", role_name);
    for (unsigned int j = 0u; j < axes; ++j) fprintf(out, "%s\"%s\"", j ? "," : "", row[j].name);
    fprintf(out, "],\"guards\":[");
    for (unsigned int r = axes; r < rows; ++r) fprintf(out, "%s\"%s\"", (r > axes) ? "," : "", row[r].name);
    fprintf(out, "],\"layers\":[");
    for (unsigned int j = 0u; j < axes; ++j) {
        fprintf(out, "%s[", j ? "," : "");
        for (unsigned int layer = 0u, n = 0u; layer < AOTX_CONDUCT_LAYERS; ++layer) if (row_at(&vector[j], layer) < vector[j].layer_count) fprintf(out, "%s%u", n++ ? "," : "", layer);
        fputc(']', out);
    }
    fprintf(out, "],\"delta\":%.9g,\"rows\":[", (double)dose);
    for (unsigned int r = 0u; r < rows; ++r) fprintf(out, "%s\"%s\"", r ? "," : "", row[r].name);
    fprintf(out, "],\"probe_layers\":[");
    for (unsigned int r = 0u; r < rows; ++r) fprintf(out, "%s%u", r ? "," : "", row[r].layer);
    fprintf(out, "],\"M\":[");
    for (unsigned int r = 0u; r < rows; ++r) { if (r) fputc(',', out); print_list(out, M[r], axes); }
    fprintf(out, "],\"K\":[");
    for (unsigned int j = 0u; j < axes; ++j) { if (j) fputc(',', out); print_list(out, K[j], axes); }
    fprintf(out, "],\"ratio\":"); print_list(out, ratio, axes);
    fprintf(out, ",\"perplexity\":"); print_list(out, perplexity, axes);
    fprintf(out, ",\"perplexity_twice\":"); print_list(out, perplexity_twice, axes);
    fprintf(out, ",\"surgical\":%.9g,\"composite\":[", (double)surgical);
    for (unsigned int j = 0u; j < axes; ++j) fprintf(out, "%s\"affect/composite-%s.aotxvec\"", j ? "," : "", row[j].name);
    fprintf(out, "],\"dominant\":%d,\"orthogonal\":%d}\n", dominant, orthogonal);
    int state = fclose(out) != 0;
    printf("calibration line: %s, figures %s, dominant %d, orthogonal %d\n", path, finite ? "finite" : "not finite", dominant, orthogonal);
    free(host_probe); aotx_steer_run_close(&run);
    return state;
}
