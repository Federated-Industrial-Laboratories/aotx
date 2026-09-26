/* Purpose: Measure the closed-loop response and the potency matrix of the affect axes.
 * Owns: The calibration line, the composite vector files and the buffers of the passes.
 * Launch shape: Host glue only; every figure comes from a kernel over rows, texts or width.
 * Lifetime: One program run. */
#include "tools/steer_set.h"

#define AOTX_CALIBRATE_ROWS  6u   /* probe rows at the most: the axes and the guards */
#define AOTX_CALIBRATE_AXES  2u   /* steered axes at the most: the how row holds two slots */
#define AOTX_CALIBRATE_RUNS  9u   /* passes at two axes: plain, two singles, one pair, two doubles, then the composite singles and pair */

/* One pass of the calibration: the vector and the strength in each steer slot. */
typedef struct aotx_calibrate_pass {
    unsigned int id[AOTX_CALIBRATE_AXES];
    float strength[AOTX_CALIBRATE_AXES];
} aotx_calibrate_pass;

static unsigned int add_pass(aotx_calibrate_pass *plan, unsigned int n, unsigned int a, float sa,
                             unsigned int b, float sb)
{
    plan[n].id[0] = a; plan[n].strength[0] = sa; plan[n].id[1] = b; plan[n].strength[1] = sb;
    return n + 1u;
}

/* The composite of one axis, in a device buffer. Its rows are orthogonalized against the
 * other axis at every layer both hold, and copied at every layer it holds alone. The layer
 * list comes back with the count. */
static unsigned int compose(const aotx_steer_run *run, const aotx_steer_vector *own,
                            const aotx_steer_vector *other, float *wa, float *wb, double *gram,
                            float *out, unsigned int *layers)
{
    unsigned int hidden = run->desc.hidden, count = 0u;
    for (unsigned int layer = 0u; layer < AOTX_CONDUCT_LAYERS && count < own->layer_count; ++layer) {
        unsigned int at = aotx_steer_row_at(own, layer), across = other ? aotx_steer_row_at(other, layer) : 0u;
        if (at == own->layer_count) continue;
        const float *a = (const float *)own->value + (size_t)at * hidden;
        if (other != 0 && across < other->layer_count) {
            const float *b = (const float *)other->value + (size_t)across * hidden;
            aotx_steer_gram<<<1, 256u>>>(a, b, hidden, gram);
            aotx_steer_compose<<<(hidden + 255u) / 256u, 256u>>>(a, b, gram, hidden, wa, wb);
            a = wa;
        }
        aotx_check_runtime(cudaMemcpy(out + (size_t)count * hidden, a, hidden * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy");
        layers[count++] = layer;
    }
    return count;
}

/* Write the composite file of one axis from its device rows. The head potency is the K of
 * the composite over two, the quadratic estimate at unit dose. */
static int composite_write(const char *models, const char *name, const unsigned int *layers,
                           unsigned int count, unsigned int hidden, const float *device, float potency)
{
    char path[AOTX_STEER_PATH]; size_t bytes = (size_t)count * hidden * sizeof(float);
    float *host = (float *)malloc(bytes);
    aotx_check_runtime(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
    snprintf(path, sizeof path, "%s/affect", models);
    if (mkdir(path, 0755) != 0 && errno != EEXIST) { fprintf(stderr, "the directory %s does not open\n", path); free(host); return 1; }
    snprintf(path, sizeof path, "%s/affect/composite-%s.aotxvec", models, name);
    int state = aotx_steer_write_values(path, layers, count, hidden, host, potency);
    free(host);
    return state;
}

int aotx_steer_calibrate(const char *models, const char *role_name, const char *axes_text,
                         const char *guards_text, const char *neutral_path, float dose, float surgical)
{
    char names[AOTX_CALIBRATE_ROWS][AOTX_STEER_NAME]; aotx_calibrate_row row[AOTX_CALIBRATE_ROWS];
    aotx_steer_vector vector[AOTX_CALIBRATE_AXES]; unsigned int id[AOTX_CALIBRATE_AXES], cid[AOTX_CALIBRATE_AXES];
    unsigned int clayers[AOTX_CALIBRATE_AXES][AOTX_CONDUCT_LAYERS], ccount[AOTX_CALIBRATE_AXES];
    aotx_calibrate_pass plan[AOTX_CALIBRATE_RUNS]; unsigned int single[2], twice[2], csingle[2], dual = 0u, cdual = 0u;
    aotx_steer_set neutral; aotx_steer_run run; char path[AOTX_STEER_PATH];
    int k = aotx_steer_names_of(axes_text, names, AOTX_CALIBRATE_AXES);
    if (k <= 0) { fprintf(stderr, "the axes list holds one or two names, the how row has two steer slots\n"); return 2; }
    int g = aotx_steer_names_of(guards_text, names + k, AOTX_CALIBRATE_ROWS - (unsigned int)k);
    if (g < 0) { fprintf(stderr, "the guard list holds at most %u names\n", AOTX_CALIBRATE_ROWS - (unsigned int)k); return 2; }
    unsigned int rows = (unsigned int)(k + g), axes = (unsigned int)k;
    if (aotx_steer_set_read(&neutral, neutral_path, 0)) return 2;
    if (aotx_steer_run_open(&run, models, role_name)) return 1;
    unsigned int hidden = run.desc.hidden, vocab = run.desc.vocab, N = neutral.texts;
    aotx_steer_measure candidate[AOTX_CALIBRATE_AXES] = {};
    for (unsigned j = 0; j < axes; ++j) memcpy(candidate[j].name, names[j], sizeof(candidate[j].name));
    if (aotx_steer_measure_vectors(candidate, axes)) return 1;
    for (unsigned j = 0; j < axes; ++j) {
        if (candidate[j].id == AOTX_MODEL_CONDUCT_NONE) return 1;
        id[j] = candidate[j].id; vector[j] = candidate[j].vector;
    }
    float *host_probe = (float *)malloc((size_t)rows * hidden * sizeof(float));
    for (unsigned int r = 0u; r < rows; ++r) {
        memset(&row[r], 0, sizeof row[r]); memcpy(row[r].name, names[r], AOTX_STEER_NAME);
        if (aotx_steer_probe_of(models, &row[r], hidden, run.desc.layers, run.desc.probe_layer, host_probe + (size_t)r * hidden)) return 1;
        for (unsigned int i = 0u; i < r; ++i) if (row[i].axis == row[r].axis) { fprintf(stderr, "the rows %s and %s name one axis\n", row[i].name, row[r].name); return 1; }
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
    unsigned int *device_layers = (unsigned int *)aotx_steer_run_take(&run, captures * sizeof(unsigned int));
    float *probe = (float *)aotx_steer_run_take(&run, (size_t)rows * hidden * sizeof(float));
    float *plain_all = (float *)aotx_steer_run_take(&run, (size_t)AOTX_MODEL_MAX_TOKENS * vocab * sizeof(float));
    float *steered_all = (float *)aotx_steer_run_take(&run, (size_t)AOTX_MODEL_MAX_TOKENS * vocab * sizeof(float));
    float *plain_last = (float *)aotx_steer_run_take(&run, (size_t)AOTX_STEER_PASS_TEXTS * vocab * sizeof(float));
    float *steered_last = (float *)aotx_steer_run_take(&run, (size_t)AOTX_STEER_PASS_TEXTS * vocab * sizeof(float));
    float *capture_pass = (float *)aotx_steer_run_take(&run, (size_t)captures * AOTX_STEER_PASS_TEXTS * hidden * sizeof(float));
    float *sum = (float *)aotx_steer_run_take(&run, sizeof(float));
    double *nll_sum = (double *)aotx_steer_run_take(&run, AOTX_CALIBRATE_RUNS * sizeof(double));
    unsigned int *nll_count = (unsigned int *)aotx_steer_run_take(&run, AOTX_CALIBRATE_RUNS * sizeof(unsigned int));
    float *shift = (float *)aotx_steer_run_take(&run, (size_t)rows * axes * sizeof(float));
    float *wa = (float *)aotx_steer_run_take(&run, hidden * sizeof(float));
    float *wb = (float *)aotx_steer_run_take(&run, hidden * sizeof(float));
    double *gram = (double *)aotx_steer_run_take(&run, 3u * sizeof(double));
    float *composite[AOTX_CALIBRATE_AXES];
    aotx_check_runtime(cudaMemcpy(device_layers, capture_layer, captures * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(probe, host_probe, (size_t)rows * hidden * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
    /* The composites go in the table as vectors, so the passes apply what a run applies. */
    for (unsigned int j = 0u; j < axes; ++j) {
        char name[2u * AOTX_STEER_NAME]; aotx_conduct_table table;
        composite[j] = (float *)aotx_steer_run_take(&run, (size_t)vector[j].layer_count * hidden * sizeof(float));
        ccount[j] = compose(&run, &vector[j], (axes == 2u) ? &vector[1u - j] : 0, wa, wb, gram, composite[j], clayers[j]);
        snprintf(name, sizeof name, "composite-%.31s", row[j].name);
        aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_conduct, sizeof table), "cudaMemcpyFromSymbol");
        cid[j] = table.vectors;
        if (aotx_conduct_register_measurement(name, clayers[j], ccount[j], hidden, composite[j], 0.0f)) return 1;
    }
    /* The passes: the plain pass, the raw passes and the composite passes. The raw passes
     * give M, the raw K, the dose-response and the perplexity. The composite passes give
     * the K that ships. */
    unsigned int none = AOTX_MODEL_CONDUCT_NONE, n = add_pass(plan, 0u, none, 0.0f, none, 0.0f);
    for (unsigned int j = 0u; j < axes; ++j) { single[j] = n; n = add_pass(plan, n, id[j], dose, none, 0.0f); }
    if (axes == 2u) { dual = n; n = add_pass(plan, n, id[0], dose, id[1], dose); }
    for (unsigned int j = 0u; j < axes; ++j) { twice[j] = n; n = add_pass(plan, n, id[j], 2.0f * dose, none, 0.0f); }
    unsigned int raw = n;
    for (unsigned int j = 0u; j < axes; ++j) { csingle[j] = n; n = add_pass(plan, n, cid[j], dose, none, 0.0f); }
    if (axes == 2u) { cdual = n; n = add_pass(plan, n, cid[0], dose, cid[1], dose); }
    unsigned int variants = n;
    float *readout = (float *)aotx_steer_run_take(&run, (size_t)raw * rows * N * sizeof(float));
    printf("calibrate: role %s, %u layers, hidden %u, dose %.9g, %u axes, %u guards, %u variants (%u raw, %u composite)\n",
           role_name, run.desc.layers, hidden, (double)dose, axes, rows - axes, variants, raw, variants - raw);
    printf("calibrate: the all-row logits take two buffers of %u rows by %u, %.1f MB each\n", AOTX_MODEL_MAX_TOKENS, vocab,
           (double)AOTX_MODEL_MAX_TOKENS * vocab * sizeof(float) / (1024.0 * 1024.0));
    float kl[AOTX_CALIBRATE_RUNS] = { 0.0f };
    for (unsigned int p = 0u; p < neutral.passes; ++p) {
        unsigned int seqs = aotx_steer_set_seqs(&neutral, p), first = neutral.first[p], pass_rows = aotx_steer_set_rows(&neutral, p);
        for (unsigned int v = 0u; v < variants; ++v) {
            aotx_model_how one; aotx_steer_how_plain(&one);
            for (unsigned int s = 0u; s < AOTX_CALIBRATE_AXES; ++s) { one.steer[s] = plan[v].id[s]; one.steer_strength[s] = plan[v].strength[s]; }
            float *logits = (v == 0u) ? plain_all : steered_all, *last = (v == 0u) ? plain_last : steered_last;
            if (aotx_steer_run_pass(&run, &neutral, p, (v == 0u) ? 0 : aotx_steer_run_how(&run, &one), logits,
                                    AOTX_MODEL_ROWS_ALL, capture_pass, device_layers, captures)) return 1;
            aotx_steer_last<<<seqs, 256u>>>(logits, run.offset, vocab, last);
            if (v != 0u) kl[v] += aotx_steer_run_kl(&run, seqs, plain_last, steered_last, sum) * (float)seqs / (float)N;
            if (v >= raw) continue;
            aotx_steer_nll<<<pass_rows, 256u>>>(logits, run.ids, run.offset, seqs, pass_rows, vocab, nll_sum + v, nll_count + v);
            for (unsigned int r = 0u; r < rows; ++r) {
                aotx_probe_read<<<dim3(seqs, 1u), 256u>>>(capture_pass + (size_t)row[r].at * seqs * hidden, probe + (size_t)r * hidden,
                                                          seqs, hidden, readout + ((size_t)v * rows + r) * N + first);
            }
        }
    }
    /* The figures: M over the rows and the steered axes, K raw and composite, the ratios,
     * the perplexities. */
    float M[AOTX_CALIBRATE_ROWS][AOTX_CALIBRATE_AXES], K[AOTX_CALIBRATE_AXES][AOTX_CALIBRATE_AXES] = { { 0.0f } };
    float K_raw[AOTX_CALIBRATE_AXES][AOTX_CALIBRATE_AXES] = { { 0.0f } };
    float ratio[AOTX_CALIBRATE_AXES], perplexity[AOTX_CALIBRATE_AXES], perplexity_twice[AOTX_CALIBRATE_AXES];
    double host_nll[AOTX_CALIBRATE_RUNS]; unsigned int host_count[AOTX_CALIBRATE_RUNS];
    aotx_check_runtime(cudaMemcpy(host_nll, nll_sum, raw * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(host_count, nll_count, raw * sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy");
    for (unsigned int r = 0u; r < rows; ++r) for (unsigned int j = 0u; j < axes; ++j) {
        aotx_probe_shift<<<1, 256u>>>(readout + ((size_t)0u * rows + r) * N, readout + ((size_t)single[j] * rows + r) * N, N, row[r].scale, dose, shift + r * axes + j);
    }
    aotx_check_runtime(cudaMemcpy(M, shift, (size_t)rows * axes * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    int dominant = 1, orthogonal = 1, finite = 1;
    for (unsigned int j = 0u; j < axes; ++j) {
        K_raw[j][j] = 2.0f * kl[single[j]] / (dose * dose);
        K[j][j] = 2.0f * kl[csingle[j]] / (dose * dose);
        ratio[j] = kl[twice[j]] / kl[single[j]];
        perplexity[j] = (float)exp(host_nll[single[j]] / host_count[single[j]] - host_nll[0] / host_count[0]);
        perplexity_twice[j] = (float)exp(host_nll[twice[j]] / host_count[twice[j]] - host_nll[0] / host_count[0]);
        float others = 0.0f;
        for (unsigned int i = 0u; i < axes; ++i) if (i != j) others += fabsf(M[i][j]);
        if (!(M[j][j] > 0.0f && M[j][j] > others)) dominant = 0;
        if (!isfinite(K[j][j]) || !isfinite(K_raw[j][j]) || !isfinite(ratio[j]) || !isfinite(perplexity[j]) || !isfinite(perplexity_twice[j])) finite = 0;
        for (unsigned int r = 0u; r < rows; ++r) if (!isfinite(M[r][j])) finite = 0;
    }
    float normalized = 0.0f, normalized_raw = 0.0f;
    if (axes == 2u) {
        K_raw[0][1] = K_raw[1][0] = (kl[dual] - kl[single[0]] - kl[single[1]]) / (dose * dose);
        K[0][1] = K[1][0] = (kl[cdual] - kl[csingle[0]] - kl[csingle[1]]) / (dose * dose);
        normalized_raw = fabsf(K_raw[0][1]) / sqrtf(K_raw[0][0] * K_raw[1][1]);
        normalized = fabsf(K[0][1]) / sqrtf(K[0][0] * K[1][1]);
        orthogonal = isfinite(normalized) && normalized < 0.3f;
        if (!isfinite(K[0][1]) || !isfinite(K_raw[0][1])) finite = 0;
    }
    for (unsigned int r = 0u; r < rows; ++r) {
        printf("probe %s: axis %u, layer %u, accuracy %.9g %s, agreement %.9g %s, mean %.9g, scale %.9g\n", row[r].name, row[r].axis, row[r].layer,
               (double)row[r].accuracy, (r >= axes) ? "monitor" : (row[r].accuracy >= 0.8f) ? "pass" : "fail",
               (double)row[r].agreement, (r >= axes) ? "monitor" : (row[r].agreement >= 0.9f) ? "pass" : "fail", (double)row[r].mean, (double)row[r].scale);
        for (unsigned int j = 0u; j < axes; ++j) printf("M %s under %s: %.9g\n", row[r].name, row[j].name, (double)M[r][j]);
    }
    printf("M dominant: %s\n", dominant ? "pass" : "fail");
    for (unsigned int j = 0u; j < axes; ++j) {
        printf("K %s: %.9g nats per unit dose squared %s, raw %.9g\n", row[j].name, (double)K[j][j], isfinite(K[j][j]) ? "pass" : "fail", (double)K_raw[j][j]);
        printf("dose-response %s: %.9g %s\n", row[j].name, (double)ratio[j], (ratio[j] >= 3.0f && ratio[j] <= 5.0f) ? "pass" : "fail");
        printf("perplexity %s: %.9g at the dose %s (bound %.9g), %.9g at twice the dose\n", row[j].name, (double)perplexity[j],
               (perplexity[j] < surgical) ? "pass" : "fail", (double)surgical, (double)perplexity_twice[j]);
    }
    if (axes == 2u) printf("K off-diagonal: %.9g, normalized %.9g %s, raw %.9g normalized %.9g\n", (double)K[0][1], (double)normalized,
                           orthogonal ? "pass" : "fail", (double)K_raw[0][1], (double)normalized_raw);
    for (unsigned int j = 0u; j < axes; ++j) {
        if (composite_write(models, row[j].name, clayers[j], ccount[j], hidden, composite[j], 0.5f * K[j][j])) return 1;
        printf("composite %s: affect/composite-%s.aotxvec, %u layers, potency %.9g nats\n", row[j].name, row[j].name, ccount[j], (double)(0.5f * K[j][j]));
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
        for (unsigned int l = 0u; l < ccount[j]; ++l) fprintf(out, "%s%u", l ? "," : "", clayers[j][l]);
        fputc(']', out);
    }
    fprintf(out, "],\"delta\":%.9g,\"rows\":[", (double)dose);
    for (unsigned int r = 0u; r < rows; ++r) fprintf(out, "%s\"%s\"", r ? "," : "", row[r].name);
    fprintf(out, "],\"probe_layers\":[");
    for (unsigned int r = 0u; r < rows; ++r) fprintf(out, "%s%u", r ? "," : "", row[r].layer);
    fprintf(out, "],\"M\":[");
    for (unsigned int r = 0u; r < rows; ++r) { if (r) fputc(',', out); aotx_steer_print_list(out, M[r], axes); }
    fprintf(out, "],\"K\":[");
    for (unsigned int j = 0u; j < axes; ++j) { if (j) fputc(',', out); aotx_steer_print_list(out, K[j], axes); }
    fprintf(out, "],\"K_raw\":[");
    for (unsigned int j = 0u; j < axes; ++j) { if (j) fputc(',', out); aotx_steer_print_list(out, K_raw[j], axes); }
    fprintf(out, "],\"ratio\":"); aotx_steer_print_list(out, ratio, axes);
    fprintf(out, ",\"perplexity\":"); aotx_steer_print_list(out, perplexity, axes);
    fprintf(out, ",\"perplexity_twice\":"); aotx_steer_print_list(out, perplexity_twice, axes);
    fprintf(out, ",\"surgical\":%.9g,\"composite\":[", (double)surgical);
    for (unsigned int j = 0u; j < axes; ++j) fprintf(out, "%s\"affect/composite-%s.aotxvec\"", j ? "," : "", row[j].name);
    fprintf(out, "],\"composite_sha256\":[");
    for (unsigned j = 0; j < axes; ++j) {
        char file[256], hash[65]; snprintf(file, sizeof file, "affect/composite-%s.aotxvec", row[j].name);
        if (aotx_control_digest(models, file, hash)) { fclose(out); return 1; }
        fprintf(out, "%s\"%s\"", j ? "," : "", hash);
    }
    fprintf(out, "],\"dominant\":%d,\"orthogonal\":%d}\n", dominant, orthogonal);
    int state = fclose(out) != 0;
    if (!state) state = aotx_control_save(path, AOTX_CONTROL_CALIBRATION);
    printf("calibration line: %s, figures %s, dominant %d, orthogonal %d\n", path, finite ? "finite" : "not finite", dominant, orthogonal);
    free(host_probe); aotx_steer_run_close(&run);
    return state;
}
