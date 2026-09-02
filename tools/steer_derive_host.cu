/* Purpose: Load contrast prompts, run derivation and write cataloged steer vectors and probes.
 * Owns: The output vector and probe files and their catalog lines.
 * Launch shape: Host glue only; all model and numeric work runs in device kernels.
 * Lifetime: One program run. */
#include "tools/steer_set.h"

/* The layer list is ascending with no repeat. The loader keeps the layers as a bit set, and
 * the rows of the file must stand in the order the set gives. */
static int layers_of(const char *text, unsigned int *layer)
{
    unsigned int count = 0u; const char *at = text;
    while (*at && count < AOTX_CONDUCT_LAYERS) {
        char *end = 0; unsigned long value = strtoul(at, &end, 10);
        if (end == at || value >= AOTX_CONDUCT_LAYERS) return -1;
        if (count != 0u && value <= layer[count - 1u]) return -1;
        layer[count++] = (unsigned int)value;
        if (*end == ',') at = end + 1; else if (*end == '\0') at = end; else return -1;
    }
    return count ? (int)count : -1;
}

/* Every named layer must be a layer of the placed model. */
static int layers_fit(const aotx_steer_run *run, const unsigned int *layers, unsigned int count)
{
    for (unsigned int l = 0u; l < count; ++l) {
        if (layers[l] >= run->desc.layers) {
            fprintf(stderr, "the layer %u is not under the %u layers of the model\n", layers[l], run->desc.layers);
            return 1;
        }
    }
    return 0;
}

static int write_vector(const char *dir, const char *trait, const unsigned int *layers,
                        unsigned int layer_count, unsigned int hidden,
                        const float *values, float potency)
{
    char path[1024], line[1024]; snprintf(path, sizeof path, "%s/%s.aotxvec", dir, trait);
    if (access(path, F_OK) == 0) {
        fprintf(stderr, "the steer vector %s is already in the model store\n", trait);
        return 1;
    }
    /* The store refuses a vector with no finite potency at the start, so no such file
     * is written. */
    if (!isfinite(potency) || potency < 0.0f) {
        fprintf(stderr, "the potency %g is not a finite figure, no vector is written\n", (double)potency);
        return 1;
    }
    if (aotx_steer_write_values(path, layers, layer_count, hidden, values, potency)) return 1;
    /* A vector file with no catalog line blocks the next run, so a failed catalog line
     * takes the file away again. */
    snprintf(line, sizeof line, "%s/steer.jsonl", dir); FILE *out = fopen(line, "a");
    int state = 1;
    if (out) {
        state = fprintf(out, "{\"name\":\"%s\",\"file\":\"%s.aotxvec\",\"potency_nats\":%.9g}\n",
                        trait, trait, (double)potency) < 0;
        if (fclose(out) != 0) state = 1;
    }
    if (state) { fprintf(stderr, "the catalog %s does not write\n", line); unlink(path); }
    return state;
}

/* The mean difference of the pairs at the named layers, as one vector, with its potency
 * at strength one over the pair texts. */
static int derive_trait(const char *models, const char *role_name, const char *trait,
                        const char *pairs_path, const unsigned int *layers, unsigned int layer_count)
{
    aotx_steer_set pairs; aotx_steer_run run; aotx_conduct_table table; float potency = 0.0f;
    if (aotx_steer_set_read(&pairs, pairs_path, 1)) return 2;
    if (aotx_steer_run_open(&run, models, role_name) || layers_fit(&run, layers, layer_count)) return 1;
    if (aotx_steer_set_count(&run, &pairs)) return 1;
    aotx_steer_set_plan(&pairs);
    unsigned int hidden = run.desc.hidden; size_t width = (size_t)layer_count * hidden;
    unsigned int *device_layers = (unsigned int *)aotx_steer_run_take(&run, layer_count * sizeof(unsigned int));
    float *capture_pass = (float *)aotx_steer_run_take(&run, width * AOTX_STEER_PASS_TEXTS * sizeof(float));
    float *capture = (float *)aotx_steer_run_take(&run, width * pairs.texts * sizeof(float));
    float *vector = (float *)aotx_steer_run_take(&run, width * sizeof(float));
    float *plain = (float *)aotx_steer_run_take(&run, (size_t)AOTX_STEER_PASS_TEXTS * run.desc.vocab * sizeof(float));
    float *steered = (float *)aotx_steer_run_take(&run, (size_t)AOTX_STEER_PASS_TEXTS * run.desc.vocab * sizeof(float));
    float *sum = (float *)aotx_steer_run_take(&run, sizeof(float));
    aotx_check_runtime(cudaMemcpy(device_layers, layers, layer_count * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");
    if (aotx_steer_run_capture(&run, &pairs, device_layers, layer_count, capture_pass, capture)) return 1;
    aotx_steer_mean<<<(unsigned int)((width + 255u) / 256u), 256u>>>(capture, pairs.pairs, layer_count, hidden, vector);
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_conduct, sizeof table), "cudaMemcpyFromSymbol");
    unsigned int id = table.vectors;
    if (aotx_conduct_register_vector(trait, layers, layer_count, hidden, vector, 0.0f)) return 1;
    if (aotx_steer_run_potency(&run, &pairs, &id, 1u, 1.0f, plain, steered, sum, &potency)) return 1;
    float *host_vector = (float *)malloc(width * sizeof(float));
    aotx_check_runtime(cudaMemcpy(host_vector, vector, width * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    int state = write_vector(models, trait, layers, layer_count, hidden, host_vector, potency);
    printf("trait %s: %u pairs, %u layers, potency %.9g nats\n", trait, pairs.pairs, layer_count, (double)potency);
    free(host_vector); aotx_steer_run_close(&run);
    return state;
}

/* One axis: the steer direction and the probe direction at each named layer. The
 * standardization set gives the mean and the scale, and the held-out set gives the two
 * figures against that mean. The vector goes to the layer with the highest held-out
 * accuracy; an equal accuracy takes the higher agreement, then the earlier layer. The
 * probe goes to the probe layer, whatever layer the vector takes. Every probe of a store
 * then reads at one layer at or after every steered layer. */
static int derive_axis(const char *models, const char *role_name, const char *axis,
                       const char *pairs_path, const char *standard_path, const char *heldout_path,
                       const unsigned int *layers, unsigned int layer_count, unsigned int probe_at,
                       int print_readouts)
{
    aotx_steer_set pairs, standard, heldout; aotx_steer_run run; char path[AOTX_STEER_PATH];
    unsigned int number = aotx_steer_axis_of(axis), id[AOTX_CONDUCT_LAYERS];
    if (number == AOTX_STEER_NO_AXIS) { fprintf(stderr, "the axis %s is not an axis of the probe table\n", axis); return 2; }
    if (aotx_steer_set_read(&pairs, pairs_path, 1) || aotx_steer_set_read(&standard, standard_path, 0)
        || aotx_steer_set_read(&heldout, heldout_path, 1)) return 2;
    if (standard.texts < 2u) { fprintf(stderr, "the standardization set %s holds one text, the scale needs two\n", standard_path); return 2; }
    snprintf(path, sizeof path, "%s/%s.aotxvec", models, axis);
    if (access(path, F_OK) == 0) { fprintf(stderr, "the steer vector %s is already in the model store\n", axis); return 1; }
    snprintf(path, sizeof path, "%s/affect/%s.aotxprb", models, axis);
    if (access(path, F_OK) == 0) { fprintf(stderr, "the probe file %s is already in the model store\n", axis); return 1; }
    if (aotx_steer_run_open(&run, models, role_name) || layers_fit(&run, layers, layer_count)) return 1;
    unsigned int hidden = run.desc.hidden; size_t width = (size_t)layer_count * hidden;
    printf("axis %s: role %s, %u layers, hidden %u, vocabulary %u, probe layer %u\n", axis, role_name, run.desc.layers, hidden,
           run.desc.vocab, layers[probe_at]);
    if (aotx_steer_set_count(&run, &pairs) || aotx_steer_set_count(&run, &standard) || aotx_steer_set_count(&run, &heldout)) return 1;
    aotx_steer_set_plan(&pairs); aotx_steer_set_plan(&standard); aotx_steer_set_plan(&heldout);
    unsigned int *device_layers = (unsigned int *)aotx_steer_run_take(&run, layer_count * sizeof(unsigned int));
    float *capture_pass = (float *)aotx_steer_run_take(&run, width * AOTX_STEER_PASS_TEXTS * sizeof(float));
    float *cap_pairs = (float *)aotx_steer_run_take(&run, width * pairs.texts * sizeof(float));
    float *cap_standard = (float *)aotx_steer_run_take(&run, width * standard.texts * sizeof(float));
    float *cap_heldout = (float *)aotx_steer_run_take(&run, width * heldout.texts * sizeof(float));
    float *vector = (float *)aotx_steer_run_take(&run, width * sizeof(float));
    float *direction = (float *)aotx_steer_run_take(&run, width * sizeof(float));
    float *variance = (float *)aotx_steer_run_take(&run, width * sizeof(float));
    float *read_standard = (float *)aotx_steer_run_take(&run, (size_t)layer_count * standard.texts * sizeof(float));
    float *read_heldout = (float *)aotx_steer_run_take(&run, (size_t)layer_count * heldout.texts * sizeof(float));
    float *mean = (float *)aotx_steer_run_take(&run, layer_count * sizeof(float));
    float *scale = (float *)aotx_steer_run_take(&run, layer_count * sizeof(float));
    float *figure = (float *)aotx_steer_run_take(&run, 2u * layer_count * sizeof(float));
    float *plain = (float *)aotx_steer_run_take(&run, (size_t)AOTX_STEER_PASS_TEXTS * run.desc.vocab * sizeof(float));
    float *steered = (float *)aotx_steer_run_take(&run, (size_t)AOTX_STEER_PASS_TEXTS * run.desc.vocab * sizeof(float));
    float *sum = (float *)aotx_steer_run_take(&run, sizeof(float));
    aotx_check_runtime(cudaMemcpy(device_layers, layers, layer_count * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");
    if (aotx_steer_run_capture(&run, &pairs, device_layers, layer_count, capture_pass, cap_pairs)
        || aotx_steer_run_capture(&run, &standard, device_layers, layer_count, capture_pass, cap_standard)
        || aotx_steer_run_capture(&run, &heldout, device_layers, layer_count, capture_pass, cap_heldout)) return 1;
    aotx_steer_mean<<<(unsigned int)((width + 255u) / 256u), 256u>>>(cap_pairs, pairs.pairs, layer_count, hidden, vector);
    aotx_probe_fit<<<layer_count, 256u>>>(cap_pairs, pairs.pairs, layer_count, hidden, variance, direction);
    aotx_probe_read<<<dim3(standard.texts, layer_count), 256u>>>(cap_standard, direction, standard.texts, hidden, read_standard);
    aotx_probe_scale<<<layer_count, 256u>>>(read_standard, standard.texts, mean, scale);
    aotx_probe_read<<<dim3(heldout.texts, layer_count), 256u>>>(cap_heldout, direction, heldout.texts, hidden, read_heldout);
    aotx_probe_count<<<layer_count, 256u>>>(read_heldout, heldout.pairs, mean, figure);
    /* The potency of each layer is the potency of that layer's direction on its own. */
    for (unsigned int l = 0u; l < layer_count; ++l) {
        char name[AOTX_CONDUCT_NAME_BYTES]; aotx_conduct_table table;
        snprintf(name, sizeof name, "%s-%u", axis, layers[l]);
        aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_conduct, sizeof table), "cudaMemcpyFromSymbol");
        id[l] = table.vectors;
        if (aotx_conduct_register_vector(name, &layers[l], 1u, hidden, vector + (size_t)l * hidden, 0.0f)) return 1;
    }
    float potency[AOTX_CONDUCT_LAYERS], host_mean[AOTX_CONDUCT_LAYERS], host_scale[AOTX_CONDUCT_LAYERS], host_figure[2u * AOTX_CONDUCT_LAYERS];
    if (aotx_steer_run_potency(&run, &pairs, id, layer_count, 1.0f, plain, steered, sum, potency)) return 1;
    float *host_vector = (float *)malloc(width * sizeof(float)), *host_direction = (float *)malloc(width * sizeof(float));
    aotx_check_runtime(cudaMemcpy(host_vector, vector, width * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(host_direction, direction, width * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(host_mean, mean, layer_count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(host_scale, scale, layer_count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(host_figure, figure, 2u * layer_count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    /* The readouts of the standardization texts at the probe layer are the source of the
     * mean and the scale. A check computes the two again from the printed lines. */
    if (print_readouts) {
        float *host_read = (float *)malloc(standard.texts * sizeof(float));
        aotx_check_runtime(cudaMemcpy(host_read, read_standard + (size_t)probe_at * standard.texts,
                                      standard.texts * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
        for (unsigned int t = 0u; t < standard.texts; ++t) {
            printf("readout %s: text %u, layer %u, %.9g\n", axis, t + 1u, layers[probe_at], (double)host_read[t]);
        }
        free(host_read);
    }
    /* The vector's layer is the one with the highest accuracy, ties by the agreement, then
     * the earlier layer. A layer whose readouts have no spread over the standardization
     * set is not chosen. The probe layer must have a spread, because the probe is written
     * there. */
    int chosen = -1;
    for (unsigned int l = 0u; l < layer_count; ++l) {
        float accuracy = host_figure[2u * l], agreement = host_figure[2u * l + 1u];
        printf("axis %s layer %u: accuracy %.9g agreement %.9g potency %.9g nats mean %.9g scale %.9g\n",
               axis, layers[l], (double)accuracy, (double)agreement, (double)potency[l], (double)host_mean[l], (double)host_scale[l]);
        if (!(host_scale[l] > 0.0f) || !isfinite(host_scale[l]) || !isfinite(host_mean[l])) continue;
        if (chosen < 0 || accuracy > host_figure[2u * chosen]
            || (accuracy == host_figure[2u * chosen] && agreement > host_figure[2u * chosen + 1u])) chosen = (int)l;
    }
    if (chosen < 0) { fprintf(stderr, "no layer of the axis %s has a spread over the standardization set\n", axis); return 1; }
    if (!(host_scale[probe_at] > 0.0f) || !isfinite(host_scale[probe_at]) || !isfinite(host_mean[probe_at])) {
        fprintf(stderr, "the probe layer %u of the axis %s has no spread over the standardization set\n", layers[probe_at], axis);
        return 1;
    }
    unsigned int best = (unsigned int)chosen; float accuracy = host_figure[2u * probe_at], agreement = host_figure[2u * probe_at + 1u];
    int state = write_vector(models, axis, &layers[best], 1u, hidden, host_vector + (size_t)best * hidden, potency[best]);
    if (state == 0) {
        state = aotx_steer_write_probe(models, axis, number, hidden, layers[probe_at], accuracy, agreement,
                                       host_mean[probe_at], host_scale[probe_at], host_direction + (size_t)probe_at * hidden);
    }
    printf("axis %s: layer %u chosen, accuracy %.9g, agreement %.9g, %u pairs, %u standardization texts, %u held-out pairs\n",
           axis, layers[best], (double)host_figure[2u * best], (double)host_figure[2u * best + 1u], pairs.pairs, standard.texts, heldout.pairs);
    printf("axis %s: probe layer %u, accuracy %.9g, agreement %.9g, mean %.9g, scale %.9g, standardized on %s%s\n",
           axis, layers[probe_at], (double)accuracy, (double)agreement, (double)host_mean[probe_at], (double)host_scale[probe_at],
           standard_path, (accuracy < 0.8f) ? ", the accuracy is under 0.8 and the loader marks the row a monitor" : "");
    free(host_vector); free(host_direction); aotx_steer_run_close(&run);
    return state;
}

static int usage(void)
{
    fprintf(stderr, "usage: aotx_steer_derive --models DIR --trait NAME --pairs FILE --layers LIST [--role NAME]\n"
                    "       aotx_steer_derive --models DIR --axis NAME --pairs FILE --neutral FILE --heldout FILE --layers LIST\n"
                    "                         [--probe-layer L] [--standardise FILE] [--print-readouts] [--role NAME]\n"
                    "       aotx_steer_derive --models DIR --calibrate --axes LIST [--guards LIST] --neutral FILE --dose D [--surgical R] [--role NAME]\n");
    return 2;
}

int main(int argc, char **argv)
{
    const char *models = 0, *trait = 0, *axis = 0, *pairs = 0, *neutral = 0, *heldout = 0, *standard = 0;
    const char *layer_text = 0, *role = "language", *axes = 0, *guards = 0, *probe_text = 0;
    float dose = 0.5f, surgical = 2.0f; int calibrate = 0, print_readouts = 0;
    /* The printed lines are the evidence of a run, so they leave the program as they come. */
    setvbuf(stdout, 0, _IOLBF, 0);
    for (int i = 1; i < argc; ) {
        if (!strcmp(argv[i], "--calibrate")) { calibrate = 1; i += 1; continue; }
        if (!strcmp(argv[i], "--print-readouts")) { print_readouts = 1; i += 1; continue; }
        /* Every other option takes one value. A last option with no value is refused by name. */
        if (i + 1 >= argc) { fprintf(stderr, "the option %s has no value\n", argv[i]); return 2; }
        const char *value = argv[i + 1];
        if (!strcmp(argv[i], "--models")) models = value;
        else if (!strcmp(argv[i], "--trait")) trait = value;
        else if (!strcmp(argv[i], "--axis")) axis = value;
        else if (!strcmp(argv[i], "--pairs")) pairs = value;
        else if (!strcmp(argv[i], "--neutral")) neutral = value;
        else if (!strcmp(argv[i], "--heldout")) heldout = value;
        else if (!strcmp(argv[i], "--layers")) layer_text = value;
        else if (!strcmp(argv[i], "--role")) role = value;
        else if (!strcmp(argv[i], "--axes")) axes = value;
        else if (!strcmp(argv[i], "--guards")) guards = value;
        else if (!strcmp(argv[i], "--dose")) dose = strtof(value, 0);
        else if (!strcmp(argv[i], "--surgical")) surgical = strtof(value, 0);
        else if (!strcmp(argv[i], "--standardise")) standard = value;
        else if (!strcmp(argv[i], "--probe-layer")) probe_text = value;
        else { fprintf(stderr, "the option %s is not known\n", argv[i]); return 2; }
        i += 2;
    }
    if (calibrate) {
        if (!models || !axes || !neutral || !(dose > 0.0f) || !(surgical > 0.0f)) return usage();
        return aotx_steer_calibrate(models, role, axes, guards, neutral, dose, surgical);
    }
    unsigned int layers[AOTX_CONDUCT_LAYERS];
    int layer_count = layer_text ? layers_of(layer_text, layers) : -1;
    if (!models || !pairs || layer_count < 0) return usage();
    if (axis) {
        if (!neutral || !heldout) return usage();
        /* The probe layer is one of the named layers, the last one when none is named. The
         * standardization set is the neutral set when none is named. */
        int probe_at = layer_count - 1;
        if (probe_text) {
            char *end = 0; unsigned long value = strtoul(probe_text, &end, 10);
            probe_at = -1;
            for (int l = 0; end != probe_text && *end == '\0' && l < layer_count; ++l) if (layers[l] == value) probe_at = l;
            if (probe_at < 0) { fprintf(stderr, "the probe layer %s is not a layer of the list\n", probe_text); return 2; }
        }
        return derive_axis(models, role, axis, pairs, standard ? standard : neutral, heldout, layers,
                           (unsigned int)layer_count, (unsigned int)probe_at, print_readouts);
    }
    if (!trait) return usage();
    return derive_trait(models, role, trait, pairs, layers, (unsigned int)layer_count);
}
