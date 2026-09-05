/* Purpose: Hold the fixture store of the affect check and the check of its loader.
 * Owns: The fixture directions, the probe files written and the loader cases.
 * Launch shape: Host code; the loader is host glue and the check reads its tables back.
 * Lifetime: One run of the test program. */
#ifndef AOTX_TESTS_AFFECT_STORE_H
#define AOTX_TESTS_AFFECT_STORE_H


/* The axes of the fixture and the layer each one reads. */
static const unsigned int aotx_affect_test_axis[4] = { 0u, 1u, 4u, 5u };
static const unsigned int aotx_affect_test_at[4] = {
    AOTX_AFFECT_TEST_LAYER, AOTX_AFFECT_TEST_LAYER, AOTX_AFFECT_TEST_LAYER,
    AOTX_AFFECT_TEST_LAYER
};
static const float aotx_affect_test_accuracy[4] = { 0.9f, 0.7f, 0.95f, 0.9f };

static float aotx_affect_test_mean(unsigned int axis)
{
    return 0.5f * (float)(axis + 1u);
}

static float aotx_affect_test_scale(unsigned int axis)
{
    return (axis < AOTX_AFFECT_GUARD_AXIS) ? 2.0f : 0.5f;
}

/* The direction of one axis: a sign pattern over the width, of unit length. Two axes give
 * two patterns that are orthogonal, so the readout of one axis is exact whatever the
 * others carry. */
static void aotx_affect_test_direction(unsigned int axis, unsigned int hidden, float *out)
{
    float scale = 1.0f / sqrtf((float)hidden);
    for (unsigned int i = 0u; i < hidden; ++i) {
        out[i] = (((i >> axis) & 1u) != 0u) ? -scale : scale;
    }
}

/* The fixture store: a directory, its probe directory and the files written into it. */
typedef struct aotx_affect_test_store {
    char dir[AOTX_AFFECT_TEST_PATH];
    char file[AOTX_AFFECT_TEST_FILES][64];
    unsigned int files;
} aotx_affect_test_store;

static int aotx_affect_test_open_store(aotx_affect_test_store *store)
{
    const char *base = getenv("TMPDIR");
    char path[AOTX_AFFECT_TEST_PATH + 320u];
    memset(store, 0, sizeof *store);
    snprintf(store->dir, sizeof store->dir, "%s/aotx-affect-XXXXXX",
             (base != 0 && base[0] != '\0') ? base : "/tmp");
    if (mkdtemp(store->dir) == 0) {
        return 1;
    }
    snprintf(path, sizeof path, "%s/affect", store->dir);
    return mkdir(path, 0700) != 0;
}

static void aotx_affect_test_shut_store(aotx_affect_test_store *store)
{
    char path[AOTX_AFFECT_TEST_PATH + 320u];
    for (unsigned int i = 0u; i < store->files; ++i) {
        snprintf(path, sizeof path, "%s/%s", store->dir, store->file[i]);
        unlink(path);
    }
    snprintf(path, sizeof path, "%s/probes.jsonl", store->dir);
    unlink(path);
    snprintf(path, sizeof path, "%s/affect/calibration.jsonl", store->dir);
    unlink(path);
    snprintf(path, sizeof path, "%s/affect", store->dir);
    rmdir(path);
    rmdir(store->dir);
}

/* Write one probe file with the head of the table and a direction of the given width. */
static int aotx_affect_test_probe(aotx_affect_test_store *store, const char *file,
                                  const char *magic, unsigned int hidden, unsigned int layer,
                                  unsigned int axis, float accuracy)
{
    char path[AOTX_AFFECT_TEST_PATH + 320u];
    float *direction = (float *)malloc((size_t)hidden * sizeof(float));
    snprintf(path, sizeof path, "%s/%s", store->dir, file);
    FILE *out = fopen(path, "wb");
    if (out == 0 || direction == 0) {
        free(direction);
        return 1;
    }
    if (store->files < AOTX_AFFECT_TEST_FILES) {
        snprintf(store->file[store->files], sizeof store->file[0], "%s", file);
        store->files += 1u;
    }
    unsigned int reserved = 0u;
    float agreement = 0.93f;
    float mean = aotx_affect_test_mean(axis);
    float scale = aotx_affect_test_scale(axis);
    aotx_affect_test_direction(axis, hidden, direction);
    int bad = fwrite(magic, 1u, 8u, out) != 8u
           || fwrite(&hidden, sizeof hidden, 1u, out) != 1u
           || fwrite(&layer, sizeof layer, 1u, out) != 1u
           || fwrite(&axis, sizeof axis, 1u, out) != 1u
           || fwrite(&accuracy, sizeof accuracy, 1u, out) != 1u
           || fwrite(&agreement, sizeof agreement, 1u, out) != 1u
           || fwrite(&mean, sizeof mean, 1u, out) != 1u
           || fwrite(&scale, sizeof scale, 1u, out) != 1u
           || fwrite(&reserved, sizeof reserved, 1u, out) != 1u
           || fwrite(direction, sizeof(float), hidden, out) != hidden;
    fclose(out);
    free(direction);
    return bad;
}

/* Write the catalog from the given rows. The accuracy prints with nine digits, so the
 * loader reads the float the file holds. */
static int aotx_affect_test_catalog(const aotx_affect_test_store *store,
                                    const char *const *file, const unsigned int *axis,
                                    const unsigned int *layer, const float *accuracy,
                                    unsigned int count)
{
    char path[AOTX_AFFECT_TEST_PATH + 320u];
    snprintf(path, sizeof path, "%s/probes.jsonl", store->dir);
    FILE *out = fopen(path, "w");
    if (out == 0) {
        return 1;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        fprintf(out, "{\"name\":\"axis%u\",\"file\":\"%s\",\"axis\":%u,\"layer\":%u,"
                     "\"accuracy\":%.9g}\n", axis[i], file[i], axis[i], layer[i],
                (double)accuracy[i]);
    }
    fclose(out);
    return 0;
}

static void aotx_affect_test_table(aotx_affect_table *table, const float **probe)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_affect_rows, sizeof *table),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(probe, aotx_affect_probe, sizeof *probe),
                       "cudaMemcpyFromSymbol");
}

/* The descriptor of the language role: the width and the layer count the loader reads. */
static void aotx_affect_test_model(void)
{
    aotx_model_desc desc;
    memset(&desc, 0, sizeof desc);
    desc.role = AOTX_AFFECT_TEST_ROLE;
    desc.hidden = AOTX_AFFECT_TEST_HIDDEN;
    desc.layers = AOTX_AFFECT_TEST_LAYERS;
    desc.probe_layer = AOTX_AFFECT_TEST_LAYER;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &desc, sizeof desc,
                                          (size_t)AOTX_AFFECT_TEST_ROLE * sizeof desc),
                       "cudaMemcpyToSymbol");
}

/* Write the four good files and the good catalog, in a shuffled catalog order. */
static int aotx_affect_test_good_store(aotx_affect_test_store *store)
{
    static const char *const file[4] = {
        "affect/valence.aotxprb", "affect/arousal.aotxprb", "affect/sycophancy.aotxprb",
        "affect/refusal.aotxprb"
    };
    static const unsigned int order[4] = { 2u, 0u, 3u, 1u };
    const char *shuffled[4];
    unsigned int axis[4], layer[4];
    float accuracy[4];
    for (unsigned int i = 0u; i < 4u; ++i) {
        if (aotx_affect_test_probe(store, file[i], "AOTXPRB1", AOTX_AFFECT_TEST_HIDDEN,
                                   aotx_affect_test_at[i], aotx_affect_test_axis[i],
                                   aotx_affect_test_accuracy[i]) != 0) {
            return 1;
        }
        shuffled[i] = file[order[i]];
        axis[i] = aotx_affect_test_axis[order[i]];
        layer[i] = aotx_affect_test_at[order[i]];
        accuracy[i] = aotx_affect_test_accuracy[order[i]];
    }
    return aotx_affect_test_catalog(store, shuffled, axis, layer, accuracy, 4u);
}

/* Write one composite vector in the conduct file format. */
static int aotx_affect_test_composite_file(aotx_affect_test_store *store,
                                           const char *file, unsigned int axis)
{
    char path[AOTX_AFFECT_TEST_PATH + 320u];
    float direction[AOTX_AFFECT_TEST_HIDDEN];
    unsigned int hidden = AOTX_AFFECT_TEST_HIDDEN;
    unsigned int layers = 1u, layer = AOTX_AFFECT_TEST_LAYER, reserved = 0u;
    float potency = 1.0f;
    snprintf(path, sizeof path, "%s/%s", store->dir, file);
    FILE *out = fopen(path, "wb");
    if (out == 0) return 1;
    aotx_affect_test_direction(axis, hidden, direction);
    int bad = fwrite("AOTXSTV1", 1u, 8u, out) != 8u
           || fwrite(&hidden, sizeof hidden, 1u, out) != 1u
           || fwrite(&layers, sizeof layers, 1u, out) != 1u
           || fwrite(&potency, sizeof potency, 1u, out) != 1u
           || fwrite(&reserved, sizeof reserved, 1u, out) != 1u
           || fwrite(&layer, sizeof layer, 1u, out) != 1u
           || fwrite(direction, sizeof(float), hidden, out) != hidden;
    fclose(out);
    if (store->files < AOTX_AFFECT_TEST_FILES) {
        snprintf(store->file[store->files], sizeof store->file[0], "%s", file);
        store->files += 1u;
    }
    return bad;
}

/* Write the fixture calibration as the last line, with or without both trust marks. */
static int aotx_affect_test_calibration(aotx_affect_test_store *store, unsigned int marked)
{
    char path[AOTX_AFFECT_TEST_PATH + 320u];
    snprintf(path, sizeof path, "%s/affect/calibration.jsonl", store->dir);
    FILE *out = fopen(path, "w");
    if (out == 0) return 1;
    fprintf(out, "{\"axes\":[\"valence\",\"arousal\"],\"K\":[[4,0],[0,1]],"
                 "\"composite\":[\"affect/composite-valence.aotxvec\","
                 "\"affect/composite-arousal.aotxvec\"],\"dominant\":%u,"
                 "\"orthogonal\":%u}\n", marked, marked);
    fclose(out);
    return 0;
}

static int aotx_affect_test_composite_store(aotx_affect_test_store *store,
                                             unsigned int marked)
{
    if (aotx_affect_test_composite_file(store, "affect/composite-valence.aotxvec", 0u)
        || aotx_affect_test_composite_file(store, "affect/composite-arousal.aotxvec", 1u))
        return 1;
    return aotx_affect_test_calibration(store, marked);
}

/* One refused store: a catalog of one row that names a file with the given fault. */
static void aotx_affect_test_refusal(aotx_affect_test_store *store, const char *name,
                                     const char *file, const char *magic,
                                     unsigned int hidden, unsigned int layer,
                                     unsigned int axis, float file_accuracy,
                                     float catalog_accuracy)
{
    const char *files[2] = { file, file };
    unsigned int axes[2] = { axis, axis };
    unsigned int layers[2] = { layer, layer };
    float accuracy[2] = { catalog_accuracy, catalog_accuracy };
    unsigned int rows = (strcmp(name, "a second row of the same axis is refused") == 0)
                      ? 2u : 1u;
    aotx_affect_table table;
    const float *probe = 0;
    int wrote = aotx_affect_test_probe(store, file, magic, hidden, layer, axis,
                                       file_accuracy)
             || aotx_affect_test_catalog(store, files, axes, layers, accuracy, rows);
    int refused = (wrote == 0) && aotx_affect_load_store(store->dir) != 0;
    aotx_affect_test_table(&table, &probe);
    aotx_affect_note(name, refused && table.count == 0u && probe == 0, "rows",
                     (double)table.count, 0.0);
}

/* One catalog must fit every resident language role, including the alternate-only case. */
static void aotx_affect_test_resident_roles(aotx_affect_test_store *store)
{
    static const char *const name[] = {
        "no resident language model leaves the probe shape unchecked",
        "the primary language role alone accepts matching probes",
        "the alternate language role alone accepts matching probes",
        "the primary language role alone rejects a stale probe layer",
        "the alternate language role alone rejects a stale probe layer",
        "both language roles accept the same probe shape",
        "both language roles reject different selected probe layers",
        "both language roles reject different hidden widths",
        "the alternate language role rejects a probe beyond its layers"
    };
    static const unsigned int resident[] = { 0u, 1u, 2u, 1u, 2u, 3u, 3u, 3u, 2u };
    static const unsigned int accepts[] = { 1u, 1u, 1u, 0u, 0u, 1u, 0u, 0u, 0u };
    aotx_model_desc saved[2], desc[2];
    aotx_check_runtime(cudaMemcpyFromSymbol(saved, aotx_model, sizeof saved,
                        AOTX_MODEL_LANGUAGE * sizeof saved[0]), "cudaMemcpyFromSymbol");
    for (unsigned int c = 0u; c < sizeof resident / sizeof resident[0]; ++c) {
        memset(desc, 0, sizeof desc);
        for (unsigned int i = 0u; i < 2u; ++i) {
            desc[i].role = AOTX_MODEL_LANGUAGE + i;
            desc[i].hidden = AOTX_AFFECT_TEST_HIDDEN;
            desc[i].layers = (resident[c] & (1u << i)) ? AOTX_AFFECT_TEST_LAYERS : 0u;
            desc[i].probe_layer = AOTX_AFFECT_TEST_LAYER;
        }
        if (c == 3u || c == 4u) desc[c - 3u].probe_layer += 1u;
        if (c == 6u) desc[1].probe_layer += 1u;
        if (c == 7u) desc[1].hidden += 32u;
        if (c == 8u) desc[1].layers = AOTX_AFFECT_TEST_LAYER;
        if (c == 0u) desc[0].hidden = desc[1].hidden = 1u;
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, desc, sizeof desc,
                            AOTX_MODEL_LANGUAGE * sizeof desc[0]), "cudaMemcpyToSymbol");
        int loaded = aotx_affect_load_store(store->dir);
        aotx_affect_table table;
        const float *probe = 0;
        aotx_affect_test_table(&table, &probe);
        int right = accepts[c]
            ? loaded == 0 && table.count == 4u && table.hidden == AOTX_AFFECT_TEST_HIDDEN
                && table.layers == (1ull << AOTX_AFFECT_TEST_LAYER) && probe != 0
            : loaded != 0 && table.count == 0u && probe == 0;
        aotx_affect_note(name[c], right, "rows", (double)table.count,
                         accepts[c] ? 4.0 : 0.0);
    }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, saved, sizeof saved,
                        AOTX_MODEL_LANGUAGE * sizeof saved[0]), "cudaMemcpyToSymbol");
}

static void aotx_affect_test_loader(aotx_affect_test_store *store)
{
    aotx_affect_table table;
    const float *probe = 0;
    char path[AOTX_AFFECT_TEST_PATH + 320u];
    float direction[AOTX_AFFECT_TEST_HIDDEN];
    float placed[AOTX_AFFECT_TEST_HIDDEN];

    /* No catalog gives zero rows. */
    int loaded = aotx_affect_load_store(store->dir);
    aotx_affect_test_table(&table, &probe);
    aotx_affect_note("a store with no catalog loads zero rows",
                     loaded == 0 && table.count == 0u && probe == 0, "rows",
                     (double)table.count, 0.0);

    /* The good store: four rows in axis order, the monitors marked, the layers set. */
    int wrote = aotx_affect_test_good_store(store);
    loaded = aotx_affect_load_store(store->dir);
    aotx_affect_test_table(&table, &probe);
    unsigned int right = 0u;
    for (unsigned int i = 0u; i < 4u && table.count == 4u; ++i) {
        const aotx_affect_row *row = &table.row[i];
        unsigned int monitor = (aotx_affect_test_accuracy[i] < AOTX_AFFECT_MONITOR_ACCURACY
                                || aotx_affect_test_axis[i] >= AOTX_AFFECT_GUARD_AXIS)
                             ? 1u : 0u;
        right += (row->axis == aotx_affect_test_axis[i] && row->layer == aotx_affect_test_at[i]
                  && row->monitor == monitor
                  && row->mean == aotx_affect_test_mean(row->axis)
                  && row->scale == aotx_affect_test_scale(row->axis)) ? 1u : 0u;
    }
    unsigned long long layers = 1ull << AOTX_AFFECT_TEST_LAYER;
    aotx_affect_note("the good store loads four rows in axis order",
                     wrote == 0 && loaded == 0 && table.count == 4u && right == 4u
                     && table.hidden == AOTX_AFFECT_TEST_HIDDEN && table.layers == layers
                     && probe != 0, "rows", (double)right, 4.0);
    aotx_affect_note("a row under the accuracy bound is a monitor",
                     table.count == 4u && table.row[1].monitor == 1u
                     && table.row[0].monitor == 0u, "monitors",
                     (double)(table.row[1].monitor + table.row[0].monitor), 1.0);
    aotx_affect_note("the guard axes are monitors whatever their accuracy",
                     table.count == 4u && table.row[2].monitor == 1u
                     && table.row[3].monitor == 1u, "monitors",
                     (double)(table.row[2].monitor + table.row[3].monitor), 2.0);

    /* The matrix holds the directions in row order. */
    unsigned int same = 0u;
    for (unsigned int i = 0u; i < 4u && probe != 0; ++i) {
        aotx_affect_test_direction(aotx_affect_test_axis[i], AOTX_AFFECT_TEST_HIDDEN,
                                   direction);
        aotx_check_runtime(cudaMemcpy(placed, probe + (size_t)i * AOTX_AFFECT_TEST_HIDDEN,
                                      sizeof placed, cudaMemcpyDeviceToHost), "cudaMemcpy");
        same += (memcmp(placed, direction, sizeof placed) == 0) ? 1u : 0u;
    }
    aotx_affect_note("the matrix holds every direction", same == 4u, "rows", (double)same,
                     4.0);
    aotx_affect_test_resident_roles(store);

    /* The refusals. Each one leaves zero rows. */
    aotx_affect_test_refusal(store, "a file of another width is refused",
                             "affect/wide.aotxprb", "AOTXPRB1",
                             AOTX_AFFECT_TEST_HIDDEN + 32u, AOTX_AFFECT_TEST_LAYER, 0u, 0.9f,
                             0.9f);
    aotx_affect_test_refusal(store, "a file with another magic is refused",
                             "affect/magic.aotxprb", "AOTXVEC1", AOTX_AFFECT_TEST_HIDDEN,
                             AOTX_AFFECT_TEST_LAYER, 0u, 0.9f, 0.9f);
    aotx_affect_test_refusal(store, "a catalog accuracy that differs is refused",
                             "affect/figure.aotxprb", "AOTXPRB1", AOTX_AFFECT_TEST_HIDDEN,
                             AOTX_AFFECT_TEST_LAYER, 0u, 0.9f, 0.5f);
    aotx_affect_test_refusal(store, "a layer beyond the language model is refused",
                             "affect/deep.aotxprb", "AOTXPRB1", AOTX_AFFECT_TEST_HIDDEN,
                             AOTX_AFFECT_TEST_LAYERS + 4u, 0u, 0.9f, 0.9f);
    aotx_affect_test_refusal(store, "a probe at another selected layer is refused",
                             "affect/stale.aotxprb", "AOTXPRB1", AOTX_AFFECT_TEST_HIDDEN,
                             AOTX_AFFECT_TEST_LAYER + 1u, 0u, 0.9f, 0.9f);
    aotx_affect_test_refusal(store, "a second row of the same axis is refused",
                             "affect/twice.aotxprb", "AOTXPRB1", AOTX_AFFECT_TEST_HIDDEN,
                             AOTX_AFFECT_TEST_LAYER, 1u, 0.9f, 0.9f);

    /* A catalog that is gone loads zero rows and gives the matrix back. */
    snprintf(path, sizeof path, "%s/probes.jsonl", store->dir);
    unlink(path);
    loaded = aotx_affect_load_store(store->dir);
    aotx_affect_test_table(&table, &probe);
    aotx_affect_note("a catalog that is gone loads zero rows again",
                     loaded == 0 && table.count == 0u && probe == 0, "rows",
                     (double)table.count, 0.0);

    /* The good store again, for the read branch. */
    wrote = aotx_affect_test_good_store(store);
    loaded = aotx_affect_load_store(store->dir);
    aotx_affect_test_table(&table, &probe);
    aotx_affect_note("the good store loads again", wrote == 0 && loaded == 0
                     && table.count == 4u, "rows", (double)table.count, 4.0);
}

#endif
