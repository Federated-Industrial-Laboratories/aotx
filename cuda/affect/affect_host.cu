/* Purpose: Load the probe rows of a model store and place them on the device.
 * Owns: The device allocation that holds the probe matrix.
 * Launch shape: Host glue only; the turn node is captured here.
 * Lifetime: From model load to model release. */
#include <cuda_runtime.h>

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include "disk/runtime/assets.h"
#include <stdlib.h>
#include <string.h>

#include "affect/affect.cuh"
#include "boot/check.h"

#define AOTX_PROBE_MAGIC "AOTXPRB1"
#define AOTX_PROBE_PATH  1024u
#define AOTX_PROBE_LINE  512u
#define AOTX_PROBE_NAME  32u

/* The head of a probe file, in the order and the sizes of its table. */
typedef struct aotx_probe_head {
    char magic[8];
    unsigned int hidden;
    unsigned int layer;
    unsigned int axis;
    float accuracy;
    float agreement;
    float mean;
    float scale;
    unsigned int reserved;
} aotx_probe_head;

/* The rows a load collects before it places them, one for each axis at the most. */
typedef struct aotx_probe_load {
    aotx_affect_row row[AOTX_AFFECT_AXES];
    float *direction[AOTX_AFFECT_AXES];
    unsigned int count;
    unsigned int hidden;
} aotx_probe_load;

static float *aotx_affect_matrix;

/* Every refusal states its reason, so a start that stops here says why. */
static int aotx_affect_refuse(const char *name, const char *reason)
{
    fprintf(stderr, "the probe row %s is refused: %s\n", name, reason);
    return 1;
}

static void aotx_affect_drop(aotx_probe_load *load)
{
    for (unsigned int i = 0u; i < load->count; ++i) {
        free(load->direction[i]);
    }
    load->count = 0u;
}

/* Read one probe file and keep its row. The catalog line names the axis, the layer and
 * the accuracy, and the head of the file must state the same three. A row of another
 * width or layer than any resident language model does not load. With no resident
 * language model, no model shape is available for comparison. */
static int aotx_affect_probe_file(aotx_probe_load *load, const char *dir, const char *file,
                                  const char *name, unsigned int axis, unsigned int layer,
                                  float accuracy)
{
    FILE *in = aotx_asset_stream(dir, file);
    aotx_probe_head head;
    if (in == 0 || fread(&head, sizeof head, 1u, in) != 1u
        || memcmp(head.magic, AOTX_PROBE_MAGIC, 8u) != 0) {
        if (in != 0) fclose(in);
        return aotx_affect_refuse(name, "the file does not read as a probe file");
    }
    if (head.axis != axis || head.layer != layer || axis >= AOTX_AFFECT_AXES
        || layer >= AOTX_MODEL_MAX_LAYERS || head.hidden == 0u) {
        fclose(in);
        return aotx_affect_refuse(name, "the axis or the layer is out of range or differs");
    }
    if (head.accuracy != accuracy) {
        fclose(in);
        return aotx_affect_refuse(name, "the file has no matching accuracy figure");
    }
    if (!isfinite(head.mean) || !isfinite(head.scale) || head.scale <= 0.0f
        || !isfinite(accuracy) || !isfinite(head.agreement)) {
        fclose(in);
        return aotx_affect_refuse(name, "the mean, the scale, the accuracy or the agreement "
                                        "is not a figure");
    }
    if (head.reserved != 0u) {
        fclose(in);
        return aotx_affect_refuse(name, "the reserved field is not zero");
    }
    aotx_model_desc language;
    for (unsigned int role = AOTX_MODEL_LANGUAGE; role <= AOTX_MODEL_LANGUAGE_Q4; ++role) {
        aotx_check_runtime(cudaMemcpyFromSymbol(&language, aotx_model,
                           offsetof(aotx_model_desc, ffn), role * sizeof language), "cudaMemcpyFromSymbol");
        if (language.layers == 0u) continue;
        if (head.hidden != language.hidden) {
            fclose(in);
            fprintf(stderr, "the probe row %s has the width %u, language role %u has %u\n",
                    name, head.hidden, role, language.hidden);
            return 1;
        }
        if (layer >= language.layers) {
            fclose(in);
            fprintf(stderr, "the probe row %s reads the layer %u, language role %u has %u layers\n",
                    name, layer, role, language.layers);
            return 1;
        }
        if (layer != language.probe_layer) {
            fclose(in);
            fprintf(stderr, "the probe row %s reads the layer %u, language role %u selects %u; derive the probe again\n",
                    name, layer, role, language.probe_layer);
            return 1;
        }
    }
    if (load->count != 0u && load->hidden != head.hidden) {
        fclose(in);
        return aotx_affect_refuse(name, "the width differs from the rows before it");
    }
    for (unsigned int i = 0u; i < load->count; ++i) {
        if (load->row[i].axis == axis) {
            fclose(in);
            return aotx_affect_refuse(name, "the axis is in the table");
        }
    }
    float *host = (float *)malloc((size_t)head.hidden * sizeof(float));
    int bad = host == 0 || fread(host, sizeof(float), head.hidden, in) != head.hidden;
    fclose(in);
    double length = 0.0;
    for (unsigned int i = 0u; !bad && i < head.hidden; ++i) {
        bad = !isfinite(host[i]);
        length += (double)host[i] * (double)host[i];
    }
    if (bad) {
        free(host);
        return aotx_affect_refuse(name, "the file does not hold its declared direction");
    }
    /* The direction is a unit vector, so a readout is a length along it. */
    if (fabs(sqrt(length) - 1.0) > 1.0e-3) {
        free(host);
        return aotx_affect_refuse(name, "the direction does not have unit length");
    }
    /* The rows stand in the order of their axes, so the trace reads a fixed row order. */
    unsigned int at = load->count;
    while (at > 0u && load->row[at - 1u].axis > axis) {
        load->row[at] = load->row[at - 1u];
        load->direction[at] = load->direction[at - 1u];
        at -= 1u;
    }
    load->row[at].layer = layer;
    load->row[at].axis = axis;
    load->row[at].mean = head.mean;
    load->row[at].scale = head.scale;
    /* A guard axis is a monitor whatever its accuracy: it is measured and never steered. */
    load->row[at].monitor = (accuracy < AOTX_AFFECT_MONITOR_ACCURACY
                             || axis >= AOTX_AFFECT_GUARD_AXIS) ? 1u : 0u;
    load->direction[at] = host;
    load->count += 1u;
    load->hidden = head.hidden;
    return 0;
}

/* Place the collected rows: one matrix of count rows of hidden floats and the row table. */
static void aotx_affect_place(const aotx_probe_load *load)
{
    aotx_affect_table table;
    memset(&table, 0, sizeof table);
    size_t bytes = (size_t)load->count * load->hidden * sizeof(float);
    float *matrix = 0;
    if (load->count != 0u) {
        aotx_check_runtime(cudaMalloc(&matrix, bytes), "cudaMalloc");
        for (unsigned int i = 0u; i < load->count; ++i) {
            aotx_check_runtime(cudaMemcpy(matrix + (size_t)i * load->hidden,
                                          load->direction[i],
                                          (size_t)load->hidden * sizeof(float),
                                          cudaMemcpyHostToDevice), "cudaMemcpy");
            table.row[i] = load->row[i];
            table.layers |= 1ull << load->row[i].layer;
        }
        table.count = load->count;
        table.hidden = load->hidden;
    }
    aotx_affect_matrix = matrix;
    printf("probes: %u rows\n", load->count);
    const float *device = matrix;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_affect_probe, &device, sizeof device),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_affect_rows, &table, sizeof table),
                       "cudaMemcpyToSymbol");
}

/* Read the catalog of the store. A store with no catalog loads zero rows, and the
 * substrate then runs on the events alone. A refused row leaves zero rows as well. */
int aotx_affect_load_store(const char *dir)
{
    aotx_probe_load load;
    char path[AOTX_PROBE_PATH];
    char line[AOTX_PROBE_LINE];
    memset(&load, 0, sizeof load);
    aotx_affect_release();
    snprintf(path, sizeof path, "%s/probes.jsonl", dir);
    FILE *in = aotx_asset_stream(dir, "probes.jsonl");
    if (in == 0) {
        return aotx_affect_load_calibration(dir);
    }
    int bad = 0;
    while (!bad && fgets(line, sizeof line, in) != 0) {
        char name[AOTX_PROBE_NAME], file[256];
        unsigned int axis, layer;
        float accuracy;
        if (sscanf(line, "{\"name\":\"%31[a-z0-9_-]\",\"file\":\"%255[^\"]\",\"axis\":%u,"
                         "\"layer\":%u,\"accuracy\":%f}",
                   name, file, &axis, &layer, &accuracy) != 5) {
            fprintf(stderr, "a probe row without axis, layer and accuracy does not enter "
                            "the table\n");
            bad = 1;
        } else {
            bad = aotx_affect_probe_file(&load, dir, file, name, axis, layer, accuracy);
        }
    }
    fclose(in);
    if (!bad) {
        aotx_affect_place(&load);
        bad = aotx_affect_load_calibration(dir);
    }
    aotx_affect_drop(&load);
    return bad;
}

void aotx_affect_release(void)
{
    aotx_affect_table none;
    const float *device = 0;
    if (aotx_affect_matrix != 0) {
        cudaFree(aotx_affect_matrix);
        aotx_affect_matrix = 0;
    }
    aotx_affect_release_calibration();
    memset(&none, 0, sizeof none);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_affect_probe, &device, sizeof device),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_affect_rows, &none, sizeof none),
                       "cudaMemcpyToSymbol");
}

/* The affect node and then the quality node follow the agent step. */
int aotx_affect_capture(void *stream)
{
    cudaStream_t on = (cudaStream_t)stream;
    aotx_affect_turn<<<1, AOTX_SLOTS, 0, on>>>();
    aotx_quality_capture(stream);
    return 0;
}
