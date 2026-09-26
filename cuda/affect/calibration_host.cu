/* Purpose: Load the marked affect composite and its budget matrix.
 * Owns: The composite directions and the built agent rows on the device.
 * Launch shape: Host glue only; the build kernel uses the placed rows.
 * Lifetime: From model load to model release. */
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include "disk/runtime/assets.h"
#include "disk/runtime/qualification.h"
#include "disk/modelfile/manifest.h"
#include <stdlib.h>
#include <string.h>

#include "affect/affect.cuh"
#include "boot/check.h"

#define AOTX_CALIBRATION_LINE 4096u
#define AOTX_CALIBRATION_PATH 1024u
#define AOTX_VECTOR_MAGIC "AOTXSTV1"

typedef struct aotx_calibration_vector_head {
    char magic[8];
    unsigned int hidden;
    unsigned int layers;
    float potency;
    unsigned int reserved;
} aotx_calibration_vector_head;

typedef struct aotx_calibration_vector {
    unsigned int hidden;
    unsigned int count;
    unsigned int layer[AOTX_MODEL_MAX_LAYERS];
    float *value;
} aotx_calibration_vector;

__device__ aotx_affect_composite_desc aotx_affect_composite_table;
__device__ const float *aotx_affect_composite;
__device__ float *aotx_affect_steer;

static float *aotx_calibration_basis;
static float *aotx_calibration_rows;

static int aotx_calibration_path_ok(const char *path)
{
    return path[0] != '\0' && path[0] != '/' && strstr(path, "..") == 0;
}

static void aotx_calibration_drop(aotx_calibration_vector *vector)
{
    free(vector->value);
    memset(vector, 0, sizeof *vector);
}

/* Read one vector in the catalog format. */
static int aotx_calibration_vector_read(const char *dir, const char *file,
                                        aotx_calibration_vector *vector)
{
    char path[AOTX_CALIBRATION_PATH];
    aotx_calibration_vector_head head;
    memset(vector, 0, sizeof *vector);
    if (!aotx_calibration_path_ok(file)) return 1;
    snprintf(path, sizeof path, "%s/%s", dir, file);
    FILE *in = aotx_asset_stream(dir, file);
    if (aotx_control_check(dir, file, AOTX_CONTROL_VECTOR, in)) {
        if (in) fclose(in);
        return 1;
    }
    if (in == 0 || fread(&head, sizeof head, 1u, in) != 1u
        || memcmp(head.magic, AOTX_VECTOR_MAGIC, 8u) != 0 || head.hidden == 0u
        || head.layers == 0u || head.layers > AOTX_MODEL_MAX_LAYERS
        || head.reserved != 0u || !isfinite(head.potency)) {
        if (in != 0) fclose(in);
        return 1;
    }
    size_t cells = (size_t)head.layers * head.hidden;
    vector->value = (float *)malloc(cells * sizeof(float));
    int bad = vector->value == 0
           || fread(vector->layer, sizeof(unsigned int), head.layers, in) != head.layers
           || fread(vector->value, sizeof(float), cells, in) != cells || fgetc(in) != EOF || ferror(in);
    fclose(in);
    for (size_t i = 0u; !bad && i < cells; ++i) bad = !isfinite(vector->value[i]);
    for (unsigned int i = 0u; !bad && i < head.layers; ++i) {
        bad = vector->layer[i] >= AOTX_MODEL_MAX_LAYERS;
        for (unsigned int j = 0u; !bad && j < i; ++j) bad = vector->layer[j] == vector->layer[i];
    }
    if (bad) {
        aotx_calibration_drop(vector);
        return 1;
    }
    vector->hidden = head.hidden;
    vector->count = head.layers;
    return 0;
}

static int aotx_calibration_parse(const char *line, float K[2][2], char file[2][256])
{
    const char *at = strstr(line, "\"K\":[[");
    const char *named = strstr(line, "\"composite\":[");
    int marks = strstr(line, "\"dominant\":1") != 0
             && strstr(line, "\"orthogonal\":1") != 0;
    int axes = strstr(line, "\"axes\":[\"valence\",\"arousal\"]") != 0;
    if (!marks || !axes || at == 0 || named == 0
        || sscanf(at, "\"K\":[[%f,%f],[%f,%f]]", &K[0][0], &K[0][1],
                  &K[1][0], &K[1][1]) != 4
        || sscanf(named, "\"composite\":[\"%255[^\"]\",\"%255[^\"]\"]",
                  file[0], file[1]) != 2) return 1;
    for (unsigned int i = 0u; i < 2u; ++i)
        for (unsigned int j = 0u; j < 2u; ++j)
            if (!isfinite(K[i][j])) return 1;
    return K[0][0] < 0.0f || K[1][1] < 0.0f;
}

static int aotx_calibration_place(const char *dir, const char *line, const aotx_control_permit *permit)
{
    aotx_calibration_vector vector[2];
    aotx_affect_composite_desc table;
    char file[2][256], hash[2][65]; unsigned char expected[2][32], actual[2][32];
    memset(vector, 0, sizeof vector);
    memset(&table, 0, sizeof table);
    table.permit = *permit;
    if (aotx_calibration_parse(line, table.K, file) != 0 || aotx_control_pair(line, expected) ||
        aotx_control_digest(dir, file[0], hash[0]) || aotx_control_digest(dir, file[1], hash[1]) ||
        aotx_manifest_digest(hash[0], actual[0]) || aotx_manifest_digest(hash[1], actual[1]) ||
        memcmp(actual, expected, sizeof actual)
        || aotx_calibration_vector_read(dir, file[0], &vector[0]) != 0
        || aotx_calibration_vector_read(dir, file[1], &vector[1]) != 0
        || vector[0].hidden != vector[1].hidden) {
        aotx_calibration_drop(&vector[0]);
        aotx_calibration_drop(&vector[1]);
        return 1;
    }
    aotx_model_desc language;
    aotx_model_default_desc(&language);
    if (language.hidden != 0u && vector[0].hidden != language.hidden) {
        aotx_calibration_drop(&vector[0]); aotx_calibration_drop(&vector[1]); return 1;
    }
    unsigned int layer[AOTX_MODEL_MAX_LAYERS], layers = 0u;
    for (unsigned int l = 0u; l < AOTX_MODEL_MAX_LAYERS; ++l) {
        unsigned int held = 0u;
        for (unsigned int j = 0u; j < 2u; ++j)
            for (unsigned int i = 0u; i < vector[j].count; ++i) held |= vector[j].layer[i] == l;
        if (held != 0u) layer[layers++] = l;
    }
    size_t cells = (size_t)layers * vector[0].hidden;
    float *host = (float *)calloc(2u * cells, sizeof(float));
    if (host == 0) { aotx_calibration_drop(&vector[0]); aotx_calibration_drop(&vector[1]); return 1; }
    for (unsigned int j = 0u; j < 2u; ++j)
        for (unsigned int i = 0u; i < vector[j].count; ++i)
            for (unsigned int l = 0u; l < layers; ++l)
                if (vector[j].layer[i] == layer[l])
                    memcpy(host + (size_t)j * cells + (size_t)l * vector[j].hidden,
                           vector[j].value + (size_t)i * vector[j].hidden,
                           vector[j].hidden * sizeof(float));
    aotx_check_runtime(cudaMalloc(&aotx_calibration_basis, 2u * cells * sizeof(float)), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(aotx_calibration_basis, host, 2u * cells * sizeof(float),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMalloc(&aotx_calibration_rows,
                                  (size_t)AOTX_SLOTS * cells * sizeof(float)), "cudaMalloc");
    aotx_check_runtime(cudaMemset(aotx_calibration_rows, 0,
                                  (size_t)AOTX_SLOTS * cells * sizeof(float)), "cudaMemset");
    aotx_control_current(&table.identity);
    table.hidden = vector[0].hidden; table.layer_count = layers; table.trusted = 1u;
    for (unsigned int l = 0u; l < layers; ++l) table.layers |= 1ull << layer[l];
    const float *basis = aotx_calibration_basis; float *rows = aotx_calibration_rows;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_affect_composite, &basis, sizeof basis),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_affect_steer, &rows, sizeof rows),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_affect_composite_table, &table, sizeof table),
                       "cudaMemcpyToSymbol");
    free(host); aotx_calibration_drop(&vector[0]); aotx_calibration_drop(&vector[1]);
    printf("affect composite: %u layers, hidden %u, trusted\n", layers, table.hidden);
    return 0;
}

int aotx_affect_load_calibration(const char *dir)
{
    char path[AOTX_CALIBRATION_PATH], line[AOTX_CALIBRATION_LINE], last[AOTX_CALIBRATION_LINE];
    snprintf(path, sizeof path, "%s/affect/calibration.jsonl", dir);
    FILE *in = aotx_asset_stream(dir, "affect/calibration.jsonl");
    if (in && aotx_control_check(dir, "affect/calibration.jsonl", AOTX_CONTROL_CALIBRATION, in)) {
        fclose(in); return 1;
    }
    aotx_control_permit permit;
    if (in && aotx_qualification_read(dir, "affect/calibration.jsonl", AOTX_CONTROL_CALIBRATION, &permit)) {
        fclose(in); return 1;
    }
    if (in && permit.status != AOTX_QUALIFICATION_ACCEPTED) {
        fclose(in); fprintf(stderr, "affect composite: accepted evidence is unavailable\n"); return 0;
    }
    last[0] = '\0';
    while (in != 0 && fgets(line, sizeof line, in) != 0)
        if (line[0] != '\n' && line[0] != '\r') snprintf(last, sizeof last, "%s", line);
    if (in != 0) fclose(in);
    if (last[0] == '\0' || aotx_calibration_place(dir, last, &permit) != 0) {
        fprintf(stderr, "affect composite: the last calibration is not trusted\n");
    }
    return 0;
}

void aotx_affect_release_calibration(void)
{
    aotx_affect_composite_desc table;
    const float *basis = 0; float *rows = 0;
    if (aotx_calibration_basis != 0) cudaFree(aotx_calibration_basis);
    if (aotx_calibration_rows != 0) cudaFree(aotx_calibration_rows);
    aotx_calibration_basis = 0; aotx_calibration_rows = 0; memset(&table, 0, sizeof table);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_affect_composite, &basis, sizeof basis), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_affect_steer, &rows, sizeof rows), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_affect_composite_table, &table, sizeof table), "cudaMemcpyToSymbol");
}
