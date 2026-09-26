/* Purpose: Load steer vectors and voice profiles and place their tables on the device.
 * Owns: Device allocations that hold registered vector values.
 * Launch shape: Host glue only; one lookup kernel for each profile string.
 * Lifetime: From model load to model release. */
#include <cuda_runtime.h>

#include "disk/runtime/assets.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "model/conduct.cuh"
#include "model/control.cuh"
#include "model/roles.h"
#include "text/text.cuh"

#define AOTX_VECTOR_MAGIC "AOTXSTV1"
#define AOTX_VECTOR_FILE  1024u
#define AOTX_PROFILE_LINE 512u

typedef struct aotx_vector_head {
    char magic[8];
    unsigned int hidden;
    unsigned int layers;
    float potency;
    unsigned int reserved;
} aotx_vector_head;

static void *aotx_conduct_piece[AOTX_CONDUCT_VECTORS];
static unsigned int aotx_conduct_pieces;

/* Every refusal states its reason, so a start that stops here says why. */
static int aotx_conduct_refuse(const char *kind, const char *name, const char *reason)
{
    fprintf(stderr, "the %s %s is refused: %s\n", kind, name, reason);
    return 1;
}

static int aotx_conduct_name_ok(const char *name)
{
    size_t n = strlen(name);
    if (n == 0u || n >= AOTX_CONDUCT_NAME_BYTES) return 0;
    for (size_t i = 0u; i < n; ++i) {
        if (!((name[i] >= 'a' && name[i] <= 'z') || (name[i] >= '0' && name[i] <= '9')
              || name[i] == '-' || name[i] == '_')) return 0;
    }
    return 1;
}

int aotx_conduct_register_vector(const char *name, const unsigned int *layers,
                                 unsigned int layer_count, unsigned int hidden,
                                 const float *device_values, float potency, unsigned positions,
                                 const aotx_control_permit *permit)
{
    aotx_conduct_table table;
    const char *kind = "steer vector";
    if (!aotx_conduct_name_ok(name)) return aotx_conduct_refuse(kind, name, "the name");
    if (layer_count == 0u || layer_count > AOTX_CONDUCT_LAYERS || hidden == 0u
        || device_values == 0) return aotx_conduct_refuse(kind, name, "the shape");
    if (positions > AOTX_CONTROL_RESPONSE) return aotx_conduct_refuse(kind, name, "the position mode");
    if (!isfinite(potency) || potency < 0.0f) return aotx_conduct_refuse(kind, name, "no finite potency");
    if (aotx_control_values(device_values, (unsigned long long)layer_count * hidden))
        return aotx_conduct_refuse(kind, name, "a value is not finite");
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_conduct, sizeof table),
                       "cudaMemcpyFromSymbol");
    if (table.vectors >= AOTX_CONDUCT_VECTORS)
        return aotx_conduct_refuse(kind, name, "the table is full");
    for (unsigned int i = 0u; i < table.vectors; ++i)
        if (strcmp(name, table.vector[i].name) == 0)
            return aotx_conduct_refuse(kind, name, "the name is in the table");
    aotx_steer_vector row;
    memset(&row, 0, sizeof row);
    if (permit) row.permit = *permit;
    for (unsigned int i = 0u; i < layer_count; ++i) {
        if (layers[i] >= AOTX_CONDUCT_LAYERS || ((row.layers >> layers[i]) & 1ull) != 0ull)
            return aotx_conduct_refuse(kind, name, "a layer is out of range or named twice");
        row.layers |= 1ull << layers[i];
    }
    size_t bytes = (size_t)layer_count * hidden * sizeof(float);
    float *copy = 0;
    aotx_check_runtime(cudaMalloc(&copy, bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(copy, device_values, bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpy");
    aotx_control_current(&row.identity);
    row.value = (unsigned long long)copy;
    row.hidden = hidden;
    row.layer_count = layer_count;
    row.potency = potency; row.positions = positions;
    snprintf(row.name, sizeof row.name, "%s", name);
    size_t at = offsetof(aotx_conduct_table, vector)
              + (size_t)table.vectors * sizeof(aotx_steer_vector);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_conduct, &row, sizeof row, at),
                       "cudaMemcpyToSymbol");
    table.vectors += 1u;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_conduct, &table.vectors,
                       sizeof table.vectors, offsetof(aotx_conduct_table, vectors)),
                       "cudaMemcpyToSymbol");
    aotx_conduct_piece[aotx_conduct_pieces++] = copy;
    return 0;
}

int aotx_conduct_register_voice(const char *name, const unsigned int *tokens,
                                const float *bias, unsigned int count)
{
    aotx_conduct_table table;
    const char *kind = "voice profile";
    if (!aotx_conduct_name_ok(name)) return aotx_conduct_refuse(kind, name, "the name");
    if (count == 0u || count > AOTX_CONDUCT_BIASES)
        return aotx_conduct_refuse(kind, name, "the line count");
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_conduct, sizeof table),
                       "cudaMemcpyFromSymbol");
    if (table.voices >= AOTX_CONDUCT_VOICES)
        return aotx_conduct_refuse(kind, name, "the table is full");
    for (unsigned int i = 0u; i < table.voices; ++i)
        if (strcmp(name, table.voice[i].name) == 0)
            return aotx_conduct_refuse(kind, name, "the name is in the table");
    for (unsigned int i = 0u; i < count; ++i) {
        if (!isfinite(bias[i])) return aotx_conduct_refuse(kind, name, "a bias is not finite");
        for (unsigned int j = 0u; j < i; ++j)
            if (tokens[j] == tokens[i])
                return aotx_conduct_refuse(kind, name, "a token is named twice");
    }
    aotx_voice_bias row;
    memset(&row, 0, sizeof row);
    memcpy(row.token, tokens, count * sizeof(unsigned int));
    memcpy(row.bias, bias, count * sizeof(float));
    aotx_control_current(&row.identity);
    row.count = count;
    snprintf(row.name, sizeof row.name, "%s", name);
    size_t at = offsetof(aotx_conduct_table, voice)
              + (size_t)table.voices * sizeof(aotx_voice_bias);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_conduct, &row, sizeof row, at),
                       "cudaMemcpyToSymbol");
    table.voices += 1u;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_conduct, &table.voices,
                       sizeof table.voices, offsetof(aotx_conduct_table, voices)),
                       "cudaMemcpyToSymbol");
    return 0;
}

static int aotx_conduct_vector_file(const char *dir, const char *file, const char *name,
                                    float catalog_potency)
{
    FILE *in = aotx_asset_stream(dir, file);
    aotx_vector_head head; unsigned positions = AOTX_CONTROL_ALL;
    if (aotx_control_check(dir, file, AOTX_CONTROL_VECTOR, in, &positions)) {
        if (in) fclose(in);
        return 1;
    }
    if (in == 0 || fread(&head, sizeof head, 1u, in) != 1u
        || memcmp(head.magic, AOTX_VECTOR_MAGIC, 8u) != 0
        || head.potency != catalog_potency || head.layers == 0u
        || head.layers > AOTX_CONDUCT_LAYERS || head.reserved != 0u) {
        if (in != 0) fclose(in);
        fprintf(stderr, "the steer vector %s has no matching potency figure\n", name);
        return 1;
    }
    /* A vector of another width is a dial that does nothing, so it does not load. A run
     * with no language model placed has no width to compare with. */
    aotx_model_desc language;
    aotx_model_default_desc(&language);
    if (language.hidden != 0u && head.hidden != language.hidden) {
        fclose(in);
        fprintf(stderr, "the steer vector %s has the width %u, the language model has %u\n",
                name, head.hidden, language.hidden);
        return 1;
    }
    unsigned int layers[AOTX_CONDUCT_LAYERS];
    size_t values = (size_t)head.layers * head.hidden;
    float *host = (float *)malloc(values * sizeof(float));
    float *device = 0;
    int bad = host == 0 || fread(layers, sizeof(unsigned int), head.layers, in) != head.layers
           || fread(host, sizeof(float), values, in) != values || fgetc(in) != EOF || ferror(in);
    for (unsigned i = 0; !bad && i < head.layers; ++i) bad = layers[i] >= language.layers;
    fclose(in);
    if (bad) {
        fprintf(stderr, "the steer vector %s does not hold its declared values\n", name);
        free(host);
        return 1;
    }
    aotx_check_runtime(cudaMalloc(&device, values * sizeof(float)), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device, host, values * sizeof(float), cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_control_permit permit;
    int state = aotx_qualification_read(dir, file, AOTX_CONTROL_VECTOR, &permit);
    if (!state) state = aotx_conduct_register_vector(name, layers, head.layers, head.hidden,
                                              device, head.potency, positions, &permit);
    if (!state && !permit.status) fprintf(stderr, "the steer vector %s is unavailable without accepted evidence\n", name);
    cudaFree(device);
    free(host);
    return state;
}

static int aotx_conduct_vectors(const char *dir)
{
    FILE *in = aotx_asset_stream(dir, "steer.jsonl");
    if (in == 0) return 0;
    char line[AOTX_PROFILE_LINE];
    int bad = 0;
    while (!bad && fgets(line, sizeof line, in) != 0) {
        char name[AOTX_CONDUCT_NAME_BYTES], file[256];
        float potency;
        if (sscanf(line, "{\"name\":\"%31[a-z0-9_-]\",\"file\":\"%255[^\"]\",\"potency_nats\":%f}",
                   name, file, &potency) != 3) {
            fprintf(stderr, "a steer vector without potency does not enter the catalog\n");
            bad = 1;
        } else bad = aotx_conduct_vector_file(dir, file, name, potency);
    }
    fclose(in);
    return bad;
}

static int aotx_conduct_profile(const char *path, const char *file)
{
    char full[AOTX_VECTOR_FILE], name[AOTX_CONDUCT_NAME_BYTES], line[AOTX_PROFILE_LINE];
    snprintf(full, sizeof full, "voice/%s", file);
    FILE *in = aotx_asset_stream(path, full);
    if (in == 0 || fgets(name, sizeof name, in) == 0) {
        if (in) fclose(in);
        fprintf(stderr, "the voice profile %s does not read\n", file);
        return 1;
    }
    name[strcspn(name, "\r\n")] = '\0';
    unsigned int token[AOTX_CONDUCT_BIASES], count = 0u;
    float bias[AOTX_CONDUCT_BIASES];
    while (count < AOTX_CONDUCT_BIASES && fgets(line, sizeof line, in) != 0) {
        char *tab = strchr(line, '\t');
        if (tab == 0) {
            fclose(in);
            fprintf(stderr, "the voice profile %s has a line without a tab\n", name);
            return 1;
        }
        *tab++ = '\0';
        tab[strcspn(tab, "\r\n")] = '\0';
        char *end = 0;
        bias[count] = strtof(line, &end);
        if (end == line || *end != '\0') {
            fclose(in);
            fprintf(stderr, "the voice profile %s has a bias that is not a number: %s\n",
                    name, line);
            return 1;
        }
        unsigned char *text = 0; unsigned int *out = 0;
        aotx_check_runtime(cudaMalloc(&text, strlen(tab)), "cudaMalloc");
        aotx_check_runtime(cudaMalloc(&out, sizeof *out), "cudaMalloc");
        aotx_check_runtime(cudaMemcpy(text, tab, strlen(tab), cudaMemcpyHostToDevice),
                           "cudaMemcpy");
        aotx_conduct_token<<<1, 1>>>(text, (unsigned int)strlen(tab), out);
        aotx_check_runtime(cudaMemcpy(&token[count], out, sizeof *out, cudaMemcpyDeviceToHost),
                           "cudaMemcpy");
        cudaFree(text); cudaFree(out);
        if (token[count] == AOTX_TEXT_NONE) {
            fclose(in);
            fprintf(stderr, "the voice profile %s names the token %s outside the vocabulary\n",
                    name, tab);
            return 1;
        }
        count += 1u;
    }
    fclose(in);
    return aotx_conduct_register_voice(name, token, bias, count);
}

int aotx_conduct_load_store(const char *dir)
{
    if (aotx_conduct_vectors(dir) != 0) return 1;
    char names[AOTX_CONDUCT_VOICES][AOTX_RUNTIME_NAME];
    int count = aotx_asset_names(dir, "voice/", ".profile", names, AOTX_CONDUCT_VOICES);
    if (count < 0) return 1;
    int bad = 0;
    for (int i = 0; i < count && !bad; ++i) bad = aotx_conduct_profile(dir, names[i]);
    return bad;
}

void aotx_conduct_release(void)
{
    for (unsigned int i = 0u; i < aotx_conduct_pieces; ++i) cudaFree(aotx_conduct_piece[i]);
    aotx_conduct_pieces = 0u;
    aotx_conduct_table none;
    memset(&none, 0, sizeof none);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_conduct, &none, sizeof none),
                       "cudaMemcpyToSymbol");
}
