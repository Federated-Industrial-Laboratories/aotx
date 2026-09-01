/* Purpose: Load contrast prompts, run derivation and write one cataloged steer vector.
 * Owns: The output vector file and its catalog line.
 * Launch shape: Host glue only; all model and numeric work runs in device kernels.
 * Lifetime: One program run. */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "kvcache/kvcache.cuh"
#include "mem/mem.cuh"
#include "model/conduct.cuh"
#include "model/forward.cuh"
#include "model/roles.h"
#include "profile/fit.h"
#include "tools/steer_text.h"

#define AOTX_DERIVE_PAIRS 32u
typedef struct aotx_vector_head { char magic[8]; unsigned int hidden, layers; float potency; unsigned int reserved; } aotx_vector_head;
__global__ void aotx_steer_mean(const float *, unsigned int, unsigned int, unsigned int, float *);
__global__ void aotx_steer_kl(const float *, const float *, unsigned int, unsigned int, float *);

static int layers_of(const char *text, unsigned int *layer)
{
    unsigned int count = 0u; const char *at = text;
    while (*at && count < AOTX_CONDUCT_LAYERS) {
        char *end = 0; unsigned long value = strtoul(at, &end, 10);
        if (end == at || value >= AOTX_CONDUCT_LAYERS) return -1;
        layer[count++] = (unsigned int)value;
        if (*end == ',') at = end + 1; else if (*end == '\0') at = end; else return -1;
    }
    return count ? (int)count : -1;
}

static int prompts_of(const char *path, char **text)
{
    FILE *in = fopen(path, "r"); char line[8192]; unsigned int count = 0u;
    if (!in) return -1;
    while (count < AOTX_DERIVE_PAIRS && fgets(line, sizeof line, in)) {
        char *tab = strchr(line, '\t');
        if (!tab) { fclose(in); return -1; }
        *tab++ = '\0'; tab[strcspn(tab, "\r\n")] = '\0';
        text[2u * count] = strdup(line); text[2u * count + 1u] = strdup(tab);
        if (!text[2u * count] || !text[2u * count + 1u]) { fclose(in); return -1; }
        count++;
    }
    fclose(in); return count ? (int)count : -1;
}

static int write_vector(const char *dir, const char *trait, const unsigned int *layers,
                        unsigned int layer_count, unsigned int hidden,
                        const float *values, float potency)
{
    char path[1024]; snprintf(path, sizeof path, "%s/%s.aotxvec", dir, trait);
    if (access(path, F_OK) == 0) {
        fprintf(stderr, "the steer vector %s is already in the model store\n", trait);
        return 1;
    }
    FILE *out = fopen(path, "wb");
    aotx_vector_head head;
    memset(&head, 0, sizeof head); memcpy(head.magic, "AOTXSTV1", 8u);
    head.hidden = hidden; head.layers = layer_count; head.potency = potency;
    size_t count = (size_t)layer_count * hidden;
    if (!out || fwrite(&head, sizeof head, 1u, out) != 1u
        || fwrite(layers, sizeof *layers, layer_count, out) != layer_count
        || fwrite(values, sizeof *values, count, out) != count || fclose(out) != 0) return 1;
    snprintf(path, sizeof path, "%s/steer.jsonl", dir); out = fopen(path, "a");
    if (!out) return 1;
    int state = fprintf(out, "{\"name\":\"%s\",\"file\":\"%s.aotxvec\",\"potency_nats\":%.9g}\n",
                        trait, trait, (double)potency) < 0 || fclose(out) != 0;
    return state;
}

static int run_pass(unsigned int role, aotx_kv_map *pages, int *ids, unsigned int *offset,
                    unsigned int seqs, unsigned int *agent, aotx_model_how *how,
                    float *logits, float *capture, unsigned int *layers, unsigned int count)
{
    aotx_model_forget();
    if (aotx_model_pages(role, offset, seqs, agent) != 0 || aotx_kv_serve(pages, 0) < 0) return 1;
    return aotx_model_probe(role, ids, offset, seqs, agent, how, logits, capture, layers, count);
}

int main(int argc, char **argv)
{
    const char *models = 0, *trait = 0, *pairs_file = 0, *layer_text = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--models")) models = argv[i + 1];
        else if (!strcmp(argv[i], "--trait")) trait = argv[i + 1];
        else if (!strcmp(argv[i], "--pairs")) pairs_file = argv[i + 1];
        else if (!strcmp(argv[i], "--layers")) layer_text = argv[i + 1]; else return 2;
    }
    unsigned int layers[AOTX_CONDUCT_LAYERS]; char *text[2u * AOTX_DERIVE_PAIRS] = { 0 };
    int layer_count = layer_text ? layers_of(layer_text, layers) : -1;
    int pairs = pairs_file ? prompts_of(pairs_file, text) : -1;
    if (!models || !trait || layer_count < 0 || pairs < 0) {
        fprintf(stderr, "usage: aotx_steer_derive --models DIR --trait NAME --pairs FILE --layers LIST\n"); return 2;
    }
    unsigned int seqs = 2u * (unsigned int)pairs, role = AOTX_PROFILE_LANGUAGE_ROLE;
    aotx_check_runtime(cudaFree(0), "cudaFree"); aotx_mem_map map; aotx_kv_map pages;
    if (aotx_mem_reserve(&map) || aotx_kv_open(&pages)
        || aotx_boot_models(models, AOTX_PROFILE_LANGUAGE, 0)
        || aotx_model_open(role, AOTX_MODEL_MAX_TOKENS)) return 1;
    aotx_model_desc desc; aotx_check_runtime(cudaMemcpyFromSymbol(&desc, aotx_model, sizeof desc,
        role * sizeof desc), "cudaMemcpyFromSymbol");
    aotx_steer_text tokenizer; aotx_steer_text_open(&tokenizer);
    unsigned int *host_ids = (unsigned int *)calloc((size_t)seqs * AOTX_STEER_STRIDE, sizeof(unsigned int));
    unsigned int counts[AOTX_STEER_TEXTS];
    if (!host_ids || aotx_steer_tokenize(&tokenizer, text, seqs, host_ids, counts)) return 1;
    unsigned int total = 0u, host_offset[AOTX_STEER_TEXTS + 1u], host_agent[AOTX_STEER_TEXTS];
    for (unsigned int i = 0u; i < seqs; ++i) { host_offset[i] = total; host_agent[i] = i; total += counts[i]; }
    host_offset[seqs] = total; if (total > AOTX_MODEL_MAX_TOKENS) return 1;
    int *ids = 0; unsigned int *offset = 0, *agent = 0, *device_layers = 0;
    float *capture = 0, *vector = 0, *plain = 0, *steered = 0, *kl = 0;
    aotx_check_runtime(cudaMalloc(&ids, total * sizeof(int)), "cudaMalloc");
    int *flat = (int *)malloc(total * sizeof(int)); unsigned int at = 0u;
    for (unsigned int i = 0u; i < seqs; ++i) for (unsigned int j = 0u; j < counts[i]; ++j) flat[at++] = (int)host_ids[(size_t)i * AOTX_STEER_STRIDE + j];
    aotx_check_runtime(cudaMemcpy(ids, flat, total * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMalloc(&offset, (seqs + 1u) * sizeof(unsigned int)), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&agent, seqs * sizeof(unsigned int)), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&device_layers, layer_count * sizeof(unsigned int)), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(offset, host_offset, (seqs + 1u) * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(agent, host_agent, seqs * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(device_layers, layers, layer_count * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");
    size_t vector_count = (size_t)layer_count * desc.hidden, logits_count = (size_t)seqs * desc.vocab;
    aotx_check_runtime(cudaMalloc(&capture, vector_count * seqs * sizeof(float)), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&vector, vector_count * sizeof(float)), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&plain, logits_count * sizeof(float)), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&steered, logits_count * sizeof(float)), "cudaMalloc");
    if (run_pass(role, &pages, ids, offset, seqs, agent, 0, plain, capture, device_layers, layer_count)) return 1;
    aotx_steer_mean<<<(vector_count + 255u) / 256u, 256u>>>(capture, pairs, layer_count, desc.hidden, vector);
    aotx_conduct_table table; aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_conduct, sizeof table), "cudaMemcpyFromSymbol");
    if (aotx_conduct_register_vector(trait, layers, layer_count, desc.hidden, vector, 0.0f)) return 1;
    aotx_model_how host_how[AOTX_STEER_TEXTS]; memset(host_how, 0, sizeof host_how);
    for (unsigned int i = 0u; i < seqs; ++i) { host_how[i].steer[0] = table.vectors; host_how[i].steer[1] = AOTX_MODEL_CONDUCT_NONE; host_how[i].steer_strength[0] = 1.0f; host_how[i].voice = AOTX_MODEL_CONDUCT_NONE; }
    aotx_model_how *how = 0; aotx_check_runtime(cudaMalloc(&how, seqs * sizeof *how), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(how, host_how, seqs * sizeof *how, cudaMemcpyHostToDevice), "cudaMemcpy");
    if (run_pass(role, &pages, ids, offset, seqs, agent, how, steered, 0, 0, 0)) return 1;
    aotx_check_runtime(cudaMalloc(&kl, sizeof(float)), "cudaMalloc"); aotx_check_runtime(cudaMemset(kl, 0, sizeof(float)), "cudaMemset");
    aotx_steer_kl<<<seqs, 256u>>>(plain, steered, seqs, desc.vocab, kl);
    float potency; aotx_check_runtime(cudaMemcpy(&potency, kl, sizeof potency, cudaMemcpyDeviceToHost), "cudaMemcpy");
    float *host_vector = (float *)malloc(vector_count * sizeof(float));
    aotx_check_runtime(cudaMemcpy(host_vector, vector, vector_count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    int state = write_vector(models, trait, layers, layer_count, desc.hidden, host_vector, potency);
    printf("trait %s: %u pairs, %u layers, potency %.9g nats\n", trait, pairs, layer_count, (double)potency);
    aotx_steer_text_close(&tokenizer); aotx_model_shut(role); aotx_boot_models_release(); aotx_kv_close(&pages); aotx_mem_release(&map);
    return state;
}
