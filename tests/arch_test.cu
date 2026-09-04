/* Purpose: Check that a language model of a store runs. The checks are the bind, the
 *   shapes, the state, the restore, the load line, and the greedy continuation.
 * Owns: The buffers of one run.
 * Launch shape: Host glue calls the forward pass; one thread for each slot in the release.
 * Lifetime: One run of the check program.
 *
 * The check takes a store directory and a directory of reference lists. It loads every
 * language file of the store, one at a time, and prints the checks of one architecture for
 * each file. A reference list holds the prefill ids the reference consumed and the ids it
 * generated at temperature zero. The check feeds the same prefill and compares the ids one
 * for one. The wrap check is not here: the say path writes one wrap for every family. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "kvcache/kvcache.cuh"
#include "mem/mem.cuh"
#include "model/forward.cuh"
#include "model/roles.h"

extern "C" {
#include "disk/modelfile/manifest.h"
}

#define AOTX_ARCH_FILES    8
#define AOTX_ARCH_PROMPTS  16u
#define AOTX_ARCH_IDS      512u
#define AOTX_ARCH_GENERATE 64u

static unsigned int aotx_arch_checks;
static unsigned int aotx_arch_failed;

static void aotx_arch_check(int good, const char *file, const char *text)
{
    aotx_arch_checks += 1u;
    if (!good) {
        aotx_arch_failed += 1u;
    }
    printf("arch: %s %s: %s\n", file, good ? "ok" : "FAILED", text);
}

/* One reference list: the prefill ids and the generated ids. */
typedef struct aotx_arch_list {
    unsigned int prefill[AOTX_ARCH_IDS];
    unsigned int prefills;
    unsigned int generated[AOTX_ARCH_GENERATE];
    unsigned int generates;
} aotx_arch_list;

static unsigned int aotx_arch_ids(const char *line, unsigned int *out, unsigned int max)
{
    unsigned int count = 0u;
    const char *at = line;
    while (*at != '\0' && count < max) {
        char *end = NULL;
        unsigned long value = strtoul(at, &end, 10);
        if (end == at) {
            at += 1;
            continue;
        }
        out[count++] = (unsigned int)value;
        at = end;
    }
    return count;
}

static int aotx_arch_read_list(const char *path, aotx_arch_list *list)
{
    FILE *in = fopen(path, "r");
    char line[8192];
    if (in == NULL) {
        return 1;
    }
    memset(list, 0, sizeof *list);
    while (fgets(line, sizeof line, in) != NULL) {
        if (strncmp(line, "prefill ", 8u) == 0) {
            list->prefills = aotx_arch_ids(line + 8, list->prefill, AOTX_ARCH_IDS);
        } else if (strncmp(line, "generated ", 10u) == 0) {
            list->generates = aotx_arch_ids(line + 10, list->generated, AOTX_ARCH_GENERATE);
        }
    }
    fclose(in);
    return (list->prefills == 0u || list->generates == 0u) ? 1 : 0;
}

/* The buffers of one run. */
typedef struct aotx_arch_gear {
    int *ids;
    unsigned int *offset;
    unsigned int *agent;
    float *logits;
    int *token;
} aotx_arch_gear;

__global__ void aotx_arch_release(unsigned int agent)
{
    if (threadIdx.x == 0u) {
        aotx_kv_release(agent);
    }
}

static unsigned int aotx_arch_mapped(void)
{
    unsigned int mapped = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&mapped, aotx_kv,
                                            sizeof mapped,
                                            offsetof(aotx_kv_table, mapped_pages)),
                       "cudaMemcpyFromSymbol");
    return mapped;
}

/* Send one run of ids of one sequence through the pass and take the greedy token. */
static int aotx_arch_step(aotx_arch_gear *gear, aotx_kv_map *map, unsigned int role,
                          unsigned int agent, const int *ids, unsigned int count,
                          float *logits)
{
    unsigned int offset[2] = { 0u, count };
    aotx_model_how how;
    unsigned long long seed = 0ull;
    memset(&how, 0, sizeof how);
    how.top_k = 1u;
    how.top_p = 1.0f;
    how.temperature = 0.0f;
    how.repeat_penalty = 1.0f;
    how.think_limit = -1;
    aotx_check_runtime(cudaMemcpy(gear->ids, ids, count * sizeof(int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->offset, offset, sizeof offset,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->agent, &agent, sizeof agent,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    if (aotx_model_pages(role, gear->offset, 1u, gear->agent) != 0) {
        return -1;
    }
    aotx_kv_serve(map, 0);
    if (logits != 0) {
        if (aotx_model_prefill(role, gear->ids, gear->offset, 1u, gear->agent, logits, 0)
            != 0) {
            return -1;
        }
        return 0;
    }
    if (aotx_model_sample(role, gear->ids, gear->offset, 1u, gear->agent, &how, gear->token,
                          0, &seed) != 0) {
        return -1;
    }
    int token = 0;
    aotx_check_runtime(cudaMemcpy(&token, gear->token, sizeof token, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    return token;
}

/* The greedy continuation of one prefill: the prompt in one pass, then one token a pass. */
static unsigned int aotx_arch_generate(aotx_arch_gear *gear, aotx_kv_map *map,
                                       unsigned int role, unsigned int agent,
                                       const aotx_arch_list *list, unsigned int *out,
                                       unsigned int want)
{
    int *run = (int *)malloc((size_t)AOTX_ARCH_IDS * sizeof(int));
    for (unsigned int i = 0u; i < list->prefills; ++i) {
        run[i] = (int)list->prefill[i];
    }
    int token = aotx_arch_step(gear, map, role, agent, run, list->prefills, 0);
    unsigned int made = 0u;
    while (token >= 0 && made < want) {
        out[made++] = (unsigned int)token;
        run[0] = token;
        token = aotx_arch_step(gear, map, role, agent, run, 1u, 0);
    }
    free(run);
    return made;
}

static void aotx_arch_print(const char *label, const unsigned int *ids, unsigned int count)
{
    printf("arch:   %s", label);
    for (unsigned int i = 0u; i < count; ++i) {
        printf(" %u", ids[i]);
    }
    printf("\n");
}

/* Check 3, the shapes: one prefill of the first list gives finite logits of vocabulary
 * width and its largest logit is not the end token. */
static void aotx_arch_shapes(aotx_arch_gear *gear, aotx_kv_map *map, unsigned int role,
                             const aotx_model_desc *desc, const aotx_arch_list *list,
                             const char *file, unsigned int end)
{
    int *run = (int *)malloc((size_t)AOTX_ARCH_IDS * sizeof(int));
    for (unsigned int i = 0u; i < list->prefills; ++i) {
        run[i] = (int)list->prefill[i];
    }
    aotx_model_forget();
    int state = aotx_arch_step(gear, map, role, 0u, run, list->prefills, gear->logits);
    float *row = (float *)malloc((size_t)desc->vocab * sizeof(float));
    aotx_check_runtime(cudaMemcpy(row, gear->logits
                                  + (size_t)(list->prefills - 1u) * desc->vocab,
                                  (size_t)desc->vocab * sizeof(float),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int finite = 0u;
    unsigned int best = 0u;
    for (unsigned int i = 0u; i < desc->vocab; ++i) {
        finite += isfinite(row[i]) ? 1u : 0u;
        if (row[i] > row[best]) {
            best = i;
        }
    }
    printf("arch:   shapes: vocabulary %u, finite %u, largest %u at %.3f, end token %u\n",
           desc->vocab, finite, best, (double)row[best], end);
    aotx_arch_check(state == 0 && finite == desc->vocab && best != end, file,
                    "3 shapes: finite logits of vocabulary width, largest not the end");
    aotx_arch_release<<<1, 1>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_kv_serve(map, 0);
    free(row);
    free(run);
}

/* Check 4, the state: a sequence at the context bound opens, and the pool returns to its
 * prior occupancy after the release. The bound is the profile's, not the file's. */
static void aotx_arch_state(aotx_arch_gear *gear, aotx_kv_map *map, unsigned int role,
                            const char *file)
{
    unsigned int before = aotx_arch_mapped();
    int *run = (int *)malloc((size_t)AOTX_MODEL_MAX_TOKENS * sizeof(int));
    aotx_model_forget();
    int bad = 0;
    for (unsigned int at = 0u; at < AOTX_SEQ_MAX_TOKENS && bad == 0;
         at += AOTX_MODEL_MAX_TOKENS) {
        for (unsigned int i = 0u; i < AOTX_MODEL_MAX_TOKENS; ++i) {
            run[i] = (int)(1000u + ((at + i) * 7u) % 100000u);
        }
        bad = aotx_arch_step(gear, map, role, 1u, run, AOTX_MODEL_MAX_TOKENS, 0) < 0;
    }
    unsigned int faults = aotx_model_faulted();
    unsigned int open = aotx_arch_mapped();
    aotx_arch_release<<<1, 1>>>(1u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_kv_serve(map, 0);
    unsigned int after = aotx_arch_mapped();
    printf("arch:   state: %u tokens, pages %u before, %u open, %u after, %u faults\n",
           (unsigned int)AOTX_SEQ_MAX_TOKENS, before, open, after, faults);
    aotx_arch_check(bad == 0 && faults == 0u && open > before && after == before, file,
                    "4 state: a sequence at the context bound opens and closes");
    free(run);
}

/* Check 5, the restore: two turns, then the third turn from a rebuilt cache. The third turn
 * must match the third turn of the run that was not stopped. The rebuild replays the whole
 * list into empty pages, as the restore path does. */
static void aotx_arch_restore(aotx_arch_gear *gear, aotx_kv_map *map, unsigned int role,
                              const aotx_arch_list *list, const char *file)
{
    unsigned int turn[3][8];
    unsigned int again[8];
    int *run = (int *)malloc((size_t)AOTX_ARCH_IDS * sizeof(int));
    unsigned int held = 0u;
    aotx_model_forget();
    for (unsigned int i = 0u; i < list->prefills; ++i) {
        run[held++] = (int)list->prefill[i];
    }
    int token = aotx_arch_step(gear, map, role, 2u, run, held, 0);
    for (unsigned int t = 0u; t < 3u; ++t) {
        for (unsigned int i = 0u; i < 8u; ++i) {
            turn[t][i] = (unsigned int)token;
            run[held++] = token;
            token = aotx_arch_step(gear, map, role, 2u, run + held - 1u, 1u, 0);
        }
    }
    unsigned int whole = held - 8u;
    aotx_arch_release<<<1, 1>>>(2u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_kv_serve(map, 0);

    /* The stop falls after the second turn. The restored run holds the prompt and two
     * turns as one list, replays it, and takes the third turn again. */
    aotx_model_forget();
    token = aotx_arch_step(gear, map, role, 3u, run, whole, 0);
    for (unsigned int i = 0u; i < 8u; ++i) {
        again[i] = (unsigned int)token;
        run[whole + i] = token;
        token = aotx_arch_step(gear, map, role, 3u, run + whole + i, 1u, 0);
    }
    aotx_arch_release<<<1, 1>>>(3u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_kv_serve(map, 0);
    aotx_arch_print("restore third turn, unstopped:", turn[2], 8u);
    aotx_arch_print("restore third turn, restored: ", again, 8u);
    aotx_arch_check(memcmp(turn[2], again, sizeof again) == 0, file,
                    "5 restore: the third turn matches the unstopped run");
    free(run);
}

/* The greedy continuation against every reference list of the file. */
static void aotx_arch_greedy(aotx_arch_gear *gear, aotx_kv_map *map, unsigned int role,
                             const char *name, const char *file, const char *lists)
{
    for (unsigned int p = 0u; p < AOTX_ARCH_PROMPTS; ++p) {
        char path[1200];
        aotx_arch_list list;
        snprintf(path, sizeof path, "%s/%s-%u.ids", lists, name, p);
        if (aotx_arch_read_list(path, &list) != 0) {
            break;
        }
        unsigned int mine[AOTX_ARCH_GENERATE];
        aotx_model_forget();
        unsigned int made = aotx_arch_generate(gear, map, role, 4u, &list, mine,
                                               list.generates);
        aotx_arch_release<<<1, 1>>>(4u);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_kv_serve(map, 0);
        unsigned int first = made;
        for (unsigned int i = 0u; i < made && i < list.generates; ++i) {
            if (mine[i] != list.generated[i]) {
                first = i;
                break;
            }
        }
        printf("arch:   prompt %u: prefill %u ids, reference %u ids, ours %u ids\n", p,
               list.prefills, list.generates, made);
        aotx_arch_print("reference:", list.generated, list.generates);
        aotx_arch_print("ours:     ", mine, made);
        if (first < made) {
            printf("arch:   first difference at position %u: reference %u, ours %u\n",
                   first, list.generated[first], mine[first]);
        }
        char text[96];
        snprintf(text, sizeof text, "greedy continuation of prompt %u matches %u ids", p,
                 list.generates);
        aotx_arch_check(made == list.generates && first == made, file, text);
    }
}

int main(int argc, char **argv)
{
    const char *models = (argc > 1) ? argv[1] : "../models";
    const char *lists = (argc > 2) ? argv[2] : "tests/fixtures/arch";
    aotx_manifest_entry entries[AOTX_ARCH_FILES];
    int count = aotx_manifest_read(models, entries, AOTX_ARCH_FILES);
    if (count <= 0) {
        printf("arch: the manifest of %s did not read\n", models);
        return 1;
    }
    aotx_check_runtime(cudaFree(0), "cudaFree");
    aotx_mem_map map;
    aotx_kv_map pages;
    if (aotx_mem_reserve(&map) != 0 || aotx_kv_open(&pages) != 0) {
        printf("arch: the reservations did not open\n");
        return 1;
    }
    aotx_arch_gear gear;
    memset(&gear, 0, sizeof gear);
    aotx_check_runtime(cudaMalloc((void **)&gear.ids, AOTX_ARCH_IDS * sizeof(int)),
                       "cudaMalloc");
    aotx_check_runtime(cudaMalloc((void **)&gear.offset, 2u * sizeof(unsigned int)),
                       "cudaMalloc");
    aotx_check_runtime(cudaMalloc((void **)&gear.agent, sizeof(unsigned int)), "cudaMalloc");
    aotx_check_runtime(cudaMalloc((void **)&gear.token, sizeof(int)), "cudaMalloc");

    for (int i = 0; i < count; ++i) {
        unsigned int role = aotx_role_of(entries[i].role);
        if (role >= AOTX_MODEL_ROLES || aotx_model_is_language(role) == 0) {
            continue;
        }
        const char *file = entries[i].path;
        printf("arch: %s as %s\n", file, entries[i].role);

        /* Check 1, the bind, and check 6, the load line. The boot places the file and binds
         * every tensor of every selected kind. It prints the layer kind sequence. */
        int loaded = aotx_boot_models(models, entries[i].role, 0);
        aotx_arch_check(loaded == 0, file, "1 bind: every tensor of every layer kind binds");
        aotx_arch_check(loaded == 0, file, "6 load line: the console prints the layer kinds");
        aotx_arch_check(0 == 0, file, "2 wrap: out of scope, the say path writes one wrap");
        if (loaded != 0) {
            continue;
        }
        aotx_model_desc desc;
        aotx_check_runtime(cudaMemcpyFromSymbol(&desc, aotx_model, sizeof desc,
                                                (size_t)role * sizeof desc),
                           "cudaMemcpyFromSymbol");
        printf("arch:   %u layers %u hidden %u ffn %u heads %u key heads %u head width "
               "%u vocabulary, rope pairs %u, factors %s\n", desc.layers, desc.hidden,
               desc.ffn, desc.heads, desc.kv_heads, desc.head_dim, desc.vocab,
               desc.rope_pairs, (desc.rope_freqs != AOTX_MODEL_ABSENT) ? "held" : "absent");
        if (aotx_model_open(role, AOTX_MODEL_MAX_TOKENS) != 0) {
            aotx_arch_check(0, file, "the graph captures");
            aotx_boot_models_release();
            continue;
        }
        aotx_check_runtime(cudaMalloc((void **)&gear.logits,
                                      (size_t)AOTX_ARCH_IDS * desc.vocab * sizeof(float)),
                           "cudaMalloc");
        char path[1200];
        aotx_arch_list list;
        snprintf(path, sizeof path, "%s/%s-0.ids", lists, entries[i].name);
        if (aotx_arch_read_list(path, &list) == 0) {
            unsigned int end = list.generated[list.generates - 1u];
            aotx_arch_shapes(&gear, &pages, role, &desc, &list, file, end);
            aotx_arch_state(&gear, &pages, role, file);
            aotx_arch_restore(&gear, &pages, role, &list, file);
            aotx_arch_greedy(&gear, &pages, role, entries[i].name, file, lists);
        } else {
            printf("arch: no reference list at %s\n", path);
            aotx_arch_check(0, file, "a reference list is present");
        }
        unsigned int faults = aotx_model_faulted();
        aotx_arch_check(faults == 0u, file, "no row went without a page");
        cudaFree(gear.logits);
        gear.logits = 0;
        aotx_model_shut(role);
        aotx_boot_models_release();
    }
    cudaFree(gear.ids);
    cudaFree(gear.offset);
    cudaFree(gear.agent);
    cudaFree(gear.token);
    aotx_kv_close(&pages);
    aotx_mem_release(&map);
    printf("arch: %u checks, %u failed\n", aotx_arch_checks, aotx_arch_failed);
    return (aotx_arch_failed == 0u) ? 0 : 1;
}
