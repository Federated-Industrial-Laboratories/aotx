/* Purpose: Check that a language model of a store runs. The checks cover the bind,
 *   wrap, shapes, state, cache rebuild, and greedy continuation.
 * Owns: The buffers of one run.
 * Launch shape: Host glue calls the forward pass; one thread for each slot in the release.
 * Lifetime: One run of the check program.
 *
 * The check takes a store directory and a directory of reference lists. It loads every
 * language file of the store, one at a time, and prints the checks of one architecture for
 * each file. A reference list holds the prefill ids the reference consumed and the ids it
 * generated at temperature zero. The check feeds the same prefill and compares the ids one
 * for one. A cache rebuild does not check journal restore after a process stop. */
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "kvcache/kvcache.cuh"
#include "mem/mem.cuh"
#include "model/forward.cuh"
#include "model/roles.h"
#include "model/wrap.cuh"

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
    while (*at != '\0') {
        char *end = NULL;
        errno = 0;
        unsigned long value = strtoul(at, &end, 10);
        if (end == at) {
            at += 1;
            continue;
        }
        if (count == max || errno == ERANGE || value > INT_MAX) return max + 1u;
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
        return errno == ENOENT ? 2 : 1;
    }
    memset(list, 0, sizeof *list);
    while (fgets(line, sizeof line, in) != NULL) {
        if (strncmp(line, "prefill ", 8u) == 0) {
            list->prefills = aotx_arch_ids(line + 8, list->prefill, AOTX_ARCH_IDS);
        } else if (strncmp(line, "generated ", 10u) == 0) {
            list->generates = aotx_arch_ids(line + 10, list->generated, AOTX_ARCH_GENERATE);
        }
    }
    int bad = ferror(in);
    bad |= fclose(in) != 0;
    return bad || list->prefills == 0u || list->prefills > AOTX_ARCH_IDS
        || list->generates == 0u || list->generates > AOTX_ARCH_GENERATE;
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
    aotx_kv_release(agent);
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
    if (count == 0u || count > AOTX_ARCH_IDS || count > AOTX_MODEL_MAX_TOKENS) return -1;
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
        return aotx_model_faulted() == 0u ? 0 : -1;
    }
    if (aotx_model_sample(role, gear->ids, gear->offset, 1u, gear->agent, &how, gear->token,
                          0, &seed) != 0) {
        return -1;
    }
    int token = 0;
    aotx_check_runtime(cudaMemcpy(&token, gear->token, sizeof token, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    return aotx_model_faulted() == 0u ? token : -1;
}

/* The greedy continuation of one prefill: the prompt in one pass, then one token a pass. */
static unsigned int aotx_arch_generate(aotx_arch_gear *gear, aotx_kv_map *map,
                                       unsigned int role, unsigned int agent,
                                       const aotx_arch_list *list, unsigned int *out,
                                       unsigned int want)
{
    int *run = (int *)malloc((size_t)AOTX_ARCH_IDS * sizeof(int));
    if (run == NULL) return 0u;
    for (unsigned int i = 0u; i < list->prefills; ++i) {
        run[i] = (int)list->prefill[i];
    }
    int token = aotx_arch_step(gear, map, role, agent, run, list->prefills, 0);
    unsigned int made = 0u;
    while (token >= 0 && made < want) {
        out[made++] = (unsigned int)token;
        if (made == want) break;
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
                             const char *file, const aotx_wrap *wrap)
{
    int *run = (int *)malloc((size_t)AOTX_ARCH_IDS * sizeof(int));
    if (run == NULL || desc->vocab == 0u) {
        aotx_arch_check(0, file, "3 shapes: prefill buffers have space");
        free(run);
        return;
    }
    for (unsigned int i = 0u; i < list->prefills; ++i) {
        run[i] = (int)list->prefill[i];
    }
    aotx_model_forget();
    int state = aotx_arch_step(gear, map, role, 0u, run, list->prefills, gear->logits);
    float *row = (float *)malloc((size_t)desc->vocab * sizeof(float));
    if (state != 0 || row == NULL) {
        aotx_arch_check(0, file, "3 shapes: prefill returns logits");
    } else {
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
    unsigned int end = 0u;
    for (unsigned int i = 0u; i < wrap->end_count; ++i) {
        end |= best == wrap->end_ids[i];
    }
    printf("arch:   shapes: vocabulary %u, finite %u, largest %u at %.3f, end token %s\n",
           desc->vocab, finite, best, (double)row[best], end ? "yes" : "no");
    aotx_arch_check(finite == desc->vocab && !end, file,
                    "3 shapes: finite logits of vocabulary width, largest not an end token");
    }
    aotx_arch_release<<<1, 1>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_kv_serve(map, 0);
    free(row);
    free(run);
}

/* Check 4, the state: a sequence at the context bound opens, and the pool returns to its
 * prior occupancy after the release. The bound is the profile's, not the file's. */
static void aotx_arch_state(aotx_arch_gear *gear, aotx_kv_map *map, unsigned int role,
                            const char *file, unsigned int vocab)
{
    unsigned int before = aotx_arch_mapped();
    int *run = (int *)malloc((size_t)AOTX_MODEL_MAX_TOKENS * sizeof(int));
    if (run == NULL || vocab == 0u) {
        aotx_arch_check(0, file, "4 state: context buffers have space");
        free(run);
        return;
    }
    aotx_model_forget();
    int bad = 0;
    for (unsigned int at = 0u; at < AOTX_SEQ_MAX_TOKENS && bad == 0;
         at += AOTX_MODEL_MAX_TOKENS) {
        unsigned int count = AOTX_SEQ_MAX_TOKENS - at;
        if (count > AOTX_MODEL_MAX_TOKENS) count = AOTX_MODEL_MAX_TOKENS;
        for (unsigned int i = 0u; i < count; ++i) {
            run[i] = (int)((1000u + (at + i) * 7u) % vocab);
        }
        bad = aotx_arch_step(gear, map, role, 1u, run, count, 0) < 0;
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

/* Compare 24 greedy tokens with the last eight tokens from an empty cache.
 * The empty cache takes the prompt and the first 16 tokens. No process stops here. */
static void aotx_arch_cache_rebuild(aotx_arch_gear *gear, aotx_kv_map *map,
                                    unsigned int role, const aotx_arch_list *list,
                                    const char *file)
{
    unsigned int whole = list->prefills + 16u;
    int run[AOTX_ARCH_IDS + 24u];
    unsigned int continuous[24], rebuilt[8];
    unsigned int made = 0u, again = 0u;
    aotx_model_forget();
    for (unsigned int i = 0u; i < list->prefills; ++i) run[i] = (int)list->prefill[i];
    int token = aotx_arch_step(gear, map, role, 2u, run, list->prefills, 0);
    while (token >= 0 && made < 24u) {
        continuous[made] = (unsigned int)token;
        run[list->prefills + made++] = token;
        if (made < 24u) token = aotx_arch_step(gear, map, role, 2u, &token, 1u, 0);
    }
    aotx_arch_release<<<1, 1>>>(2u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_kv_serve(map, 0);

    if (made == 24u) {
        aotx_model_forget();
        for (unsigned int at = 0u; at < whole;) {
            unsigned int count = whole - at;
            if (count > AOTX_MODEL_MAX_TOKENS) count = AOTX_MODEL_MAX_TOKENS;
            token = aotx_arch_step(gear, map, role, 3u, run + at, count, 0);
            if (token < 0) break;
            at += count;
        }
        while (token >= 0 && again < 8u) {
            rebuilt[again++] = (unsigned int)token;
            if (again < 8u) token = aotx_arch_step(gear, map, role, 3u, &token, 1u, 0);
        }
        aotx_arch_release<<<1, 1>>>(3u);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_kv_serve(map, 0);
    }
    aotx_arch_print("cache rebuild, continuous:", continuous, made);
    aotx_arch_print("cache rebuild, last eight:", rebuilt, again);
    aotx_arch_check(made == 24u && again == 8u
                    && memcmp(continuous + 16u, rebuilt, sizeof rebuilt) == 0, file,
                    "cache rebuild: eight tokens match the continuous run");
}

/* The greedy continuation against every reference list of the file. */
static void aotx_arch_greedy(aotx_arch_gear *gear, aotx_kv_map *map, unsigned int role,
                             const char *name, const char *file, const char *lists)
{
    for (unsigned int p = 0u; p < AOTX_ARCH_PROMPTS; ++p) {
        char path[1200];
        aotx_arch_list list;
        snprintf(path, sizeof path, "%s/%s-%u.ids", lists, name, p);
        int read = aotx_arch_read_list(path, &list);
        if (read != 0) {
            if (read != 2 || p == 0u) aotx_arch_check(0, file, "the reference list reads");
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
        for (unsigned int i = 0u; i < made; ++i) {
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

#include "arch_batch.h"
#include "arch_process.h"

int main(int argc, char **argv)
{
    const char *models = (argc > 1) ? argv[1] : "../models";
    const char *lists = (argc > 2) ? argv[2] : "tests/fixtures/arch";
    aotx_manifest_entry entries[AOTX_ARCH_FILES];
    char load_lines[AOTX_ARCH_FILES][1024] = {};
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
    aotx_check_runtime(cudaMalloc((void **)&gear.offset, (AOTX_SLOTS + 1u) * sizeof(unsigned int)),
                       "cudaMalloc");
    aotx_check_runtime(cudaMalloc((void **)&gear.agent, AOTX_SLOTS * sizeof(unsigned int)), "cudaMalloc");
    aotx_check_runtime(cudaMalloc((void **)&gear.token, AOTX_SLOTS * sizeof(int)), "cudaMalloc");

    unsigned int languages = 0u;
    for (int i = 0; i < count; ++i) {
        unsigned int role = aotx_role_of(entries[i].role);
        if (role >= AOTX_MODEL_ROLES || aotx_model_is_language(role) == 0) {
            continue;
        }
        const char *file = entries[i].path;
        languages += 1u;
        unsigned int matches = 0u;
        for (int j = 0; j < count; ++j) matches += strcmp(entries[j].role, entries[i].role) == 0;
        if (matches != 1u) {
            aotx_arch_check(0, file, "one file has the requested language role");
            continue;
        }
        printf("arch: %s as %s\n", file, entries[i].role);

        /* Check 1: the load binds every required tensor of each selected layer kind. */
        int loaded = aotx_boot_models(models, entries[i].role, 0);
        aotx_arch_check(loaded == 0, file, "1 bind: every tensor of every layer kind binds");
        if (loaded != 0) {
            continue;
        }
        if (argc < 5) {
            printf("arch: %s not checked: 6 load line requires a process script\n", file);
            printf("arch: %s not checked: 5 restore requires a process script\n", file);
        }
        char model_path[AOTX_MANIFEST_PATH];
        aotx_modelfile *model = NULL;
        aotx_wrap wrap;
        int wrap_bad = aotx_manifest_path(model_path, sizeof model_path, models, file) != 0;
        if (!wrap_bad) wrap_bad = aotx_modelfile_open(model_path, &model) != 0;
        if (!wrap_bad) wrap_bad = aotx_wrap_read(model, &entries[i], &wrap) != 0;
        if (model != NULL) aotx_modelfile_close(model);
        if (!wrap_bad) wrap_bad = aotx_model_wrap_check(role, &wrap, file);
        aotx_arch_check(wrap_bad == 0, file, "2 wrap: spans, end tokens, and prefill pass");
        if (wrap_bad) {
            aotx_boot_models_release();
            continue;
        }
        aotx_model_desc desc;
        aotx_check_runtime(cudaMemcpyFromSymbol(&desc, aotx_model, sizeof desc,
                                                (size_t)role * sizeof desc),
                           "cudaMemcpyFromSymbol");
        aotx_arch_load_line(&desc, load_lines[i], sizeof load_lines[i]);
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
            aotx_arch_shapes(&gear, &pages, role, &desc, &list, file, &wrap);
            aotx_arch_state(&gear, &pages, role, file, desc.vocab);
            aotx_arch_cache_rebuild(&gear, &pages, role, &list, file);
            aotx_arch_greedy(&gear, &pages, role, entries[i].name, file, lists);
            aotx_arch_batch(&gear, &pages, role, &list, file, desc.vocab);
        } else {
            printf("arch: no reference list at %s\n", path);
            aotx_arch_check(0, file, "a reference list is present");
        }
        cudaFree(gear.logits);
        gear.logits = 0;
        aotx_model_shut(role);
        aotx_boot_models_release();
    }
    if (languages == 0u) aotx_arch_check(0, models, "the manifest has a language file");
    cudaFree(gear.ids);
    cudaFree(gear.offset);
    cudaFree(gear.agent);
    cudaFree(gear.token);
    aotx_kv_close(&pages);
    aotx_mem_release(&map);
    if (argc >= 5) {
        for (int i = 0; i < count; ++i) {
            if (load_lines[i][0] != '\0')
                aotx_arch_process(argv[0], argv[3], argv[4], models,
                                  &entries[i], load_lines[i]);
        }
    }
    printf("arch: %u checks, %u failed\n", aotx_arch_checks, aotx_arch_failed);
    return (aotx_arch_failed == 0u) ? 0 : 1;
}
