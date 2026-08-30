/* Purpose: Fill the descriptor of each model role from the file metadata and the tensors.
 * Owns: Nothing that lasts; the descriptor lives in device memory.
 * Launch shape: Host glue only; the bind kernel finds the tensors.
 * Lifetime: One model load.  */
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#include "boot/check.h"
#include "mem/mem.cuh"
#include "embed/embed.cuh"
#include "model/forward.cuh"
#include "model/names.h"
#include "model/roles.h"
#include "rerank/rerank.cuh"

extern "C" {
#include "disk/modelfile/manifest.h"
#include "disk/modelfile/modelfile.h"
}

#define AOTX_DESC_MAX_FILES  8
#define AOTX_DESC_KEY        96

/* The two name lists of names.h, which the bind kernel also holds. The host reads them to
 * print the name of a tensor the model does not have; it forms no name for a lookup. */
static const char aotx_desc_whole_name[AOTX_DESC_WHOLE][AOTX_DESC_NAME] =
    AOTX_DESC_WHOLE_LIST;
static const char aotx_desc_layer_name[AOTX_DESC_PER_LAYER][AOTX_DESC_NAME] =
    AOTX_DESC_LAYER_LIST;

/* The name of one place of the name list. */
static void aotx_desc_name(char *out, unsigned int size, unsigned int at)
{
    if (at < AOTX_DESC_WHOLE) {
        snprintf(out, size, "%s", aotx_desc_whole_name[at]);
        return;
    }
    unsigned int layer = (at - AOTX_DESC_WHOLE) / AOTX_DESC_PER_LAYER;
    unsigned int which = (at - AOTX_DESC_WHOLE) % AOTX_DESC_PER_LAYER;
    snprintf(out, size, "blk.%u.%s.weight", layer, aotx_desc_layer_name[which]);
}

/* One metadata value of the file, under the name of the architecture. */
static int aotx_desc_u32(const aotx_modelfile *file, const char *arch, const char *tail,
                         unsigned int *value, int needed)
{
    char key[AOTX_DESC_KEY];
    uint32_t got = 0u;
    snprintf(key, sizeof key, "%s.%s", arch, tail);
    if (aotx_modelfile_u32(file, key, &got) != 0) {
        if (needed != 0) {
            fprintf(stderr, "the model file does not hold %s\n", key);
            return 1;
        }
        return 0;
    }
    *value = (unsigned int)got;
    return 0;
}

static int aotx_desc_f32(const aotx_modelfile *file, const char *arch, const char *tail,
                         float *value)
{
    char key[AOTX_DESC_KEY];
    snprintf(key, sizeof key, "%s.%s", arch, tail);
    if (aotx_modelfile_f32(file, key, value) != 0) {
        fprintf(stderr, "the model file does not hold %s\n", key);
        return 1;
    }
    return 0;
}

/* Read the shape of one model into the descriptor. */
static int aotx_desc_shape(const aotx_modelfile *file, aotx_model_desc *desc)
{
    const char *arch = NULL;
    size_t length = 0u;
    char name[AOTX_DESC_KEY];
    if (aotx_modelfile_string(file, "general.architecture", &arch, &length) != 0
        || length == 0u || length >= sizeof name) {
        fprintf(stderr, "the model file does not name an architecture\n");
        return 1;
    }
    memcpy(name, arch, length);
    name[length] = '\0';
    if (aotx_desc_u32(file, name, "block_count", &desc->layers, 1) != 0
        || aotx_desc_u32(file, name, "embedding_length", &desc->hidden, 1) != 0
        || aotx_desc_u32(file, name, "feed_forward_length", &desc->ffn, 1) != 0
        || aotx_desc_u32(file, name, "attention.head_count", &desc->heads, 1) != 0
        || aotx_desc_u32(file, name, "attention.head_count_kv", &desc->kv_heads, 1) != 0
        || aotx_desc_u32(file, name, "attention.key_length", &desc->head_dim, 1) != 0
        || aotx_desc_u32(file, name, "context_length", &desc->context, 1) != 0
        || aotx_desc_u32(file, name, "pooling_type", &desc->pooling, 0) != 0
        || aotx_desc_f32(file, name, "rope.freq_base", &desc->rope_theta) != 0
        || aotx_desc_f32(file, name, "attention.layer_norm_rms_epsilon", &desc->rms_eps) != 0) {
        return 1;
    }
    if (desc->layers > AOTX_MODEL_MAX_LAYERS || desc->heads == 0u || desc->kv_heads == 0u
        || (desc->heads % desc->kv_heads) != 0u) {
        fprintf(stderr, "the layer count or the head count is outside the bounds\n");
        return 1;
    }

    /* A head must fit one warp and hold a whole number of lanes. The attention kernel
     * gives one warp to a head and the same count of elements to each lane. */
    if (desc->head_dim == 0u || desc->head_dim > AOTX_MODEL_HEAD_MAX
        || (desc->head_dim % 32u) != 0u) {
        fprintf(stderr, "the head width %u is not a multiple of 32 up to %u\n",
                desc->head_dim, AOTX_MODEL_HEAD_MAX);
        return 1;
    }

    /* A pooling the heads of this system do not give is refused. A file which asks for
     * another one must not run as if it asked for none. */
    unsigned int asked = desc->pooling;
    unsigned int gives = (desc->role == AOTX_MODEL_EMBEDDING) ? AOTX_EMBED_POOL_LAST
                       : ((desc->role == AOTX_MODEL_RERANKER) ? AOTX_RERANK_POOL_RANK : 0u);
    if (asked != gives) {
        fprintf(stderr, "the model of role %u asks for the pooling %u and this system "
                "gives %u\n", desc->role, asked, gives);
        return 1;
    }
    return 0;
}

/* Let the device find every tensor of one model. The kernel forms each name, mixes it as
 * the tensor table does, and writes the offsets. */
static int aotx_desc_bind(unsigned int role, unsigned int model, unsigned int count)
{
    unsigned int *missing = 0;
    unsigned int report[2] = { 0u, ~0u };
    aotx_check_runtime(cudaMalloc((void **)&missing, sizeof report), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(missing, report, sizeof report, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    unsigned int blocks = (count + 127u) / 128u;
    aotx_model_bind<<<blocks, 128>>>(role, model, count, missing);
    aotx_check_runtime(cudaMemcpy(report, missing, sizeof report, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    cudaFree(missing);
    if (report[0] != 0u) {
        char name[AOTX_DESC_KEY];
        aotx_desc_name(name, sizeof name, report[1]);
        fprintf(stderr, "the model of role %u has no tensor %s, and %u more are absent\n",
                role, name, report[0] - 1u);
        return 1;
    }
    return 0;
}

/* Open one entry and bind its tensor rows to the descriptor role the caller names. */
static int aotx_desc_one(const char *dir, const aotx_manifest_entry *entry,
                         unsigned int model, unsigned int role)
{
    char path[AOTX_MANIFEST_PATH];
    aotx_modelfile *file = NULL;
    if (aotx_manifest_path(path, sizeof path, dir, entry->path) != 0
        || aotx_modelfile_open(path, &file) != 0) {
        fprintf(stderr, "the file %s did not open\n", entry->path);
        return 1;
    }

    aotx_model_desc desc;
    memset(&desc, 0, sizeof desc);
    desc.role = role;
    desc.tied_output = 1u;
    desc.token_embd = AOTX_MODEL_ABSENT;
    desc.output_norm = AOTX_MODEL_ABSENT;
    desc.output = AOTX_MODEL_ABSENT;
    desc.cls_output = AOTX_MODEL_ABSENT;
    int bad = aotx_desc_shape(file, &desc);
    aotx_modelfile_close(file);
    if (bad != 0) {
        return 1;
    }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &desc, sizeof desc,
                                          (size_t)role * sizeof desc),
                       "cudaMemcpyToSymbol");
    return aotx_desc_bind(role, model, AOTX_DESC_NAMES(desc.layers));
}

int aotx_model_describe_one(const char *dir, const char *name, unsigned int target)
{
    if (target >= AOTX_MODEL_ROLES) {
        fprintf(stderr, "the target model role %u is outside the table\n", target);
        return 1;
    }
    aotx_manifest_entry entries[AOTX_DESC_MAX_FILES];
    int count = aotx_manifest_read(dir, entries, AOTX_DESC_MAX_FILES);
    if (count <= 0) {
        fprintf(stderr, "the model file list in %s did not read\n", dir);
        return 1;
    }
    for (int i = 0; i < count; ++i) {
        if (strcmp(entries[i].name, name) == 0) {
            return aotx_desc_one(dir, &entries[i], (unsigned int)i, target);
        }
    }
    fprintf(stderr, "the model file list has no entry for %s\n", name);
    return 1;
}

int aotx_model_describe(const char *dir, const char *roles)
{
    char unknown[64];
    if (aotx_role_unknown(roles, unknown, sizeof unknown) != 0) {
        fprintf(stderr, "the role %s is not a role of this system\n", unknown);
        return 1;
    }
    aotx_manifest_entry entries[AOTX_DESC_MAX_FILES];
    int count = aotx_manifest_read(dir, entries, AOTX_DESC_MAX_FILES);
    if (count <= 0) {
        fprintf(stderr, "the model file list in %s did not read\n", dir);
        return 1;
    }
    unsigned int want[AOTX_MODEL_ROLES];
    unsigned int wanted = aotx_role_list(roles, want);
    unsigned int found = 0u;
    unsigned int done = 0u;
    for (int i = 0; i < count; ++i) {
        unsigned int role = aotx_role_of(entries[i].name);
        if (role >= AOTX_MODEL_ROLES || aotx_role_wanted(roles, entries[i].name) == 0) {
            continue;
        }
        if (aotx_desc_one(dir, &entries[i], (unsigned int)i, role) != 0) {
            return 1;
        }
        found |= 1u << role;
        done += 1u;
    }
    if (done != wanted) {
        for (unsigned int k = 0u; k < wanted; ++k) {
            if ((found & (1u << want[k])) == 0u) {
                fprintf(stderr, "the model file list has no entry for the role %s\n",
                        aotx_role_name[want[k]]);
            }
        }
        return 1;
    }
    return 0;
}
