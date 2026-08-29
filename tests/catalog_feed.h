/* Purpose: Build the import records of a module directory and give them to the device.
 * Owns: The bytes of one module the caller reads from a directory.
 * Threading: One host thread reads the directory and writes the ring.
 * Lifetime: One run of the test program. */
#ifndef AOTX_TESTS_CATALOG_FEED_H
#define AOTX_TESTS_CATALOG_FEED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "catalog/catalog.cuh"
#include "seam_feed.h"

/* Records one import takes: the head and one part for each run of text. */
#define AOTX_TEST_IMPORT_MAX  256u

/* One module the check reads from a directory or builds from text. */
typedef struct aotx_test_module {
    unsigned int  kind;
    char          name[AOTX_CATALOG_NAME_BYTES];
    char         *manifest;
    unsigned int  manifest_len;
    char         *body;
    unsigned int  body_len;
    unsigned char digest[32];
} aotx_test_module;

/* Read a whole file. The caller frees the bytes. */
static char *aotx_test_read_file(const char *path, unsigned int *length)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long bytes = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (bytes < 0) {
        fclose(file);
        return NULL;
    }
    char *text = (char *)calloc((size_t)bytes + 1u, 1u);
    if (text != NULL && bytes > 0 && fread(text, 1u, (size_t)bytes, file) != (size_t)bytes) {
        free(text);
        text = NULL;
    }
    fclose(file);
    if (text == NULL) {
        return NULL;
    }
    *length = (unsigned int)bytes;
    return text;
}

/* Give the value of a key of a manifest text, into out. The return is 1 when the key
 * stands in the text. */
static int aotx_test_manifest_value(const char *text, unsigned int length, const char *key,
                                    char *out, unsigned int bytes)
{
    unsigned int at = 0u;
    unsigned int span = (unsigned int)strlen(key);
    while (at < length) {
        unsigned int line = at;
        while (at < length && text[at] != '\n') {
            at += 1u;
        }
        if (at > line + span + 1u && strncmp(text + line, key, span) == 0
            && text[line + span] == ':') {
            unsigned int from = line + span + 1u;
            while (from < at && (text[from] == ' ' || text[from] == '\t')) {
                from += 1u;
            }
            unsigned int stop = at;
            while (stop > from && (text[stop - 1u] == ' ' || text[stop - 1u] == '\r')) {
                stop -= 1u;
            }
            unsigned int held = stop - from;
            if (held >= bytes) {
                held = bytes - 1u;
            }
            memcpy(out, text + from, held);
            out[held] = '\0';
            return 1;
        }
        at += 1u;
    }
    return 0;
}

/* Build one module from a manifest text and a body text. */
static void aotx_test_module_text(aotx_test_module *module, unsigned int kind,
                                  const char *name, const char *manifest, const char *body)
{
    memset(module, 0, sizeof *module);
    module->kind = kind;
    snprintf(module->name, sizeof module->name, "%s", name);
    module->manifest_len = (unsigned int)strlen(manifest);
    module->manifest = (char *)calloc(module->manifest_len + 1u, 1u);
    memcpy(module->manifest, manifest, module->manifest_len);
    if (body != NULL) {
        module->body_len = (unsigned int)strlen(body);
        module->body = (char *)calloc(module->body_len + 1u, 1u);
        memcpy(module->body, body, module->body_len);
    }
}

/* Read one module directory: the manifest and the file its body key names. A directory
 * with a skill file and no manifest gives the head of that file and the text after it. */
static int aotx_test_module_dir(aotx_test_module *module, const char *dir)
{
    char path[1024];
    char value[256];
    memset(module, 0, sizeof *module);
    const char *tail = strrchr(dir, '/');
    snprintf(module->name, sizeof module->name, "%s", (tail != NULL) ? tail + 1 : dir);

    snprintf(path, sizeof path, "%s/module.manifest", dir);
    module->manifest = aotx_test_read_file(path, &module->manifest_len);
    if (module->manifest != NULL) {
        if (aotx_test_manifest_value(module->manifest, module->manifest_len, "kind",
                                     value, sizeof value) == 0) {
            return 1;
        }
        module->kind = (strcmp(value, "skill") == 0) ? AOTX_MODULE_SKILL
                     : ((strcmp(value, "role") == 0) ? AOTX_MODULE_ROLE
                        : AOTX_MODULE_TOOL);
        if (aotx_test_manifest_value(module->manifest, module->manifest_len, "body",
                                     value, sizeof value) != 0 && value[0] != '\0') {
            snprintf(path, sizeof path, "%s/%s", dir, value);
            module->body = aotx_test_read_file(path, &module->body_len);
        }
        return 0;
    }
    /* A skill directory that holds the skill file alone. The feeder reads that file whole
     * and sends it as the second file with no manifest beside it. The head then names one
     * file whose byte count stands in the second place. The check builds that shape and
     * no other, because a fixture that splits the file is not the writer. */
    snprintf(path, sizeof path, "%s/SKILL.md", dir);
    unsigned int whole = 0u;
    char *text = aotx_test_read_file(path, &whole);
    if (text == NULL) {
        return 1;
    }
    module->kind = AOTX_MODULE_SKILL;
    module->manifest_len = 0u;
    module->manifest = (char *)calloc(1u, 1u);
    module->body_len = whole;
    module->body = text;
    return 0;
}

static void aotx_test_module_free(aotx_test_module *module)
{
    free(module->manifest);
    free(module->body);
    module->manifest = NULL;
    module->body = NULL;
}

/* Build the head and the parts of one import into a run of record bodies. The return is
 * the record count, and sizes takes the body length of each record. */
static unsigned int aotx_test_import_build(const aotx_test_module *module,
                                           unsigned int import, unsigned char *out,
                                           unsigned int *sizes)
{
    aotx_import_head head;
    memset(&head, 0, sizeof head);
    head.import = import;
    head.part = 0u;
    head.kind = module->kind;
    /* The head counts the files that carry bytes, as the feeder counts them. */
    head.file_bytes[0] = module->manifest_len;
    head.file_bytes[1] = module->body_len;
    head.files = ((module->manifest_len != 0u) ? 1u : 0u)
               + ((module->body_len != 0u) ? 1u : 0u);
    memcpy(head.digest, module->digest, sizeof head.digest);
    snprintf(head.name, sizeof head.name, "%s", module->name);
    snprintf(head.path, sizeof head.path, "%s", module->name);
    memset(out, 0, AOTX_BODY_BYTES);
    memcpy(out, &head, sizeof head);
    sizes[0] = (unsigned int)sizeof head;

    unsigned int made = 1u;
    unsigned int number = 1u;
    for (unsigned int file = 0u; file < (unsigned int)AOTX_IMPORT_FILES; ++file) {
        const char *text = (file == 0u) ? module->manifest : module->body;
        unsigned int length = (file == 0u) ? module->manifest_len : module->body_len;
        for (unsigned int at = 0u; at < length; at += AOTX_IMPORT_TEXT_BYTES) {
            unsigned int span = length - at;
            if (span > (unsigned int)AOTX_IMPORT_TEXT_BYTES) {
                span = (unsigned int)AOTX_IMPORT_TEXT_BYTES;
            }
            aotx_import_part part;
            memset(&part, 0, sizeof part);
            part.import = import;
            part.part = number++;
            part.file = file;
            part.offset = at;
            part.length = span;
            memcpy(part.text, text + at, span);
            memcpy(out + (size_t)made * AOTX_BODY_BYTES, &part, sizeof part);
            sizes[made] = (unsigned int)sizeof part;
            made += 1u;
        }
    }
    return made;
}

/* Put the records of one import in the inbound ring, as the feeder does. The head and the
 * parts go in as one run of records of the same type. */
static void aotx_test_import_feed(aotx_seam_rings *rings, const aotx_test_module *module,
                                  unsigned int import, unsigned long long boot_id)
{
    unsigned char *bodies =
        (unsigned char *)calloc(AOTX_TEST_IMPORT_MAX, AOTX_BODY_BYTES);
    unsigned int *sizes = (unsigned int *)calloc(AOTX_TEST_IMPORT_MAX, sizeof(unsigned int));
    unsigned int made = aotx_test_import_build(module, import, bodies, sizes);
    for (unsigned int i = 0u; i < made; ++i) {
        aotx_test_feed_records(rings, AOTX_REC_IMPORT, AOTX_CLASS_A, AOTX_WRITER_FEEDER,
                               0u, bodies + (size_t)i * AOTX_BODY_BYTES, sizes[i], 1u,
                               boot_id);
    }
    free(bodies);
    free(sizes);
}

/* Put one remove record in the inbound ring. */
static void aotx_test_remove_feed(aotx_seam_rings *rings, const char *name,
                                  unsigned long long boot_id)
{
    aotx_remove_body body;
    memset(&body, 0, sizeof body);
    snprintf(body.name, sizeof body.name, "%s", name);
    aotx_test_feed_records(rings, AOTX_REC_REMOVE, AOTX_CLASS_A, AOTX_WRITER_FEEDER, 0u,
                           &body, (unsigned int)sizeof body, 1u, boot_id);
}

/* Apply the records of one import on the device with no ring. The setup of a check that
 * has no pump takes this path; every check of the import itself takes the ring. */
__global__ void aotx_test_catalog_apply(const unsigned char *bodies,
                                        const unsigned int *sizes, unsigned int count)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_catalog_apply(AOTX_REC_IMPORT, bodies + (size_t)i * AOTX_BODY_BYTES,
                           sizes[i], 0ull);
    }
}

static void aotx_test_import_direct(const aotx_test_module *module, unsigned int import)
{
    unsigned char *bodies =
        (unsigned char *)calloc(AOTX_TEST_IMPORT_MAX, AOTX_BODY_BYTES);
    unsigned int *sizes = (unsigned int *)calloc(AOTX_TEST_IMPORT_MAX, sizeof(unsigned int));
    unsigned int made = aotx_test_import_build(module, import, bodies, sizes);
    unsigned char *on = NULL;
    unsigned int *counts = NULL;
    cudaMalloc((void **)&on, (size_t)made * AOTX_BODY_BYTES);
    cudaMalloc((void **)&counts, (size_t)made * sizeof(unsigned int));
    cudaMemcpy(on, bodies, (size_t)made * AOTX_BODY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(counts, sizes, (size_t)made * sizeof(unsigned int), cudaMemcpyHostToDevice);
    aotx_test_catalog_apply<<<1, 1>>>(on, counts, made);
    cudaDeviceSynchronize();
    cudaFree(on);
    cudaFree(counts);
    free(bodies);
    free(sizes);
}

/* Read the catalog back from the device. The caller frees the block. */
static aotx_catalog_state *aotx_test_catalog_read(void)
{
    aotx_catalog_state *state = (aotx_catalog_state *)calloc(1, sizeof *state);
    cudaMemcpyFromSymbol(state, aotx_catalog, sizeof *state);
    return state;
}

/* Give the entry of an installed module of a name, or AOTX_MODULE_SLOTS. */
static unsigned int aotx_test_catalog_entry(const char *name, unsigned int kind)
{
    aotx_catalog_state *state = aotx_test_catalog_read();
    unsigned int found = AOTX_MODULE_SLOTS;
    unsigned int length = (unsigned int)strlen(name);
    for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i) {
        if (state->entry[i].state == AOTX_CATALOG_INSTALLED
            && state->entry[i].kind == kind && state->entry[i].name_len == length
            && strncmp(state->entry[i].name, name, length) == 0) {
            found = i;
            break;
        }
    }
    free(state);
    return found;
}

/* Import every module directory below a directory, in name order, with no ring. The
 * return is the count that landed. */
static unsigned int aotx_test_modules_direct(const char *root, const char *const *names,
                                             unsigned int count)
{
    char dir[1024];
    unsigned int made = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_test_module module;
        snprintf(dir, sizeof dir, "%s/%s", root, names[i]);
        if (aotx_test_module_dir(&module, dir) != 0) {
            continue;
        }
        aotx_test_import_direct(&module, i + 1u);
        aotx_test_module_free(&module);
        made += 1u;
    }
    return made;
}

/* Put the built-in tools and the three role directories of one directory in the catalog.
 * The return is zero when the four tools and the three roles stand. */
static int aotx_test_roles_of(const char *dir)
{
    static const char *const names[3] = { "conductor", "verifier", "worker" };
    if (aotx_catalog_open() != 0) {
        printf("the built-in tools did not go in the catalog\n");
        return 1;
    }
    unsigned int landed = aotx_test_modules_direct(dir, names, 3u);
    if (landed != 3u
        || aotx_test_catalog_entry("conductor", AOTX_MODULE_ROLE) >= AOTX_MODULE_SLOTS
        || aotx_test_catalog_entry("worker", AOTX_MODULE_ROLE) >= AOTX_MODULE_SLOTS
        || aotx_test_catalog_entry("verifier", AOTX_MODULE_ROLE) >= AOTX_MODULE_SLOTS) {
        printf("%u of 3 role directories of %s went in the catalog\n", landed, dir);
        return 1;
    }
    return 0;
}

/* Put the built-in tools and the three role directories of the repository in the catalog.
 * A check that drives agents, panels or commands calls this after it binds the ring. The
 * call comes before the first case, because a role is a module and not a constant. */
static int aotx_test_catalog_setup(void)
{
    return aotx_test_roles_of(AOTX_MODULES_DIR);
}

/* Report whether an entry is the installed role of a name. */
static int aotx_test_catalog_is_role(unsigned int entry, const char *name)
{
    return aotx_test_catalog_entry(name, AOTX_MODULE_ROLE) == entry
        && entry < AOTX_MODULE_SLOTS;
}

/* Give the entry of the role of a place in the catalog order, or AOTX_MODULE_SLOTS. The
 * fixture of the panel check reads the roles this way, because a role is a module. */
static unsigned int aotx_test_catalog_role_at(unsigned int which)
{
    aotx_catalog_state *state = aotx_test_catalog_read();
    unsigned int seen = 0u;
    unsigned int found = AOTX_MODULE_SLOTS;
    for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i) {
        if (state->entry[i].state == AOTX_CATALOG_INSTALLED
            && state->entry[i].kind == AOTX_MODULE_ROLE) {
            if (seen == which) {
                found = i;
                break;
            }
            seen += 1u;
        }
    }
    free(state);
    return found;
}

#endif
