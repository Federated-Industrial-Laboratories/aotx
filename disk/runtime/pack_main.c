/* Purpose: Create a complete runtime file from selected prepared components.
 * Owns: The creation batch and its source descriptors.
 * Threading: One process; files stream through bounded disk buffers.
 * Lifetime: One exclusive output creation; source files stay unchanged. */
#include "disk/runtime/pack.h"
#include "disk/runtime/activate.h"
#include "disk/runtime/appraisal.h"
#include "disk/policy/file.h"
#include <stdio.h>
#include "cognitive/format.h"
#include "profile/profile.cuh"
#include "cuda/seam/wire.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void section(aotx_runtime_pack *p, uint32_t type, uint16_t schema,
                     const void *data, uint64_t bytes) {
    aotx_ccir_input *in = p->inputs + p->count++;
    memset(in, 0, sizeof(*in)); in->fd = -1;
    in->section.type = type; in->section.schema = schema;
    in->section.flags = AOTX_CCIR_REQUIRED; in->section.id[0] = (unsigned char)type;
    in->section.alignment = 128; in->section.bytes = bytes; in->data = data;
}
static int has_asset(const aotx_runtime_pack *p, const char *name) {
    for (uint32_t i = 0; i < p->index.count; ++i)
        if (!strcmp((const char *)p->index.rows[i] + 64, name)) return 1;
    return 0;
}
static int policy_asset(aotx_runtime_pack *p, const char *path, aotx_policy_file *file) {
    if (p->count >= AOTX_CCIR_SECTIONS || p->index.count >= AOTX_CCIR_SECTIONS)
        return AOTX_CCIR_LIMIT;
    if (has_asset(p, "policy.bin")) return AOTX_CCIR_INVALID;
    int rc = aotx_policy_file_read(path, NULL, 0, file);
    if (rc) return rc == AOTX_POLICY_FILE_DIGEST ? AOTX_CCIR_INVALID : rc;
    if (file->config.mode == AOTX_POLICY_NATIVE &&
        file->config.architecture != AOTX_RUNTIME_ARCH) return AOTX_CCIR_UNSUPPORTED;
    aotx_ccir_input *in = &p->inputs[p->count++];
    memset(in, 0, sizeof(*in)); in->fd = -1;
    in->section.type = AOTX_CCIR_ASSET; in->section.schema = 1;
    in->section.flags = AOTX_CCIR_REQUIRED; in->section.alignment = 4096;
    in->section.id[0] = AOTX_CCIR_ASSET;
    aotx_ccir_put(in->section.id + 8, p->index.count + 1, 8);
    in->section.bytes = file->buffer_bytes;
    memcpy(in->section.digest, file->digest, 32);
    in->source = AOTX_CCIR_MEMORY; in->data = file->buffer;
    unsigned char *row = p->index.rows[p->index.count++];
    memset(row, 0, AOTX_RUNTIME_ROW);
    memcpy(row, in->section.id, 16); aotx_ccir_put(row + 16, 3, 4);
    aotx_ccir_put(row + 24, in->section.bytes, 8);
    memcpy(row + 32, in->section.digest, 32); strcpy((char *)row + 64, "policy.bin");
    return 0;
}
static int create(aotx_runtime_pack *p, const char *output, const char *roles, int shared) {
    unsigned char *h = p->index.header;
    memcpy(h, "AOTXRT01", 8); aotx_ccir_put(h + 8, 1, 4);
    aotx_ccir_put(h + 12, AOTX_RUNTIME_ROW, 4); aotx_ccir_put(h + 16, p->index.count, 4);
#ifdef AOTX_AFFECT
    aotx_ccir_put(h + 20, AOTX_RUNTIME_AFFECT, 4);
    if (!has_asset(p, "quality/refusal-phrases.txt")) return AOTX_CCIR_INVALID;
#endif
    if (has_asset(p, "audio.jsonl"))
        aotx_ccir_put(h + 20, aotx_ccir_u32(h + 20) | AOTX_RUNTIME_AUDIO, 4);
    if (has_asset(p, "vision.jsonl"))
        aotx_ccir_put(h + 20, aotx_ccir_u32(h + 20) | AOTX_RUNTIME_VISION, 4);
    if (has_asset(p, "policy.bin"))
        aotx_ccir_put(h + 20, aotx_ccir_u32(h + 20) | AOTX_RUNTIME_POLICY, 4);
    aotx_ccir_put(h + 24, AOTX_WIRE_LAYOUT, 4); aotx_ccir_put(h + 28, AOTX_SLOTS, 4);
    aotx_ccir_put(h + 32, AOTX_COG_OBJECTS, 4); aotx_ccir_put(h + 36, AOTX_RUNTIME_ARCH, 4);
    aotx_ccir_put(h + 40, AOTX_COG_PAYLOAD, 8); aotx_ccir_put(h + 48, AOTX_RUNTIME_ABI, 4);
    if (shared) {
        aotx_runtime_shared_profile profile;
        aotx_runtime_shared_current(&profile);
        aotx_runtime_shared_write(h, &profile);
    }
    strcpy((char *)h + 64, roles); h[128] = AOTX_CCIR_REPLAY;
    int rc = aotx_runtime_pack_appraisal(p);
    if (rc) return rc;
    memcpy(p->replay, "AOTXRPL1", 8); aotx_ccir_put(p->replay + 8, 1, 4);
    aotx_ccir_put(p->replay + 12, 1, 4);
    unsigned char checkpoint[16] = {AOTX_CCIR_CHECKPOINT}, live[16] = {AOTX_CCIR_LIVE};
    aotx_ccir_live_manifest(p->manifest, checkpoint, live);
    aotx_ccir_put(p->manifest + 8, 3, 4);
    aotx_ccir_put(p->manifest + 20, aotx_ccir_u32(p->memory + 8), 4);
    p->manifest[72] = AOTX_CCIR_RUNTIME;
    /* The first five directory entries are reserved before asset collection. */
    uint32_t count = p->count; p->count = 0;
    section(p, AOTX_CCIR_MANIFEST, 3, p->manifest, sizeof(p->manifest));
    section(p, AOTX_CCIR_CHECKPOINT, (uint16_t)aotx_ccir_u32(p->memory + 8), p->memory, p->memory_bytes);
    section(p, AOTX_CCIR_LIVE, 1, p->live, sizeof(p->live));
    section(p, AOTX_CCIR_RUNTIME, aotx_runtime_schema(aotx_ccir_u32(h + 20)), h,
        AOTX_RUNTIME_HEADER + p->index.count * AOTX_RUNTIME_ROW);
    section(p, AOTX_CCIR_REPLAY, 1, p->replay, sizeof(p->replay));
    p->count = count;
    aotx_ccir_limits limits; aotx_ccir_default_limits(&limits);
    aotx_ccir_view view; int fd = -1;
    rc = aotx_ccir_lock(output, 1, 1, &fd);
    if (rc) return rc;
    rc = aotx_ccir_initialize(fd, p->source.lineage, NULL, p->inputs, p->count,
                               &p->source.meta, &limits, &view);
    if (!rc) rc = aotx_runtime_dependencies(&view);
    if (!rc) rc = aotx_ccir_parent_sync(output);
    if (rc) unlink(output);
    close(fd);
    return rc;
}
int main(int argc, char **argv) {
    const char *memory = NULL, *models = NULL, *modules = NULL, *settings = NULL;
    const char *output = NULL, *roles = NULL, *phrases = NULL, *runtime = NULL, *policy = NULL;
    int shared = 0;
    for (int i = 1; i < argc;) {
        if (!strcmp(argv[i], "--shared")) {
            if (shared) goto usage;
            shared = 1; ++i; continue;
        }
        if (i + 1 == argc) goto usage;
        if (!strcmp(argv[i], "--memory")) memory = argv[i + 1];
        else if (!strcmp(argv[i], "--models")) models = argv[i + 1];
        else if (!strcmp(argv[i], "--modules")) modules = argv[i + 1];
        else if (!strcmp(argv[i], "--settings")) settings = argv[i + 1];
        else if (!strcmp(argv[i], "--output")) output = argv[i + 1];
        else if (!strcmp(argv[i], "--roles")) roles = argv[i + 1];
        else if (!strcmp(argv[i], "--phrases")) phrases = argv[i + 1];
        else if (!strcmp(argv[i], "--runtime")) runtime = argv[i + 1];
        else if (!strcmp(argv[i], "--policy")) { if (policy) goto usage; policy = argv[i + 1]; }
        else goto usage;
        i += 2;
    }
    if (runtime) {
        if (!shared || memory || models || modules || settings || output || roles || phrases || policy) goto usage;
        int rc = aotx_runtime_promote_shared(runtime);
        if (rc) fprintf(stderr, "shared runtime update refused: %s (%d)\n", aotx_ccir_status_text(rc), rc);
        else puts("shared runtime profile is saved");
        return rc ? 1 : 0;
    }
    if (!memory || !models || !modules || !output || !roles || !*roles || strlen(roles) >= 64) goto usage;
    aotx_runtime_pack *p = calloc(1, sizeof(*p));
    if (!p) return 1;
    aotx_policy_file policy_file = {0};
    p->source.fd = -1; p->count = 5;
    int rc = aotx_runtime_pack_memory(p, memory);
    if (!rc) rc = aotx_runtime_pack_models(p, models);
    if (!rc) rc = aotx_runtime_pack_tree(p, modules, "", "modules/", 2);
    if (!rc) rc = aotx_runtime_pack_settings(p, settings);
    if (!rc && phrases && !has_asset(p, "quality/refusal-phrases.txt"))
        rc = aotx_runtime_pack_asset(p, phrases, "quality/refusal-phrases.txt", 1);
    if (!rc && policy) rc = policy_asset(p, policy, &policy_file);
    if (!rc) rc = create(p, output, roles, shared);
    if (rc) fprintf(stderr, "runtime file creation refused: %s (%d)\n", aotx_ccir_status_text(rc), rc);
    else printf("runtime file created: %u assets\n", p->index.count);
    aotx_policy_file_close(&policy_file); aotx_runtime_pack_close(p); free(p);
    return rc ? 1 : 0;
usage:
    fputs("Use: aotx_ccir_pack --memory FILE --models DIR --roles LIST --modules DIR --output FILE\n"
          "                    [--settings FILE] [--phrases FILE] [--shared] [--policy FILE]\n"
          "     aotx_ccir_pack --runtime FILE --shared\n"
          "Create a new runtime from a complete prepared memory checkpoint.\n"
          "Device settings and data modules define the initial identity. No users are bound.\n"
          "The shared option requires shared tables when the complete runtime starts.\n"
          "The policy option stores one complete bundle. Packing does not execute code or grant trust.\n"
          "The runtime option updates an existing complete file with its asset extents unchanged.\n"
          "An affect build requires a refusal phrase asset in the store or --phrases.\n", stderr);
    return 2;
}
