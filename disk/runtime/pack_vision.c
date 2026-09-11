/* Purpose: Package the validated vision weights and relocate their paired manifest.
 * Owns: A generated manifest and the component's source descriptor.
 * Threading: One packager processes the model component batch.
 * Lifetime: One complete runtime file creation. */
#include "disk/runtime/pack.h"
#include "disk/modelfile/vision.h"
#include "disk/modelfile/manifest.h"
#include "disk/modelfile/media_profile.h"
#include <stdio.h>
#include <string.h>

int aotx_runtime_pack_vision(aotx_runtime_pack *p, const char *store,
    const aotx_manifest_entry *original, const aotx_manifest_entry *relocated, unsigned count)
{
    aotx_manifest_entry pair[2];
    int read = aotx_vision_manifest(store, pair);
    if (!read) return 0;
    if (read != 2) return AOTX_CCIR_INVALID;
    int parent = aotx_vision_pair(pair, original, count);
    if (parent < 0) return AOTX_CCIR_INVALID;
    aotx_modelfile *file = NULL; aotx_vision_desc desc;
    int rc = aotx_modelfile_open_entry(store, pair + 1, &file);
    if (!rc) rc = aotx_vision_file(file, &desc);
    aotx_modelfile_close(file);
    if (rc) return AOTX_CCIR_INVALID;
    char path[AOTX_MANIFEST_PATH]; unsigned char digest[32];
    if (aotx_manifest_digest(pair[1].sha256, digest) ||
        aotx_manifest_path(path, sizeof(path), store, pair[1].path)) return AOTX_CCIR_INVALID;
    rc = aotx_runtime_pack_asset(p, path, "weights/vision.gguf", 1);
    if (rc) return rc;
    const aotx_ccir_section *asset = &p->inputs[p->count-1].section;
    if (asset->bytes != pair[1].bytes || memcmp(asset->digest, digest, 32)) return AOTX_CCIR_INVALID;
    pair[0] = relocated[parent]; strcpy(pair[1].path, "weights/vision.gguf");
    FILE *manifest = tmpfile();
    if (!manifest) return AOTX_CCIR_IO;
    for (unsigned i = 0; i < 2 && !rc; ++i) {
        char line[AOTX_MANIFEST_LINE];
        if (aotx_manifest_write_line(line, sizeof(line), pair + i) || fputs(line, manifest) == EOF)
            rc = AOTX_CCIR_IO;
    }
    if (!rc && fflush(manifest)) rc = AOTX_CCIR_IO;
    if (!rc) {
        snprintf(path, sizeof(path), "/proc/self/fd/%d", fileno(manifest));
        rc = aotx_runtime_pack_asset(p, path, "vision.jsonl", 1);
    }
    fclose(manifest);
    if (rc) return rc;
    aotx_media_profile profile; unsigned char bytes[AOTX_MEDIA_PROFILE_BYTES];
    if (aotx_media_profile_store(store, &profile)) return AOTX_CCIR_INVALID;
    aotx_media_profile_write(&profile, bytes);
    FILE *profile_file = tmpfile();
    if (!profile_file) return AOTX_CCIR_IO;
    if (fwrite(bytes, 1, sizeof(bytes), profile_file) != sizeof(bytes) || fflush(profile_file)) rc = AOTX_CCIR_IO;
    if (!rc) {
        snprintf(path, sizeof(path), "/proc/self/fd/%d", fileno(profile_file));
        rc = aotx_runtime_pack_asset(p, path, "media.profile", 1);
    }
    fclose(profile_file); return rc;
}
