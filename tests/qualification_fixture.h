/* Purpose: Write explicit synthetic evidence for structural loader checks.
 * Owns: Small qualification files and seven bounded source references.
 * Threading: One test process writes each temporary store.
 * Lifetime: No fixture is used as model qualification evidence. */
#ifndef AOTX_TEST_QUALIFICATION_FIXTURE_H
#define AOTX_TEST_QUALIFICATION_FIXTURE_H
#include "disk/runtime/qualification.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
static const char *aotx_qualification_test_keys[] = {
    "binding", "source", "commitments", "examples", "calibration", "acceptance", "consumer"};
static int aotx_qualification_test_write(const char *dir, const char *asset, unsigned kind, const char *doses) {
    char json[8192], path[2048], name[256], digest[65], part[512];
    int n = snprintf(json, sizeof(json), "{\"schema\":1,\"kind\":%u,\"status\":\"accepted\",\"checks\":255,\"doses\":%s", kind, doses);
    if (n < 0 || (size_t)n >= sizeof(json)) return 1;
    for (unsigned i = 0; i < 7; ++i) {
        if (!i) snprintf(name, sizeof(name), "%s.binding", asset);
        else {
            snprintf(name, sizeof(name), "qualification-fixture-%s", aotx_qualification_test_keys[i]);
            snprintf(path, sizeof(path), "%s/%s", dir, name);
            FILE *out = fopen(path, "w"); if (!out) return 1;
            fprintf(out, "Synthetic %s for a structural loader test.\n", aotx_qualification_test_keys[i]);
            if (fclose(out)) return 1;
        }
        if (aotx_control_digest(dir, name, digest)) return 1;
        snprintf(part, sizeof(part), ",\"%s\":{\"file\":\"%s\",\"sha256\":\"%s\"}",
            aotx_qualification_test_keys[i], name, digest);
        if (strlen(json) + strlen(part) + 2 > sizeof(json)) return 1;
        strcat(json, part);
    }
    strcat(json, "}"); snprintf(path, sizeof(path), "%s/%s.qualification", dir, asset);
    FILE *out = fopen(path, "w"); if (!out) return 1;
    int bad = fputs(json, out) < 0; return fclose(out) || bad;
}
static void aotx_qualification_test_drop(const char *dir) {
    char path[2048];
    for (unsigned i = 1; i < 7; ++i) {
        snprintf(path, sizeof(path), "%s/qualification-fixture-%s", dir, aotx_qualification_test_keys[i]);
        unlink(path);
    }
}
#endif
