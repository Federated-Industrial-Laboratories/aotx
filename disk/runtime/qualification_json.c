/* Purpose: Decode control qualification without searching inside JSON values.
 * Owns: A bounded result with exact evidence references and accepted doses.
 * Threading: One caller for each independent control in the input batch.
 * Lifetime: One parse; a failed parse leaves the output empty. */
#include "disk/runtime/qualification.h"
#include "disk/runtime/runtime.h"
#include "disk/modelfile/manifest.h"
#include "disk/modelfile/manifest_json.h"
#include <string.h>

static int string(aotx_manifest_json *j, char *out, size_t room) {
    size_t n = 0;
    if (aotx_manifest_json_string(j, (unsigned char *)out, room - 1, &n) ||
        memchr(out, 0, n)) return 1;
    out[n] = 0; return 0;
}
static int key(aotx_manifest_json *j, const char *const *keys, unsigned count, unsigned *seen) {
    char name[32];
    if (string(j, name, sizeof(name)) || aotx_manifest_json_take(j, ':')) return -1;
    for (unsigned i = 0; i < count; ++i) if (!strcmp(name, keys[i])) {
        if (*seen & (1u << i)) return -1;
        *seen |= 1u << i; return (int)i;
    }
    return -1;
}
static int reference(aotx_manifest_json *j, aotx_control_reference_file *out) {
    const char *keys[] = {"file", "sha256"}; unsigned seen = 0;
    char hex[65];
    if (aotx_manifest_json_take(j, '{')) return 1;
    for (unsigned i = 0; i < 2; ++i) {
        if (i && aotx_manifest_json_take(j, ',')) return 1;
        int k = key(j, keys, 2, &seen);
        if (k < 0 || (k == 0 && (string(j, out->name, sizeof(out->name)) ||
            !aotx_runtime_name(out->name))) || (k == 1 &&
            (string(j, hex, sizeof(hex)) || aotx_manifest_digest(hex, out->digest)))) return 1;
    }
    unsigned nonzero = 0;
    for (unsigned i = 0; i < 32; ++i) nonzero |= out->digest[i];
    return !nonzero || aotx_manifest_json_take(j, '}');
}
static int doses(aotx_manifest_json *j, aotx_control_permit *out) {
    if (aotx_manifest_json_take(j, '[')) return 1;
    if (!aotx_manifest_json_take(j, ']')) return 0;
    do {
        uint64_t n; int minus = !aotx_manifest_json_take(j, '-');
        if (minus && (j->at == j->end || *j->at < '0' || *j->at > '9')) return 1;
        if (out->count == AOTX_QUALIFICATION_DOSES ||
            aotx_manifest_json_number(j, &n) || !n || n > 40000) return 1;
        int value = minus ? -(int)n : (int)n;
        out->dose[out->count++] = value;
        if (!aotx_manifest_json_take(j, ']')) return 0;
    } while (!aotx_manifest_json_take(j, ','));
    return 1;
}
int aotx_qualification_parse(const void *bytes, size_t length, aotx_qualification *out) {
    const char *keys[] = {"schema", "kind", "status", "checks", "doses", "binding",
        "source", "commitments", "examples", "calibration", "acceptance", "consumer"};
    aotx_qualification row = {0}; unsigned seen = 0; uint64_t n;
    memset(out, 0, sizeof(*out));
    if (!bytes || !length || length > AOTX_QUALIFICATION_BYTES) return 1;
    aotx_manifest_json j = {bytes, (const unsigned char *)bytes + length};
    if (aotx_manifest_json_take(&j, '{')) return 1;
    for (unsigned i = 0; i < sizeof(keys) / sizeof(*keys); ++i) {
        if (i && aotx_manifest_json_take(&j, ',')) return 1;
        int k = key(&j, keys, sizeof(keys) / sizeof(*keys), &seen);
        if (k < 0) return 1;
        if (k == 0) {
            if (aotx_manifest_json_number(&j, &n) || n != 1) return 1;
        } else if (k == 1) {
            if (aotx_manifest_json_number(&j, &n) || n < AOTX_CONTROL_VECTOR || n > AOTX_CONTROL_CALIBRATION) return 1;
            row.kind = (unsigned)n;
        } else if (k == 2) {
            char status[32];
            if (string(&j, status, sizeof(status))) return 1;
            if (!strcmp(status, "accepted")) row.permit.status = AOTX_QUALIFICATION_ACCEPTED;
            else if (strcmp(status, "unavailable")) return 1;
        } else if (k == 3) {
            if (aotx_manifest_json_number(&j, &n) || n > AOTX_QUALIFICATION_CHECKS) return 1;
            row.checks = (unsigned)n;
        } else if (k == 4) {
            if (doses(&j, &row.permit)) return 1;
        } else if (reference(&j, &row.reference[k - 5])) return 1;
    }
    if (aotx_manifest_json_take(&j, '}') || aotx_manifest_json_end(&j)) return 1;
    if (!row.permit.status && row.permit.count) return 1;
    if (row.permit.status) {
        if (row.checks != AOTX_QUALIFICATION_CHECKS) return 1;
        if (row.kind == AOTX_CONTROL_VECTOR) {
            if (!row.permit.count) return 1;
            for (unsigned i = 0; i < row.permit.count; ++i)
                for (unsigned j = 0; j < i; ++j) if (row.permit.dose[i] == row.permit.dose[j]) return 1;
        } else if (row.kind == AOTX_CONTROL_PROBE) {
            if (row.permit.count) return 1;
        } else {
            /* Absolute axis bounds and a divergence budget, in that order. */
            if (row.permit.count != 3) return 1;
            for (unsigned i = 0; i < 3; ++i) if (row.permit.dose[i] <= 0) return 1;
        }
    }
    *out = row; return 0;
}
