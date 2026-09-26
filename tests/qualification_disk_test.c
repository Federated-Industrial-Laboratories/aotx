/* Purpose: Check control evidence, exact settings and invalid file refusal.
 * Owns: Distinct model, source and evidence files at batch sizes one and 64.
 * Threading: One disk test process with no model execution.
 * Lifetime: Temporary files are removed before exit. */
#include "tests/disk_fake.h"
#include "disk/runtime/qualification.h"
#include "disk/ccir/internal.h"

static const char *keys[] = {"binding", "source", "commitments", "examples", "calibration", "acceptance", "consumer"};
static const char *names[] = {"vector.binding", "source.txt", "commitments.txt", "examples.tsv", "calibration.json", "acceptance.json", "consumer.txt"};
static char refs[4096];
static void write_bytes(const char *dir, const char *name, const void *bytes, size_t length) {
    char path[512]; snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *out = fopen(path, "wb"); CHECK(out != NULL, "fixture does not open");
    if (!out) return;
    CHECK(fwrite(bytes, 1, length, out) == length, "fixture does not write");
    CHECK(!fclose(out), "fixture does not close");
}
static void format(char *out, size_t room, unsigned schema, unsigned kind, const char *status,
    unsigned checks, const char *doses) {
    int n = snprintf(out, room, "{\"schema\":%u,\"kind\":%u,\"status\":\"%s\",\"checks\":%u,\"doses\":%s,%s}",
        schema, kind, status, checks, doses, refs);
    CHECK(n > 0 && (size_t)n < room, "qualification buffer is short");
}
static void batch(unsigned count) {
    char dir[] = "/tmp/aotx-qualification-XXXXXX", path[512], json[8192], changed[8192];
    CHECK(mkdtemp(dir) != NULL, "fixture directory does not open");
    snprintf(path, sizeof(path), "%s/vector", dir);
    for (unsigned i = 0; i < count; ++i) {
        char value[128]; snprintf(value, sizeof(value), "Distinct vector bytes for source %u.\n", i);
        write_bytes(dir, "vector", value, strlen(value));
        aotx_control_identity identity = {0}; identity.model[i % 32] = i + 1;
        identity.wrap.end_count = 1; identity.wrap.end_ids[0] = i + 7;
        identity.wrap.think_open_id = identity.wrap.think_close_id = UINT32_MAX;
        CHECK(!aotx_control_write(path, 1, &identity, AOTX_CONTROL_RESPONSE), "binding does not write");
        refs[0] = 0;
        for (unsigned j = 0; j < AOTX_QUALIFICATION_REFERENCES; ++j) {
            if (j) {
                snprintf(value, sizeof(value), "Distinct file %u, source %u.\n", j, i);
                write_bytes(dir, names[j], value, strlen(value));
            }
            char hash[65], part[512];
            CHECK(!aotx_control_digest(dir, names[j], hash), "reference digest does not read");
            snprintf(part, sizeof(part), "%s\"%s\":{\"file\":\"%s\",\"sha256\":\"%s\"}", j ? "," : "", keys[j], names[j], hash);
            strcat(refs, part);
        }
        aotx_control_permit permit; aotx_qualification parsed;
        char qualified[512]; snprintf(qualified, sizeof(qualified), "%s/vector.qualification", dir); unlink(qualified);
        CHECK(!aotx_qualification_read(dir, "vector", 1, &permit) && !permit.status,
            "an absent qualification enables a vector");
        format(json, sizeof(json), 1, 1, "accepted", 255, "[5000,10000]");
        CHECK(!aotx_qualification_parse(json, strlen(json), &parsed) && parsed.permit.count == 2 &&
            parsed.permit.dose[0] == 5000 && parsed.permit.dose[1] == 10000, "accepted doses do not parse");
        write_bytes(dir, "vector.qualification", json, strlen(json));
        CHECK(!aotx_qualification_read(dir, "vector", 1, &permit) && permit.status == 1 && permit.count == 2,
            "accepted exact evidence does not load");
        unsigned char digest[32]; aotx_ccir_hash(json, strlen(json), digest);
        CHECK(!memcmp(digest, permit.digest, 32), "qualification digest differs");
        CHECK(aotx_qualification_read(dir, "vector", 2, &permit) && !permit.status, "wrong control kind loads");
        write_bytes(dir, names[1], "changed", 7);
        CHECK(aotx_qualification_read(dir, "vector", 1, &permit) && !permit.status, "changed source loads");
        snprintf(value, sizeof(value), "Distinct file 1, source %u.\n", i);
        write_bytes(dir, names[1], value, strlen(value));
        write_bytes(dir, "vector", "changed", 7);
        CHECK(aotx_qualification_read(dir, "vector", 1, &permit) && !permit.status, "changed vector loads");
        snprintf(value, sizeof(value), "Distinct vector bytes for source %u.\n", i);
        write_bytes(dir, "vector", value, strlen(value));
        for (unsigned j = 0; j < AOTX_QUALIFICATION_REFERENCES; ++j) {
            snprintf(path, sizeof(path), "%s/%s", dir, names[j]);
            char backup[520]; snprintf(backup, sizeof(backup), "%s.saved", path);
            CHECK(!rename(path, backup), "reference does not move");
            CHECK(aotx_qualification_read(dir, "vector", 1, &permit) && !permit.status, "missing evidence loads");
            CHECK(!rename(backup, path), "reference does not return");
        }
        const char *invalid[] = {"[]", "[0]", "[5000,5000]", "[40001]", "[-40001]", "[0.5]", "[- 5]",
            "[+5000]", "[05000]", "[5000,]", "[5000,10000,15000,20000,25000,30000,35000,40000,-1,-2,-3,-4,-5,-6,-7,-8,-9]"};
        for (unsigned j = 0; j < sizeof(invalid) / sizeof(*invalid); ++j) {
            format(changed, sizeof(changed), 1, 1, "accepted", 255, invalid[j]);
            CHECK(aotx_qualification_parse(changed, strlen(changed), &parsed) && !parsed.permit.status,
                "invalid dose list parses");
        }
        for (unsigned bit = 1; bit < 256; bit *= 2) {
            format(changed, sizeof(changed), 1, 1, "accepted", 255 ^ bit, "[5000]");
            CHECK(aotx_qualification_parse(changed, strlen(changed), &parsed), "missing acceptance requirement parses");
        }
        format(changed, sizeof(changed), 2, 1, "accepted", 255, "[5000]");
        CHECK(aotx_qualification_parse(changed, strlen(changed), &parsed), "unknown schema parses");
        format(changed, sizeof(changed), 1, 4, "accepted", 255, "[5000]");
        CHECK(aotx_qualification_parse(changed, strlen(changed), &parsed), "unknown kind parses");
        format(changed, sizeof(changed), 1, 1, "measurement", 255, "[5000]");
        CHECK(aotx_qualification_parse(changed, strlen(changed), &parsed), "file grants measurement access");
        format(changed, sizeof(changed), 1, 1, "unavailable", 0, "[5000]");
        CHECK(aotx_qualification_parse(changed, strlen(changed), &parsed), "unavailable control has accepted doses");
        format(changed, sizeof(changed), 1, 1, "unavailable", 0, "[]");
        CHECK(!aotx_qualification_parse(changed, strlen(changed), &parsed) && !parsed.permit.status,
            "unavailable control does not parse");
        format(changed, sizeof(changed), 1, 2, "accepted", 255, "[]");
        CHECK(!aotx_qualification_parse(changed, strlen(changed), &parsed), "probe qualification does not parse");
        format(changed, sizeof(changed), 1, 3, "accepted", 255, "[]");
        CHECK(aotx_qualification_parse(changed, strlen(changed), &parsed), "continuous setting without bounds parses");
        format(changed, sizeof(changed), 1, 3, "accepted", 255, "[5000,5000,1000]");
        CHECK(!aotx_qualification_parse(changed, strlen(changed), &parsed), "bounded continuous setting does not parse");
        format(changed, sizeof(changed), 1, 3, "accepted", 255, "[-5000,5000,1000]");
        CHECK(aotx_qualification_parse(changed, strlen(changed), &parsed), "negative axis bound parses");
        strcpy(changed, json); strcpy(strstr(changed, "\"schema\""), "\"schema\":1,\"schema\":1}");
        CHECK(aotx_qualification_parse(changed, strlen(changed), &parsed), "duplicate key parses");
        strcpy(changed, json); strcat(changed, "x");
        CHECK(aotx_qualification_parse(changed, strlen(changed), &parsed), "trailing bytes parse");
        strcpy(changed, json); changed[30] = 0;
        CHECK(aotx_qualification_parse(changed, strlen(json), &parsed), "embedded zero parses");
        CHECK(aotx_qualification_read(dir, "../vector", 1, &permit), "outside asset path loads");
        snprintf(path, sizeof(path), "%s/vector", dir);
    }
    aotx_remove_tree(dir);
}
int main(void) { batch(1); batch(64); return aotx_report("qualification disk", 5000); }
