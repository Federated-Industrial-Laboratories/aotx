/* Purpose: Check exact control bindings and damaged asset refusal.
 * Owns: Distinct file, model and turn-format fixtures at batch sizes one and 64.
 * Threading: One disk test process.
 * Lifetime: Temporary assets are removed before exit. */
#include "tests/disk_fake.h"
#include "disk/runtime/control.h"
#include "disk/ccir/internal.h"
#include <limits.h>

static int read_control(const char *dir, const aotx_control_identity *identity, unsigned kind, unsigned *positions) {
    char path[512]; snprintf(path, sizeof(path), "%s/vector.aotxvec", dir);
    FILE *in = fopen(path, "rb");
    CHECK(in != NULL, "asset does not open");
    if (!in) return 1;
    CHECK(fseek(in, 3, SEEK_SET) == 0, "stream position does not set");
    int rc = aotx_control_read(dir, "vector.aotxvec", kind, identity, in, positions);
    CHECK(ftell(in) == 3, "binding check changed the input position");
    fclose(in); return rc;
}
static void batch(unsigned count) {
    char dir[] = "/tmp/aotx-control-XXXXXX", path[512], bound[512];
    CHECK(mkdtemp(dir) != NULL, "fixture directory does not open");
    snprintf(path, sizeof(path), "%s/vector.aotxvec", dir);
    snprintf(bound, sizeof(bound), "%s/vector.aotxvec.binding", dir);
    for (unsigned i = 0; i < count; ++i) {
        aotx_control_identity identity = {0}, wrong;
        identity.model[i % 32] = (unsigned char)(i + 1);
        identity.wrap.end_count = 1; identity.wrap.end_ids[0] = i + 7;
        identity.wrap.bytes[0] = (unsigned char)(i + 32); identity.wrap.length[0] = 1;
        identity.wrap.think_open_id = identity.wrap.think_close_id = UINT32_MAX;
        FILE *out = fopen(path, "wb"); CHECK(out != NULL, "asset does not write");
        if (!out) break;
        fprintf(out, "asset %u with distinct values\n", i); fclose(out);
        unlink(bound);
        CHECK(read_control(dir, &identity, AOTX_CONTROL_VECTOR, NULL) != 0, "unbound asset loads");
        CHECK(aotx_control_write(path, AOTX_CONTROL_VECTOR, &identity, AOTX_CONTROL_ALL) == 0, "binding does not write");
        CHECK(read_control(dir, &identity, AOTX_CONTROL_VECTOR, NULL) == 0, "matching asset refuses");
        CHECK(read_control(dir, &identity, AOTX_CONTROL_PROBE, NULL) != 0, "wrong asset kind loads");
        wrong = identity; wrong.model[(i + 1) % 32] ^= 17;
        CHECK(read_control(dir, &wrong, AOTX_CONTROL_VECTOR, NULL) != 0, "same-width wrong model loads");
        wrong = identity; ++wrong.wrap.end_ids[0];
        CHECK(read_control(dir, &wrong, AOTX_CONTROL_VECTOR, NULL) != 0, "changed end token loads");
        wrong = identity; wrong.wrap.bytes[0] ^= 1;
        CHECK(read_control(dir, &wrong, AOTX_CONTROL_VECTOR, NULL) != 0, "changed turn span loads");
        unsigned char raw[AOTX_CONTROL_BYTES], digest[32];
        FILE *in = fopen(bound, "rb"); CHECK(in != NULL, "binding does not open");
        if (!in) break;
        CHECK(fread(raw, 1, sizeof(raw), in) == sizeof(raw), "binding is short"); fclose(in);
        CHECK(!aotx_control_decode(raw, AOTX_CONTROL_VECTOR, &wrong, digest, NULL), "binding does not decode");
        CHECK(!memcmp(&wrong, &identity, sizeof(identity)), "binding loses identity bytes");
        raw[12] = 2;
        CHECK(aotx_control_decode(raw, AOTX_CONTROL_VECTOR, &wrong, digest, NULL), "unknown hook loads");
        raw[12] = 1; raw[600] = 1;
        CHECK(aotx_control_decode(raw, AOTX_CONTROL_VECTOR, &wrong, digest, NULL), "reserved bytes load");
        out = fopen(path, "ab"); CHECK(out != NULL, "asset does not reopen");
        if (out) { fputc('x', out); fclose(out); }
        CHECK(read_control(dir, &identity, AOTX_CONTROL_VECTOR, NULL) != 0, "changed asset bytes load");
        CHECK(aotx_control_write(path, AOTX_CONTROL_VECTOR, &identity, AOTX_CONTROL_ALL) == 0, "new binding does not write");
        CHECK(read_control(dir, &identity, AOTX_CONTROL_VECTOR, NULL) == 0, "new exact binding refuses");
        CHECK(!truncate(bound, AOTX_CONTROL_BYTES - 1), "binding does not truncate");
        CHECK(read_control(dir, &identity, AOTX_CONTROL_VECTOR, NULL) != 0, "short binding loads");
        unsigned positions = 99;
        CHECK(!aotx_control_write(path, AOTX_CONTROL_VECTOR, &identity, AOTX_CONTROL_RESPONSE), "response binding does not write");
        CHECK(read_control(dir, &identity, AOTX_CONTROL_VECTOR, NULL), "response vector loads in a legacy consumer");
        CHECK(!read_control(dir, &identity, AOTX_CONTROL_VECTOR, &positions) && positions == AOTX_CONTROL_RESPONSE,
            "response mode does not load");
        in = fopen(bound, "rb"); CHECK(in != NULL, "response binding does not open");
        if (!in) break;
        CHECK(fread(raw, 1, sizeof(raw), in) == sizeof(raw), "response binding is short"); fclose(in);
        CHECK(!memcmp(raw, "AOTXCTL2", 8) && aotx_ccir_u32(raw + 8) == 2 && aotx_ccir_u32(raw + 20) == 1,
            "response version bytes differ");
        raw[20] = 0;
        CHECK(aotx_control_decode(raw, AOTX_CONTROL_VECTOR, &wrong, digest, &positions), "new version accepts legacy positions");
        raw[20] = 2;
        CHECK(aotx_control_decode(raw, AOTX_CONTROL_VECTOR, &wrong, digest, &positions), "unknown positions load");
        raw[20] = 1; raw[8] = 1;
        CHECK(aotx_control_decode(raw, AOTX_CONTROL_VECTOR, &wrong, digest, &positions), "mixed version loads");
        CHECK(aotx_control_write(path, AOTX_CONTROL_PROBE, &identity, AOTX_CONTROL_RESPONSE), "response probe writes");
        CHECK(aotx_control_write(path, AOTX_CONTROL_VECTOR, &identity, 2), "unknown position mode writes");
        CHECK(!aotx_control_write(path, AOTX_CONTROL_VECTOR, &identity, AOTX_CONTROL_ALL), "legacy binding does not write");
        CHECK(!read_control(dir, &identity, AOTX_CONTROL_VECTOR, &positions) && positions == AOTX_CONTROL_ALL,
            "legacy mode does not load in the new reader");
        memset(identity.model, 0, 32);
        CHECK(aotx_control_write(path, AOTX_CONTROL_VECTOR, &identity, AOTX_CONTROL_ALL) != 0, "missing model identity writes");
    }
    unlink(path); unlink(bound); CHECK(!rmdir(dir), "fixture directory is not empty");
}
int main(void) {
    batch(1); batch(64);
    return aotx_report("control disk", 2000);
}
