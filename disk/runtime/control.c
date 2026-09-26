/* Purpose: Read and write exact control bindings without loading model code.
 * Owns: Bounded binding bytes and short file leases.
 * Threading: One caller for each independent asset batch.
 * Lifetime: Each call closes its own files and leaves the input stream at its position. */
#include "disk/runtime/control.h"
#include "disk/runtime/assets.h"
#include "disk/ccir/internal.h"
#include "disk/wire/diskwire.h"
#include "disk/modelfile/manifest.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int identity_valid(const aotx_control_identity *identity) {
    return !aotx_ccir_zero(identity->model, 32) && aotx_wrap_valid(&identity->wrap) &&
        !identity->wrap.usable && identity->wrap.think_open_id == UINT32_MAX &&
        identity->wrap.think_close_id == UINT32_MAX;
}
int aotx_control_decode(const unsigned char raw[AOTX_CONTROL_BYTES], unsigned kind,
    aotx_control_identity *identity, unsigned char digest[32], unsigned *positions) {
    memset(identity, 0, sizeof(*identity));
    unsigned mode = aotx_ccir_u32(raw + 20);
    int legacy = !memcmp(raw, "AOTXCTL1", 8) && aotx_ccir_u32(raw + 8) == 1 && mode == AOTX_CONTROL_ALL;
    int response = !memcmp(raw, "AOTXCTL2", 8) && aotx_ccir_u32(raw + 8) == 2 &&
        mode == AOTX_CONTROL_RESPONSE && kind == AOTX_CONTROL_VECTOR && positions;
    if ((!legacy && !response) ||
        aotx_ccir_u32(raw + 12) != AOTX_CONTROL_HOOK ||
        kind < AOTX_CONTROL_VECTOR || kind > AOTX_CONTROL_CALIBRATION ||
        aotx_ccir_u32(raw + 16) != kind ||
        !aotx_ccir_zero(raw + 600, 8)) return 1;
    if (positions) *positions = mode;
    memcpy(identity->model, raw + 24, 32);
    memcpy(&identity->wrap, raw + 56, sizeof(identity->wrap));
    memcpy(digest, raw + 568, 32);
    return !identity_valid(identity) || aotx_ccir_zero(digest, 32);
}
static int stream_hash(FILE *in, unsigned char digest[32]) {
    fpos_t position;
    if (fgetpos(in, &position) || fseek(in, 0, SEEK_SET)) return 1;
    aotx_sha256 state; aotx_sha256_init(&state);
    unsigned char bytes[16384]; size_t got;
    while ((got = fread(bytes, 1, sizeof(bytes), in))) aotx_sha256_update(&state, bytes, got);
    int bad = ferror(in) != 0;
    aotx_sha256_final(&state, digest);
    clearerr(in);
    return fsetpos(in, &position) || bad;
}
int aotx_control_read(const char *store, const char *name, unsigned kind,
    const aotx_control_identity *expected, FILE *asset, unsigned *positions) {
    char path[AOTX_RUNTIME_NAME]; unsigned char raw[AOTX_CONTROL_BYTES], wanted[32], actual[32];
    aotx_control_identity identity;
    if (!asset || !identity_valid(expected) || !aotx_runtime_name(name)) return 1;
    int n = snprintf(path, sizeof(path), "%s.binding", name);
    if (n < 0 || (size_t)n >= sizeof(path)) return 1;
    FILE *in = aotx_asset_stream(store, path);
    if (!in) return 1;
    int bad = fread(raw, 1, sizeof(raw), in) != sizeof(raw) || fgetc(in) != EOF || ferror(in);
    if (fclose(in)) bad = 1;
    return bad || aotx_control_decode(raw, kind, &identity, wanted, positions) ||
        memcmp(&identity, expected, sizeof(identity)) || stream_hash(asset, actual) ||
        memcmp(wanted, actual, 32);
}
int aotx_control_write(const char *path, unsigned kind, const aotx_control_identity *identity, unsigned positions) {
    unsigned char raw[AOTX_CONTROL_BYTES] = {0}; char target[2048], temporary[2048];
    if (!identity_valid(identity) || kind < AOTX_CONTROL_VECTOR || kind > AOTX_CONTROL_CALIBRATION) return 1;
    if (positions > AOTX_CONTROL_RESPONSE || (positions && kind != AOTX_CONTROL_VECTOR)) return 1;
    int a = snprintf(target, sizeof(target), "%s.binding", path);
    int b = snprintf(temporary, sizeof(temporary), "%s.binding.XXXXXX", path);
    if (a < 0 || (size_t)a >= sizeof(target) || b < 0 || (size_t)b >= sizeof(temporary)) return 1;
    FILE *in = fopen(path, "rb");
    if (!in) return 1;
    int bad = stream_hash(in, raw + 568);
    if (fclose(in)) bad = 1;
    if (bad) return 1;
    memcpy(raw, positions ? "AOTXCTL2" : "AOTXCTL1", 8);
    aotx_ccir_put(raw + 20, positions, 4);
    aotx_ccir_put(raw + 8, positions ? 2 : 1, 4); aotx_ccir_put(raw + 12, AOTX_CONTROL_HOOK, 4);
    aotx_ccir_put(raw + 16, kind, 4);
    memcpy(raw + 24, identity->model, 32); memcpy(raw + 56, &identity->wrap, sizeof(identity->wrap));
    int fd = mkstemp(temporary);
    if (fd < 0) return 1;
    FILE *out = fdopen(fd, "wb");
    if (!out) { close(fd); unlink(temporary); return 1; }
    bad = fwrite(raw, 1, sizeof(raw), out) != sizeof(raw) || fflush(out) || fsync(fd);
    if (fclose(out)) bad = 1;
    if (!bad && rename(temporary, target)) bad = 1;
    if (bad) unlink(temporary);
    return bad;
}

int aotx_control_digest(const char *store, const char *name, char text[65]) {
    FILE *in = aotx_asset_stream(store, name); unsigned char digest[32];
    if (!in) return 1;
    int bad = stream_hash(in, digest);
    if (fclose(in)) bad = 1;
    if (!bad) aotx_sha256_text(digest, text);
    return bad;
}
int aotx_control_pair(const char *line, unsigned char digest[2][32]) {
    const char *at = strstr(line, "\"composite_sha256\":["); char text[2][65];
    if (!at || sscanf(at, "\"composite_sha256\":[\"%64[0-9a-f]\",\"%64[0-9a-f]\"]", text[0], text[1]) != 2)
        return 1;
    return aotx_manifest_digest(text[0], digest[0]) || aotx_manifest_digest(text[1], digest[1]);
}
