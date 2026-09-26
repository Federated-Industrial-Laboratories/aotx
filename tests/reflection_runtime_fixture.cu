/* Purpose: Write distinct prepared evidence for complete runtime file checks.
 * Owns: Synthetic task outcomes and an explicit selected model identity.
 * Launch shape: Host test construction for N=1 or N=64; no model interpretation.
 * Lifetime: Caller-owned fixture files, separate from measured model qualification. */
#include "appraisal_recall_fixture.h"
#include <fstream>
int main(int argc, char **argv) {
    if (argc != 4 || strlen(argv[3]) != 64) return 2;
    unsigned n = (unsigned)strtoul(argv[2], nullptr, 10);
    if (n != 1 && n != 64) return 2;
    unsigned char digest[32];
    for (unsigned i = 0; i < 32; ++i) {
        unsigned byte; if (sscanf(argv[3] + 2 * i, "%2x", &byte) != 1) return 2;
        digest[i] = (unsigned char)byte;
    }
    auto f = aotx_ar_corpus(n);
    for (unsigned i = 0; i < f.rows.size(); ++i) {
        auto &p = f.payloads[i]; unsigned kind = aotx_get(f.rows[i].data() + AOTX_CO_KIND, 2);
        if (kind == AOTX_COG_POLICY && p.size() == AOTX_APPRAISAL_QUEUE_BYTES) memcpy(p.data() + 96, digest, 32);
        if (kind == AOTX_COG_APPRAISAL) memcpy(p.data() + 64, digest, 32);
        if (kind == AOTX_COG_RELATIONSHIP) memcpy(p.data() + 104, digest, 32);
    }
    auto bytes = f.wire(false, f.rows.size());
    std::ofstream out(argv[1], std::ios::binary | std::ios::out); out.write((const char *)bytes.data(), bytes.size());
    return out ? 0 : 1;
}
