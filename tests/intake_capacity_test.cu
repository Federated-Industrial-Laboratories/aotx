/* Purpose: Verify candidate scaling and atomic store capacity refusal.
 * Owns: Large distinct candidate batches and independently sized pressure stores.
 * Launch shape: N=1 and N=64 through the complete live admission path.
 * Lifetime: One test process with supplied model outputs. */
#include "intake_fixture.h"

static void aotx_intake_capacity(unsigned n, unsigned mode) {
    const unsigned candidates = 100;
    aotx_fixture seed;
    if (mode) {
        unsigned count = mode == 1 ? AOTX_COG_OBJECTS + 1 - (candidates + 3) * n : 1;
        for (unsigned i = 0; i < count; ++i)
            seed.add(aotx_memory_row(i, AOTX_COG_COMPONENT, 600000 + i, i + 1, 2),
                aotx_bytes(mode == 2 ? AOTX_COG_PAYLOAD - 8192 * n : 1, 0x41));
    }
    const unsigned cut = seed.rows.size();
    aotx_intake_device d(n); d.send(aotx_live_load_bytes(seed.wire(false, cut)), 1);
    aotx_check(!d.state().status && d.state().ready, "capacity seed is a valid admitted store");
    auto bind = aotx_intake_bind(n); aotx_put(bind.data() + 32, cut);
    d.send(bind, 3); aotx_check(!d.state().status, "capacity binding batch is admitted");
    auto input = aotx_intake_query(n, cut, 1); std::vector<std::string> replies;
    for (unsigned i = 0; i < n; ++i) {
        std::string source, reply = "[";
        for (unsigned j = 0; j < candidates; ++j) {
            std::string quote = "item_" + std::to_string(i) + "_" + std::to_string(j) + ".";
            source += quote + " "; if (j) reply += ",";
            reply += "[3,\"" + quote + "\",0]";
        }
        reply += "]"; replies.push_back(reply);
        auto q = input.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
        aotx_check(source.size() <= AOTX_RECALL_TEXT && reply.size() < AOTX_INTAKE_REPLY,
            "large candidate fixture fits the declared source and response bounds");
        memset(q + 4640, 0, AOTX_RECALL_TEXT); memcpy(q + 4640, source.data(), source.size());
        aotx_put(q + 148, source.size(), 4);
    }
    auto before = aotx_retain_store(); auto bindings = d.bindings(n);
    auto result = aotx_retain_result(d.intake(input, replies), AOTX_INTAKE_CHOICE);
    auto bytes = aotx_retain_store(); auto s = (const aotx_cognitive_store *)bytes.data();
    if (!mode) {
        aotx_check(!d.state().status && s->count == (candidates + 3) * n,
            "one hundred candidates per source publish with the complete source batch");
        if (d.state().status) return;
        for (unsigned i = 0; i < n; ++i) {
            auto meta = result.data() + 64 + i * AOTX_LIVE_INTAKE_ROW + AOTX_LIVE_AUTO_ROW;
            aotx_check(aotx_get(meta + 72, 4) == candidates, "the recorded decision includes every candidate");
            for (unsigned j = 0; j < candidates; ++j) {
                auto row = s->objects[3 * n + i * candidates + j];
                auto p = s->payload + aotx_get(row + AOTX_CO_OFFSET);
                std::string quote = "item_" + std::to_string(i) + "_" + std::to_string(j) + ".";
                aotx_check(aotx_get(p + 12, 4) == quote.size() && !memcmp(p + 96, quote.data(), quote.size()),
                    "large candidate batches preserve every distinct quote in order");
            }
        }
    } else {
        aotx_check(d.state().status == AOTX_COG_CAPACITY && bytes == before,
            "whole candidate capacity refusal preserves every store byte");
        auto after = d.bindings(n);
        aotx_check(!memcmp(bindings.data(), after.data(), n * sizeof(after[0])),
            "capacity refusal preserves every conversation binding");
        aotx_check(result.size() == 64 && !aotx_get(result.data() + 8, 4),
            "capacity refusal records no accepted rows or partial tail");
    }
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) for (unsigned mode = 0; mode < 3; ++mode) aotx_intake_capacity(n, mode);
    printf("interpretation capacity: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
