/* Purpose: Verify GPU text preparation, atomic refusal and exact recorded replay.
 * Owns: Independent input, authority and context expectations across distinct slots.
 * Launch shape: Real inbound, tool planning, completion and prompt kernels at N=1/N=64.
 * Lifetime: One bounded test process; actual model encoding has a separate boot check. */
#include "text_fixture.h"
#include "text_cleanup.h"

static void aotx_text_roundtrip(unsigned n) {
    auto f = aotx_text_corpus(n); uint64_t seq = 2 * n;
    aotx_live_records start, records;
    std::vector<std::string> expected;
    std::vector<aotx_live_binding> bound;
    uint64_t hash = 0;
    {
        aotx_text_device d(n);
        start = d.send(aotx_live_load_bytes(f.wire(false, seq)), AOTX_LIVE_LOAD);
        auto bind = d.send(aotx_live_binding_bytes(n, seq), AOTX_LIVE_BIND);
        start.insert(start.end(), bind.begin(), bind.end());
        records = d.text(aotx_text_input(n, seq));
        aotx_check(d.state().status == 0 && d.state().phase == AOTX_LIVE_IDLE, "text batch reaches a complete recorded choice");
        aotx_check(d.encoded() == n && d.state().searches == n, "every text row encodes and searches once");
        auto choice = aotx_text_choice(records);
        aotx_check(choice.size() == 64 + n * AOTX_LIVE_TEXT_CHOICE_ROW && !memcmp(choice.data(), "AOTXTCH1", 8), "text choice carries full prepared queries");
        expected = d.prompt(n); bound = d.bindings(n);
        for (unsigned i = 0; i < n; ++i) {
            auto q = bound[i].query;
            aotx_check(aotx_get(q + 128, 4) == 3 && !memcmp(q + 96, aotx_test_processor, 32), "prepared width and processor identity");
            for (unsigned j = 0; j < 32; ++j) aotx_check(q[64 + j] == 0x12, "resident model digest is copied exactly");
            float values[3]; memcpy(values, q + 160, 12);
            aotx_check(values[0] == 2 + i && values[1] == 3 && values[2] == 1, "recorded vector belongs to its own batch row");
            aotx_check(bound[i].choice.count == 1 && aotx_selected(bound[i].choice, 0) == 10000 + i * 3, "semantic selection obeys private scope");
            aotx_check(expected[i].find("fact " + std::to_string(i) + " item 0") != std::string::npos, "text-derived choice reaches actual wrapped prompt");
            aotx_check(expected[i].find("AUDIT_OLD") == std::string::npos, "text mode excludes old transcript state");
        }
        hash = d.seam().apply.state_hash;
    }
    {
        aotx_text_device d(n); d.process_text(start, true); d.process_text(records, true);
        auto actual = d.prompt(n); auto rows = d.bindings(n);
        aotx_check(!d.state().fatal && d.encoded() == 0 && d.state().searches == 0 && d.state().replays == n, "text replay does no embedding or search");
        aotx_check(d.seam().apply.state_hash == hash, "text replay preserves exact journal state hash");
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(actual[i] == expected[i], "text replay produces exact wrapped prompt");
            aotx_check(!memcmp(rows[i].query, bound[i].query, AOTX_RECALL_QUERY), "prepared query bytes survive replay");
        }
        d.idle(n);
        aotx_live_records choices;
        for (auto &r : records) if (aotx_get(r.data() + 68, 4) == AOTX_LIVE_TEXT_CHOICE) choices.push_back(r);
        d.process_text(choices, true);
        aotx_check(d.state().fatal && d.state().replays == n, "duplicate text choice cannot requeue a request");
        for (auto &b : d.bindings(n)) aotx_check(b.ordinal == 1, "duplicate text choice preserves ordinal");
    }
    for (unsigned field : {0u, 16u, 64u, 96u, 128u, 140u, 152u, 160u, 4640u, 8191u}) {
        aotx_text_device d(n); d.process_text(start, true);
        auto raw = aotx_text_input(n, seq); auto request = aotx_live_parts(raw, AOTX_LIVE_TEXT, 3);
        d.process_text(request, true);
        auto choice = aotx_text_choice(records);
        auto p = choice.data() + 128 + (n - 1) * AOTX_LIVE_TEXT_CHOICE_ROW;
        if (field == 160) aotx_put(p + field, 0x7fc00000, 4); else p[field] ^= 1;
        d.process_text(aotx_live_parts(choice, AOTX_LIVE_TEXT_CHOICE, 3), true);
        aotx_check(d.state().fatal && !d.state().replays && !d.encoded(), "tampered prepared query refuses the whole replay batch");
        for (auto &b : d.bindings(n)) aotx_check(!b.ordinal && !b.context_bytes, "failed text replay publishes no partial context");
    }
    {
        aotx_text_device d(n); d.process_text(start, true);
        d.process_text(aotx_live_parts(aotx_text_input(n, seq), AOTX_LIVE_TEXT, 3), true);
        unsigned *ok; AOTX_CUDA(cudaMallocManaged(&ok, sizeof(*ok)));
        aotx_live_test_end<<<1,1>>>(ok); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_check(!*ok && !d.encoded(), "missing text choice refuses restore without encoding"); cudaFree(ok);
    }
}
__global__ void aotx_text_unavailable(unsigned mode, unsigned n) {
    if (threadIdx.x) return;
    if (mode == 1) aotx_tool_embed.ready = 0;
    if (mode == 2) aotx_tool_embed.width = 1025;
    if (mode == 3) aotx_model_load.resident[AOTX_MODEL_EMBEDDING].active = 0;
    if (mode == 4) aotx_requests.slot[n - 1].request = 1;
    if (mode == 5) aotx_model_load.pending_count = 1;
    if (mode == 6) aotx_tool_embed.state[n - 1] = AOTX_TOOL_EMBED_DONE;
}
static void aotx_text_refusals(unsigned n) {
    auto f = aotx_text_corpus(n); unsigned seq = 2 * n;
    for (unsigned bad = 0; bad < 18; ++bad) {
        aotx_text_device d(n);
        d.send(aotx_live_load_bytes(f.wire(false, seq)), AOTX_LIVE_LOAD);
        d.send(aotx_live_binding_bytes(n, seq), AOTX_LIVE_BIND);
        auto p = aotx_text_input(n, seq); auto q = p.data() + 128 + (n - 1) * AOTX_LIVE_QUERY_ROW;
        if (bad == 0) { aotx_put(q + 148, 193, 4); memset(q + 4640, 'a', 193); }
        if (bad == 1) q[4640] = 0xff;
        if (bad == 2) q[64] = 1;
        if (bad == 3) aotx_put(q + 128, 3, 4);
        if (bad == 4) q[160] = 1;
        if (bad == 5) q[16] ^= 1;
        if (bad >= 6 && bad < 12) aotx_text_unavailable<<<1,1>>>(bad - 5, n);
        if (bad >= 12) aotx_text_failure = bad < 16 ? bad - 11 : bad - 10;
        auto records = d.text(p);
        aotx_check(d.state().status && d.state().phase == AOTX_LIVE_IDLE, "invalid or unavailable text request terminates with refusal");
        aotx_check(!d.state().searches, "failed text preparation performs no recall");
        for (auto &b : d.bindings(n)) aotx_check(!b.ordinal && !b.context_bytes, "failed text batch does not publish any row");
        auto choice = aotx_text_choice(records);
        aotx_check(choice.size() == 64 && aotx_get(choice.data() + 44, 4) != 0, "text refusal is a recorded count-zero choice");
        if (bad < 12) aotx_check(d.encoded() == 0, "admission refusal does no encoding");
        if (bad >= 12) {
            aotx_tool_embed_batch embed; AOTX_CUDA(cudaMemcpyFromSymbol(&embed, aotx_tool_embed, sizeof(embed)));
            for (unsigned i = 0; i < n; ++i) aotx_check(embed.state[i] == AOTX_TOOL_EMBED_NONE, "failed batch releases all embedding leases");
        }
    }
}
static void aotx_text_followon(unsigned n) {
    auto f = aotx_text_corpus(n); aotx_text_device d(n);
    d.send(aotx_live_load_bytes(f.wire(false, 2 * n)), AOTX_LIVE_LOAD);
    d.send(aotx_live_binding_bytes(n, 2 * n), AOTX_LIVE_BIND);
    for (unsigned turn = 1; turn <= 3; ++turn) {
        aotx_text_failure = turn == 2 ? 5 : 0;
        auto p = aotx_text_input(n, 2 * n, turn);
        if (turn == 3) for (unsigned i = 0; i < n; ++i) {
            auto q = p.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
            memset(q + 4640, 'a' + i % 26, 192); aotx_put(q + 148, 192, 4);
        }
        d.text(p);
        aotx_check(!d.state().status && d.encoded() == (uint64_t)n * turn, "turnover and a full release queue do not repeat encoding");
        for (auto &b : d.bindings(n)) aotx_check(b.ordinal == turn && b.context_bytes <= 256, "text turnover retains a fixed memory budget");
        d.idle(n);
    }
    auto before = d.bindings(n);
    auto bad = aotx_text_input(n, 2 * n, 4); bad[128 + 16] ^= 1; d.text(bad);
    auto after = d.bindings(n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!memcmp(&before[i], &after[i], sizeof(before[i])), "refused text preserves the preceding context and ordinal");
    d.text(aotx_text_input(n, 2 * n, 4));
    aotx_check(!d.state().status && d.encoded() == 4ull * n, "a valid request follows a refused text batch");
}
int main(int argc, char **argv) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    bool cleanup_only = argc == 2 && !strcmp(argv[1], "--cleanup");
    for (unsigned n : {1u, 64u}) {
        if (!cleanup_only) { aotx_text_roundtrip(n); aotx_text_refusals(n); aotx_text_followon(n); }
        for (unsigned failure : {0u, 1u, 3u}) aotx_text_cleanup(n, failure);
    }
    printf("text memory: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
