/* Purpose: Drive typed live transfers through the real inbound and prompt consumers.
 * Owns: Independent byte fixtures, device rings and collected journal records.
 * Launch shape: Distinct conversation batches at one and all profile slots.
 * Lifetime: One test process without model weights. */
#ifndef AOTX_TEST_LIVE_FIXTURE_H
#define AOTX_TEST_LIVE_FIXTURE_H
#include "recall_fixture.h"
#include "agent/prompt.cuh"
#include "sched/sched.cuh"
#include "wrap_fixture.h"

using aotx_live_record = std::array<unsigned char, AOTX_SLOT_BYTES>;
using aotx_live_records = std::vector<aotx_live_record>;
#define AOTX_LIVE_TEST_SLOTS 32768u
#define AOTX_LIVE_TEST_ROLE (AOTX_MODULE_SLOTS - 1u)
#define AOTX_LIVE_CLEAR(symbol) do { void *p; AOTX_CUDA(cudaGetSymbolAddress(&p, symbol)); \
    AOTX_CUDA(cudaMemset(p, 0, sizeof(symbol))); } while (0)

static aotx_bytes aotx_live_envelope(const char *magic, unsigned n, uint64_t sequence, unsigned row) {
    aotx_bytes p(64 + (size_t)n * row, 0); memcpy(p.data(), magic, 8);
    aotx_put(p.data() + 8, n, 4); aotx_put(p.data() + 12, 1, 4); aotx_id(p.data() + 16, 9000);
    aotx_put(p.data() + 32, sequence); aotx_put(p.data() + 40, row, 4); return p;
}
static aotx_bytes aotx_live_load_bytes(const aotx_bytes &checkpoint, const aotx_bytes &tail = {}) {
    aotx_bytes p(16, 0); aotx_put(p.data(), checkpoint.size()); aotx_put(p.data() + 8, tail.size());
    p.insert(p.end(), checkpoint.begin(), checkpoint.end()); p.insert(p.end(), tail.begin(), tail.end()); return p;
}
static aotx_bytes aotx_live_binding_bytes(unsigned n, uint64_t sequence, unsigned scope = 0) {
    auto p = aotx_live_envelope("AOTXBND1", n, sequence, AOTX_LIVE_BIND_ROW);
    for (unsigned i = 0; i < n; ++i) {
        auto r = p.data() + 64 + i * AOTX_LIVE_BIND_ROW;
        aotx_put(r, i, 4); aotx_put(r + 4, scope, 4); aotx_id(r + 8, 1000 + i);
        if (scope == 1) aotx_id(r + 24, 2000 + i);
        aotx_id(r + 40, 8000 + i); aotx_put(r + 56, 16, 4);
    }
    return p;
}
static aotx_bytes aotx_live_query_bytes(unsigned n, uint64_t sequence, uint64_t ordinal, unsigned scope = 0) {
    auto p = aotx_live_envelope("AOTXLIV1", n, sequence, AOTX_LIVE_QUERY_ROW);
    auto prepared = aotx_memory_queries(n, sequence, scope);
    for (unsigned i = 0; i < n; ++i) {
        auto r = p.data() + 64 + i * AOTX_LIVE_QUERY_ROW;
        aotx_put(r, i, 4); aotx_id(r + 16, 8000 + i); aotx_put(r + 32, ordinal);
        memcpy(r + 64, aotx_query_at(prepared, i), AOTX_RECALL_QUERY);
        aotx_id(r + 64, 100000 + ordinal * 64 + i); aotx_id(r + 112, 200000 + ordinal * 64 + i);
        aotx_pin(r + 64, 0, 0, 10000 + i * 3);
        aotx_put(r + 64 + 136, 256, 4);
        std::string text = "turn " + std::to_string(ordinal) + " request " + std::to_string(i);
        memset(r + 64 + 4640, 0, AOTX_RECALL_TEXT);
        aotx_put(r + 64 + 148, text.size(), 4); memcpy(r + 64 + 4640, text.data(), text.size());
    }
    return p;
}
static aotx_live_records aotx_live_parts(const aotx_bytes &p, unsigned op, uint64_t id) {
    aotx_live_records records;
    for (size_t offset = 0; offset < p.size(); offset += AOTX_LIVE_DATA) {
        aotx_live_record r = {}; auto h = (aotx_record_header *)r.data(); auto body = r.data() + 64;
        unsigned bytes = (unsigned)std::min((size_t)AOTX_LIVE_DATA, p.size() - offset);
        h->magic = AOTX_WIRE_MAGIC; h->layout = 1; h->header_bytes = 64;
        h->cls = AOTX_CLASS_A; h->type = AOTX_LIVE_RECORD; h->writer = AOTX_WRITER_FEEDER;
        h->body_len = 32 + bytes; h->seq = records.size() + 1;
        aotx_put(body, 1, 4); aotx_put(body + 4, op, 4); aotx_id(body + 8, id);
        aotx_put(body + 24, p.size(), 4); aotx_put(body + 28, offset, 4);
        memcpy(body + 32, p.data() + offset, bytes); records.push_back(r);
    }
    return records;
}
__global__ void aotx_live_test_setup(unsigned n) {
    if (threadIdx.x) return;
    aotx_settings_reset();
    aotx_setting_table.row[AOTX_SET_RECALL_K].value = 0;
    aotx_setting_table.row[AOTX_SET_COMPACT_AT].value = 0;
    aotx_catalog_built_in(); aotx_catalog_anchor();
    aotx_catalog_entry *role = &aotx_catalog.entry[AOTX_LIVE_TEST_ROLE];
    role->state = AOTX_CATALOG_INSTALLED; role->kind = AOTX_MODULE_ROLE; role->role.budget = 8;
    aotx_model[AOTX_MODEL_LANGUAGE].layers = 1;
    aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, 1, 1, 8);
    aotx_tool_embed.ready = 1;
    for (unsigned i = 0; i < n; ++i) {
        aotx_agents.agent[i].state = AOTX_AGENT_STATE_IDLE;
        aotx_agents.agent[i].role = AOTX_LIVE_TEST_ROLE; aotx_agents.agent[i].task = ~0u;
        aotx_agent_gear[i].call.entry = AOTX_CATALOG_NO_ENTRY;
        aotx_tool_policies[i].choices = 349525u;
        aotx_transcript[i].summary_len = 9;
        for (unsigned j = 0; j < 9; ++j) aotx_transcript[i].summary[j] = "AUDIT_OLD"[j];
    }
}
__global__ void aotx_live_test_start(unsigned available) {
    unsigned count = available - (unsigned)aotx_seam.in.consumed;
    aotx_seam.apply.available = count;
    if (count > 256) count = 256;
    aotx_seam.apply.this_tick = aotx_live_window(aotx_seam.in.consumed, count);
    aotx_seam.apply.first_seq = aotx_seam.dev.tail + 1;
    aotx_seam.dev.tail += 2 * aotx_seam.apply.this_tick;
    ++aotx_time_tick;
}
__global__ void aotx_live_test_idle(unsigned n) {
    unsigned i = threadIdx.x; if (i >= n) return;
    aotx_agents.agent[i].state = AOTX_AGENT_STATE_IDLE;
    aotx_agent_gear[i].has_message = 0; aotx_agent_gear[i].stop_requested = 0;
    aotx_say.slot[i].wanted = 0;
}
__global__ void aotx_live_test_continuation(unsigned n, unsigned *out) {
    unsigned i = threadIdx.x; if (i >= n) return;
    aotx_agent_work *g = aotx_agent_gear + i;
    aotx_say.slot[i].wanted = 0;
    out[i] = aotx_agent_prompt(i, 0, g->message, g->message_len, 0, 0, 0, "tool result ok", 14);
}
__global__ void aotx_live_test_end(unsigned *out) { *out = aotx_live_restore_end(); }

struct aotx_live_view {
    uint32_t ready, phase, op, total, received, count, status, written, choice_bytes, fatal;
    uint64_t source_seq, request_seq, accepted, refused, searches, replays;
    unsigned char transfer_id[16], query_id[16];
};
static_assert(sizeof(aotx_live_view) == offsetof(aotx_live_state, input), "state prefix");
struct aotx_live_device {
    unsigned char *in, *out;
    aotx_inbound_preamble *preamble;
    uint64_t next_id = 1;
    explicit aotx_live_device(unsigned n) {
        AOTX_LIVE_CLEAR(aotx_live); AOTX_LIVE_CLEAR(aotx_live_store); AOTX_LIVE_CLEAR(aotx_live_bindings);
        AOTX_LIVE_CLEAR(aotx_agents); AOTX_LIVE_CLEAR(aotx_agent_gear); AOTX_LIVE_CLEAR(aotx_transcript);
        AOTX_LIVE_CLEAR(aotx_say); AOTX_LIVE_CLEAR(aotx_catalog); AOTX_LIVE_CLEAR(aotx_tool_policies);
        AOTX_LIVE_CLEAR(aotx_sched); AOTX_LIVE_CLEAR(aotx_agent_count);
        AOTX_CUDA(cudaMalloc(&in, AOTX_LIVE_TEST_SLOTS * AOTX_SLOT_BYTES));
        AOTX_CUDA(cudaMalloc(&out, 262144 * AOTX_SLOT_BYTES));
        AOTX_CUDA(cudaMalloc(&preamble, sizeof(*preamble)));
        AOTX_CUDA(cudaMemset(preamble, 0, sizeof(*preamble)));
        aotx_seam_state s = {}; s.dev.base = out; s.dev.slot_count = 262144; s.dev.mask = 262143;
        s.in.slots = in; s.in.slot_count = AOTX_LIVE_TEST_SLOTS; s.in.mask = AOTX_LIVE_TEST_SLOTS - 1;
        s.in.preamble = (unsigned char *)preamble; s.apply.state_hash = 14695981039346656037ull;
        AOTX_CUDA(cudaMemcpyToSymbol(aotx_seam, &s, sizeof(s)));
        aotx_test_wrap_open(); aotx_live_test_setup<<<1,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize());
    }
    ~aotx_live_device() { cudaFree(preamble); cudaFree(out); cudaFree(in); }
    aotx_live_view state() {
        aotx_live_view s; AOTX_CUDA(cudaMemcpyFromSymbol(&s, aotx_live, sizeof(s))); return s;
    }
    aotx_seam_state seam() { aotx_seam_state s; AOTX_CUDA(cudaMemcpyFromSymbol(&s, aotx_seam, sizeof(s))); return s; }
    std::vector<aotx_live_binding> bindings(unsigned n) {
        std::vector<aotx_live_binding> b(n); AOTX_CUDA(cudaMemcpyFromSymbol(b.data(), aotx_live_bindings, n * sizeof(b[0]))); return b;
    }
    aotx_live_records process(aotx_live_records records, bool replay = false, bool finish = true, void (*hook)(bool) = nullptr) {
        aotx_check(records.size() <= AOTX_LIVE_TEST_SLOTS, "fixture inbound capacity");
        auto before = seam(); before.in.consumed = 0; before.replaying = replay;
        uint64_t first = before.dev.tail;
        for (auto &r : records) {
            auto h = (aotx_record_header *)r.data();
            if (replay) { h->flags |= AOTX_FLAG_REPLAYED; h->source_seq[0] = (uint32_t)h->seq; h->source_seq[1] = (uint32_t)(h->seq >> 32); }
        }
        AOTX_CUDA(cudaMemcpyToSymbol(aotx_seam, &before, sizeof(before)));
        AOTX_CUDA(cudaMemcpy(in, records.data(), records.size() * AOTX_SLOT_BYTES, cudaMemcpyHostToDevice));
        unsigned ticks = 0;
        for (; ticks < 1000; ++ticks) {
            aotx_live_test_start<<<1,1>>>(records.size());
            aotx_seam_apply_inbound<<<AOTX_APPLY_BLOCKS,AOTX_APPLY_THREADS>>>();
            if (hook) hook(replay);
            aotx_live_stage<<<1,64>>>(); aotx_live_prepare<<<1,64>>>(); aotx_live_search<<<64,64>>>();
            aotx_live_decide<<<1,64>>>(); aotx_live_commit<<<1,64>>>();
            AOTX_CUDA(cudaGetLastError());
            auto now = seam(); auto s = state();
            if (now.in.consumed == records.size() && (!finish || s.phase == AOTX_LIVE_IDLE || s.phase == AOTX_LIVE_WAIT)) break;
        }
        aotx_check(ticks < 1000, "bounded transfer completion");
        auto after = seam(); aotx_check(after.apply.rejected == before.apply.rejected, "typed records pass inbound framing");
        aotx_check(after.dev.tail < 262144, "fixture journal capacity");
        aotx_live_records all(after.dev.tail - first), journal;
        AOTX_CUDA(cudaMemcpy(all.data(), out + first * AOTX_SLOT_BYTES, all.size() * AOTX_SLOT_BYTES, cudaMemcpyDeviceToHost));
        for (const auto &r : all) if (((const aotx_record_header *)r.data())->cls == AOTX_CLASS_A) journal.push_back(r);
        return journal;
    }
    aotx_live_records send(const aotx_bytes &p, unsigned op) { return process(aotx_live_parts(p, op, next_id++)); }
    std::vector<std::string> prompt(unsigned n) {
        aotx_agent_step<<<1,AOTX_SLOTS>>>(0); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_say_state s; AOTX_CUDA(cudaMemcpyFromSymbol(&s, aotx_say, sizeof(s)));
        std::vector<std::string> values;
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(s.slot[i].wanted && s.slot[i].length && s.slot[i].length <= AOTX_SAY_BYTES, "real agent prompt is queued");
            aotx_check(s.slot[i].page_limit == 16, "bound page budget reaches the tokenizer slot");
            std::string value(AOTX_SAY_BYTES, '\0');
            AOTX_CUDA(cudaMemcpyFromSymbol(value.data(), aotx_say, AOTX_SAY_BYTES,
                offsetof(aotx_say_state, prompt) + i * AOTX_SAY_BYTES));
            value.resize(std::min(s.slot[i].length, AOTX_SAY_BYTES)); values.push_back(value);
        }
        return values;
    }
    void idle(unsigned n) { aotx_live_test_idle<<<1,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize()); }
};
#endif
