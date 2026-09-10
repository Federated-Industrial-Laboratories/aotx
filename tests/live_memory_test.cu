/* Purpose: Verify live scoped recall, prompt use, turnover and exact journal replay.
 * Owns: Independent expected prompt content and malformed-transfer controls.
 * Launch shape: Real inbound and cognitive kernels at N=1 and N=64.
 * Lifetime: One bounded device test process with no model weights. */
#include "live_fixture.h"

static aotx_fixture aotx_live_correction(unsigned n, uint64_t first) {
    aotx_fixture f;
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_memory_row(i, AOTX_COG_ASSERTION, 10000 + i * 3, first + i);
        aotx_put(r.data() + AOTX_CO_VERSION, 2);
        aotx_put(r.data() + AOTX_CO_CREATED, n + i + 1);
        aotx_id(r.data() + AOTX_CO_EMBEDDING, 900000 + i);
        aotx_put(r.data() + AOTX_CO_EMBED_VERSION, 1);
        f.add(r, aotx_memory_text("corrected fact " + std::to_string(i)));
    }
    return f;
}
static void aotx_live_turnover(unsigned n) {
    auto corpus = aotx_memory_corpus(n);
    aotx_live_records start;
    std::vector<aotx_live_records> journal;
    std::vector<std::vector<std::string>> expected;
    uint64_t hash = 0, sequence = corpus.rows.size();
    unsigned turns = n == 1 ? 70 : 4;
    unsigned correction_at = n == 1 ? 35 : 2;
    aotx_live_records correction;
    {
        aotx_live_device d(n);
        auto load = d.send(aotx_live_load_bytes(corpus.wire(false, sequence)), AOTX_LIVE_LOAD);
        aotx_check(d.state().ready && !d.state().status, "live checkpoint admission");
        auto bind = d.send(aotx_live_binding_bytes(n, sequence), AOTX_LIVE_BIND);
        aotx_check(!d.state().status, "fresh conversation bindings");
        start = load; start.insert(start.end(), bind.begin(), bind.end());
        for (unsigned turn = 1; turn <= turns; ++turn) {
            if (turn == correction_at) {
                auto update = aotx_live_correction(n, sequence + 1);
                correction = d.send(update.wire(true, sequence + 1, 6), AOTX_LIVE_UPDATE);
                aotx_check(!d.state().status, "idle correction is admitted"); sequence += n;
            }
            auto query = aotx_live_query_bytes(n, sequence, turn);
            if (turn >= correction_at) for (unsigned i = 0; i < n; ++i)
                aotx_put(query.data() + 64 + i * AOTX_LIVE_QUERY_ROW + 64 + 4256 + 16, 2);
            auto records = d.send(query, AOTX_LIVE_QUERY);
            aotx_check(!d.state().status && d.state().phase == AOTX_LIVE_IDLE, "complete choice precedes request delivery");
            auto prompts = d.prompt(n); auto rows = d.bindings(n);
            for (unsigned i = 0; i < n; ++i) {
                std::string fact = turn < correction_at ? "fact " + std::to_string(i) + " item 0" : "corrected fact " + std::to_string(i);
                std::string input = "turn " + std::to_string(turn) + " request " + std::to_string(i);
                aotx_check(rows[i].ordinal == turn && rows[i].choice.count == 1, "current bounded request replaces prior state");
                aotx_check(rows[i].context_bytes <= 256 && rows[i].choice.searches == 1, "fixed memory budget and real search");
                aotx_check(prompts[i].find(fact) != std::string::npos && prompts[i].find(input) != std::string::npos,
                    "real agent prompt contains exact scoped memory and current input");
                aotx_check(prompts[i].find("AUDIT_OLD") == std::string::npos, "audit summary cannot become cognitive history");
                aotx_check(prompts[i].find("<|im_start|>user\n[memory id=") != std::string::npos,
                    "selected memory uses the loaded model wrap");
                if (i + 1 < n) aotx_check(prompts[i].find("fact " + std::to_string(i + 1) + " item") == std::string::npos,
                    "another principal's memory stays out of the prompt");
            }
            if (turn == 1) {
                auto bad_update = aotx_live_correction(n, sequence + 1);
                auto refused = d.send(bad_update.wire(true, sequence + 1, 6), AOTX_LIVE_UPDATE);
                aotx_check(d.state().status == AOTX_COG_DENIED, "store changes refuse during active prompts");
                unsigned *result; AOTX_CUDA(cudaMallocManaged(&result, n * sizeof(*result)));
                aotx_live_test_continuation<<<1,64>>>(n, result); AOTX_CUDA(cudaDeviceSynchronize());
                aotx_say_state say; AOTX_CUDA(cudaMemcpyFromSymbol(&say, aotx_say, sizeof(say)));
                for (unsigned i = 0; i < n; ++i) {
                    std::string value((char *)say.prompt[i], say.slot[i].length);
                    aotx_check(result[i] && value.find("tool result ok") != std::string::npos && value.find("[memory id=") != std::string::npos,
                        "tool continuation keeps bound memory and current result");
                }
                cudaFree(result);
                /* The refused update is an ordered journal input, even though it changes no state. */
                records.insert(records.end(), refused.begin(), refused.end());
            }
            journal.push_back(records); expected.push_back(prompts); d.idle(n);
        }
        auto state = d.state(); aotx_check(state.searches == (uint64_t)n * turns, "all live requests searched once");
        uint32_t stored_count; uint64_t stored_sequence;
        AOTX_CUDA(cudaMemcpyFromSymbol(&stored_count, aotx_live_store, sizeof(stored_count), offsetof(aotx_cognitive_store, count)));
        AOTX_CUDA(cudaMemcpyFromSymbol(&stored_sequence, aotx_live_store, sizeof(stored_sequence), offsetof(aotx_cognitive_store, sequence)));
        aotx_check(stored_count == 3 * n && stored_sequence == 3 * n, "turnover does not append a saved request archive");
        hash = d.seam().apply.state_hash;
    }
    {
        aotx_live_device d(n);
        d.process(start, true);
        for (unsigned turn = 1; turn <= turns; ++turn) {
            if (turn == correction_at) d.process(correction, true);
            /* Keep the original active-prompt refusal after prompt construction. */
            auto records = journal[turn - 1]; aotx_live_records after;
            if (turn == 1) {
                auto split = std::find_if(records.begin(), records.end(), [](const aotx_live_record &r) {
                    return aotx_get(r.data() + 64 + 4, 4) == AOTX_LIVE_UPDATE;
                });
                after.assign(split, records.end()); records.erase(split, records.end());
            }
            d.process(records, true);
            aotx_check(d.state().fatal == 0, "complete typed replay remains valid");
            auto actual = d.prompt(n);
            for (unsigned i = 0; i < n; ++i) aotx_check(actual[i] == expected[turn - 1][i], "journal replay reproduces exact wrapped prompt bytes");
            if (!after.empty()) d.process(after, true);
            d.idle(n);
        }
        aotx_check(d.state().searches == 0 && d.state().replays == (uint64_t)n * turns, "restore performs zero searches across turnover");
        aotx_check(d.seam().apply.state_hash == hash, "typed state hash survives exact ordered replay");
        unsigned *ok; AOTX_CUDA(cudaMallocManaged(&ok, sizeof(*ok)));
        aotx_live_test_end<<<1,1>>>(ok); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_check(*ok == 1, "restore completion accepts complete live state"); cudaFree(ok);
    }
}

__global__ void aotx_live_pending_task(unsigned enabled) {
    aotx_task_used[0] = enabled;
    aotx_agents.task[0].agent = 0;
    aotx_agents.task[0].state = AOTX_TASK_PENDING;
}
static void aotx_live_refusals(unsigned n) {
    auto corpus = aotx_memory_corpus(n); auto image = aotx_live_load_bytes(corpus.wire(false, 2 * n));
    aotx_live_device d(n); d.send(image, AOTX_LIVE_LOAD);
    auto bind = aotx_live_binding_bytes(n, 2 * n);
    auto bad = bind; aotx_put(bad.data() + 64 + (n - 1) * 64 + 56, 0, 4);
    d.send(bad, AOTX_LIVE_BIND); aotx_check(d.state().status == 1, "one malformed binding refuses the batch");
    for (const auto &b : d.bindings(n)) aotx_check(!b.active, "binding refusal publishes no rows");
    aotx_live_pending_task<<<1,1>>>(1);
    d.send(bind, AOTX_LIVE_BIND);
    aotx_check(d.state().status == AOTX_COG_DENIED, "binding cannot adopt a pending base task");
    for (const auto &b : d.bindings(n)) aotx_check(!b.active, "pending task keeps the binding batch unchanged");
    aotx_live_pending_task<<<1,1>>>(0);
    d.send(bind, AOTX_LIVE_BIND); d.send(bind, AOTX_LIVE_BIND);
    aotx_check(d.state().status == AOTX_COG_DENIED, "binding cannot silently replace a conversation");
    auto query = aotx_live_query_bytes(n, 2 * n, 1);
    for (unsigned mode = 0; mode < 8; ++mode) {
        bad = query; auto r = bad.data() + 64 + (n - 1) * AOTX_LIVE_QUERY_ROW; auto q = r + 64;
        if (mode == 0) q[16] ^= 1;
        if (mode == 1) r[16] ^= 1;
        if (mode == 2) aotx_put(r + 32, 2);
        if (mode == 3) aotx_put(q + 136, 1, 4);
        if (mode == 4) q[64] ^= 1;
        if (mode == 5) q[8191] = 1;
        if (mode == 6) aotx_put(bad.data() + 32, 2 * n - 1);
        if (mode == 7) aotx_put(q + 4256 + 16, 2);
        auto records = d.send(bad, AOTX_LIVE_QUERY);
        const unsigned expected[8] = {11,11,6,2,7,1,10,10};
        aotx_check(d.state().status == expected[mode] && d.state().phase == AOTX_LIVE_IDLE, "invalid query has a complete refused choice");
        aotx_check(!records.empty() && aotx_get(records.back().data() + 64 + 4,4) == AOTX_LIVE_CHOICE,
            "query refusal is journaled for deterministic completion");
        for (const auto &b : d.bindings(n)) aotx_check(!b.ordinal && !b.context_bytes, "query refusal preserves all current rows");
    }
    d.send(query, AOTX_LIVE_QUERY); d.prompt(n); d.idle(n);
    d.send(query, AOTX_LIVE_QUERY); aotx_check(d.state().status == 6, "old request ordinal cannot run twice");
}
static void aotx_live_replay_failures(unsigned n) {
    auto corpus = aotx_memory_corpus(n); auto image = aotx_live_load_bytes(corpus.wire(false, 2 * n));
    aotx_live_records load, bind, query;
    {
        aotx_live_device d(n); load = d.send(image, AOTX_LIVE_LOAD);
        bind = d.send(aotx_live_binding_bytes(n, 2 * n), AOTX_LIVE_BIND);
        query = d.send(aotx_live_query_bytes(n, 2 * n, 1), AOTX_LIVE_QUERY);
    }
    for (unsigned mode = 0; mode < 3; ++mode) {
        aotx_live_device d(n); d.process(load, true); d.process(bind, true);
        auto bad = query;
        if (mode == 0) bad.erase(std::remove_if(bad.begin(), bad.end(), [](const aotx_live_record &r) {
            return aotx_get(r.data() + 68,4) == AOTX_LIVE_CHOICE;
        }), bad.end());
        if (mode == 1) bad.pop_back();
        if (mode == 2) {
            /* The selected ID begins at choice offset 144, in its first part. */
            for (auto &r : bad) if (aotx_get(r.data() + 68,4) == AOTX_LIVE_CHOICE && !aotx_get(r.data() + 92,4)) {
                r[64 + 32 + 144] ^= 1; break;
            }
        }
        d.process(bad, true);
        unsigned *ok; AOTX_CUDA(cudaMallocManaged(&ok, sizeof(*ok)));
        aotx_live_test_end<<<1,1>>>(ok); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_check(!*ok && d.state().fatal && !d.state().searches, "missing, partial or changed choices fail restore without search");
        for (const auto &b : d.bindings(n)) aotx_check(!b.ordinal, "invalid replay cannot queue part of a batch");
        cudaFree(ok);
    }
}
__global__ void aotx_live_test_controls(unsigned n, unsigned *result) {
    if (threadIdx.x) return;
    for (unsigned i = 0; i < n; ++i) {
        result[i * 4] = aotx_agent_message(i, (const unsigned char *)"untyped", 7, 0);
        result[i * 4 + 1] = aotx_transcript_pages(i, 32);
        result[i * 4 + 2] = aotx_transcript_compact(i);
        result[i * 4 + 3] = aotx_transcript_page_limit(i);
    }
}
static void aotx_live_authority(unsigned n) {
    /* The foreign request is valid for another admitted principal, so recall alone cannot refuse it. */
    auto corpus = aotx_memory_corpus(n == 1 ? 2 : n);
    uint64_t sequence = corpus.rows.size();
    aotx_live_device d(n);
    d.send(aotx_live_load_bytes(corpus.wire(false, sequence)), AOTX_LIVE_LOAD);
    d.send(aotx_live_binding_bytes(n, sequence), AOTX_LIVE_BIND);
    auto query = aotx_live_query_bytes(n, sequence, 1);
    auto q = query.data() + 64 + (n - 1) * AOTX_LIVE_QUERY_ROW + 64;
    unsigned foreign = n == 1 ? 1 : 0;
    aotx_id(q + 16, 1000 + foreign); aotx_pin(q, 0, 0, 10000 + foreign * 3);
    d.send(query, AOTX_LIVE_QUERY);
    aotx_check(d.state().status == AOTX_COG_DENIED && d.state().searches == 0,
        "another valid principal cannot replace the conversation authority");
    for (const auto &b : d.bindings(n)) aotx_check(!b.ordinal, "foreign authority publishes no row");
    unsigned *out; AOTX_CUDA(cudaMallocManaged(&out, n * 4 * sizeof(*out)));
    aotx_live_test_controls<<<1,1>>>(n, out); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(out[i*4] && out[i*4+1] && out[i*4+2], "base message and history controls cannot change a bound conversation");
        aotx_check(out[i*4+3] == 16, "page report names the effective fixed cap");
    }
    cudaFree(out);
}
static void aotx_live_replay_once(unsigned n) {
    auto corpus = aotx_memory_corpus(n);
    for (unsigned refused = 0; refused < 2; ++refused) {
        aotx_live_records load, bind, query, choice;
        {
            aotx_live_device d(n);
            load = d.send(aotx_live_load_bytes(corpus.wire(false, 2 * n)), AOTX_LIVE_LOAD);
            bind = d.send(aotx_live_binding_bytes(n, 2 * n), AOTX_LIVE_BIND);
            query = d.send(aotx_live_query_bytes(n, 2 * n, refused ? 2 : 1), AOTX_LIVE_QUERY);
            for (const auto &r : query)
                if (aotx_get(r.data() + 68, 4) == AOTX_LIVE_CHOICE) choice.push_back(r);
            aotx_check(!choice.empty(), "both accepted and refused requests record a choice");
        }
        aotx_live_device d(n);
        d.process(load, true); d.process(bind, true); d.process(query, true);
        if (!refused) d.prompt(n);
        d.idle(n);
        unsigned *ok; AOTX_CUDA(cudaMallocManaged(&ok, sizeof(*ok)));
        aotx_live_test_end<<<1,1>>>(ok); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_check(*ok && !d.state().fatal, "one matching choice completes valid restore");
        uint64_t replays = refused ? 0 : n;
        aotx_check(d.state().replays == replays, "only admitted requests count as replayed");
        d.process(choice, true);
        aotx_live_test_end<<<1,1>>>(ok); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_check(!*ok && d.state().fatal && d.state().replays == replays && !d.state().searches,
            "a consumed choice cannot replay again without an outstanding query");
        std::vector<aotx_agent_work> gear(n); aotx_say_state say;
        AOTX_CUDA(cudaMemcpyFromSymbol(gear.data(), aotx_agent_gear, n * sizeof(gear[0])));
        AOTX_CUDA(cudaMemcpyFromSymbol(&say, aotx_say, sizeof(say)));
        for (const auto &b : d.bindings(n))
            aotx_check(b.ordinal == (refused ? 0u : 1u), "duplicate choice preserves the current ordinal");
        for (unsigned i = 0; i < n; ++i)
            aotx_check(!gear[i].has_message && !say.slot[i].wanted, "duplicate choice cannot queue another prompt");
        cudaFree(ok);
    }
}
int main(void) {
    for (unsigned n : {1u, 64u}) {
        aotx_live_turnover(n); aotx_live_refusals(n); aotx_live_replay_failures(n); aotx_live_authority(n);
        aotx_live_replay_once(n);
    }
    printf("live memory: %u checks, %u failed\n", aotx_checks, aotx_failures);
    return aotx_failures || aotx_checks != 6367 ? 1 : 0;
}
