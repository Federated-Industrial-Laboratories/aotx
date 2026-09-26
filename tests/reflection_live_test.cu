/* Purpose: Check idle review scheduling, actual task use and journal recovery.
 * Owns: Distinct sources, complete journals and damaged result controls.
 * Launch shape: Data and native policy graphs at N=1 and N=64.
 * Lifetime: Bounded live work, interruption and repeated exact replay. */
#include "policy_fixture.h"
#include "appraisal_recall_fixture.h"
#include "reflection/state.cuh"
#include "policy/control.cuh"
#include "appraisal/appraisal.cuh"
#include <chrono>

static unsigned aotx_review_n;
static const char *aotx_review_ptx;
static std::unique_ptr<aotx_review_state> aotx_review_read(void) {
    auto s = std::make_unique<aotx_review_state>();
    AOTX_CUDA(cudaMemcpyFromSymbol(s.get(), aotx_review, sizeof(*s))); return s;
}
static void aotx_review_hook(bool replay) {
    if (replay) aotx_live_test_idle<<<1,64>>>(aotx_review_n);
    aotx_policy_active_graph->tick();
}
static aotx_live_records aotx_review_command(const char *text) {
    aotx_live_records records(1); auto &r = records[0]; auto h = (aotx_record_header *)r.data();
    h->magic = AOTX_WIRE_MAGIC; h->layout = AOTX_WIRE_LAYOUT; h->header_bytes = 64;
    h->cls = AOTX_CLASS_A; h->type = AOTX_REC_INPUT_LINE; h->writer = AOTX_WRITER_CONSOLE;
    h->seq = 1; h->body_len = strlen(text); memcpy(r.data() + 64, text, h->body_len); return records;
}
static aotx_bytes aotx_review_query_rows(unsigned n, uint64_t cut, unsigned ordinal) {
    auto p = aotx_live_envelope("AOTXLIV1", n, cut, AOTX_LIVE_QUERY_ROW);
    auto q = aotx_ar_queries(n, cut);
    for (unsigned i = 0; i < n; ++i) {
        auto r = p.data() + 64 + i * AOTX_LIVE_QUERY_ROW;
        aotx_put(r, i, 4); aotx_id(r + 16, 8000 + i); aotx_put(r + 32, ordinal);
        aotx_put(aotx_query_at(q, i) + 132, AOTX_RECALL_LIMIT, 4);
        memcpy(r + 64, aotx_query_at(q, i), AOTX_RECALL_QUERY);
        aotx_id(r + 64, 100000 + ordinal * 64 + i); aotx_id(r + 112, 200000 + ordinal * 64 + i);
    }
    return p;
}
static void aotx_review_reset(void) {
    AOTX_LIVE_CLEAR(aotx_checkpoint); AOTX_LIVE_CLEAR(aotx_appraisal);
    aotx_checkpoint_test_clear<<<1,AOTX_SLOTS>>>();
}
static void aotx_review_live_case(unsigned n, unsigned mode) {
    aotx_review_n = n;
    aotx_policy_asset asset(mode, 16, "aotx_creator_maintenance", aotx_review_ptx, 1, 255, AOTX_ARCH, 3);
    aotx_live_records journal, work, prefix;
    std::unique_ptr<aotx_cognitive_store> expected;
    uint64_t frontier = 0, maximum = 0;
    {
        aotx_live_device d(n); aotx_review_reset(); asset.open();
        aotx_policy_graph graph; aotx_policy_active_graph = &graph;
        auto f = aotx_ar_corpus(n);
        aotx_maint_append(journal, d.send(aotx_live_load_bytes(f.wire(false, f.rows.size())), AOTX_LIVE_LOAD));
        aotx_check(!d.state().status, "qualified completed sources pass live admission");
        aotx_maint_append(journal, d.send(aotx_live_binding_bytes(n, f.rows.size()), AOTX_LIVE_BIND));
        uint64_t tail = d.seam().dev.tail;
        for (unsigned i = 0; i < 4; ++i) graph.tick();
        aotx_check(!aotx_review_read()->completed && d.seam().dev.tail == tail, "review remains off until explicit operator selection");
        aotx_maint_append(journal, d.process(aotx_review_command("policy review on")));
        aotx_check(aotx_review_read()->enabled && aotx_review_read()->control_revision == 1, "recorded local control enables review once");
        aotx_policy_test_control<<<1,1>>>(3); graph.tick();
        aotx_check(!aotx_policy_read_state()->calls, "foreground work suppresses policy entry");
        aotx_policy_test_control<<<1,1>>>(0);
        prefix = journal;
        work = d.process({}, false, n == 1, aotx_review_hook);
        if (n == 64) {
            auto pending = d.state(); uint64_t first_tick = 0, last_tick = 0;
            AOTX_CUDA(cudaMemcpyFromSymbol(&first_tick, aotx_time_tick, sizeof(first_tick)));
            aotx_check(aotx_review_read()->active && pending.written < pending.choice_bytes,
                "the maximum batch exposes its bounded publication interval");
            aotx_maint_append(work, d.process(aotx_review_command("policy pause"), false, true, aotx_review_hook));
            AOTX_CUDA(cudaMemcpyFromSymbol(&last_tick, aotx_time_tick, sizeof(last_tick)));
            unsigned bound = (pending.choice_bytes - pending.written + AOTX_LIVE_EMIT * AOTX_LIVE_DATA - 1) /
                (AOTX_LIVE_EMIT * AOTX_LIVE_DATA);
            aotx_check(last_tick - first_tick <= bound && !aotx_review_read()->active && aotx_policy_read_state()->paused,
                "a pause during recording finishes within the remaining publication tick bound");
            aotx_maint_append(work, d.process(aotx_review_command("policy resume")));
            printf("reflection yield n=%u mode=%u ticks=%llu bound=%u\n", n, mode,
                (unsigned long long)(last_tick - first_tick), bound);
        }
        aotx_maint_append(journal, work);
        auto review = aotx_review_read(); expected = aotx_maint_store();
        frontier = review->frontier; maximum = review->maximum_ns;
        aotx_check(maximum < 1000000000ull, "the admitted reference batch completes within the measured one-second budget");
        aotx_check(!d.state().fatal && !review->status && review->completed == n && !review->active,
            "the selected policy completes every supported review row");
        aotx_check(expected->count == f.rows.size() + 2 * n && frontier && review->calls == 1,
            "one idle batch records exactly one cue and dependency set per source");
        tail = d.seam().dev.tail;
        for (unsigned i = 0; i < 8; ++i) {
            auto quiet = d.process({}, false, true, aotx_review_hook);
            aotx_check(quiet.empty(), "unchanged completed sources generate no additional records");
        }
        aotx_check(d.seam().dev.tail == tail && aotx_review_read()->completed == n, "quiet ticks preserve the complete result");
        auto queries = d.send(aotx_review_query_rows(n, expected->sequence, 1), AOTX_LIVE_QUERY);
        aotx_maint_append(journal, queries);
        aotx_check(!d.state().status, "later matching task queries succeed through the live consumer");
        auto prompts = d.prompt(n);
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(prompts[i].find(AOTX_REVIEW_TEXT) != std::string::npos &&
                prompts[i].find(aotx_ar_text(i, false)) != std::string::npos,
                "actual agent prompts include the review and its exact supported source");
            if (n > 1) aotx_check(prompts[i].find(aotx_ar_text((i + 1) % n, false)) == std::string::npos,
                "actual task prompts exclude another participant's private evidence");
        }
        d.idle(n);
    }
    aotx_policy_close();
    for (unsigned pass = 0; pass < 2; ++pass) {
        aotx_live_device d(n); aotx_review_reset(); asset.open();
        aotx_policy_graph graph; aotx_policy_active_graph = &graph;
        d.process(journal, true, true, aotx_review_hook);
        auto review = aotx_review_read(); auto s = aotx_maint_store();
        aotx_check(!d.state().fatal && !aotx_policy_read_state()->fatal && review->completed == n && review->frontier == frontier &&
            !memcmp(s.get(), expected.get(), sizeof(*s)), "full journal recovery preserves exact review objects and the processed frontier");
        aotx_check(!aotx_policy_read_state()->calls, "recovery never executes the native policy entry");
        d.idle(n); auto quiet = d.process({}, false, true, aotx_review_hook);
        aotx_check(quiet.empty() && aotx_review_read()->completed == n, "recovered completed work is not nominated again");
        aotx_policy_close();
    }
    // Changed result bytes and repeated results cannot alter committed memory.
    for (unsigned fault = 0; fault < 2; ++fault) {
        aotx_live_device d(n); aotx_review_reset(); asset.open();
        aotx_policy_graph graph; aotx_policy_active_graph = &graph;
        d.process(prefix, true, true, aotx_review_hook);
        auto damaged = work; bool changed = false;
        if (!fault) {
            for (auto it = damaged.rbegin(); it != damaged.rend(); ++it) {
                auto h = (aotx_record_header *)it->data();
                if (h->type == AOTX_LIVE_RECORD && aotx_get(it->data() + 68, 4) == AOTX_REVIEW_RESULT) {
                    (*it)[64 + h->body_len - 1] ^= 1; changed = true; break;
                }
            }
        } else {
            d.process(work, true, true, aotx_review_hook); damaged.clear();
            for (const auto &r : work) {
                auto h = (const aotx_record_header *)r.data();
                if (h->type == AOTX_LIVE_RECORD && aotx_get(r.data() + 68, 4) == AOTX_REVIEW_RESULT)
                    damaged.push_back(r);
            }
            changed = !damaged.empty();
        }
        aotx_check(changed, "the replay defect targets actual recorded result bytes");
        auto before = aotx_maint_store();
        d.process(damaged, true, true, aotx_review_hook);
        auto after = aotx_maint_store();
        aotx_check(d.state().fatal && !memcmp(before.get(), after.get(), sizeof(*before)),
            "a changed or repeated result fails recovery without publishing memory");
        aotx_policy_close();
    }
    // A partial result must recover as one recorded interruption.
    aotx_live_records request, partial;
    for (const auto &r : work) {
        auto h = (const aotx_record_header *)r.data();
        if (h->type == AOTX_LIVE_RECORD && aotx_get(r.data() + 68, 4) == AOTX_REVIEW_RESULT) {
            if (partial.empty()) partial.push_back(r);
        } else request.push_back(r);
    }
    aotx_check(!request.empty() && !partial.empty(), "the test captured separate request and result boundaries");
    aotx_live_records interrupted;
    {
        aotx_live_device d(n); aotx_review_reset(); asset.open();
        aotx_policy_graph graph; aotx_policy_active_graph = &graph;
        d.process(prefix, true, true, aotx_review_hook); d.process(request, true, true, aotx_review_hook);
        auto before = aotx_maint_store();
        if (n == 64) d.process(partial, true);
        unsigned *ok, result = 0; AOTX_CUDA(cudaMalloc(&ok, sizeof(unsigned)));
        aotx_live_test_end<<<1,1>>>(ok); AOTX_CUDA(cudaMemcpy(&result, ok, sizeof(result), cudaMemcpyDeviceToHost)); cudaFree(ok);
        aotx_check(result && aotx_review_read()->recovery, "incomplete recorded work reaches the explicit interruption boundary");
        interrupted = d.process({}, false, true, aotx_review_hook);
        auto after = aotx_maint_store(); auto review = aotx_review_read();
        aotx_check(!d.state().fatal && review->interrupted == n && !review->completed && review->frontier == frontier &&
            !memcmp(before.get(), after.get(), sizeof(*before)), "interruption changes no source or review memory and processes the source frontier");
        aotx_policy_close();
    }
    {
        aotx_live_device d(n); aotx_review_reset(); asset.open();
        aotx_policy_graph graph; aotx_policy_active_graph = &graph;
        d.process(prefix, true, true, aotx_review_hook); d.process(request, true, true, aotx_review_hook);
        if (n == 64) d.process(partial, true);
        d.process(interrupted, true, true, aotx_review_hook);
        aotx_check(!d.state().fatal && aotx_review_read()->interrupted == n && !aotx_review_read()->completed,
            "a recorded interruption supersedes a partial result during repeated recovery");
        aotx_policy_close();
    }
    printf("reflection live n=%u mode=%u completed=%u maximum_ns=%llu records=%zu\n", n, mode, n,
        (unsigned long long)maximum, journal.size());
}
static void aotx_review_no_work(unsigned n, unsigned mode) {
    aotx_review_n = n;
    aotx_policy_asset asset(AOTX_POLICY_RULES, 16, "aotx_creator_maintenance", aotx_review_ptx, 1, 255, AOTX_ARCH, 3);
    aotx_live_device d(n); aotx_review_reset(); asset.open();
    aotx_policy_graph graph; aotx_policy_active_graph = &graph;
    auto f = aotx_ar_corpus(n);
    for (unsigned i = 0; i < f.rows.size(); ++i) {
        unsigned kind = aotx_get(f.rows[i].data() + AOTX_CO_KIND, 2);
        if (!mode && kind == AOTX_COG_APPRAISAL) {
            aotx_put(f.payloads[i].data() + 4, 0, 4); aotx_put(f.payloads[i].data() + 8, 0, 4);
        }
        if (mode == 1 && kind == AOTX_COG_EVENT) {
            unsigned owner = (i - 1) / 8;
            aotx_put(f.rows[i].data() + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
            aotx_id(f.rows[i].data() + AOTX_CO_SOURCE, aotx_ar_id(owner, 2));
            aotx_put(f.rows[i].data() + AOTX_CO_SOURCE_VERSION, 1);
        }
    }
    d.send(aotx_live_load_bytes(f.wire(false, f.rows.size(), mode == 2 ? UINT64_MAX : 5)), AOTX_LIVE_LOAD);
    aotx_check(d.state().status == (mode == 1 ? AOTX_COG_SOURCE : 0),
        "generated evidence is refused; zero outcomes and exhausted capacity remain valid state");
    d.process(aotx_review_command("policy review on"));
    auto before = aotx_maint_store(); uint64_t tail = d.seam().dev.tail;
    if (mode == 2) {
        d.process({}, false, true, aotx_review_hook);
        auto review = aotx_review_read();
        aotx_check(review->status == AOTX_COG_CAPACITY && review->refused == n && !review->frontier &&
            review->calls == 1 && !review->completed, "capacity refusal records one result without advancing the source frontier");
        tail = d.seam().dev.tail;
    }
    uint64_t calls = aotx_policy_read_state()->calls, reviews = aotx_review_read()->calls;
    for (unsigned i = 0; i < 8; ++i) d.process({}, false, true, aotx_review_hook);
    auto after = aotx_maint_store(); auto review = aotx_review_read();
    aotx_check(aotx_policy_read_state()->calls == calls && review->calls == reviews && !review->pending &&
        (mode == 2 || (!calls && !reviews)) && d.seam().dev.tail == tail &&
        !memcmp(before.get(), after.get(), sizeof(*before)),
        "unsupported, generated and capacity-blocked sources cause no repeated work or memory change");
    aotx_policy_close();
}
static void aotx_review_arrival(unsigned n, unsigned mode) {
    uint64_t ticks[2] = {}, nanos[2] = {};
    unsigned bound = 0;
    for (unsigned active = 0; active < 2; ++active) {
        aotx_review_n = n;
        aotx_policy_asset asset(mode, 16, "aotx_creator_maintenance", aotx_review_ptx, 1, 255, AOTX_ARCH, 3);
        aotx_live_device d(n); aotx_review_reset(); asset.open();
        aotx_policy_graph graph; aotx_policy_active_graph = &graph;
        auto f = aotx_ar_corpus(n);
        d.send(aotx_live_load_bytes(f.wire(false, f.rows.size())), AOTX_LIVE_LOAD);
        d.send(aotx_live_binding_bytes(n, f.rows.size()), AOTX_LIVE_BIND);
        if (active) d.process(aotx_review_command("policy review on"));
        graph.tick(); aotx_live_stage<<<1,64>>>(); aotx_live_decide<<<1,64>>>();
        AOTX_CUDA(cudaDeviceSynchronize());
        auto pending = d.state();
        aotx_check(!pending.status && !!aotx_review_read()->active == !!active,
            "foreground arrival has an explicit active and inactive control");
        if (active) {
            aotx_check(pending.phase == AOTX_LIVE_WRITE && pending.choice_bytes && !pending.written,
                "foreground input arrives before the first result publication fragment");
            bound = (pending.choice_bytes + AOTX_LIVE_EMIT * AOTX_LIVE_DATA - 1) /
                (AOTX_LIVE_EMIT * AOTX_LIVE_DATA);
        }
        auto input = aotx_live_parts(aotx_review_query_rows(1, f.rows.size() + active * 2 * n, 1),
            AOTX_LIVE_QUERY, 17000);
        uint64_t first = 0, last = 0;
        AOTX_CUDA(cudaMemcpyFromSymbol(&first, aotx_time_tick, sizeof(first)));
        auto start = std::chrono::steady_clock::now();
        d.process(input, false, true, aotx_review_hook);
        auto prompts = d.prompt(1);
        nanos[active] = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count();
        AOTX_CUDA(cudaMemcpyFromSymbol(&last, aotx_time_tick, sizeof(last))); ticks[active] = last - first;
        aotx_check(!d.state().status && !d.state().fatal && prompts[0].find(aotx_ar_text(0, false)) != std::string::npos,
            "arriving foreground input reaches its actual prompt with supported evidence");
        aotx_check(aotx_review_read()->completed == active * n && !aotx_review_read()->active,
            "foreground use leaves one complete result and no active review");
        aotx_policy_close();
    }
    aotx_check(ticks[1] > ticks[0] && ticks[1] - ticks[0] <= bound,
        "added foreground wait stays within the complete publication tick bound");
    aotx_check(nanos[1] < nanos[0] + 1000000000ull,
        "added foreground latency stays within the reference one-second budget");
    printf("reflection arrival n=%u mode=%u source_rows=%u publication_bound=%u idle_ticks=%llu active_ticks=%llu idle_ns=%llu active_ns=%llu added_ns=%lld\n",
        n, mode, 1 + 8 * n, bound, (unsigned long long)ticks[0], (unsigned long long)ticks[1],
        (unsigned long long)nanos[0], (unsigned long long)nanos[1], (long long)nanos[1] - (long long)nanos[0]);
}

int main(int argc, char **argv) {
    if (argc > 3 || (argc == 3 && strcmp(argv[2], "--yield-only"))) return 2;
    aotx_review_ptx = argc >= 2 ? argv[1] : AOTX_POLICY_TEST_PTX;
    if (argc == 3) {
        for (unsigned n : {1u, 64u}) for (unsigned mode : {AOTX_POLICY_RULES, AOTX_POLICY_NATIVE})
            aotx_review_arrival(n, mode);
        printf("reflection arrival: %u checks, %u failures\n", aotx_checks, aotx_failures);
        return aotx_failures ? 1 : 0;
    }
    for (unsigned n : {1u, 64u}) {
        for (unsigned mode = 0; mode < 3; ++mode) aotx_review_no_work(n, mode);
        for (unsigned mode : {AOTX_POLICY_RULES, AOTX_POLICY_NATIVE}) aotx_review_live_case(n, mode);
    }
    printf("reflection live: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
