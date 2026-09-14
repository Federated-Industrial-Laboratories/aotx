/* Purpose: Verify atomic appraisal writes, independent admission and exact recovery.
 * Owns: Whole-batch source, result, corruption and interrupted-journal expectations.
 * Launch shape: Real device state transitions at N=1 and N=64.
 * Lifetime: One maintained test process with controlled decoder responses. */
#include "appraisal_work_fixture.h"
#include "appraisal_recall_fixture.h"

static unsigned aotx_appraisal_pair_row(const aotx_fixture &f, uint64_t id) {
    for (unsigned j = 0; j < f.rows.size(); ++j)
        if (aotx_get(f.rows[j].data() + AOTX_CO_ID) == id) return j;
    aotx_check(false, "paired correction fixture contains each required object");
    return 0;
}
static void aotx_appraisal_paired_correction(unsigned n, bool protect) {
    aotx_live_records start, request, decision;
    aotx_bytes expected;
    {
        aotx_appraisal_device d(n); auto f = aotx_ar_corpus(n);
        for (unsigned i = 0; i < n; ++i) {
            if (protect && i + 1 == n)
                aotx_put(f.rows[aotx_appraisal_pair_row(f, aotx_ar_id(i, 7))].data() + AOTX_CO_FLAGS, AOTX_COG_PROTECTED, 4);
            aotx_ar_episode(f, i, true, 0);
            f.rows.resize(f.rows.size() - 2); f.payloads.resize(f.payloads.size() - 2);
            aotx_put(f.payloads.back().data() + 12, 0, 4); memset(f.payloads.back().data() + 96, 0, 32);
        }
        start = d.send(aotx_live_load_bytes(f.wire(false, f.rows.size())), 1);
        aotx_check(!d.state().status && !d.state().fatal, "paired correction source checkpoint is valid");
        auto before = aotx_retain_store();
        aotx_bytes input(64 + n * 32, 0); memcpy(input.data(), "AOTXAPR1", 8);
        aotx_put(input.data() + 8, 1, 4); aotx_put(input.data() + 12, n, 4);
        aotx_put(input.data() + 16, f.rows.size()); memcpy(input.data() + 24, f.rows[0].data() + AOTX_CO_ID, 16);
        aotx_put(input.data() + 40, 1);
        aotx_bytes replies(n * AOTX_INTAKE_REPLY, 0); std::vector<unsigned> sizes(n);
        for (unsigned i = 0; i < n; ++i) {
            const auto &r = f.rows[aotx_appraisal_pair_row(f, aotx_ar_id(i, 14))];
            memcpy(input.data() + 64 + i * 32, r.data() + AOTX_CO_ID, 16);
            aotx_put(input.data() + 80 + i * 32, 1); aotx_put(input.data() + 88 + i * 32, i, 4);
            std::string text = "{\"benefit\":" + std::to_string(100000 + i) +
                ",\"harm\":0,\"arousal\":4294967295,\"consequence\":1,\"confidence\":500000,"
                "\"regard_gain\":700000,\"regard_loss\":0,\"trust_gain\":4294967295,\"trust_loss\":4294967295,"
                "\"evidence\":\"" + aotx_ar_text(i, true) + "\",\"task\":\"\",\"commitment\":\"\",\"correction\":1}";
            sizes[i] = text.size(); memcpy(replies.data() + i * AOTX_INTAKE_REPLY, text.data(), text.size());
        }
        request = d.process(aotx_live_parts(input, 16, d.next_id++), false, false);
        aotx_check(d.appraisal().active, "paired correction reaches the controlled decoder boundary");
        unsigned char *p; unsigned *lengths;
        AOTX_CUDA(cudaMalloc(&p, replies.size())); AOTX_CUDA(cudaMalloc(&lengths, n * sizeof(unsigned)));
        AOTX_CUDA(cudaMemcpy(p, replies.data(), replies.size(), cudaMemcpyHostToDevice));
        AOTX_CUDA(cudaMemcpy(lengths, sizes.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));
        aotx_appraisal_test_output<<<1,64>>>(p, lengths, 0); AOTX_CUDA(cudaDeviceSynchronize());
        cudaFree(lengths); cudaFree(p); decision = d.process({});
        aotx_check(!d.state().fatal, "a denied paired correction leaves live memory available");
        if (d.state().fatal) return;
        expected = aotx_retain_store(); auto app = d.appraisal();
        auto old = (const aotx_cognitive_store *)before.data(), now = (const aotx_cognitive_store *)expected.data();
        aotx_check(!memcmp(old->objects, now->objects, old->count * AOTX_COG_OBJECT) &&
            !memcmp(old->payload, now->payload, old->bytes), "paired correction preserves all prior object and payload bytes");
        aotx_check(!app.active && app.completed == (protect ? 0 : n) && app.interrupted == (protect ? n : 0) &&
            app.last_status == (protect ? AOTX_COG_DENIED : AOTX_COG_OK), "one protected pair denies the complete batch without partial updates");
        aotx_check(now->count == old->count + n * (protect ? 1 : 3), "denial writes only queue versions and no derived evidence");
        for (unsigned i = 0; i < n; ++i) {
            unsigned at = old->count + i * (protect ? 1 : 3);
            const auto *queue = now->objects[at], *qp = now->payload + aotx_get(queue + AOTX_CO_OFFSET);
            aotx_check(aotx_get(queue + AOTX_CO_ID) == aotx_ar_id(i, 14) && aotx_get(queue + AOTX_CO_VERSION) == 2 &&
                aotx_get(qp + 12, 4) == (protect ? AOTX_APPRAISAL_INTERRUPTED : AOTX_APPRAISAL_COMPLETE),
                "each distinct source retains its exact queue and terminal state");
            if (!protect) {
                const auto *a = now->objects[at + 1], *r = now->objects[at + 2];
                aotx_check(aotx_get(a + AOTX_CO_SUPERSEDES) == aotx_ar_id(i, 6) &&
                    aotx_get(r + AOTX_CO_SUPERSEDES) == aotx_ar_id(i, 7), "allowed correction supersedes the exact assessment and relationship pair");
                aotx_check(aotx_get(now->payload + aotx_get(a + AOTX_CO_OFFSET) + 4, 4) == 100000 + i,
                    "allowed paired corrections retain each source value");
            }
        }
    }
    {
        aotx_appraisal_device d(n); d.process(start, true); d.process(request, true); d.process(decision, true);
        aotx_check(!d.state().fatal && !d.appraisal().calls && aotx_retain_store() == expected,
            "paired correction completion and denial recover exactly without generation");
    }
}

static void aotx_appraisal_manual(unsigned n) {
    aotx_live_records start, decision; aotx_bytes expected;
    {
        aotx_appraisal_device d(n); aotx_fixture empty;
        start = d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1);
        aotx_appraisal_join(start, d.send(aotx_appraisal_config_bytes(), 15));
        aotx_appraisal_join(start, d.send(aotx_live_binding_bytes(n, 1), 3));
        aotx_appraisal_join(start, d.send(aotx_retain_query(n, 1, 1, false, 0, 3, 80), 4));
        d.idle(n);
        auto request = aotx_retain_bytes(n, 1, 1);
        for (unsigned i = 1; i < n; i += 2) memset(request.data() + 64 + i * 160 + 112, 0, 16);
        decision = d.send(request, 8); expected = aotx_retain_store();
        auto s = (const aotx_cognitive_store *)expected.data();
        aotx_check(!d.state().status && s->count == 1 + 3 * n + (n + 1) / 2,
            "explicit retention adds queues only for sources with an admitted subject");
        for (unsigned j = 0; j < (n + 1) / 2; ++j) {
            auto q = s->objects[1 + 3 * n + j], source = s->objects[1 + 6 * j];
            aotx_check(!memcmp(q + AOTX_CO_SOURCE, source + AOTX_CO_ID, 16) &&
                !memcmp(q + AOTX_CO_SUBJECT, source + AOTX_CO_SUBJECT, 16),
                "sparse explicit queues preserve the correct retained source row");
        }
    }
    {
        aotx_appraisal_device d(n); d.process(start, true); d.idle(n); d.process(decision, true);
        aotx_check(!d.state().fatal && aotx_retain_store() == expected,
            "explicit source and sparse queue admission recover exactly");
    }
}
static void aotx_appraisal_roundtrip(unsigned n, unsigned scope) {
    aotx_live_records start, request, decision; aotx_bytes expected, input, result;
    std::vector<aotx_live_binding> bindings;
    {
        aotx_appraisal_device d(n); start = aotx_appraisal_sources(d, n, scope);
        bindings = d.bindings(n); input = aotx_appraisal_work_request(n);
        request = d.process(aotx_live_parts(input, 16, d.next_id++), false, false);
        aotx_check(d.appraisal().active && d.state().phase == AOTX_INTAKE_RUN, "request reaches the controlled decoder boundary");
        auto before = aotx_retain_store(); aotx_appraisal_output(n);
        aotx_live_decide<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_check(d.state().phase == AOTX_LIVE_WRITE && aotx_retain_store() == before,
            "prepared results publish no interpretation before recording");
        decision = d.process({}, false, false);
        if (d.state().phase == AOTX_LIVE_WRITE)
            aotx_check(aotx_retain_store() == before, "partial result recording publishes no interpretation");
        aotx_appraisal_join(decision, d.process({})); expected = aotx_retain_store();
        result = aotx_retain_result(decision, 17);
        auto s = (const aotx_cognitive_store *)expected.data(); auto state = d.appraisal();
        aotx_check(!d.state().fatal && !state.last_status && !state.active && state.completed == n,
            "complete recorded result publishes the whole source batch");
        aotx_check(s->count == 1 + n * 7 && s->sequence == 1 + n * 7, "each source gains one completed queue, assessment and relationship");
        auto after = d.bindings(n);
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(!memcmp(&after[i], &bindings[i], sizeof(after[i])), "internal work changes no user conversation or selection");
            auto queue = s->objects[1 + n * 4 + i * 3], assessment = s->objects[2 + n * 4 + i * 3];
            auto relation = s->objects[3 + n * 4 + i * 3], event = s->objects[1 + i * 3];
            auto qp = s->payload + aotx_get(queue + AOTX_CO_OFFSET);
            auto ap = s->payload + aotx_get(assessment + AOTX_CO_OFFSET), rp = s->payload + aotx_get(relation + AOTX_CO_OFFSET);
            aotx_check(aotx_get(qp + 12, 4) == 1 && aotx_get(queue + AOTX_CO_VERSION) == 2, "one queue version completes one source");
            aotx_check(aotx_get(ap + 4, 4) == 100000 + i && aotx_get(ap + 8, 4) == 800000 - i,
                "distinct source values retain positive and negative dimensions");
            aotx_check(aotx_get(rp + 12, 4) == 1 && aotx_get(rp + 24, 4) == UINT32_MAX && aotx_get(rp + 28, 4) == UINT32_MAX,
                "each external source contributes one exposure and no unsupported task trust");
            aotx_check(!memcmp(assessment + AOTX_CO_SOURCE, event + AOTX_CO_ID, 16) &&
                !memcmp(relation + AOTX_CO_SUBJECT, event + AOTX_CO_SUBJECT, 16), "accepted evidence retains its exact event and actor");
        }
        aotx_put(input.data() + 16, s->sequence);
        d.process(aotx_live_parts(input, 16, d.next_id++));
        aotx_check(!d.appraisal().active && d.state().status && aotx_retain_store() == expected,
            "a completed queue cannot create a second exposure");
    }
    {
        aotx_appraisal_device d(n); d.process(start, true); d.idle(n); d.process(request, true);
        aotx_check(d.appraisal().active && d.state().phase == AOTX_LIVE_WAIT, "recovery waits for its recorded result without a decoder call");
        d.process(decision, true);
        aotx_check(!d.state().fatal && !d.appraisal().calls && aotx_retain_store() == expected,
            "ordinary recovery reproduces every store byte with zero appraisal generation");
    }
    if (scope) return;
    for (unsigned mode = 0; mode < 8; ++mode) {
        aotx_appraisal_device d(n); d.process(start, true); d.idle(n); d.process(request, true);
        auto before = aotx_retain_store(); auto changed = result;
        unsigned row = 64 + (n - 1) * AOTX_APPRAISAL_RESULT_ROW;
        if (mode < 6) {
            unsigned offsets[] = {0, 16, 24, 56, 64, 4095}; changed[row + offsets[mode]] ^= 1;
        } else if (mode == 6) changed.back() ^= 1;
        else changed[32] = 1;
        auto parts = aotx_live_parts(changed, 17, 5);
        d.process(parts, true);
        aotx_check(d.state().fatal && aotx_retain_store() == before, "altered source, version, model, output or accepted tail cannot publish");
    }
    {
        aotx_appraisal_device d(n); d.process(start, true); d.idle(n); d.process(request, true);
        auto partial = decision; partial.resize(1);
        d.process(partial, true); auto before = aotx_retain_store();
        unsigned *out; AOTX_CUDA(cudaMalloc(&out, 4)); aotx_live_test_end<<<1,1>>>(out); AOTX_CUDA(cudaDeviceSynchronize()); cudaFree(out);
        aotx_check(!d.state().fatal, "raw recovery permits an interrupted internal result");
        auto interrupted = d.process({}); auto saved = aotx_retain_store();
        aotx_check(!d.appraisal().active && d.appraisal().interrupted == n && saved != before,
            "raw recovery records interrupted queues without publishing partial interpretations");
        aotx_appraisal_device again(n); again.process(start, true); again.idle(n); again.process(request, true);
        again.process(partial, true); again.process(interrupted, true);
        aotx_check(!again.state().fatal && aotx_retain_store() == saved && !again.appraisal().calls,
            "a second recovery accepts the exact interruption marker after old partial bytes");
    }
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) {
        aotx_appraisal_manual(n);
        for (unsigned scope = 0; scope < 3; ++scope) aotx_appraisal_roundtrip(n, scope);
        aotx_appraisal_paired_correction(n, false);
        aotx_appraisal_paired_correction(n, true);
    }
    printf("appraisal work: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures || aotx_checks < 1000 ? 1 : 0;
}
