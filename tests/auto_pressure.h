/* Purpose: Verify automatic retention refusal, focus pressure and ID allocation.
 * Owns: Admitted pressure stores and exact no-publication expectations.
 * Launch shape: One and 64 distinct input rows through live device kernels.
 * Lifetime: One bounded test process. */
#ifndef AOTX_TEST_AUTO_PRESSURE_H
#define AOTX_TEST_AUTO_PRESSURE_H
#include "retain_pressure.h"

static std::vector<uint32_t> aotx_auto_queues(unsigned n) {
    std::vector<uint32_t> values(n);
    for (unsigned i = 0; i < n; ++i) AOTX_CUDA(cudaMemcpyFromSymbol(&values[i], aotx_agent_gear,
        sizeof(values[i]), i * sizeof(aotx_agent_work) + offsetof(aotx_agent_work, has_message)));
    return values;
}
static void aotx_auto_refused(aotx_live_device &d, const aotx_bytes &query, unsigned n, unsigned status) {
    auto store = aotx_retain_store(); auto bindings = d.bindings(n); auto queues = aotx_auto_queues(n);
    auto records = d.send(query, 4); auto result = aotx_retain_result(records, 10);
    if (d.state().status != status) printf("automatic refusal: expected %u, received %u\n", status, d.state().status);
    aotx_check(d.state().status == status && d.state().phase == AOTX_LIVE_IDLE, "automatic batch reports the exact refusal");
    aotx_check(store == aotx_retain_store() && queues == aotx_auto_queues(n), "refusal preserves store and every input queue");
    aotx_auto_same(bindings, d.bindings(n));
    aotx_check(result.size() == 64 && aotx_get(result.data() + 44, 4) == status &&
        !aotx_get(result.data() + 8, 4) && !aotx_get(result.data() + 48), "refusal records no accepted rows or tail");
}
static void aotx_auto_pressure(unsigned n) {
    for (unsigned mode = 0; mode < 5; ++mode) {
        aotx_fixture f; uint64_t cut = 0, tick = 5;
        if (mode < 2) {
            unsigned count = mode ? 1 : AOTX_COG_OBJECTS + 1 - 3 * n;
            for (unsigned i = 0; i < count; ++i) f.add(aotx_memory_row(i, AOTX_COG_COMPONENT, 600000 + i, i + 1, 2),
                aotx_bytes(mode ? AOTX_COG_PAYLOAD - 64 : 1, 0x41));
            cut = count;
        } else if (mode == 2) {
            f.add(aotx_memory_row(n - 1, AOTX_COG_EVENT, 100064 + n - 1, 1), aotx_memory_text("existing event")); cut = 1;
        } else if (mode == 3) tick = UINT64_MAX;
        else cut = UINT64_MAX - 3 * n + 1;
        aotx_live_device d(n); d.send(aotx_live_load_bytes(f.wire(false, cut, tick)), 1);
        aotx_check(!d.state().status && d.state().ready, "pressure checkpoint is valid");
        d.send(aotx_auto_bind(n, cut), 3); aotx_check(!d.state().status, "pressure bindings are valid");
        aotx_auto_refused(d, aotx_retain_query(n, cut, 1), n, mode == 2 ? AOTX_COG_VERSION : AOTX_COG_CAPACITY);
    }
    for (unsigned mode = 0; mode < 4; ++mode) {
        aotx_fixture f;
        for (unsigned j = 0; j < 8; ++j) {
            auto row = aotx_memory_row(0, AOTX_COG_WORKING, 500000 + j, j + 1, 2);
            aotx_put(row.data() + AOTX_CO_RETENTION, mode < 3 ? mode : 0, 4);
            if (mode == 3) aotx_put(row.data() + AOTX_CO_FLAGS, AOTX_COG_PROTECTED, 4);
            f.add(row, aotx_memory_text("focus " + std::to_string(j)));
        }
        aotx_live_device d(n); d.send(aotx_live_load_bytes(f.wire(false, 8)), 1); d.send(aotx_auto_bind(n, 8, 2), 3);
        aotx_check(!d.state().status, "focus checkpoint and bindings are valid");
        aotx_retain_seed_focus<<<1,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize());
        auto query = aotx_retain_query(n, 8, 1, false, 2);
        if (mode) { aotx_auto_refused(d, query, n, AOTX_COG_CAPACITY); continue; }
        auto result = aotx_retain_result(d.send(query, 4), 10);
        aotx_check(!d.state().status, "ordinary focus permits automatic input");
        if (d.state().status) continue;
        auto b = d.bindings(n);
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(b[i].focus_count == 8, "replacement keeps eight focus references");
            for (unsigned j = 0; j < 7; ++j)
                aotx_check(aotx_get(b[i].focus[j]) == 500000 + (i + j + 1) % 8, "oldest ordinary reference leaves this binding");
            aotx_check(!memcmp(b[i].focus[7], result.data() + 64 + i * AOTX_LIVE_AUTO_ROW + AOTX_LIVE_TEXT_CHOICE_ROW + 48, 16),
                "new reference enters the correct binding last");
        }
        auto bytes = aotx_retain_store();
        aotx_check(((const aotx_cognitive_store *)bytes.data())->count == 8 + 3 * n, "focus replacement keeps stored history");
    }
}
__global__ void aotx_auto_busy(unsigned slot) { aotx_agent_gear[slot].has_message = 1; }
static void aotx_auto_admission(unsigned n) {
    {
        aotx_live_device d(n); aotx_fixture f; d.send(aotx_live_load_bytes(f.wire(false, 0)), 1);
        auto bad = aotx_auto_bind(n, 0); aotx_put(bad.data() + 124 + (n - 1) * 64, 2, 4);
        auto before = d.bindings(n); d.send(bad, 3);
        aotx_check(d.state().status == AOTX_COG_FORMAT, "unknown automatic mode refuses the bind batch");
        aotx_auto_same(before, d.bindings(n));
    }
    {
        unsigned slots = n == 1 ? 2 : n, inputs = slots - 1;
        aotx_live_device d(slots); aotx_fixture f; d.send(aotx_live_load_bytes(f.wire(false, 0)), 1);
        d.send(aotx_auto_bind(slots, 0), 3); aotx_auto_busy<<<1,1>>>(slots - 1); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_auto_refused(d, aotx_retain_query(inputs, 0, 1), slots, AOTX_COG_DENIED);
    }
    {
        auto f = aotx_memory_corpus(n); aotx_put(f.rows[2 * n - 1].data() + AOTX_CO_EXPIRY, 2 * n + 1);
        aotx_live_device d(n); d.send(aotx_live_load_bytes(f.wire(false, 2 * n)), 1); d.send(aotx_auto_bind(n, 2 * n), 3);
        aotx_check(!d.state().status, "expiry fixture and bindings are admitted");
        aotx_auto_refused(d, aotx_live_query_bytes(n, 2 * n, 1), n, AOTX_COG_DENIED);
        aotx_check(d.state().searches == n, "expiry refusal follows successful pre-write searches");
    }
    {
        aotx_fixture f; auto row = aotx_memory_row(0, AOTX_COG_COMPONENT, 600000, 1, 2);
        memcpy(row.data() + AOTX_CO_ID, "AOTXGEN1", 8); aotx_put(row.data() + AOTX_CO_ID + 8, 2);
        f.add(row, aotx_bytes(1, 0x41));
        aotx_live_device d(n); d.send(aotx_live_load_bytes(f.wire(false, 1)), 1); d.send(aotx_auto_bind(n, 1), 3);
        aotx_check(!d.state().status, "allocated-ID collision seed is admitted");
        auto q = aotx_retain_query(n, 1, 1); auto last = q.data() + 128 + (n - 1) * AOTX_LIVE_QUERY_ROW;
        memcpy(last, "AOTXGEN1", 8); aotx_put(last + 8, 3);
        auto result = aotx_retain_result(d.send(q, 4), 10);
        aotx_check(!d.state().status, "allocation skips stored IDs and every pending event ID");
        if (!d.state().status) for (unsigned i = 0; i < n; ++i) {
            auto out = result.data() + 64 + i * AOTX_LIVE_AUTO_ROW + AOTX_LIVE_TEXT_CHOICE_ROW;
            aotx_check(!memcmp(out + 48, "AOTXGEN1", 8) && aotx_get(out + 56) == 4 + 2 * i &&
                aotx_get(out + 72) == 5 + 2 * i, "generated references are distinct and skip both occupied candidates");
            aotx_check(!memcmp(out + 32, q.data() + 128 + i * AOTX_LIVE_QUERY_ROW, 16), "collision handling preserves the source event ID");
        }
    }
}
static void aotx_auto_turnover(unsigned n) {
    aotx_live_device d(n); aotx_fixture f; d.send(aotx_live_load_bytes(f.wire(false, 0)), 1); d.send(aotx_auto_bind(n, 0), 3);
    std::vector<std::vector<aotx_bytes>> references(n);
    for (unsigned turn = 1; turn <= 10; ++turn) {
        auto result = aotx_retain_result(d.send(aotx_retain_query(n, (turn - 1) * 3 * n, turn), 4), 10);
        aotx_check(!d.state().status, "automatic input remains usable beyond initial focus capacity");
        if (d.state().status) return;
        auto rows = d.bindings(n);
        for (unsigned i = 0; i < n; ++i) {
            auto out = result.data() + 64 + i * AOTX_LIVE_AUTO_ROW + AOTX_LIVE_TEXT_CHOICE_ROW;
            references[i].emplace_back(out + 48, out + 64); unsigned count = std::min(turn, 8u);
            aotx_check(rows[i].ordinal == turn && rows[i].focus_count == count, "focus capacity does not cap accepted turns");
            for (unsigned j = 0; j < count; ++j)
                aotx_check(!memcmp(rows[i].focus[j], references[i][turn - count + j].data(), 16), "each binding retains its ordered recent references");
        }
        d.idle(n);
    }
    auto bytes = aotx_retain_store(); auto s = (const aotx_cognitive_store *)bytes.data();
    aotx_check(s->count == 30 * n && s->sequence == 30 * n, "ten turns retain every immutable source object");
}
static void aotx_auto_text_failure(unsigned n) {
    for (unsigned failure : {1u, 3u}) {
        aotx_live_records start, records;
        {
            aotx_text_device d(n); auto f = aotx_text_corpus(n);
            start = d.send(aotx_live_load_bytes(f.wire(false, 2 * n)), 1);
            auto b = d.send(aotx_auto_bind(n, 2 * n), 3); start.insert(start.end(), b.begin(), b.end());
            auto store = aotx_retain_store(); auto bindings = d.bindings(n); auto queues = aotx_auto_queues(n);
            aotx_text_failure = failure; records = d.text(aotx_text_input(n, 2 * n));
            aotx_check(d.state().status && !d.state().searches, "encoder failure refuses automatic admission before recall");
            aotx_check(aotx_retain_store() == store && aotx_auto_queues(n) == queues, "encoder failure keeps the store and message queues");
            aotx_auto_same(bindings, d.bindings(n));
            auto r = aotx_retain_result(records, 10);
            aotx_check(r.size() == 64 && aotx_get(r.data() + 44, 4), "encoder failure records an automatic refusal");
        }
        {
            aotx_text_device d(n); d.process_text(start, true); auto store = aotx_retain_store(); auto bindings = d.bindings(n);
            d.process_text(records, true);
            aotx_check(!d.state().fatal && !d.encoded() && !d.state().searches, "refusal recovery does not encode or search");
            aotx_check(aotx_retain_store() == store, "refusal recovery creates no memory"); aotx_auto_same(bindings, d.bindings(n));
        }
    }
}
#endif
