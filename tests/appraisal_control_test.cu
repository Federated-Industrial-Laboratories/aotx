/* Purpose: Verify staged controls, eligible work and bounded cancellation of appraisal.
 * Owns: Distinct operator values, retained sources and exact lease expectations.
 * Launch shape: Real CUDA admission and commit nodes at one and 64 source rows.
 * Lifetime: One maintained test process without model weights or forward execution. */
#include "appraisal_control_fixture.h"

static void aotx_control_settings(unsigned n) {
    aotx_appraisal_device d(n); aotx_control_reset(); aotx_retain_open(d, n);
    for (unsigned i = 0; i < n; ++i) {
        auto before = aotx_retain_store(); uint64_t accepted = d.state().accepted;
        aotx_control_line("appraisal on");
        aotx_control_line("appraisal limits " + std::to_string(160 + i) + " " + std::to_string(512 + i) +
            " " + std::to_string(1600 + i) + " " + std::to_string(1 + i));
        aotx_control_line("appraisal priority " + std::to_string(100 + i) + " " + std::to_string(200 + i));
        auto staged = d.appraisal();
        aotx_check(staged.control_pending && aotx_get(staged.control + 12, 4) == 3 &&
            aotx_retain_store() == before && d.state().accepted == accepted,
            "CLI changes remain staged until one recorded configuration is admitted");
        auto good = staged;
        aotx_control_line("appraisal limits 0 1 1 1"); aotx_control_line("appraisal limits 1 4097 1 1");
        aotx_control_line("appraisal priority 1000001 0"); aotx_control_line("appraisal limits 42949672960 1 1 1");
        aotx_control_line("appraisal background maybe"); staged = d.appraisal();
        aotx_check(!memcmp(staged.control, good.control, sizeof(staged.control)), "invalid commands cannot alter the staged configuration");
        for (unsigned guard : {1u, 2u, 6u}) {
            aotx_control_guard<<<1,1>>>(guard, n); d.process({}, false, false);
            aotx_check(d.appraisal().control_pending && aotx_retain_store() == before && d.state().accepted == accepted,
                "held, foreground and incomplete policy work postpone configuration admission");
        }
        aotx_control_guard<<<1,1>>>(0, n); auto recorded = d.process({});
        auto payload = aotx_retain_result(recorded, AOTX_APPRAISAL_CONTROL);
        aotx_check(payload.size() == 96 && !memcmp(payload.data(), good.control, 96), "one operation records the exact combined CLI settings");
        auto current = aotx_control_read();
        aotx_check(!d.appraisal().control_pending && current.flags == 3 && !current.enabled && !current.pending,
            "writes and recall do not enable background work");
        auto saved = aotx_retain_store(); auto store = (const aotx_cognitive_store *)saved.data();
        auto row = store->objects[current.current], p = store->payload + aotx_get(row + AOTX_CO_OFFSET);
        aotx_check(aotx_get(row + AOTX_CO_VERSION) == i + 1 && aotx_get(p + 16, 4) == 160 + i &&
            aotx_get(p + 20, 4) == 512 + i && aotx_get(p + 24, 4) == 1600 + i && aotx_get(p + 36, 4) == i + 1 &&
            aotx_get(p + 28, 4) == 100 + i && aotx_get(p + 32, 4) == 200 + i,
            "each configuration version keeps its distinct limits and recall priorities");
    }
    aotx_control_line("appraisal background on"); d.process({});
    aotx_check(aotx_control_read().enabled, "background can be selected independently");
    aotx_control_line("appraisal off"); d.process({}); auto disabled = aotx_control_read();
    aotx_check(!disabled.flags && !disabled.enabled && !disabled.pending, "current disabled configuration cannot fall back to an old enabled version");
    aotx_control_mutate<<<1,1>>>(9, n); disabled = aotx_control_read();
    aotx_check(disabled.current == UINT32_MAX && !disabled.flags && !disabled.enabled,
        "unavailable current configuration cannot revive an older enabled version");
}
static void aotx_control_callbacks(unsigned n) {
    aotx_appraisal_device d(n); aotx_control_reset(); aotx_appraisal_sources(d, n);
    auto before = aotx_retain_store(); auto first = aotx_control_read();
    aotx_check(d.appraisal().pending == n && !first.pending && !first.enabled, "retained pending sources do not imply enabled background work");
    for (unsigned i = 0; i < 4; ++i) d.process({});
    aotx_check(!d.appraisal().calls && !d.appraisal().active && aotx_retain_store() == before, "background off causes no request or memory write");
    aotx_control_policy<<<1,1>>>(1, 0); aotx_control_line("appraisal background on");
    aotx_check(!d.appraisal().control_pending && d.appraisal().last_status == AOTX_COG_LAYOUT && aotx_retain_store() == before,
        "an ABI 1 creator refuses background appraisal without a policy fallback");
    aotx_control_policy<<<1,1>>>(2, 0); aotx_control_line("appraisal background on");
    aotx_control_guard<<<1,1>>>(4, n); d.process({}); aotx_control_guard<<<1,1>>>(0, n);
    auto enabled = aotx_control_read(); aotx_check(enabled.pending == n && enabled.enabled, "current source queues become eligible when background is enabled");
    for (unsigned change = 0; change < 6; ++change) {
        aotx_control_policy<<<1,1>>>(2, change); auto out = aotx_control_read(1);
        aotx_check(out.taken[0] == (change == 0) && !out.taken[1], "a supplied ABI 2 proposal consumes once and refuses stale source, root, size or work");
    }
    aotx_control_policy<<<1,1>>>(1, 0); aotx_check(!aotx_control_read(1).taken[0], "ABI 1 cannot consume eligible appraisal work");
    aotx_control_policy<<<1,1>>>(2, 0); auto prior = aotx_control_read();
    aotx_control_mutate<<<1,1>>>(1, n); auto moved = aotx_control_read(1);
    aotx_check(moved.sequence == prior.sequence && moved.root == prior.root && moved.revision == prior.revision + 1 &&
        moved.passes == prior.passes + 1 && moved.pending == n && !moved.taken[0],
        "a completed maintenance attempt changes work observation and invalidates the old proposal");
    aotx_control_policy<<<1,1>>>(2, 0); aotx_check(aotx_control_read(1).taken[0], "a proposal for the new maintenance observation can consume current work");
    aotx_control_line("appraisal off"); d.process({});
    aotx_control_policy<<<1,1>>>(2, 0); auto quiet = aotx_control_read(1);
    aotx_check(!quiet.enabled && !quiet.pending && !quiet.taken[0] && !d.appraisal().calls,
        "current disabled configuration suppresses old queues and accepted proposals");
}
static void aotx_control_explicit(unsigned n) {
    aotx_appraisal_device d(n); aotx_control_reset(); aotx_appraisal_sources(d, n);
    aotx_control_metadata<<<1,AOTX_SLOTS>>>(n, 0);
    auto before = aotx_retain_store(); auto bindings = d.bindings(n); auto user = aotx_control_read();
    aotx_control_line("appraisal run");
    aotx_control_guard<<<1,1>>>(1, n); d.process({}, false, false);
    aotx_check(d.appraisal().explicit_pending && !d.appraisal().active && aotx_retain_store() == before,
        "an explicit operator run remains pending while the scheduler holds");
    aotx_control_guard<<<1,1>>>(2, n); d.process({}, false, false);
    aotx_check(d.appraisal().explicit_pending && !d.appraisal().active, "an explicit run waits for foreground work to become quiet");
    aotx_control_guard<<<1,1>>>(0, n); auto records = d.process({}, false, false);
    auto request = aotx_retain_result(records, AOTX_APPRAISAL_REQUEST); auto state = d.appraisal();
    aotx_check(state.active && state.count == n && !state.explicit_pending && state.calls == 1 && !state.background,
        "an explicit run admits all available queues with background disabled");
    aotx_check(request.size() == 64 + n * 32 && !aotx_get(request.data() + 48, 4), "the recorded batch is explicitly foreground work");
    auto view = aotx_control_read();
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(view.owned[i] == i + 1 && view.row_state[i] == 1 && view.wanted[i] && !view.row_status[i],
            "each admitted source owns a separate internal prompt lease");
        aotx_check(state.rows[i].queue == 1 + 3 * n + i && state.rows[i].source == 1 + 3 * i && state.rows[i].slot == i,
            "batch admission preserves each exact source, queue and free slot");
        aotx_check(view.turn[i] == user.turn[i], "internal prompt admission creates no external agent turn");
    }
    auto after = d.bindings(n); aotx_check(!memcmp(after.data(), bindings.data(), n * sizeof(bindings[0])) && aotx_retain_store() == before,
        "explicit work changes no user conversation or source memory before completion");
    aotx_control_guard<<<1,1>>>(4, n); aotx_intake_step<<<1,AOTX_SLOTS>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    aotx_control_finish(d, before); aotx_check(d.appraisal().interrupted == n, "pre-tokenization pause records one interruption per queued source");
}
static void aotx_control_cancel(unsigned n, unsigned guard) {
    aotx_appraisal_device d(n); aotx_control_reset(); aotx_appraisal_sources(d, n);
    aotx_control_metadata<<<1,AOTX_SLOTS>>>(n, 0);
    d.process(aotx_live_parts(aotx_appraisal_work_request(n), 16, d.next_id++), false, false);
    aotx_check(d.appraisal().active && !d.state().status, "real admission reaches the internal prompt boundary");
    aotx_control_metadata<<<1,AOTX_SLOTS>>>(n, 1);
    auto before = aotx_retain_store(); auto bindings = d.bindings(n); auto user = aotx_control_read();
    uint64_t journal = d.seam().dev.tail;
    if (guard == 1 || guard == 7) aotx_control_line("appraisal off");
    aotx_control_guard<<<1,1>>>(guard == 7 ? 1 : guard, n);
    if (guard == 7) {
        auto prior = aotx_control_read(); journal = d.seam().dev.tail;
        aotx_intake_step<<<1,AOTX_SLOTS>>>(); aotx_decode_commit<<<1,AOTX_SLOTS>>>(1);
        auto held = aotx_control_read();
        aotx_check(!memcmp(held.ticks, prior.ticks, sizeof(held.ticks)) && !held.status && d.seam().dev.tail == journal,
            "scheduler hold suspends decode progress and lease cleanup");
        aotx_control_guard<<<1,1>>>(0, n);
    }
    if (guard == 1) aotx_control_guard<<<1,1>>>(0, n);
    journal = d.seam().dev.tail;
    aotx_intake_step<<<1,AOTX_SLOTS>>>(); auto stop = aotx_control_read();
    for (unsigned i = 0; i < n; ++i)
        aotx_check(stop.row_status[i] == AOTX_COG_DENIED && (stop.sequence_flags[i] & AOTX_DECODE_MARK_STOP) && stop.owned[i] == i + 1,
            "pending controls or foreground work stop each active lease at the next intake boundary");
    aotx_decode_commit<<<1,AOTX_SLOTS>>>(1);
    if (guard == 5) aotx_control_mutate<<<1,1>>>(3, n);
    aotx_intake_step<<<1,AOTX_SLOTS>>>(); auto released = aotx_control_read();
    if (guard == 5) {
        aotx_check(released.owned[n - 1] && released.sequence_state[n - 1] == AOTX_SEQ_STATE_DONE,
            "a full cache request queue retains the lease until release can be recorded");
        aotx_control_mutate<<<1,1>>>(4, n); aotx_intake_step<<<1,AOTX_SLOTS>>>(); released = aotx_control_read();
    }
    aotx_check(released.phase == AOTX_INTAKE_DONE && !released.live && d.seam().dev.tail == journal && aotx_retain_store() == before,
        "bounded decode cleanup emits no external token or partial interpretation");
    for (unsigned i = 0; i < n; ++i)
        aotx_check(!released.owned[i] && !released.pages[i] && !released.wanted[i] && released.row_state[i] == 4 &&
            released.sequence_state[i] == AOTX_SEQ_STATE_FREE && released.released[i] == 1 && released.turn[i] == user.turn[i],
            "each cancelled slot releases its page count and sequence exactly once");
    aotx_control_guard<<<1,1>>>(0, n); aotx_control_finish(d, before);
    auto saved = aotx_retain_store(); auto store = (const aotx_cognitive_store *)saved.data();
    aotx_check(d.appraisal().interrupted == n && store->count == 1 + n * 5,
        "a cancelled batch adds only interrupted queue versions and no inferred evidence");
    auto after = d.bindings(n); aotx_check(!memcmp(after.data(), bindings.data(), n * sizeof(bindings[0])), "cancelled work leaves user conversations unchanged");
}
static void aotx_control_capacity(unsigned n) {
    aotx_appraisal_device d(n); aotx_control_reset(); aotx_appraisal_sources(d, n);
    aotx_control_line("appraisal background on"); d.process({});
    aotx_control_line("appraisal run"); d.process({}, false, false);
    aotx_check(d.appraisal().active, "explicit admission reaches the controlled result boundary");
    auto saved = aotx_retain_store(); uint32_t bytes = ((const aotx_cognitive_store *)saved.data())->bytes;
    aotx_control_mutate<<<1,1>>>(2, n, AOTX_COG_PAYLOAD); AOTX_CUDA(cudaDeviceSynchronize());
    auto full = aotx_retain_store(); aotx_appraisal_output(n, AOTX_COG_CAPACITY);
    auto records = d.process({}); auto result = aotx_retain_result(records, AOTX_APPRAISAL_RESULT);
    aotx_check(result.size() == 64 && !aotx_get(result.data() + 12, 4) && !aotx_get(result.data() + 24),
        "capacity refusal records one complete result without an object tail");
    aotx_check(d.appraisal().last_status == AOTX_COG_CAPACITY && aotx_retain_store() == full && !aotx_control_read().pending,
        "result capacity refusal suppresses retries against the unchanged memory footprint");
    uint64_t calls = d.appraisal().calls;
    for (unsigned i = 0; i < 4; ++i) d.process({});
    aotx_check(d.appraisal().calls == calls && !d.appraisal().active, "unchanged result capacity cannot produce a repeated work loop");
    aotx_control_mutate<<<1,1>>>(1, n); aotx_check(!aotx_control_read().pending, "maintenance without reclaimed room does not bypass result capacity suppression");
    aotx_control_mutate<<<1,1>>>(2, n, bytes); aotx_check(aotx_control_read().pending == n, "reclaimed payload space permits the pending source batch again");
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) {
        aotx_control_settings(n); aotx_control_callbacks(n); aotx_control_explicit(n);
        for (unsigned guard : {1u, 2u, 3u, 4u, 5u, 7u}) aotx_control_cancel(n, guard);
        aotx_control_capacity(n);
    }
    printf("appraisal controls: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures || aotx_checks < 1000 ? 1 : 0;
}
