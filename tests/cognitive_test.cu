/* Purpose: Check atomic typed state admission, replay and current reference access.
 * Owns: Distinct object batches and malformed semantic fixtures.
 * Launch shape: N=1 and N=64, with a separate full-capacity batch.
 * Lifetime: One test process; no model or base runtime is loaded. */
#include "cognitive_fixture.h"

static void aotx_state_cases(unsigned n) {
    aotx_device d;
    auto initial = aotx_initial(n), appraisal = aotx_appraisals(n), selection = aotx_selections(n);
    auto cp = initial.wire(false, n);
    aotx_check(d.load(cp).status == 0, "initial restore");
    aotx_check(d.checkpoint() == cp, "exact initial bytes");
    auto malformed = cp; malformed[96] = 1;
    d.rejects(malformed, false, AOTX_COG_FORMAT, "reserved header bytes");
    malformed = cp; aotx_put(malformed.data() + 24, UINT64_MAX);
    d.rejects(malformed, false, AOTX_COG_CAPACITY, "payload length overflow");
    malformed = cp; aotx_put(malformed.data() + 128 + AOTX_CO_OFFSET, UINT64_MAX);
    d.rejects(malformed, false, AOTX_COG_FORMAT, "object offset overflow");
    auto bad = appraisal; bad.rows.back()[AOTX_CO_SCOPE] = AOTX_COG_INSTANCE;
    bad.rows.back()[AOTX_CO_OWNER] ^= 3;
    if ((n - 1) % 3 == AOTX_COG_ROOM) memset(bad.rows.back().data() + AOTX_CO_ROOM, 0, 16);
    d.rejects(bad.wire(true, n + 1, 6), true, AOTX_COG_SCOPE, "private source cannot become shared");
    bad = appraisal; memset(bad.rows.back().data() + AOTX_CO_SOURCE, 0, 16);
    aotx_put(bad.rows.back().data() + AOTX_CO_SOURCE_VERSION, 0);
    d.rejects(bad.wire(true, n + 1, 6), true, AOTX_COG_SOURCE, "inferred source required");
    bad = appraisal; aotx_id(bad.rows.back().data() + AOTX_CO_SOURCE, 800000);
    d.rejects(bad.wire(true, n + 1, 6), true, AOTX_COG_REFERENCE, "missing source");
    bad = appraisal; aotx_id(bad.rows.back().data() + AOTX_CO_SOURCE, 101 + n - 1);
    d.rejects(bad.wire(true, n + 1, 6), true, AOTX_COG_REFERENCE, "cyclic source");
    bad = appraisal; aotx_put(bad.payloads.back().data() + 8, AOTX_COG_SCALE + 1, 4);
    d.rejects(bad.wire(true, n + 1, 6), true, AOTX_COG_FORMAT, "harm range");
    bad = appraisal; memset(bad.rows.back().data() + AOTX_CO_SUBJECT, 0, 16);
    d.rejects(bad.wire(true, n + 1, 6), true, AOTX_COG_SOURCE, "appraisal subject required");
    bad = appraisal; aotx_id(bad.rows.back().data() + AOTX_CO_SUBJECT, 9876);
    d.rejects(bad.wire(true, n + 1, 6), true, AOTX_COG_SOURCE, "wrong appraisal subject");
    auto tail = appraisal.wire(true, n + 1, 6);
    auto r = d.load(tail, true);
    aotx_check(r.status == 0 && r.applied == n && r.sequence == 2 * n, "apply appraisal batch");
    initial.append(appraisal);
    aotx_check(d.checkpoint() == initial.wire(false, 2 * n, 6), "mixed values and subject preserved");
    r = d.load(tail, true);
    aotx_check(!r.status && !r.applied && r.sequence == 2 * n, "covered tail is not repeated");
    malformed = tail; malformed.back() ^= 1;
    d.rejects(malformed, true, AOTX_COG_VERSION, "covered tail conflict");
    bad = selection; aotx_put(bad.rows.back().data() + AOTX_CO_UPDATED, 2 * n);
    d.rejects(bad.wire(true, 2 * n + 1, 7), true, AOTX_COG_SEQUENCE, "duplicate new sequence");
    bad = selection; aotx_put(bad.payloads.back().data() + 32, 2);
    d.rejects(bad.wire(true, 2 * n + 1, 7), true, AOTX_COG_REFERENCE, "selection exact version");
    bad = selection;
    for (unsigned i = 0; i < n; ++i) {
        aotx_id(bad.rows[i].data() + AOTX_CO_SOURCE, 101 + i);
        aotx_put(bad.rows[i].data() + AOTX_CO_SOURCE_VERSION, 1);
        aotx_put(bad.rows[i].data() + AOTX_CO_SOURCE_KIND, AOTX_COG_OBSERVED, 4);
    }
    d.rejects(bad.wire(true, 2 * n + 1, 7), true, AOTX_COG_SOURCE, "inferred source cannot become observation under new ID");
    auto covered = appraisal; covered.append(selection);
    r = d.load(covered.wire(true, n + 1, 7), true);
    aotx_check(!r.status && r.applied == n && r.sequence == 3 * n, "covered prefix and new selection batch");
    initial.append(selection);
    aotx_check(d.checkpoint() == initial.wire(false, 3 * n, 7), "recorded selection order exact");
    auto matches = d.resolve(n, 201, 1);
    for (unsigned i = 0; i < n; ++i)
        aotx_check(matches[i].status == AOTX_COG_OK && matches[i].version == 1, "selection reference access");
    matches = d.resolve(n, 201, 1, true);
    for (unsigned i = 0; i < n; ++i)
        aotx_check(matches[i].status == (i % 3 == 2 ? AOTX_COG_OK : AOTX_COG_DENIED), "wrong principal or room");
    auto revisions = aotx_initial(n);
    for (unsigned i = 0; i < n; ++i) {
        aotx_put(revisions.rows[i].data() + AOTX_CO_VERSION, 2);
        aotx_put(revisions.rows[i].data() + AOTX_CO_UPDATED, 3 * n + i + 1);
        revisions.payloads[i].push_back('!');
    }
    bad = revisions; aotx_put(bad.rows.back().data() + AOTX_CO_SOURCE_KIND, AOTX_COG_OBSERVED, 4);
    d.rejects(bad.wire(true, 3 * n + 1, 8), true, AOTX_COG_SOURCE, "reported state cannot become observed");
    bad = revisions; aotx_put(bad.rows.back().data() + AOTX_CO_VERSION, 3);
    d.rejects(bad.wire(true, 3 * n + 1, 8), true, AOTX_COG_VERSION, "version gap");
    bad = revisions; aotx_id(bad.rows.back().data() + AOTX_CO_SUBJECT, 7000);
    d.rejects(bad.wire(true, 3 * n + 1, 8), true, AOTX_COG_VERSION, "subject binding cannot change");
    auto gap = revisions.wire(true, 3 * n + 2, 8);
    d.rejects(gap, true, AOTX_COG_SEQUENCE, "sequence gap");
    r = d.load(revisions.wire(true, 3 * n + 1, 8), true);
    aotx_check(!r.status && r.applied == n, "subsequent mutation");
    initial.append(revisions);
    auto expected = initial.wire(false, 4 * n, 8);
    aotx_check(d.checkpoint() == expected, "all immutable revisions remain");
    matches = d.resolve(n, 201, 1);
    for (const auto &m : matches) aotx_check(m.status == AOTX_COG_STALE, "queued selection rechecks versions");
    matches = d.resolve(n, 1, 1);
    for (const auto &m : matches) aotx_check(m.status == AOTX_COG_STALE, "old version access refused");
    matches = d.resolve(n, 1, 2);
    for (const auto &m : matches) aotx_check(!m.status && m.version == 2, "current revision access");
    aotx_check(!d.load(expected).status && d.checkpoint() == expected, "restore historical selection after revision");
    printf("state N=%u complete\n", n);
}

static void aotx_restriction_cases(unsigned n) {
    aotx_device d;
    auto initial = aotx_initial(n), derived = aotx_appraisals(n);
    initial.append(derived);
    aotx_check(!d.load(initial.wire(false, 2 * n)).status, "source restriction setup");
    aotx_fixture changes;
    for (unsigned i = 0; i < n; ++i) {
        auto row = initial.rows[i];
        aotx_put(row.data() + AOTX_CO_VERSION, 2); aotx_put(row.data() + AOTX_CO_UPDATED, 2 * n + i + 1);
        aotx_put(row.data() + AOTX_CO_FLAGS, AOTX_COG_TOMBSTONE, 4); changes.add(row, {});
    }
    aotx_check(!d.load(changes.wire(true, 2 * n + 1, 6), true).status, "source tombstone batch");
    for (const auto &m : d.resolve(n, 101, 1)) aotx_check(m.status == AOTX_COG_DENIED, "derived source tombstone access");
    auto protected_state = aotx_initial(n);
    for (auto &row : protected_state.rows) aotx_put(row.data() + AOTX_CO_FLAGS, AOTX_COG_PROTECTED, 4);
    aotx_check(!d.load(protected_state.wire(false, n)).status, "protected state setup");
    for (unsigned i = 0; i < n; ++i) aotx_put(changes.rows[i].data() + AOTX_CO_UPDATED, n + i + 1);
    d.rejects(changes.wire(true, n + 1, 6), true, AOTX_COG_DENIED, "protected deletion refused");
    auto expired = aotx_initial(n);
    for (auto &row : expired.rows) aotx_put(row.data() + AOTX_CO_EXPIRY, n);
    aotx_check(!d.load(expired.wire(false, n)).status, "expired history retained");
    for (const auto &m : d.resolve(n, 1, 1)) aotx_check(m.status == AOTX_COG_DENIED, "expiry enforced");

    auto shared = aotx_initial(n), child = aotx_appraisals(n);
    for (unsigned i = 0; i < n; ++i) {
        aotx_put(shared.rows[i].data() + AOTX_CO_SCOPE, AOTX_COG_INSTANCE, 4);
        aotx_put(child.rows[i].data() + AOTX_CO_SCOPE, AOTX_COG_INSTANCE, 4);
        memset(shared.rows[i].data() + AOTX_CO_ROOM, 0, 16);
        memset(child.rows[i].data() + AOTX_CO_ROOM, 0, 16);
    }
    shared.append(child);
    aotx_check(!d.load(shared.wire(false, 2 * n)).status, "shared dependency setup");
    for (const auto &m : d.resolve(n, 101, 1, true)) aotx_check(!m.status, "shared child initially visible");
    auto narrow = aotx_initial(n);
    for (unsigned i = 0; i < n; ++i) {
        aotx_put(narrow.rows[i].data() + AOTX_CO_SCOPE, AOTX_COG_PRIVATE, 4);
        memset(narrow.rows[i].data() + AOTX_CO_ROOM, 0, 16);
        aotx_put(narrow.rows[i].data() + AOTX_CO_VERSION, 2);
        aotx_put(narrow.rows[i].data() + AOTX_CO_UPDATED, 2 * n + i + 1);
    }
    aotx_check(!d.load(narrow.wire(true, 2 * n + 1, 6), true).status, "source scope narrows");
    for (const auto &m : d.resolve(n, 101, 1, true)) aotx_check(m.status == AOTX_COG_DENIED, "scope narrowing restricts child");
    for (const auto &m : d.resolve(n, 101, 1)) aotx_check(!m.status, "source owner retains access");

    auto original = aotx_initial(n), app = aotx_appraisals(n), selected = aotx_selections(n);
    original.append(app); original.append(selected);
    aotx_check(!d.load(original.wire(false, 3 * n)).status, "supersession setup");
    aotx_fixture replacements;
    for (unsigned i = 0; i < n; ++i) {
        auto row = aotx_object(i, AOTX_COG_ASSERTION, 401 + i, 3 * n + i + 1);
        aotx_id(row.data() + AOTX_CO_SUPERSEDES, i + 1); aotx_put(row.data() + AOTX_CO_SUPER_VERSION, 1);
        replacements.add(row, {'n', 'e', 'w', (unsigned char)i});
    }
    aotx_check(!d.load(replacements.wire(true, 3 * n + 1, 6), true).status, "new ID supersedes old assertion");
    for (const auto &m : d.resolve(n, 201, 1)) aotx_check(m.status == AOTX_COG_STALE, "superseded queued selection");
    for (const auto &m : d.resolve(n, 1, 1)) aotx_check(m.status == AOTX_COG_STALE, "superseded direct reference");
}

static aotx_bytes aotx_media_payload(unsigned i) {
    aotx_bytes p(AOTX_COG_MEDIA_HEADER + 16 + 16, 0);
    aotx_put(p.data(), 1, 4); aotx_put(p.data() + 4, AOTX_COG_FEATURE_MEDIA, 4);
    aotx_put(p.data() + 8, AOTX_COG_EXACT_FEATURES, 4); aotx_put(p.data() + 12, AOTX_COG_F32, 4);
    aotx_put(p.data() + 16, 2); aotx_put(p.data() + 24, 2); aotx_put(p.data() + 48, 16);
    aotx_put(p.data() + 56, 2); aotx_put(p.data() + 64, AOTX_COG_TEMPORAL, 4);
    aotx_put(p.data() + 68, 2, 4);
    for (unsigned j = 72; j < 168; ++j) p[j] = (unsigned char)(j + i);
    aotx_put(p.data() + 192, 0x3f000000u + i, 4); aotx_put(p.data() + 196, 0xbf000000u + i, 4);
    aotx_put(p.data() + 200, 0x3f800000u + i, 4); aotx_put(p.data() + 204, 0x3e000000u + i, 4);
    aotx_put(p.data() + 208, 100 + i); aotx_put(p.data() + 216, 200 + i);
    return p;
}
static void aotx_media_cases(unsigned n) {
    aotx_device d;
    aotx_fixture f;
    for (unsigned i = 0; i < n; ++i) f.add(aotx_object(i, AOTX_COG_MEDIA, i + 1, i + 1), aotx_media_payload(i));
    auto cp = f.wire(false, n);
    aotx_check(!d.load(cp).status && d.checkpoint() == cp, "exact feature and position bytes");
    auto bad = f; memset(bad.payloads.back().data() + 104, 0, 32);
    d.rejects(bad.wire(false, n), false, AOTX_COG_LAYOUT, "missing model identity");
    bad = f; memset(bad.payloads.back().data() + 136, 0, 32);
    d.rejects(bad.wire(false, n), false, AOTX_COG_LAYOUT, "missing processor identity");
    bad = f; aotx_put(bad.payloads.back().data() + 64, 0, 4);
    d.rejects(bad.wire(false, n), false, AOTX_COG_LAYOUT, "missing positional layout");
    bad = f; aotx_put(bad.payloads.back().data() + 16, UINT64_MAX);
    d.rejects(bad.wire(false, n), false, AOTX_COG_LAYOUT, "shape multiplication overflow");
    bad = f; aotx_put(bad.payloads.back().data() + 192, 0x7fc00000u, 4);
    d.rejects(bad.wire(false, n), false, AOTX_COG_LAYOUT, "nonfinite feature");
    for (unsigned i = 0; i < n; ++i) {
        aotx_bytes p(192 + 6, 0);
        aotx_put(p.data(), 1, 4); aotx_put(p.data() + 4, AOTX_COG_IMAGE_MEDIA, 4);
        aotx_put(p.data() + 8, AOTX_COG_SOURCE_BYTES, 4); aotx_put(p.data() + 12, AOTX_COG_U8, 4);
        aotx_put(p.data() + 16, 1); aotx_put(p.data() + 24, 2); aotx_put(p.data() + 32, 3);
        aotx_put(p.data() + 48, 6); aotx_put(p.data() + 64, AOTX_COG_SPATIAL, 4);
        aotx_put(p.data() + 68, 3, 4); p[72] = 1;
        for (unsigned j = 192; j < p.size(); ++j) p[j] = (unsigned char)(i + j);
        f.payloads[i] = p;
    }
    cp = f.wire(false, n);
    aotx_check(!d.load(cp).status && d.checkpoint() == cp, "raw RGB image bytes");
    for (unsigned i = 0; i < n; ++i) {
        auto &p = f.payloads[i]; p.resize(200, 0);
        aotx_put(p.data() + 4, AOTX_COG_AUDIO_MEDIA, 4); aotx_put(p.data() + 12, AOTX_COG_I16, 4);
        aotx_put(p.data() + 16, 2); aotx_put(p.data() + 24, 2); aotx_put(p.data() + 32, 0);
        aotx_put(p.data() + 48, 8); aotx_put(p.data() + 64, AOTX_COG_TEMPORAL, 4);
        aotx_put(p.data() + 68, 2, 4);
        aotx_put(p.data() + 168, 48000, 4);
    }
    cp = f.wire(false, n);
    aotx_check(!d.load(cp).status && d.checkpoint() == cp, "raw audio bytes");
}

static void aotx_capacity_cases() {
    aotx_device d;
    auto f = aotx_initial(AOTX_COG_OBJECTS);
    aotx_check(!d.load(f.wire(false, AOTX_COG_OBJECTS)).status, "object table at capacity");
    auto extra = aotx_initial(1);
    aotx_id(extra.rows[0].data() + AOTX_CO_ID, 9999);
    aotx_put(extra.rows[0].data() + AOTX_CO_CREATED, AOTX_COG_OBJECTS + 1); aotx_put(extra.rows[0].data() + AOTX_CO_UPDATED, AOTX_COG_OBJECTS + 1);
    d.rejects(extra.wire(true, AOTX_COG_OBJECTS + 1, 6), true, AOTX_COG_CAPACITY, "object cap refuses without offload");
    f = aotx_initial(1); f.payloads[0].resize(AOTX_COG_PAYLOAD, 37);
    aotx_check(!d.load(f.wire(false, 1)).status, "payload arena at capacity");
    aotx_put(extra.rows[0].data() + AOTX_CO_CREATED, 2); aotx_put(extra.rows[0].data() + AOTX_CO_UPDATED, 2);
    d.rejects(extra.wire(true, 2, 6), true, AOTX_COG_CAPACITY, "payload cap refuses without offload");
    aotx_cognitive_checkpoint<<<1, 64>>>(d.live, d.image, 128, d.result);
    aotx_check(d.finish().status == AOTX_COG_CAPACITY, "checkpoint output capacity");
}

static void aotx_tail_layout_cases(unsigned n) {
    aotx_device d;
    auto base = aotx_initial(n);
    aotx_check(!d.load(base.wire(false, n)).status, "tail layout setup");
    aotx_fixture f;
    for (unsigned i = 0; i < 2 * n; ++i) {
        auto row = aotx_object(i % n, AOTX_COG_ASSERTION, 10000 + i, n + i + 1);
        f.add(row, {(unsigned char)i, (unsigned char)(i + 1), (unsigned char)(i + 2), (unsigned char)(i + 3)});
    }
    auto tail = f.wire(true, n + 1, 6), unused = tail, overlap = tail;
    unused.push_back(0x5a);
    aotx_put(unused.data() + 24, aotx_get(unused.data() + 24) + 1);
    aotx_put(unused.data() + 80, unused.size());
    for (unsigned i = 0; i < n; ++i)
        aotx_put(overlap.data() + AOTX_COG_HEADER + (2 * i + 1) * AOTX_COG_OBJECT + AOTX_CO_OFFSET, 8 * i);
    d.rejects(unused, true, AOTX_COG_FORMAT, "tail cannot contain unreferenced bytes");
    d.rejects(overlap, true, AOTX_COG_FORMAT, "tail extents cannot overlap");
    auto result = d.load(tail, true);
    aotx_check(!result.status && result.applied == 2 * n && result.sequence == 3 * n, "canonical tail accepted");
    base.append(f);
    aotx_check(d.checkpoint() == base.wire(false, 3 * n, 6), "canonical tail exact state");
    d.rejects(unused, true, AOTX_COG_FORMAT, "covered tail cannot contain unreferenced bytes");
    d.rejects(overlap, true, AOTX_COG_FORMAT, "covered tail extents cannot overlap");
}

static void aotx_payload_batch_cases(unsigned n) {
    aotx_device d;
    auto base = aotx_initial(n);
    for (unsigned i = 0; i < n; ++i)
        base.payloads[i].assign((AOTX_COG_PAYLOAD - 4 * n) / n, (unsigned char)(37 + i));
    aotx_check(!d.load(base.wire(false, n)).status, "batched payload capacity setup");
    aotx_fixture exact, excess;
    for (unsigned i = 0; i < n; ++i) {
        auto row = aotx_object(i, AOTX_COG_ASSERTION, 10000 + i, n + i + 1);
        exact.add(row, {(unsigned char)i, (unsigned char)(i + 1), (unsigned char)(i + 2), (unsigned char)(i + 3)});
        excess.add(row, {(unsigned char)i, (unsigned char)(i + 1), (unsigned char)(i + 2),
                         (unsigned char)(i + 3), (unsigned char)(i + 4)});
    }
    d.rejects(excess.wire(true, n + 1, 6), true, AOTX_COG_CAPACITY, "aggregate payload capacity and rollback");
    auto result = d.load(exact.wire(true, n + 1, 6), true);
    aotx_check(!result.status && result.applied == n && result.sequence == 2 * n, "aggregate exact fit accepted");
    base.append(exact);
    aotx_check(d.checkpoint() == base.wire(false, 2 * n, 6), "aggregate exact fit bytes");
}
int main() {
    for (unsigned n : {1u, 64u}) {
        aotx_state_cases(n); aotx_restriction_cases(n); aotx_media_cases(n);
        aotx_tail_layout_cases(n); aotx_payload_batch_cases(n);
    }
    aotx_capacity_cases();
    printf("cognitive: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
