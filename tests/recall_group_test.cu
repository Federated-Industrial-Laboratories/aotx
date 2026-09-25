/* Purpose: Check shared source labels with focus and complete evidence groups.
 * Owns: Independent byte expansion, exact limits and recorded replay controls.
 * Launch shape: Distinct N=1 and N=64 queries on the device path.
 * Lifetime: One test process without model weights. */
#include "recall_compact_fixture.h"
#include <sstream>
#include "recall_group_identity.h"

static std::string aotx_expand_groups(const std::string &text) {
    std::vector<std::string> labels;
    std::istringstream input(text); std::string line, out;
    while (std::getline(input, line)) {
        if (line.rfind("[source_group=", 0) == 0) {
            size_t space = line.find(' ');
            unsigned group = std::stoul(line.substr(14, space - 14));
            aotx_check(group == labels.size(), "source groups have consecutive first occurrence numbers");
            labels.push_back(line.substr(space, line.size() - space - 1)); continue;
        }
        if (line.rfind("[memory ", 0) == 0) {
            size_t at = line.find(" source_group=");
            if (at != std::string::npos) {
                size_t end = line.find_first_of(" ]", at + 14);
                unsigned group = std::stoul(line.substr(at + 14, end - at - 14));
                aotx_check(group < labels.size(), "each memory group names an emitted exact source label");
                if (group < labels.size()) line.replace(at, end - at, labels[group]);
            }
        }
        out += line + '\n';
    }
    if (!text.empty() && text.back() != '\n') out.pop_back();
    return out;
}
static void aotx_group_case(unsigned n) {
    auto f = aotx_compact_corpus(n);
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_memory_row(i, AOTX_COG_WORKING, 980000 + i, f.rows.size() + 1);
        f.add(r, aotx_memory_text("Earlier request " + std::to_string(i) + std::string(180, 'x')));
    }
    auto image = f.wire(false, f.rows.size()); aotx_recall_device d;
    aotx_check(!d.load(image).status, "source pairs and distinct focus records load");
    auto q = aotx_memory_queries(n, f.rows.size());
    for (unsigned i = 0; i < n; ++i) {
        auto p = aotx_query_at(q, i); aotx_compact_version(p, 3); aotx_put(p + 132, 10, 4);
    }
    auto old = d.search(q, n); aotx_status_rows(old, 0, "prior rendering remains available");
    for (unsigned i = 0; i < n; ++i) aotx_compact_version(aotx_query_at(q, i), 4);
    aotx_group_identities(d, f, q, n);
    auto grouped = d.search(q, n); aotx_status_rows(grouped, 0, "group label rendering");
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(!memcmp(old[i].selection, grouped[i].selection, AOTX_RECALL_SELECTION),
            "group labels retain exact ordered evidence references");
        aotx_check(aotx_expand_groups(aotx_context(grouped[i])) == aotx_context(old[i]),
            "expanding source groups restores every prior context byte");
        aotx_check(grouped[i].context_bytes < old[i].context_bytes, "repeated source labels reduce actual context bytes");
        auto p = aotx_query_at(q, i); aotx_pin(p, 1, 0, 980000 + i); aotx_put(p + 132, 11, 4);
    }
    auto focus = d.search(q, n); aotx_status_rows(focus, 0, "focus and two complete source groups");
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(focus[i].count == 11 && aotx_selected(focus[i], 0) == 980000 + i &&
            focus[i].reason[0] == AOTX_RECALL_FOCUS, "mandatory focus stays first with all ten evidence references");
        for (unsigned source = 2 * i; source < 2 * i + 2; ++source)
            for (unsigned part = 3; part < 8; ++part)
                aotx_check(aotx_context_has(focus[i], aotx_ar_id(source, part)), "each exact source and appraisal group remains complete");
        aotx_check(focus[i].context_bytes - 8 - aotx_get(aotx_query_at(q, i) + 148, 4) <= 4096,
            "focus and all source table bytes fit the original memory bound");
        aotx_compact_version(aotx_query_at(q, i), 3);
    }
    auto disabled = d.search(q, n); aotx_status_rows(disabled, 0, "prior revision with mandatory focus");
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(disabled[i].count < 11 && aotx_selected(disabled[i], 0) == 980000 + i,
            "disabled group labels detect lost evidence coverage with focus retained");
        aotx_compact_version(aotx_query_at(q, i), 4);
        aotx_put(aotx_query_at(q, i) + 136, focus[i].context_bytes - 8 - aotx_get(aotx_query_at(q, i) + 148, 4), 4);
    }
    auto exact = d.search(q, n); aotx_status_rows(exact, 0, "exact source table byte limit");
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(aotx_context(exact[i]) == aotx_context(focus[i]), "table sizing equals final rendering at the byte boundary");
        aotx_put(aotx_query_at(q, i) + 136, aotx_get(aotx_query_at(q, i) + 136, 4) - 1, 4);
    }
    auto short_rows = d.search(q, n); aotx_status_rows(short_rows, 0, "one byte less retains atomic evidence groups");
    for (const auto &r : short_rows) aotx_check(r.count == 6, "a partial second evidence group never enters the selection");
    for (unsigned i = 0; i < n; ++i) aotx_put(aotx_query_at(q, i) + 136, 4096, 4);
    auto original = d.search(q, n); aotx_check(!d.record(q, n, true).status, "record new rendering and complete choices");
    auto saved = d.checkpoint(); aotx_check(!d.load(saved).status, "restore recorded source groups");
    auto replay = d.search(d.saved_queries(n), n, true); aotx_status_rows(replay, 0, "recorded group replay");
    for (unsigned i = 0; i < n; ++i)
        aotx_check(!replay[i].searches && aotx_context(replay[i]) == aotx_context(original[i]),
            "restored source groups are byte exact without a new search");
}
int main(void) {
    for (unsigned n : {1u, 64u}) { aotx_group_case(n); printf("source groups N=%u complete\n", n); }
    printf("source groups: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
