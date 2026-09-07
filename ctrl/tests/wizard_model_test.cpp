// Purpose: Check exact model selection and startup roles at N=1 and N=64.
// Owns: Distinct model rows, selection changes, and refusal assertions.
// Threading: One host caller checks complete row sets without a child process.
// Lifetime: Each row set and selection end after their case.
#include "wizard/models.hpp"
#include "replica/store.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {

namespace wizard = aotx::ctrl::wizard;
namespace replica = aotx::ctrl::replica;
unsigned checks = 0, bad = 0;

void check(bool condition, const char *message)
{
    ++checks;
    if (!condition) { ++bad; std::fprintf(stderr, "FAIL %s\n", message); }
}

replica::Model model(unsigned id, std::string role, bool active = true)
{
    replica::Model out;
    out.name = "model-" + std::to_string(id);
    out.role = std::move(role);
    out.file = "file-" + std::to_string(id) + ".gguf";
    out.digest = std::string(60, '0') + std::to_string(1000 + id);
    out.bytes = 10000 + id;
    out.on_disk = true;
    out.active = active;
    out.catalogued = false;
    return out;
}

void selection_cases(unsigned count)
{
    const unsigned before = checks;
    for (unsigned chosen = 0; chosen < count; ++chosen) {
        std::vector<replica::Model> rows;
        for (unsigned index = 0; index < count; ++index)
            rows.push_back(model(index, index % 2 ? "language" : "language-q4", index == chosen));
        const auto selected = rows[chosen];
        const auto identity = replica::store::key(selected);
        wizard::ModelSelection selection;
        selection.refresh(true, rows);
        check(selection.selected() != nullptr && selection.identity() == identity,
              "the initial selection uses the active present model");
        auto roles = wizard::model_roles(selection);
        check(roles.ready() && roles.roles == selected.role && !roles.embedding_available &&
              roles.notice.find("embedding") != std::string::npos,
              "language-only startup keeps the exact role and explains unavailable memory");
        check(!wizard::can_fetch(selection.selected()) && !wizard::can_activate(selection.selected()),
              "a local-only entry does not call catalog actions");
        auto complete = rows;
        complete.push_back(model(100, "embedding"));
        complete.push_back(model(101, "reranker"));
        selection.refresh(true, complete);
        roles = wizard::model_roles(selection);
        check(roles.ready() && roles.roles == "embedding,reranker," + selected.role &&
              roles.embedding_available && roles.notice.empty(),
              "present active companions retain the normal startup role order");
        aotx::ctrl::instances::Definition definition;
        definition.roles = roles.roles;
        check(definition.roles == "embedding,reranker," + selected.role,
              "the selected role replaces the instance type default");
        std::reverse(complete.begin(), complete.end());
        selection.refresh(true, complete);
        check(selection.identity() == identity && selection.selected() != nullptr &&
              selection.selected()->file == selected.file,
              "row reordering cannot select another file by index");
        selection.refresh(false, complete, "The manifest line was refused.");
        check(selection.rows().empty() && !selection.selected() && selection.identity() == identity &&
              selection.refusal() == "The manifest line was refused.",
              "a failed store read clears rows but retains selection identity and refusal");
        check(!wizard::model_roles(selection).ready(), "a failed store read cannot produce startup roles");
        selection.refresh(true, rows);
        check(selection.selected() != nullptr && selection.identity() == identity,
              "a restored exact row recovers the selection");
        for (unsigned field = 0; field < 5; ++field) {
            auto changed = rows;
            auto &replacement = changed[chosen];
            if (field == 0) replacement.name += "-changed";
            if (field == 1) replacement.file += "-changed";
            if (field == 2) replacement.digest[0] = 'f';
            if (field == 3) replacement.role = selected.role == "language" ? "language-q4" : "language";
            if (field == 4) ++replacement.bytes;
            selection.refresh(true, changed);
            check(selection.identity() == identity && !selection.selected() &&
                  !wizard::model_roles(selection).ready(),
                  "a changed identity never silently replaces the selected file");
            check(selection.select(replica::store::key(replacement)) &&
                  selection.selected()->file == replacement.file,
                  "an explicit replacement selects the new exact identity");
            selection.refresh(true, rows);
            check(!selection.selected(), "an absent replacement does not fall back to the previous row");
            check(selection.select(identity), "the original row can be selected explicitly");
        }
        auto stale = rows;
        stale.erase(stale.begin() + chosen);
        selection.refresh(true, stale);
        check(!selection.selected() && !selection.refusal().empty() && selection.identity() == identity,
              "a removed selection remains missing even when other language rows exist");
        selection.refresh(true, rows);
        auto flags = rows;
        flags[chosen].active = false;
        flags[chosen].catalogued = true;
        selection.refresh(true, flags);
        check(selection.selected() != nullptr && selection.identity() == identity &&
              wizard::can_activate(selection.selected()) && !wizard::model_roles(selection).ready(),
              "activation flags change eligibility without changing file identity");
        flags[chosen].on_disk = false;
        selection.refresh(true, flags);
        check(selection.selected() != nullptr && wizard::can_fetch(selection.selected()) &&
              !wizard::can_activate(selection.selected()) && !wizard::model_roles(selection).ready(),
              "a missing catalog file can be fetched but cannot start or activate");
        flags[chosen].catalogued = false;
        selection.refresh(true, flags);
        check(!wizard::can_fetch(selection.selected()) && !wizard::can_activate(selection.selected()),
              "a missing local-only file cannot call catalog fetch or activation");
    }
    std::printf("wizard selection N=%u: %u checks\n", count, checks - before);
}

void role_cases(unsigned count)
{
    const unsigned before = checks;
    for (unsigned index = 0; index < count; ++index) {
        auto language = model(index, index % 2 ? "language" : "language-q4");
        auto embedding = model(100 + index, "embedding");
        auto reranker = model(200 + index, "reranker");
        wizard::ModelSelection selection;
        std::vector<replica::Model> rows{language, embedding, reranker};
        selection.refresh(true, rows);
        selection.select(replica::store::key(language));
        for (unsigned companion = 1; companion < 3; ++companion) {
            for (bool missing : {false, true}) {
                auto changed = rows;
                if (missing) changed[companion].on_disk = false;
                else changed[companion].active = false;
                selection.refresh(true, changed);
                const auto roles = wizard::model_roles(selection);
                check(roles.ready() && roles.roles == (companion == 1 ? "reranker," : "embedding,") + language.role,
                      "an absent or inactive companion is omitted from startup roles");
                check(roles.embedding_available == (companion != 1) && roles.notice.empty() == (companion != 1),
                      "only an active present embedding file establishes the embedding dependency");
            }
        }
        for (unsigned role = 0; role < 3; ++role) {
            auto duplicate = rows;
            auto other = rows[role];
            other.name += "-other";
            other.file += "-other";
            other.on_disk = false;
            duplicate.push_back(other);
            selection.refresh(true, duplicate);
            const auto roles = wizard::model_roles(selection);
            check(!roles.ready() && roles.roles.empty() && !roles.embedding_available &&
                  roles.refusal.find("more than one active") != std::string::npos,
                  "ambiguous active roles are refused even when one file is absent");
        }
        auto duplicate = rows;
        duplicate.push_back(language);
        selection.refresh(true, duplicate);
        check(!selection.selected() && !wizard::model_roles(selection).ready(),
              "a repeated exact row is ambiguous rather than a successful selection");
        auto alternate = model(300 + index, language.role == "language" ? "language-q4" : "language");
        auto both = rows;
        both.push_back(alternate);
        selection.refresh(true, both);
        check(wizard::model_roles(selection).roles == "embedding,reranker," + language.role,
              "the other active language role is not added to the selected role list");
        std::string reason;
        check(selection.request(alternate.name, alternate.role, reason) && reason.empty() &&
              wizard::model_roles(selection).roles == "embedding,reranker," + alternate.role,
              "an explicit name and role select the requested language file");
        check(selection.request("", language.role, reason) && selection.selected()->name == language.name,
              "an explicit role selects its unique active present file");
        auto same_name = both;
        same_name.back().name = language.name;
        selection.refresh(true, same_name);
        check(!selection.request(language.name, "", reason) && !reason.empty(),
              "a name shared by two roles needs an exact selection");
        check(selection.request(replica::store::key(same_name.back()), "", reason) &&
              selection.selected()->file == alternate.file,
              "a full identity resolves the shared name");
        check(!selection.request("absent-model", "", reason) && !reason.empty(),
              "a requested absent name is refused");
        check(!selection.request("", "embedding", reason) && !reason.empty(),
              "a nonlanguage role is refused for language selection");
    }
    std::printf("wizard roles N=%u: %u checks\n", count, checks - before);
}

void defaults()
{
    wizard::ModelSelection selection;
    selection.refresh(true, {});
    check(!selection.selected() && !wizard::model_roles(selection).ready(), "an empty store cannot start");
    const std::vector<replica::Model> both{model(0, "language"), model(1, "language-q4")};
    selection.refresh(true, both);
    check(selection.selected() && selection.selected()->role == AOTX_CTRL_LANGUAGE_ROLE,
          "the compiled language role remains the initial preference");
    wizard::ModelSelection small("language-q4");
    small.refresh(true, both);
    check(small.selected() && small.selected()->role == "language-q4", "the preferred small-file role is selected");
    check(!wizard::language_role("embedding") && !wizard::language_role("language,reranker") &&
          wizard::language_role("language") && wizard::language_role("language-q4"),
          "only the two exact language roles qualify for selection");
    check(!wizard::can_fetch(nullptr) && !wizard::can_activate(nullptr), "an absent row exposes no model action");
}

} // namespace

int main()
{
    for (unsigned count : {1u, 64u}) { selection_cases(count); role_cases(count); }
    defaults();
    std::printf("wizard models: %u checks, %u failures\n", checks, bad);
    return bad == 0 ? 0 : 1;
}
