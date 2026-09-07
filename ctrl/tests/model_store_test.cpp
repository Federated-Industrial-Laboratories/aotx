// Purpose: Check local model visibility and exact file identity in the client store.
// Owns: Temporary catalog, store, manifest, and model files.
// Launch shape: One host process checks batches of 1 and 64 distinct model rows.
// Lifetime: Each batch removes its files before the next batch starts.
#include "replica/store.hpp"
#include "replica/schema.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
int applied = 0;
int failed = 0;

void check(bool value, const char *text)
{
    ++applied;
    if (!value) { ++failed; std::printf("model store: %s\n", text); }
}

std::string fields(const std::string &name, unsigned bytes, bool manifest)
{
    return "\"name\":\"" + name + "\",\"" + (manifest ? "path" : "file") +
        "\":\"" + name + ".gguf\",\"bytes\":" + std::to_string(bytes) +
        ",\"sha256\":\"" + std::string(64, 'a') +
        "\",\"source\":\"local/source\",\"revision\":\"revision\"";
}

std::string catalog(const std::string &name, unsigned bytes)
{
    return "{" + fields(name, bytes, false) +
        ",\"role\":\"language\",\"repository\":\"local/source\",\"license\":\"Apache-2.0\","
        "\"quant\":\"Q8_0\",\"profiles\":\"12g\",\"note\":\"\",\"verified\":false}\n";
}

std::string manifest(const std::string &name, unsigned bytes)
{
    return "{" + fields(name, bytes, true) +
        ",\"role\":\"language\",\"license\":\"Apache-2.0\",\"wrap\":{\"end_ids\":[2,3]}}\n";
}

const aotx::ctrl::replica::Model *find(const std::vector<aotx::ctrl::replica::Model> &rows,
                                     const std::string &name)
{
    const auto row = std::find_if(rows.begin(), rows.end(), [&name](const auto &item) {
        return item.name == name;
    });
    return row == rows.end() ? nullptr : &*row;
}

void batch(unsigned count)
{
    char pattern[] = "/tmp/aotx_model_store_XXXXXX";
    char *made = mkdtemp(pattern);
    check(made != nullptr, "the fixture directory did not open");
    if (!made) return;
    const std::filesystem::path root(made);
    const auto catalog_path = root / "catalog.jsonl";
    std::ofstream(catalog_path) << catalog("known", 99);
    std::ofstream saved(root / "store.jsonl");
    for (unsigned index = 0; index < count; ++index) {
        const std::string name = "local-" + std::to_string(index);
        const unsigned bytes = 17 + index;
        std::ofstream(root / (name + ".gguf")) << std::string(bytes, char('a' + index % 26));
        saved << "{" << fields(name, bytes, false)
              << ",\"date\":\"2000-01-01\",\"verified\":true}\n";
    }
    saved.close();
    std::ofstream(root / "manifest.jsonl") << manifest("local-0", 17);
    std::vector<aotx::ctrl::replica::Model> rows;
    std::string reason;
    check(aotx::ctrl::replica::store::read(catalog_path, root, rows, reason),
          "the local model sources did not read");
    check(rows.size() == count + 1, "local rows absent from the catalog were omitted");
    for (unsigned index = 0; index < count; ++index) {
        const std::string name = "local-" + std::to_string(index);
        const auto *row = find(rows, name);
        check(row && row->file == name + ".gguf" && row->bytes == 17 + index &&
                  row->on_disk && row->source == "local/source",
              "a distinct local file identity or presence was lost");
        check(row && row->active == (index == 0), "the active role did not match its file");
        check(row && !row->catalogued, "a local row gained a catalog action");
        if (row) {
            auto changed = *row;
            changed.file += ".other";
            check(aotx::ctrl::replica::store::key(changed) !=
                      aotx::ctrl::replica::store::key(*row), "a changed file kept its selection key");
            changed = *row;
            changed.active = !changed.active;
            changed.on_disk = !changed.on_disk;
            check(aotx::ctrl::replica::store::key(changed) ==
                      aotx::ctrl::replica::store::key(*row), "mutable state changed the file key");
        }
    }
    std::filesystem::remove(root / "local-0.gguf");
    check(aotx::ctrl::replica::store::read(catalog_path, root, rows, reason),
          "a missing model file hid its metadata");
    const auto *missing = find(rows, "local-0");
    check(missing && missing->active && !missing->on_disk,
          "an active row incorrectly established file presence");
    std::ofstream(root / "manifest.jsonl") << manifest("known", 99);
    check(aotx::ctrl::replica::store::read(catalog_path, root, rows, reason),
          "the stale catalog manifest did not read");
    const auto *known = find(rows, "known");
    check(known && known->active && !known->on_disk,
          "a stale catalog manifest made a missing file present");
    std::ofstream(catalog_path, std::ios::app) << catalog("known", 99);
    check(!aotx::ctrl::replica::store::read(catalog_path, root, rows, reason) && rows.empty(),
          "a duplicate catalog name kept actionable rows");
    std::filesystem::remove(catalog_path);
    check(aotx::ctrl::replica::store::read(catalog_path, root, rows, reason) &&
              rows.size() == count + 1,
          "a local store required the bundled catalog");
    std::filesystem::remove(root / "store.jsonl");
    std::ofstream(root / "manifest.jsonl") << manifest("local-0", 17);
    std::ofstream(root / "local-0.gguf") << std::string(17, 'a');
    check(aotx::ctrl::replica::store::read(catalog_path, root, rows, reason) &&
              rows.size() == 1 && rows[0].active && rows[0].on_disk,
          "a manifest-only local model was omitted");
    std::ofstream(root / "local-0.gguf") << "short";
    check(aotx::ctrl::replica::store::read(catalog_path, root, rows, reason) &&
              !rows[0].on_disk, "a truncated file retained present status");
    std::ofstream(root / "local-0.gguf") << std::string(17, 'a');
    std::string alias = catalog("local-0", 99);
    alias.replace(alias.find("local-0.gguf"), 12, "other.gguf");
    std::ofstream(catalog_path) << alias;
    check(aotx::ctrl::replica::store::read(catalog_path, root, rows, reason) && rows.size() == 2,
          "same-name rows for different files were combined");
    check(rows.size() == 2 && rows[0].catalogued && !rows[0].active &&
              !rows[0].on_disk && !rows[1].catalogued && rows[1].active && rows[1].on_disk,
          "same-name rows exchanged role, presence, or catalog state");
    if (rows.size() == 2) {
        rows[0].fetching = true;
        rows[0].fetched = 21;
        rows[1].fetch_result = "other-file";
        std::ofstream(root / "manifest.jsonl") << manifest("local-0", 18);
        check(aotx::ctrl::replica::store::read(catalog_path, root, rows, reason) &&
                  rows[0].fetching && rows[0].fetched == 21 && rows[1].fetch_result.empty(),
              "fetch progress moved to a changed file identity");
    }
    std::filesystem::remove(catalog_path);
    for (const std::string bad : {"../outside", "bad\\u0000name", "bad\\nname"}) {
        std::ofstream(root / "manifest.jsonl") << manifest(bad, 17);
        check(!aotx::ctrl::replica::store::read(catalog_path, root, rows, reason) && rows.empty(),
              "an invalid file name kept actionable rows");
    }
    std::ofstream(root / "manifest.jsonl") << manifest("local-0", 17) << manifest("local-0", 17);
    check(!aotx::ctrl::replica::store::read(catalog_path, root, rows, reason) && rows.empty(),
          "a duplicate manifest name kept actionable rows");
    std::filesystem::remove(root / "manifest.jsonl");
    std::ofstream history(root / "store.jsonl");
    for (unsigned bytes : {16u, 17u})
        history << "{" << fields("local-0", bytes, false)
                << ",\"date\":\"2000-01-01\",\"verified\":true}\n";
    history.close();
    check(aotx::ctrl::replica::store::read(catalog_path, root, rows, reason) &&
              rows.size() == 1 && rows[0].bytes == 17 && rows[0].on_disk,
          "the append log did not retain the latest complete local identity");
    std::filesystem::create_directory(root / "manifest.jsonl");
    check(!aotx::ctrl::replica::store::read(catalog_path, root, rows, reason) &&
              rows.empty() && reason.find("model manifest") != std::string::npos,
          "a manifest read error retained actionable rows or omitted its refusal");
    std::filesystem::remove(root / "manifest.jsonl");
    const auto blocked = root / "blocked";
    std::filesystem::create_directory(blocked);
    std::filesystem::permissions(blocked, std::filesystem::perms::none);
    bool refused = false;
    try {
        refused = !aotx::ctrl::replica::store::read(catalog_path, blocked, rows, reason) &&
                  rows.empty() && !reason.empty();
    } catch (const std::filesystem::filesystem_error &) {}
    std::filesystem::permissions(blocked, std::filesystem::perms::owner_all);
    check(refused, "an inaccessible store did not return a visible refusal");
    std::filesystem::remove_all(root);
}
} // namespace

int main()
{
    batch(1);
    batch(64);
    std::printf("model store: %d cases, %d failed\n", applied, failed);
    return failed == 0 ? 0 : 1;
}
