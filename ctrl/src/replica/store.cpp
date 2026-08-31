// Purpose: Read the model catalog, local store, and active manifest.
// Owns: Temporary rows from three model JSONL files.
// Launch shape: One interface thread reads all model rows in file order.
// Lifetime: Temporary rows end after one complete store read.
#include "replica/store.hpp"

#include "replica/schema.hpp"

#include <fstream>
#include <map>

namespace aotx::ctrl::replica::store {
namespace {

template <class Take>
bool lines(const std::filesystem::path &path, bool required, const char *name,
           std::string &reason, Take take)
{
    std::ifstream file(path);
    if (!file) {
        if (!required && !std::filesystem::exists(path)) return true;
        reason = "The " + std::string(name) + " file does not open.";
        return false;
    }
    std::string line;
    std::uint64_t number = 0u;
    while (std::getline(file, line)) {
        ++number;
        if (line.empty()) continue;
        if (!take(line)) {
            reason = "The " + std::string(name) + " line " + std::to_string(number) +
                     " was refused.";
            return false;
        }
    }
    return true;
}

bool same_file(const Model &left, const Model &right)
{
    return left.name == right.name && left.file == right.file && left.bytes == right.bytes &&
           left.digest == right.digest;
}

} // namespace

bool read(const std::filesystem::path &catalog, const std::filesystem::path &directory,
          std::vector<Model> &models, std::string &reason)
{
    std::vector<Model> made;
    std::map<std::string, Model> local;
    std::vector<Model> active;
    if (!lines(catalog, true, "model catalog", reason, [&made](const std::string &line) {
            Model row;
            if (!schema::model_catalog(line, row)) return false;
            made.push_back(std::move(row));
            return true;
        }) || made.empty()) {
        if (reason.empty()) reason = "The model catalog has no entries.";
        return false;
    }
    if (!lines(directory / "store.jsonl", false, "model store", reason,
               [&local](const std::string &line) {
                   Model row;
                   if (!schema::model_store(line, row)) return false;
                   local[row.name] = std::move(row);
                   return true;
               }) ||
        !lines(directory / "manifest.jsonl", false, "model manifest", reason,
               [&active](const std::string &line) {
                   Model row;
                   if (!schema::model_manifest(line, row)) return false;
                   active.push_back(std::move(row));
                   return true;
               })) return false;

    for (Model &row : made) {
        const auto saved = local.find(row.name);
        if (saved != local.end() && same_file(row, saved->second)) {
            std::error_code error;
            const std::uintmax_t bytes = std::filesystem::file_size(directory / row.file, error);
            row.on_disk = !error && bytes == row.bytes;
            row.verified = row.verified || (row.on_disk && saved->second.verified);
        }
        for (const Model &entry : active) {
            if (same_file(row, entry) && row.role == entry.role) {
                row.active = true;
                row.on_disk = true;
            }
        }
        for (const Model &old : models) {
            if (old.name == row.name) {
                row.fetching = old.fetching;
                row.fetched = old.fetched;
                row.fetch_total = old.fetch_total;
                row.fetch_result = old.fetch_result;
            }
        }
    }
    models = std::move(made);
    reason.clear();
    return true;
}

} // namespace aotx::ctrl::replica::store
