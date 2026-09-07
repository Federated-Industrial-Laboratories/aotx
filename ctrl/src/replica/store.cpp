// Purpose: Read the model catalog, local store, and active manifest.
// Owns: Temporary rows from three model JSONL files.
// Launch shape: One interface thread reads all model rows in file order.
// Lifetime: Temporary rows end after one complete store read.
#include "replica/store.hpp"

#include "replica/schema.hpp"

#include <fstream>
#include <map>
#include <algorithm>
#include <cstdlib>
#include <cmath>

namespace aotx::ctrl::replica::store {
namespace {

constexpr std::size_t model_line_bound = 2048u;

bool next_line(std::istream &file, std::string &line, bool &overflow)
{
    line.clear();
    overflow = false;
    char byte = '\0';
    while (file.get(byte)) {
        if (byte == '\n') return true;
        if (line.size() < model_line_bound) line.push_back(byte);
        else overflow = true;
    }
    return !line.empty() || overflow;
}

template <class Take>
bool lines(const std::filesystem::path &path, bool required, const char *name,
           std::string &reason, Take take)
{
    std::ifstream file(path);
    if (!file) {
        std::error_code error;
        const bool exists = std::filesystem::exists(path, error);
        if (!required && !exists && !error) return true;
        reason = "The " + std::string(name) + " file does not open.";
        return false;
    }
    std::string line;
    std::uint64_t number = 0u;
    bool overflow = false;
    while (next_line(file, line, overflow)) {
        ++number;
        if (overflow || (!line.empty() && !take(line))) {
            reason = "The " + std::string(name) + " line " + std::to_string(number) +
                     " was refused.";
            return false;
        }
        if (line.empty()) continue;
    }
    if (!file.eof() || file.bad()) {
        reason = "The " + std::string(name) + " file does not read.";
        return false;
    }
    return true;
}

bool same_file(const Model &left, const Model &right)
{
    return left.name == right.name && left.file == right.file && left.bytes == right.bytes &&
           left.digest == right.digest;
}

bool base_name(const std::string &name)
{
    return !name.empty() && name.front() != '.' &&
           std::none_of(name.begin(), name.end(), [](unsigned char byte) {
               return byte < 32 || byte == 127 || byte == '/' || byte == '\\';
           }) &&
           name.find("..") == std::string::npos;
}

bool add_source(std::vector<Model> &rows, Model row, bool append_log = false)
{
    if (!base_name(row.name) || row.name.find(' ') != std::string::npos || !base_name(row.file) ||
        (!row.role.empty() && row.role != "language" && row.role != "language-q4" &&
         row.role != "embedding" && row.role != "reranker")) return false;
    for (Model &held : rows) {
        if (held.name != row.name) continue;
        if (!append_log) return false;
        held = std::move(row);
        return true;
    }
    rows.push_back(std::move(row));
    return true;
}

} // namespace

std::string key(const Model &model)
{
    return model.name + "\n" + model.file + "\n" + model.digest + "\n" +
           model.role + "\n" + std::to_string(model.bytes);
}

bool read(const std::filesystem::path &catalog, const std::filesystem::path &directory,
          std::vector<Model> &models, std::string &reason)
{
    const std::vector<Model> prior = std::move(models);
    models.clear();
    std::vector<Model> catalog_rows;
    std::vector<Model> local;
    std::vector<Model> active;
    if (!lines(catalog, false, "model catalog", reason, [&catalog_rows](const std::string &line) {
            Model row;
            if (!schema::model_catalog(line, row)) return false;
            row.catalogued = true;
            return add_source(catalog_rows, std::move(row));
        }) || !lines(directory / "store.jsonl", false, "model store", reason,
               [&local](const std::string &line) {
                   Model row;
                   if (!schema::model_store(line, row)) return false;
                   return add_source(local, std::move(row), true);
               }) ||
        !lines(directory / "manifest.jsonl", false, "model manifest", reason,
               [&active](const std::string &line) {
                   Model row;
                   if (!schema::model_manifest(line, row)) return false;
                   return add_source(active, std::move(row));
               })) return false;

    std::vector<Model> made = std::move(catalog_rows);
    for (const Model &entry : active) {
        auto found = std::find_if(made.begin(), made.end(), [&entry](const Model &row) {
            return same_file(row, entry) && row.role == entry.role;
        });
        if (found == made.end()) made.push_back(entry);
        else found->active = true;
    }
    for (const Model &entry : local) {
        bool matched = false;
        for (Model &row : made) {
            if (same_file(row, entry)) { row.verified = entry.verified; matched = true; }
        }
        if (!matched) made.push_back(entry);
    }
    for (Model &row : made) {
        std::error_code error;
        const auto path = directory / row.file;
        const bool regular = std::filesystem::is_regular_file(path, error);
        const std::uintmax_t bytes = regular ? std::filesystem::file_size(path, error) : 0;
        row.on_disk = regular && !error && bytes == row.bytes;
        row.verified = row.on_disk && row.verified;
        for (const Model &old : prior) {
            if (same_file(old, row) && old.role == row.role) {
                row.fetching = old.fetching;
                row.fetched = old.fetched;
                row.fetch_total = old.fetch_total;
                row.fetch_result = old.fetch_result;
            }
        }
    }
    if (made.empty()) {
        reason = "The model sources have no entries.";
        return false;
    }
    models = std::move(made);
    reason.clear();
    return true;
}

bool read_controls(const std::filesystem::path &directory,
                   std::vector<ModelParameters> &parameters,
                   std::vector<SteerVector> &vectors,
                   std::vector<VoiceProfile> &profiles, std::string &reason)
{
    std::vector<ModelParameters> made_parameters;
    std::vector<SteerVector> made_vectors;
    std::vector<VoiceProfile> made_profiles;
    if (!lines(directory / "parameters.jsonl", false, "model parameters", reason,
               [&made_parameters](const std::string &line) {
                   ModelParameters row;
                   if (!schema::model_parameters(line, row)) return false;
                   made_parameters.push_back(std::move(row));
                   return true;
               }) ||
        !lines(directory / "steer.jsonl", false, "steer catalog", reason,
               [&made_vectors](const std::string &line) {
                   SteerVector row;
                   if (!schema::steer_vector(line, row)) return false;
                   made_vectors.push_back(std::move(row));
                   return true;
               })) return false;

    std::error_code error;
    const std::filesystem::path voice = directory / "voice";
    for (const auto &entry : std::filesystem::directory_iterator(voice, error)) {
        if (error) break;
        if (!entry.is_regular_file(error) || error || entry.path().extension() != ".profile") {
            error.clear();
            continue;
        }
        std::ifstream file(entry.path());
        std::string name;
        if (!std::getline(file, name) || name.empty() || name.size() > 31u) {
            reason = "A voice profile name was refused.";
            return false;
        }
        unsigned entries = 0u;
        std::string line;
        while (std::getline(file, line)) {
            const std::size_t tab = line.find('\t');
            if (tab == std::string::npos || tab == 0u || tab + 1u == line.size() ||
                ++entries > 128u) {
                reason = "A voice profile line was refused.";
                return false;
            }
            const std::string bias = line.substr(0u, tab);
            char *end = nullptr;
            const double amount = std::strtod(bias.c_str(), &end);
            if (end == nullptr || end != bias.c_str() + bias.size() ||
                !std::isfinite(amount)) {
                reason = "A voice profile bias was refused.";
                return false;
            }
        }
        if (!file.eof() || entries == 0u) {
            reason = "A voice profile was refused.";
            return false;
        }
        made_profiles.push_back({name, entry.path().filename().string(), entries});
    }
    if (error && error != std::errc::no_such_file_or_directory) {
        reason = "The voice profile directory does not read.";
        return false;
    }
    std::sort(made_profiles.begin(), made_profiles.end(),
              [](const VoiceProfile &left, const VoiceProfile &right) {
                  return left.name < right.name;
              });
    parameters = std::move(made_parameters);
    vectors = std::move(made_vectors);
    profiles = std::move(made_profiles);
    reason.clear();
    return true;
}

} // namespace aotx::ctrl::replica::store
