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
        if (!required && !std::filesystem::exists(path)) return true;
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
