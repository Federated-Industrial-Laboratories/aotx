// Purpose: Read bounded sampler preset files beside the model store.
// Owns: The engine parameter table used when model metadata omits a key.
// Launch shape: One interface or check thread reads files in name order.
// Lifetime: The constant table exists for the process lifetime.
#include "model/controls.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>

namespace aotx::ctrl::model {
namespace {

std::string trim(std::string text)
{
    const auto space = [](unsigned char byte) { return std::isspace(byte) != 0; };
    while (!text.empty() && space(static_cast<unsigned char>(text.front()))) text.erase(0u, 1u);
    while (!text.empty() && space(static_cast<unsigned char>(text.back()))) text.pop_back();
    return text;
}

const EngineParameter *find_parameter(const std::string &key)
{
    for (const EngineParameter &row : engine_parameters()) {
        if (key == row.name) return &row;
    }
    return nullptr;
}

bool valid_value(const EngineParameter &parameter, const std::string &text)
{
    if (text.empty()) return false;
    char *end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (end != text.c_str() + text.size() || !std::isfinite(value) ||
        value < parameter.least ||
        value > parameter.most) return false;
    return !parameter.whole || value == static_cast<double>(static_cast<long long>(value));
}

} // namespace

const std::vector<EngineParameter> &engine_parameters()
{
    static const std::vector<EngineParameter> rows = {
        {"temperature", "Temperature", 0.0, 0.0, 2.0, false},
        {"top_k", "Top K", 0.0, 0.0, 256.0, true},
        {"top_p", "Top P", 1.0, 0.0001, 1.0, false},
        {"min_p", "Minimum P", 0.0, 0.0, 1.0, false},
        {"repeat_penalty", "Repeat penalty", 1.0, 0.0001, 2.0, false},
        {"repeat_window", "Repeat window", 0.0, 0.0, 8191.0, true},
        {"presence_penalty", "Presence penalty", 0.0, -2.0, 2.0, false},
        {"frequency_penalty", "Frequency penalty", 0.0, -2.0, 2.0, false},
        {"seed", "Seed", 0.0, 0.0, 2147483647.0, true},
        {"think_limit", "Thinking limit", -1.0, -1.0, 8191.0, true}};
    return rows;
}

bool read_preset(const std::filesystem::path &path, Preset &preset, std::string &reason)
{
    std::ifstream file(path);
    if (!file) {
        reason = "The preset file does not open.";
        return false;
    }
    Preset made;
    made.name = path.stem().string();
    std::string line;
    unsigned number = 0u;
    while (std::getline(file, line)) {
        ++number;
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        line = trim(line);
        if (line.empty()) continue;
        const std::size_t equal = line.find('=');
        const std::string key = trim(line.substr(0u, equal));
        const std::string value = equal == std::string::npos ? std::string()
                                                              : trim(line.substr(equal + 1u));
        constexpr const char *prefix = "decode.";
        const std::string name = key.rfind(prefix, 0u) == 0u ? key.substr(7u) : std::string();
        const EngineParameter *parameter = find_parameter(name);
        if (parameter == nullptr || !valid_value(*parameter, value)) {
            reason = "The preset line " + std::to_string(number) + " was refused.";
            return false;
        }
        made.values.push_back({key, value});
    }
    if (!file.eof() || made.values.empty()) {
        reason = "The preset has no values.";
        return false;
    }
    preset = std::move(made);
    reason.clear();
    return true;
}

bool read_presets(const std::filesystem::path &directory, std::vector<Preset> &presets,
                  std::string &reason)
{
    std::vector<Preset> made;
    std::error_code error;
    for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
        if (error) break;
        if (!entry.is_regular_file(error) || error || entry.path().extension() != ".preset") {
            error.clear();
            continue;
        }
        Preset preset;
        if (!read_preset(entry.path(), preset, reason)) return false;
        made.push_back(std::move(preset));
    }
    if (error) {
        reason = "The preset directory does not read.";
        return false;
    }
    std::sort(made.begin(), made.end(),
              [](const Preset &left, const Preset &right) { return left.name < right.name; });
    presets = std::move(made);
    reason.clear();
    return true;
}

std::vector<std::string> preset_commands(unsigned agent, const Preset &preset)
{
    std::vector<std::string> commands;
    for (const auto &value : preset.values) {
        commands.push_back("agent " + std::to_string(agent) + " " + value.first + " " +
                           value.second);
    }
    return commands;
}

} // namespace aotx::ctrl::model
