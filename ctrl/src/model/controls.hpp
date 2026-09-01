// Purpose: Define model parameter defaults and preset files.
// Owns: No live state; callers own loaded preset rows.
// Launch shape: One interface or check thread reads one store directory.
// Lifetime: Returned rows remain in caller-owned storage.
#ifndef AOTX_CTRL_MODEL_CONTROLS_HPP
#define AOTX_CTRL_MODEL_CONTROLS_HPP

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace aotx::ctrl::model {

struct EngineParameter {
    const char *name;
    const char *label;
    double initial;
    double least;
    double most;
    bool whole;
};

struct Preset {
    std::string name;
    std::vector<std::pair<std::string, std::string>> values;
};

const std::vector<EngineParameter> &engine_parameters();
bool read_preset(const std::filesystem::path &path, Preset &preset, std::string &reason);
bool read_presets(const std::filesystem::path &directory, std::vector<Preset> &presets,
                  std::string &reason);
std::vector<std::string> preset_commands(unsigned agent, const Preset &preset);

} // namespace aotx::ctrl::model

#endif
