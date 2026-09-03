// Purpose: Validate and apply simulated file and live settings.
// Owns: Setting edit controls and their result notifications.
// Launch shape: One panel draws all supported setting keys.
// Lifetime: Accepted values remain in the simulated state.
#include "settings/settings.hpp"

#include "imgui.h"
#include "cuda/settings/keys.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace aotx::ctrl::settings {
namespace {

void initialize(State &view, const sim::State &state)
{
    if (view.values.size() == state.settings.size()) return;
    view.values.resize(state.settings.size());
    for (std::size_t index = 0; index < state.settings.size(); ++index) {
        std::strncpy(view.values[index].data(), state.settings[index].value.c_str(),
                     view.values[index].size() - 1);
    }
}

} // namespace

void draw(State &view, sim::State &state, toast::Lane &toasts, double now, bool *open)
{
    initialize(view, state);
    if (!ImGui::Begin("Settings", open)) {
        ImGui::End();
        return;
    }
    for (std::size_t index = 0; index < state.settings.size(); ++index) {
        const sim::Setting &item = state.settings[index];
        ImGui::PushID(static_cast<int>(index));
        ImGui::TextUnformatted(item.key.c_str());
        ImGui::SetNextItemWidth(130.0f);
        ImGui::InputText("##value", view.values[index].data(), view.values[index].size());
        ImGui::SameLine();
        if (ImGui::Button(item.live ? "Apply" : "Save")) {
            if (state.set_value(index, view.values[index].data())) {
                toasts.add(item.key + (item.live ? " was applied." : " was saved."),
                           toast::Severity::success, now);
            } else {
                toasts.add(state.refusal(), toast::Severity::error, now);
            }
        }
        ImGui::TextDisabled("Default %s; valid %s", item.default_value.c_str(),
                            item.valid_values.c_str());
        ImGui::Separator();
        ImGui::PopID();
    }
    ImGui::End();
}

namespace {

enum class Kind { number, text };
enum class Side { BOOT, DEVICE, TERMINAL };

struct Spec {
    std::string key;
    Kind kind;
    Side side;
    std::string effect;
    long long default_number;
    long long least;
    long long most;
    long long scale;
    std::string default_text;
};

std::string number_text(long long value, long long scale)
{
    if (scale == 1) return std::to_string(value);
    std::ostringstream text;
    text << std::fixed << std::setprecision(4)
         << static_cast<double>(value) / static_cast<double>(scale);
    std::string made = text.str();
    while (!made.empty() && made.back() == '0') made.pop_back();
    if (!made.empty() && made.back() == '.') made.pop_back();
    return made;
}

const std::vector<Spec> &specs()
{
    static const std::vector<Spec> table = [] {
        std::vector<Spec> made;
#define AOTX_CTRL_NUMBER(symbol, key, side, effect, def, low, high, unit) \
        made.push_back({key, Kind::number, Side::side, #effect, def, low, high, unit, {}});
        AOTX_SETTING_NUMBERS(AOTX_CTRL_NUMBER)
#undef AOTX_CTRL_NUMBER
#define AOTX_CTRL_TEXT(symbol, key, side, effect, def) \
        made.push_back({key, Kind::text, Side::side, #effect, 0, 0, 0, 1, def});
        AOTX_SETTING_TEXTS(AOTX_CTRL_TEXT)
#undef AOTX_CTRL_TEXT
        return made;
    }();
    return table;
}

std::string default_value(const Spec &spec)
{
    return spec.kind == Kind::number ? number_text(spec.default_number, spec.scale)
                                     : spec.default_text;
}

bool number_value(const Spec &spec, const char *text)
{
    if (text[0] == '\0') return false;
    char *end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(parsed)) return false;
    const double scaled = parsed * static_cast<double>(spec.scale);
    const long long whole = std::llround(scaled);
    return std::fabs(scaled - static_cast<double>(whole)) < 0.00001 &&
           whole >= spec.least && whole <= spec.most;
}

std::string trim_setting(std::string text)
{
    const std::size_t first = text.find_first_not_of(" \t\r");
    if (first == std::string::npos) return {};
    const std::size_t last = text.find_last_not_of(" \t\r");
    return text.substr(first, last - first + 1u);
}

bool save_file(const std::filesystem::path &path, const std::string &key,
               const std::string &value, std::string &reason)
{
    std::vector<std::string> lines;
    std::ifstream input(path);
    std::string line;
    bool found = false;
    while (std::getline(input, line)) {
        std::string content = line.substr(0u, line.find('#'));
        const std::size_t equal = content.find('=');
        if (equal != std::string::npos && trim_setting(content.substr(0u, equal)) == key) {
            if (!found) lines.push_back(key + " = " + value);
            found = true;
        } else {
            lines.push_back(line);
        }
    }
    if (!found) lines.push_back(key + " = " + value);
    const std::filesystem::path temporary = path.string() + ".ctrl.tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) {
        reason = "The settings file was not saved because its temporary file did not open.";
        return false;
    }
    for (const std::string &held : lines) output << held << '\n';
    output.close();
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary);
        reason = "The settings file was not saved because its replacement failed.";
        return false;
    }
    return true;
}

void initialize_live(State &view, const replica::State &state)
{
    if (view.live_initialized) return;
    view.values.resize(specs().size());
    for (std::size_t index = 0u; index < specs().size(); ++index) {
        std::string value;
        if (!replica::setting_value(state.settings(), specs()[index].key, value)) {
            value = default_value(specs()[index]);
        }
        std::strncpy(view.values[index].data(), value.c_str(), view.values[index].size() - 1u);
        view.values[index].back() = '\0';
    }
    view.live_initialized = true;
}

} // namespace

bool apply_value(replica::State &state, client::Client &client, toast::Lane &toasts,
                 double now, const std::string &key, const std::string &value)
{
    if (state.phase() == "running") {
        if (client.send_line("set " + key + " " + value)) return true;
        toasts.add("The setting was not applied because the connection is not ready.",
                   toast::Severity::error, now);
        return false;
    }
    std::string reason;
    if (save_file(state.settings(), key, value, reason)) {
        toasts.add(key + " was saved.", toast::Severity::success, now);
        return true;
    }
    toasts.add(std::move(reason), toast::Severity::error, now);
    return false;
}

void draw(State &view, replica::State &state, client::Client &client,
          toast::Lane &toasts, double now, bool *open)
{
    initialize_live(view, state);
    if (!ImGui::Begin("Settings", open)) {
        ImGui::End();
        return;
    }
    const bool running = state.phase() == "running";
    for (std::size_t index = 0u; index < specs().size(); ++index) {
        const Spec &spec = specs()[index];
        ImGui::PushID(static_cast<int>(index));
        ImGui::TextUnformatted(spec.key.c_str());
        ImGui::SetNextItemWidth(180.0f);
        ImGui::InputText("##value", view.values[index].data(), view.values[index].size());
        const bool valid = spec.kind == Kind::text || number_value(spec, view.values[index].data());
        const bool live = running && spec.side == Side::DEVICE;
        ImGui::SameLine();
        if (ImGui::Button(live ? "Apply" : "Save")) {
            if (!valid) {
                toasts.add("The setting was refused because its value is not valid.",
                           toast::Severity::error, now);
            } else if (live) {
                apply_value(state, client, toasts, now, spec.key, view.values[index].data());
            } else {
                std::string reason;
                if (save_file(state.settings(), spec.key, view.values[index].data(), reason)) {
                    toasts.add(spec.key + " was saved.", toast::Severity::success, now);
                } else {
                    toasts.add(std::move(reason), toast::Severity::error, now);
                }
            }
        }
        if (spec.kind == Kind::number) {
            ImGui::TextDisabled("Default %s; range %s to %s; effect %s",
                default_value(spec).c_str(), number_text(spec.least, spec.scale).c_str(),
                number_text(spec.most, spec.scale).c_str(), spec.effect.c_str());
        } else {
            ImGui::TextDisabled("Default %s; text; effect %s",
                                spec.default_text.empty() ? "empty" : spec.default_text.c_str(),
                                spec.effect.c_str());
        }
        ImGui::Separator();
        ImGui::PopID();
    }
    ImGui::End();
}

} // namespace aotx::ctrl::settings
