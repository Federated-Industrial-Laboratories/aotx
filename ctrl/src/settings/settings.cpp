// Purpose: Validate and apply simulated file and live settings.
// Owns: Setting edit controls and their result notifications.
// Launch shape: One panel draws all supported setting keys.
// Lifetime: Accepted values remain in the simulated state.
#include "settings/settings.hpp"

#include "imgui.h"

#include <cstring>

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
        if (ImGui::Button(item.live ? "Set live" : "Save")) {
            if (state.set_value(index, view.values[index].data())) {
                toasts.add(item.key + (item.live ? " was set live." : " was saved."),
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

} // namespace aotx::ctrl::settings
