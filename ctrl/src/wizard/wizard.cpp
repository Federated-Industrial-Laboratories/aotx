// Purpose: Guide simulated detect, build, fetch, activate, and start actions.
// Owns: Modal sequence controls and result notifications.
// Launch shape: One modal advances through five ordered actions.
// Lifetime: Completion closes the current guide sequence.
#include "wizard/wizard.hpp"

#include "imgui.h"

#include <array>

namespace aotx::ctrl::wizard {

void draw(State &view, sim::State &state, toast::Lane &toasts, double now, bool *open)
{
    if (*open) ImGui::OpenPopup("First-run guide");
    bool popup_open = true;
    if (!ImGui::BeginPopupModal("First-run guide", &popup_open,
                                ImGuiWindowFlags_AlwaysAutoResize)) {
        if (!popup_open) *open = false;
        return;
    }
    static const std::array<const char *, 5> titles = {
        "Detect", "Build", "Fetch", "Activate", "Start"};
    static const std::array<const char *, 5> results = {
        "One card and the 12g profile were detected.",
        "The build directory is ready.",
        "The simulated model fetch completed.",
        "The language model is active for the conductor.",
        "The first simulated instance started."};
    const unsigned step = view.step < titles.size() ? view.step : 0;
    ImGui::Text("Step %u of %zu", step + 1, titles.size());
    ImGui::SeparatorText(titles[step]);
    ImGui::TextWrapped("%s", results[step]);
    if (ImGui::Button(step + 1 == titles.size() ? "Finish" : "Continue")) {
        if (step == 2 && state.models.size() > 3) {
            state.models[3].state = "on disk";
            state.models[3].fetch_progress = 1.0f;
        } else if (step == 3 && state.models.size() > 3) {
            state.activate_model(3, "language");
        }
        toasts.add(results[step], toast::Severity::success, now);
        if (step + 1 == titles.size()) {
            state.set_instance_state(0, sim::InstanceState::running);
            view.step = 0;
            *open = false;
            ImGui::CloseCurrentPopup();
        } else {
            ++view.step;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
        view.step = 0;
        *open = false;
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
    if (!popup_open) {
        view.step = 0;
        *open = false;
    }
}

} // namespace aotx::ctrl::wizard
