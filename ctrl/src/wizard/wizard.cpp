// Purpose: Guide simulated detect, build, fetch, activate, and start actions.
// Owns: Modal sequence controls and result notifications.
// Launch shape: One modal advances through five ordered actions.
// Lifetime: Completion closes the current guide sequence.
#include "wizard/wizard.hpp"

#include "imgui.h"

#include <array>
#include <cstring>

namespace aotx::ctrl::wizard {

void draw(State &view, sim::State &state, toast::Lane &toasts, double now, bool *open)
{
    if (*open && !ImGui::IsPopupOpen("First run")) ImGui::OpenPopup("First run");
    bool popup_open = true;
    if (!ImGui::BeginPopupModal("First run", &popup_open,
                                ImGuiWindowFlags_AlwaysAutoResize)) {
        if (!popup_open) *open = false;
        return;
    }
    static const std::array<const char *, 5> titles = {
        "Detect", "Build", "Model", "Activate", "Start"};
    static const std::array<const char *, 5> messages = {
        "One card and the 12g profile were detected.",
        "Select a build path.",
        "Select a language model.",
        "The language model is ready for activation.",
        "The first simulated instance started."};
    static const std::array<const char *, 3> build_paths = {
        "/opt/aotx/build/12g", "/opt/aotx/build/8g", "/usr/local/lib/aotx"};
    const unsigned page = view.page < titles.size() ? view.page : 0;
    ImGui::Text("%u of %zu", page + 1, titles.size());
    ImGui::SeparatorText(titles[page]);
    ImGui::TextWrapped("%s", messages[page]);
    if (page == 1) {
        ImGui::InputText("Build path", view.build_path.data(), view.build_path.size());
        const char *preview = view.build_path[0] == '\0' ? "Select a directory"
                                                         : view.build_path.data();
        if (ImGui::BeginCombo("Browse", preview)) {
            for (const char *path : build_paths) {
                if (ImGui::Selectable(path, std::strcmp(path, view.build_path.data()) == 0)) {
                    std::strncpy(view.build_path.data(), path, view.build_path.size() - 1);
                    view.build_path.back() = '\0';
                }
            }
            ImGui::EndCombo();
        }
    } else if (page == 2 && !state.models.empty()) {
        if (view.model_index >= state.models.size()) view.model_index = 0;
        if (ImGui::BeginCombo("Language model", state.models[view.model_index].name.c_str())) {
            for (std::size_t index = 0; index < state.models.size(); ++index) {
                if (ImGui::Selectable(state.models[index].name.c_str(),
                                      index == view.model_index)) {
                    view.model_index = index;
                }
            }
            ImGui::EndCombo();
        }
    }
    if (ImGui::Button(page + 1 == titles.size() ? "Finish" : "Continue")) {
        std::string result = messages[page];
        bool accepted = true;
        if (page == 1 && view.build_path[0] == '\0') {
            result = "The build path was refused because no directory is selected.";
            accepted = false;
        } else if (page == 1) {
            result = "The build path was selected.";
        } else if (page == 2 && view.model_index < state.models.size()) {
            sim::Model &selected = state.models[view.model_index];
            if (selected.state == "catalog") {
                state.fetch_model(view.model_index);
                selected.state = "on disk";
                selected.fetch_progress = 1.0f;
                result = selected.name + " fetch completed.";
            } else {
                result = selected.name + " is available on disk.";
            }
        } else if (page == 3 && view.model_index < state.models.size()) {
            accepted = state.activate_model(view.model_index, "language");
            result = accepted ? state.models[view.model_index].name + " is active for language."
                              : state.refusal();
        }
        toasts.add(result, accepted ? toast::Severity::success : toast::Severity::error, now);
        if (!accepted) {
            ImGui::EndPopup();
            return;
        }
        if (page + 1 == titles.size()) {
            state.set_instance_state(0, sim::InstanceState::running, now);
            view.page = 0;
            *open = false;
            ImGui::CloseCurrentPopup();
        } else {
            ++view.page;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
        view.page = 0;
        *open = false;
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
    if (!popup_open) {
        view.page = 0;
        *open = false;
    }
}

} // namespace aotx::ctrl::wizard
