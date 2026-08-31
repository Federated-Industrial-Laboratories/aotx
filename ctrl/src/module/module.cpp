// Purpose: List and import simulated skills, roles, and tools.
// Owns: Directory picker simulation and import notifications.
// Launch shape: One panel draws each module catalog group.
// Lifetime: Imported entries remain in the simulated catalog.
#include "module/module.hpp"

#include "imgui.h"

#include <cstring>

namespace aotx::ctrl::module {
namespace {

void draw_group(const sim::State &state, const char *kind, const char *title)
{
    if (!ImGui::CollapsingHeader(title, ImGuiTreeNodeFlags_DefaultOpen)) return;
    for (const sim::Module &item : state.modules) {
        if (item.kind == kind) ImGui::BulletText("%s", item.name.c_str());
    }
}

} // namespace

void draw(State &view, sim::State &state, toast::Lane &toasts, double now, bool *open)
{
    if (!ImGui::Begin("Modules", open)) {
        ImGui::End();
        return;
    }
    ImGui::SeparatorText("Directory import");
    ImGui::InputText("Directory", view.directory.data(), view.directory.size());
    const char *preview = view.directory[0] == '\0' ? "Select a directory" : view.directory.data();
    if (ImGui::BeginCombo("Browse", preview)) {
        for (const sim::ModuleDirectory &item : state.module_directories) {
            if (ImGui::Selectable(item.path.c_str(), item.path == view.directory.data())) {
                std::strncpy(view.directory.data(), item.path.c_str(), view.directory.size() - 1);
                view.directory.back() = '\0';
            }
            ImGui::SameLine();
            ImGui::TextDisabled("%s", item.kind.c_str());
        }
        ImGui::EndCombo();
    }
    if (ImGui::Button("Import")) {
        if (state.import_module(view.directory.data())) {
            toasts.add("The module directory was imported.", toast::Severity::success, now);
        } else {
            toasts.add(state.refusal(), toast::Severity::error, now);
        }
    }
    draw_group(state, "skill", "Skills");
    draw_group(state, "role", "Roles");
    draw_group(state, "tool", "Tools");
    ImGui::End();
}

} // namespace aotx::ctrl::module
