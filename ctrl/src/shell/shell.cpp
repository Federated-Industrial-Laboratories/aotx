// Purpose: Draw the main menu and maintain the default dock layout.
// Owns: Dock nodes, menu actions, and the information window.
// Launch shape: One full-viewport window contains one dock space.
// Lifetime: ImGui stores the layout in the user configuration file.
#include "shell/shell.hpp"

#include "imgui.h"
#include "imgui_internal.h"

namespace aotx::ctrl::shell {
namespace {

void draw_menu(State &shell, sim::State &simulated)
{
    if (!ImGui::BeginMenuBar()) {
        return;
    }
    if (ImGui::BeginMenu("Instances")) {
        for (std::size_t index = 0; index < simulated.instances.size(); ++index) {
            if (ImGui::MenuItem(simulated.instances[index].name.c_str(), nullptr,
                                simulated.selected_instance == index)) {
                simulated.selected_instance = index;
                simulated.instance_selection_requested = true;
                shell.show_instances = true;
            }
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Windows")) {
        ImGui::MenuItem("Chat", nullptr, &shell.show_chat);
        ImGui::MenuItem("Instances", nullptr, &shell.show_instances);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        if (ImGui::MenuItem("Rebuild layout")) {
            shell.rebuild_layout = true;
            shell.show_chat = true;
            shell.show_instances = true;
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("About AOTX-CTRL")) {
            shell.show_about = true;
        }
        ImGui::EndMenu();
    }
    ImGui::EndMenuBar();
}

void rebuild(ImGuiID dock_id, const ImGuiViewport *viewport)
{
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, viewport->WorkSize);
    ImGuiID center = dock_id;
    ImGuiID left = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.27f, nullptr, &center);
    ImGui::DockBuilderDockWindow("Instances", left);
    ImGui::DockBuilderDockWindow("Chat", center);
    ImGui::DockBuilderFinish(dock_id);
}

} // namespace

void draw_dock_space(State &shell, sim::State &simulated)
{
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking |
                                   ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                                   ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus |
                                   ImGuiWindowFlags_NoNavFocus;
    ImGui::Begin("AOTX-CTRL dock space", nullptr, flags);
    ImGui::PopStyleVar(3);
    draw_menu(shell, simulated);

    const ImGuiID dock_id = ImGui::GetID("AOTX-CTRL dock space node");
    const bool needs_layout = shell.rebuild_layout || ImGui::DockBuilderGetNode(dock_id) == nullptr;
    ImGui::DockSpace(dock_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
    if (needs_layout) {
        rebuild(dock_id, viewport);
        shell.rebuild_layout = false;
    }
    ImGui::End();

    if (shell.show_about) {
        ImGui::OpenPopup("About AOTX-CTRL");
        shell.show_about = false;
    }
    if (ImGui::BeginPopupModal("About AOTX-CTRL", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("AOTX-CTRL uses simulated data.");
        ImGui::TextUnformatted("No live system is attached.");
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

} // namespace aotx::ctrl::shell
