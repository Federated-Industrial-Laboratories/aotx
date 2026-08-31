// Purpose: Draw the main menu and maintain the default dock layout.
// Owns: Dock nodes, menu actions, and the information window.
// Launch shape: One full-viewport window contains one dock space.
// Lifetime: ImGui stores the layout in the user configuration file.
#include "shell/shell.hpp"

#include "chat/chat.hpp"
#include "client/client.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include "replica/replica.hpp"

namespace aotx::ctrl::shell {
namespace {

void draw_menu(State &shell, sim::State &simulated)
{
    if (!ImGui::BeginMenuBar()) {
        return;
    }
    if (ImGui::BeginMenu("Instances")) {
        if (ImGui::MenuItem("New instance")) {
            simulated.instance_creation_requested = true;
            shell.show_instances = true;
        }
        ImGui::Separator();
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
        if (ImGui::MenuItem("New conversation")) {
            simulated.create_conversation(simulated.selected_instance);
        }
        ImGui::Separator();
        for (std::size_t index = 0; index < simulated.conversations.size(); ++index) {
            sim::Conversation &conversation = simulated.conversations[index];
            const std::string label = conversation.name + "##window-menu-" +
                                      std::to_string(index);
            if (ImGui::MenuItem(label.c_str(), nullptr, conversation.window_open)) {
                conversation.window_open = true;
                simulated.selected_conversation = index;
                simulated.conversation_selection_requested = true;
            }
        }
        ImGui::Separator();
        ImGui::MenuItem("Instances", nullptr, &shell.show_instances);
        ImGui::MenuItem("Control", nullptr, &shell.show_control);
        ImGui::MenuItem("Models", nullptr, &shell.show_models);
        ImGui::MenuItem("Modules", nullptr, &shell.show_modules);
        ImGui::MenuItem("Settings", nullptr, &shell.show_settings);
        ImGui::MenuItem("Monitor", nullptr, &shell.show_monitor);
        ImGui::MenuItem("Transcripts", nullptr, &shell.show_browser);
        ImGui::MenuItem("First run", nullptr, &shell.show_wizard);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        if (ImGui::MenuItem("Rebuild layout")) {
            shell.rebuild_layout = true;
            simulated.conversations.front().window_open = true;
            shell.show_instances = true;
            shell.show_control = true;
            shell.show_models = true;
            shell.show_modules = true;
            shell.show_settings = true;
            shell.show_monitor = true;
            shell.show_browser = true;
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

void rebuild(ImGuiID dock_id, const ImGuiViewport *viewport, const sim::State &simulated)
{
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, viewport->WorkSize);
    ImGuiID center = dock_id;
    ImGuiID left = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.27f, nullptr, &center);
    ImGuiID right = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.30f, nullptr, &center);
    ImGuiID lower = ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.30f, nullptr, &center);
    ImGui::DockBuilderDockWindow("Instances", left);
    ImGui::DockBuilderDockWindow("Control", left);
    const std::string first_chat = chat::window_name(simulated.conversations.front(), 0);
    ImGui::DockBuilderDockWindow(first_chat.c_str(), center);
    ImGui::DockBuilderDockWindow("Models", right);
    ImGui::DockBuilderDockWindow("Modules", right);
    ImGui::DockBuilderDockWindow("Settings", right);
    ImGui::DockBuilderDockWindow("Monitor", lower);
    ImGui::DockBuilderDockWindow("Transcripts", lower);
    ImGui::DockBuilderFinish(dock_id);
}

void draw_live_menu(State &shell, replica::State &state)
{
    if (!ImGui::BeginMenuBar()) return;
    if (ImGui::BeginMenu("Instances")) {
        ImGui::MenuItem(state.journal().string().c_str(), nullptr, true, false);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Windows")) {
        for (replica::Agent &agent : state.agents()) {
            const std::string label = agent.conversation + "##window-agent-" +
                                      std::to_string(agent.id);
            ImGui::MenuItem(label.c_str(), nullptr, &agent.window_open);
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        if (ImGui::MenuItem("Rebuild layout")) shell.rebuild_layout = true;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("About AOTX-CTRL")) shell.show_about = true;
        ImGui::EndMenu();
    }
    ImGui::EndMenuBar();
}

void rebuild_live(ImGuiID dock_id, const ImGuiViewport *viewport, const replica::State &state)
{
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, viewport->WorkSize);
    for (const replica::Agent &agent : state.agents()) {
        const std::string name = chat::window_name(agent);
        ImGui::DockBuilderDockWindow(name.c_str(), dock_id);
    }
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
        rebuild(dock_id, viewport, simulated);
        shell.rebuild_layout = false;
    }
    ImGui::End();

    if (shell.show_about) {
        ImGui::OpenPopup("About AOTX-CTRL");
        shell.show_about = false;
    }
    if (ImGui::BeginPopupModal("About AOTX-CTRL", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("The program runs on simulated data.");
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void draw_dock_space(State &shell, replica::State &state, const client::Client &client)
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
    draw_live_menu(shell, state);
    const ImGuiID dock_id = ImGui::GetID("AOTX-CTRL dock space node");
    const bool needs_layout = shell.rebuild_layout || ImGui::DockBuilderGetNode(dock_id) == nullptr;
    ImGui::DockSpace(dock_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
    if (needs_layout) {
        rebuild_live(dock_id, viewport, state);
        shell.rebuild_layout = false;
    }
    ImGui::End();

    if (shell.show_about) {
        ImGui::OpenPopup("About AOTX-CTRL");
        shell.show_about = false;
    }
    if (ImGui::BeginPopupModal("About AOTX-CTRL", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("The journal directory is %s. The connection is %s.",
                    state.journal().string().c_str(), client.connection());
        if (ImGui::Button("Close")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

} // namespace aotx::ctrl::shell
