// Purpose: List and import simulated skills, roles, and tools.
// Owns: Directory picker simulation and import notifications.
// Launch shape: One panel draws each module catalog group.
// Lifetime: Imported entries remain in the simulated catalog.
#include "module/module.hpp"

#include "imgui.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <vector>

namespace aotx::ctrl::module {
namespace {

void draw_group(const sim::State &state, const char *kind, const char *title)
{
    if (!ImGui::CollapsingHeader(title, ImGuiTreeNodeFlags_DefaultOpen)) return;
    for (const sim::Module &item : state.modules) {
        if (item.kind == kind) ImGui::BulletText("%s", item.name.c_str());
    }
}

void draw_group(const replica::State &state, const char *kind, const char *title)
{
    if (!ImGui::CollapsingHeader(title, ImGuiTreeNodeFlags_DefaultOpen)) return;
    for (const replica::Module &item : state.modules()) {
        if (item.kind == kind) ImGui::BulletText("%s", item.name.c_str());
    }
}

void copy_path(State &view, const std::filesystem::path &path)
{
    const std::string text = path.string();
    std::strncpy(view.directory.data(), text.c_str(), view.directory.size() - 1u);
    view.directory.back() = '\0';
}

void picker(State &view)
{
    if (view.browser.empty()) view.browser = std::filesystem::current_path();
    if (ImGui::Button("Use this directory")) {
        copy_path(view, view.browser);
        view.show_picker = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Parent")) view.browser = view.browser.parent_path();
    ImGui::TextDisabled("%s", view.browser.string().c_str());
    std::vector<std::filesystem::path> directories;
    std::error_code error;
    for (std::filesystem::directory_iterator item(view.browser, error), end;
         !error && item != end; item.increment(error)) {
        if (item->is_directory(error) && !error) directories.push_back(item->path());
        error.clear();
    }
    std::sort(directories.begin(), directories.end());
    for (const std::filesystem::path &path : directories) {
        if (ImGui::Selectable(path.filename().string().c_str())) view.browser = path;
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

void draw(State &view, replica::State &state, client::Client &client,
          toast::Lane &toasts, double now, bool *open)
{
    if (!ImGui::Begin("Modules", open)) {
        ImGui::End();
        return;
    }
    ImGui::SeparatorText("Directory import");
    ImGui::InputText("Directory", view.directory.data(), view.directory.size());
    if (ImGui::Button("Browse")) view.show_picker = !view.show_picker;
    ImGui::SameLine();
    if (ImGui::Button("Import")) {
        if (view.directory[0] == '\0') {
            toasts.add("The import was refused because no directory was selected.",
                       toast::Severity::error, now);
        } else {
            client.send_line("import " + std::string(view.directory.data()));
        }
    }
    if (view.show_picker) picker(view);
    draw_group(state, "skill", "Skills");
    draw_group(state, "role", "Roles");
    draw_group(state, "tool", "Tools");
    ImGui::End();
}

} // namespace aotx::ctrl::module
