// Purpose: List changed module and voice profile disk copies.
// Owns: No disk copy; synchronization uses existing socket commands.
// Launch shape: One interface thread scans bounded row paths each frame.
// Lifetime: The device remains authoritative outside an explicit Sync act.
#include "sync/sync.hpp"

#include "imgui.h"
#include "replica/schema.hpp"

#include <algorithm>
#include <vector>

namespace aotx::ctrl::sync {
namespace {

std::filesystem::file_time_type newest(const std::filesystem::path &path)
{
    std::error_code error;
    auto latest = std::filesystem::last_write_time(path, error);
    if (error || !std::filesystem::is_directory(path, error)) return latest;
    unsigned seen = 0u;
    for (std::filesystem::recursive_directory_iterator item(path, error), end;
         !error && item != end && seen < 4096u; item.increment(error), ++seen) {
        if (!item->is_regular_file(error) || error) {
            error.clear();
            continue;
        }
        const auto changed = item->last_write_time(error);
        if (!error && changed > latest) latest = changed;
        error.clear();
    }
    return latest;
}

State::Stamp &stamp(State &view, const std::filesystem::path &path)
{
    return view.stamps[path.lexically_normal().string()];
}

void take_results(State &view, const replica::State &state)
{
    if (!view.notes_bound) {
        view.notes_bound = true;
        view.notes_seen = state.notes().size();
        return;
    }
    if (state.notes().size() < view.notes_seen) view.notes_seen = 0u;
    for (; view.notes_seen < state.notes().size(); ++view.notes_seen) {
        const std::string &line = state.notes()[view.notes_seen].text;
        std::string name;
        std::string kind;
        if (replica::schema::import_result(line, name, kind)) {
            for (const replica::Module &module : state.modules()) {
                if (module.name == name) {
                    State::Stamp &held = stamp(view, module.directory);
                    held.last_import = newest(module.directory);
                    held.known = true;
                    view.result = line;
                    break;
                }
            }
            continue;
        }
        std::string loaded;
        if (replica::schema::model_load(line, "language", loaded) ||
            replica::schema::model_load(line, "language-q4", loaded)) {
            const std::filesystem::path voice = state.models_directory() / "voice";
            for (auto &entry : view.stamps) {
                if (entry.first.rfind(voice.lexically_normal().string(), 0u) == 0u) {
                    entry.second.last_import = newest(entry.first);
                    entry.second.known = true;
                }
            }
            view.result = line;
        }
    }
}

void state_text(const std::filesystem::path &path, State::Stamp &held)
{
    const DiskState state = held.known ? disk_state(newest(path), held.last_import)
                                       : DiskState::in_step;
    ImGui::TextUnformatted(state_word(held.known, state));
}

void module_rows(State &view, replica::State &state, client::Client &client)
{
    ImGui::SeparatorText("Modules");
    for (const replica::Module &module : state.modules()) {
        const std::filesystem::path path = module.directory;
        State::Stamp &held = stamp(view, path);
        ImGui::PushID(("module-" + module.name).c_str());
        ImGui::Text("%s (%s)", module.name.c_str(), module.kind.c_str());
        ImGui::SameLine();
        state_text(path, held);
        if (ImGui::Button("Sync")) client.send_line("import " + path.string());
        ImGui::SameLine();
        ImGui::TextDisabled("Import this disk copy into the running system.");
        ImGui::PopID();
    }
}

std::vector<std::filesystem::path> profiles(const std::filesystem::path &directory)
{
    std::vector<std::filesystem::path> out;
    std::error_code error;
    for (std::filesystem::directory_iterator item(directory, error), end;
         !error && item != end; item.increment(error)) {
        if (item->is_regular_file(error) && !error && item->path().extension() == ".profile") {
            out.push_back(item->path());
        }
        error.clear();
    }
    std::sort(out.begin(), out.end());
    return out;
}

const replica::Model *language_model(const replica::State &state)
{
    for (const replica::Model &model : state.models()) {
        if (model.active && (model.role == "language" || model.role == "language-q4")) {
            return &model;
        }
    }
    return nullptr;
}

void profile_rows(State &view, replica::State &state, client::Client &client)
{
    ImGui::SeparatorText("Profiles");
    const replica::Model *model = language_model(state);
    for (const std::filesystem::path &path : profiles(state.models_directory() / "voice")) {
        State::Stamp &held = stamp(view, path);
        ImGui::PushID(path.string().c_str());
        ImGui::TextUnformatted(path.filename().string().c_str());
        ImGui::SameLine();
        state_text(path, held);
        if (model == nullptr) ImGui::BeginDisabled();
        if (ImGui::Button("Sync") && model != nullptr) {
            client.send_line("model load " + model->role + " " + model->name);
        }
        if (model == nullptr) ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled("Reload the model to import this profile disk copy.");
        ImGui::PopID();
    }
}

} // namespace

DiskState disk_state(std::filesystem::file_time_type disk,
                     std::filesystem::file_time_type last_import)
{
    return disk > last_import ? DiskState::disk_newer : DiskState::in_step;
}

const char *state_word(bool known, DiskState state)
{
    if (!known) return "not known";
    return state == DiskState::disk_newer ? "disk newer" : "in step";
}

bool verify_state_logic()
{
    const auto base = std::filesystem::file_time_type{};
    const auto later = base + std::chrono::seconds(1);
    const bool fixture = disk_state(base, base) == DiskState::in_step &&
                         disk_state(later, base) == DiskState::disk_newer;
    const bool mutation_caught = !(disk_state(base, later) == DiskState::disk_newer);
    const bool words = std::string(state_word(false, DiskState::in_step)) == "not known" &&
                       std::string(state_word(false, DiskState::disk_newer)) == "not known" &&
                       std::string(state_word(true, DiskState::disk_newer)) == "disk newer" &&
                       std::string(state_word(true, DiskState::in_step)) == "in step";
    return fixture && mutation_caught && words;
}

void draw(State &view, replica::State &state, client::Client &client, bool *open)
{
    if (!ImGui::Begin("Sync", open)) {
        ImGui::End();
        return;
    }
    take_results(view, state);
    ImGui::TextWrapped("Sync imports a disk edit into the running system.");
    module_rows(view, state, client);
    profile_rows(view, state, client);
    if (!view.result.empty()) {
        ImGui::SeparatorText("Result");
        ImGui::TextWrapped("%s", view.result.c_str());
    }
    ImGui::End();
}

} // namespace aotx::ctrl::sync
