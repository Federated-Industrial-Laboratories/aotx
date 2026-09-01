// Purpose: Guide simulated detect, build, fetch, activate, and start actions.
// Owns: Modal sequence controls and result notifications.
// Launch shape: One modal advances through five ordered actions.
// Lifetime: Completion closes the current guide sequence.
#include "wizard/wizard.hpp"

#include "replica/store.hpp"

#include "imgui.h"
#include "process/child.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <sstream>
#include <system_error>
#include <utility>

namespace aotx::ctrl::wizard {

struct DetectAction::Impl {
    pid_t child = -1;
    int output = -1;
    std::string bytes;
    std::string profile;
    std::string architecture;
    std::string result;
    bool ready = false;

    ~Impl()
    {
        if (output >= 0) close(output);
        if (child < 0) return;
        const process::End ended = process::end_child(child);
        result = ended == process::End::kill
            ? "The detection child received SIGKILL after the bounded wait."
            : "The detection child ended after SIGTERM.";
    }

    void parse()
    {
        std::istringstream lines(bytes);
        std::string line;
        while (std::getline(lines, line)) {
            std::istringstream fields(line);
            std::string profile_word;
            std::string arch_word;
            std::string extra;
            if (fields >> profile_word >> profile >> arch_word >> architecture &&
                !(fields >> extra) && profile_word == "profile" && arch_word == "arch") {
                break;
            }
            profile.clear();
            architecture.clear();
        }
        unsigned arch = 0u;
        const auto parsed = std::from_chars(architecture.data(),
                                            architecture.data() + architecture.size(), arch);
        ready = (profile == "8g" || profile == "12g" || profile == "24g" ||
                 profile == "48g") && parsed.ec == std::errc() &&
                parsed.ptr == architecture.data() + architecture.size() && arch != 0u;
        result = ready ? "The card selected the " + profile + " profile and architecture " +
                             architecture + "."
                       : "The profile detection result was refused.";
    }
};

DetectAction::DetectAction() : impl_(std::make_unique<Impl>()) {}
DetectAction::~DetectAction() = default;
bool DetectAction::start(const std::filesystem::path &script)
{
    if (impl_->child >= 0) return false;
    int pipe_fd[2];
    if (pipe2(pipe_fd, O_CLOEXEC | O_NONBLOCK) != 0) {
        impl_->result = "The profile detection child does not start.";
        return false;
    }
    const pid_t child = fork();
    if (child < 0) {
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        impl_->result = "The profile detection child does not start.";
        return false;
    }
    if (child == 0) {
        dup2(pipe_fd[1], STDOUT_FILENO);
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        execl(script.c_str(), script.c_str(), static_cast<char *>(nullptr));
        _exit(127);
    }
    close(pipe_fd[1]);
    impl_->child = child;
    impl_->output = pipe_fd[0];
    impl_->bytes.clear();
    impl_->profile.clear();
    impl_->architecture.clear();
    impl_->ready = false;
    impl_->result = "The profile detection started.";
    return true;
}
void DetectAction::tick()
{
    if (impl_->child < 0) return;
    std::array<char, 1024> buffer{};
    while (true) {
        const ssize_t count = read(impl_->output, buffer.data(), buffer.size());
        if (count > 0) {
            impl_->bytes.append(buffer.data(), static_cast<std::size_t>(count));
            continue;
        }
        if (count < 0 && errno == EINTR) continue;
        break;
    }
    int status = 0;
    if (waitpid(impl_->child, &status, WNOHANG) <= 0) return;
    impl_->child = -1;
    close(impl_->output);
    impl_->output = -1;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        impl_->result = "The profile detection failed.";
        return;
    }
    impl_->parse();
}
bool DetectAction::running() const { return impl_->child >= 0; }
bool DetectAction::ready() const { return impl_->ready; }
const std::string &DetectAction::profile() const { return impl_->profile; }
const std::string &DetectAction::architecture() const { return impl_->architecture; }
const std::string &DetectAction::result() const { return impl_->result; }

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

/* The sequence lists the store at its own model directory, which the created instance
 * uses, not the store of the selected instance. */
void refresh_store(LiveState &view, double now)
{
    if (now < view.store_read_at) return;
    view.store_read_at = now + 1.0;
    std::string reason;
    std::vector<replica::Model> models;
    if (replica::store::read(AOTX_CTRL_MODEL_CATALOG, view.models_path.data(), models,
                             reason)) {
        view.store_models = std::move(models);
        view.store_reason.clear();
    } else {
        view.store_reason = reason;
    }
}

void draw(LiveState &view, DetectAction &detect, model::StoreAction &model_action,
          instances::Lifecycle &lifecycle, replica::State &state, toast::Lane &toasts,
          double now, bool *open)
{
    (void)state;
    if (*open && !ImGui::IsPopupOpen("First run")) ImGui::OpenPopup("First run");
    bool popup_open = true;
    if (!ImGui::BeginPopupModal("First run", &popup_open,
                                ImGuiWindowFlags_AlwaysAutoResize)) {
        if (!popup_open) *open = false;
        return;
    }
    static const std::array<const char *, 6> titles = {
        "Detect", "Build", "Model", "Activate", "Start", "First say"};
    if (view.page >= titles.size()) view.page = 0u;

    /* Each page runs its own work and reports every frame. The Continue control stays
     * disabled until the page completes, so the order always holds. */
    if (view.entered != view.page) {
        view.entered = view.page;
        view.acted = false;
    }
    detect.tick();
    model_action.tick();
    ImGui::Text("%u of %zu", view.page + 1u, titles.size());
    ImGui::SeparatorText(titles[view.page]);
    bool ready = false;
    std::string status;

    if (view.page == 0u) {
        if (!detect.ready() && !detect.running() && !view.acted) {
            view.acted = true;
            const std::filesystem::path script =
                std::filesystem::path(view.build_path.data()).parent_path() /
                "tools/profile-detect.sh";
            detect.start(script);
        }
        ready = detect.ready();
        status = detect.running() ? "The card detection runs." : detect.result();
        if (!ready && !detect.running() && ImGui::Button("Detect again")) view.acted = false;
    } else if (view.page == 1u) {
        ImGui::InputText("Build directory", view.build_path.data(), view.build_path.size());
        const std::filesystem::path build = view.build_path.data();
        ready = std::filesystem::is_regular_file(build / "aotx_boot") &&
                std::filesystem::is_regular_file(build / "aotx_models");
        status = ready ? "The build directory is ready."
                       : "The build directory does not hold the programs.";
    } else if (view.page == 2u) {
        refresh_store(view, now);
        std::size_t chosen = view.store_models.size();
        for (std::size_t index = 0u; index < view.store_models.size(); ++index) {
            const replica::Model &row = view.store_models[index];
            if (row.role != "language" && row.role != "language-q4") continue;
            if (chosen == view.store_models.size()) chosen = index;
            if (row.name == view.model_name) {
                chosen = index;
                break;
            }
        }
        if (chosen < view.store_models.size()) {
            view.model_name = view.store_models[chosen].name;
            view.model_index = chosen;
        }
        if (ImGui::BeginCombo("Language model", view.model_name.c_str())) {
            for (std::size_t index = 0u; index < view.store_models.size(); ++index) {
                const replica::Model &row = view.store_models[index];
                if (row.role != "language" && row.role != "language-q4") continue;
                if (ImGui::Selectable(row.name.c_str(), row.name == view.model_name)) {
                    view.model_name = row.name;
                    view.model_index = index;
                }
            }
            ImGui::EndCombo();
        }
        const replica::Model *selected = view.model_index < view.store_models.size()
            ? &view.store_models[view.model_index] : nullptr;
        if (selected != nullptr && selected->on_disk) {
            ready = true;
            status = selected->name + " is on disk.";
        } else if (model_action.running()) {
            status = model_action.progress().empty() ? "The model fetch runs."
                                                     : model_action.progress();
        } else if (selected != nullptr) {
            if (ImGui::Button("Fetch")) {
                if (!model_action.fetch(view.build_path.data(), view.models_path.data(),
                                        selected->name)) {
                    toasts.add(model_action.refusal(), toast::Severity::error, now);
                }
            }
            status = "The model is not on disk. Fetch it to continue.";
        } else {
            status = view.store_reason.empty() ? "The store lists no language model."
                                               : view.store_reason;
        }
    } else if (view.page == 3u) {
        refresh_store(view, now);
        const replica::Model *selected = view.model_index < view.store_models.size()
            ? &view.store_models[view.model_index] : nullptr;
        if (selected != nullptr && selected->active) {
            ready = true;
            status = selected->name + " is active for " + selected->role + ".";
        } else if (model_action.running()) {
            status = "The model activation runs.";
        } else if (selected != nullptr && !view.acted) {
            view.acted = true;
            if (!model_action.activate(view.build_path.data(), view.models_path.data(),
                                       selected->role, selected->name)) {
                toasts.add(model_action.refusal(), toast::Severity::error, now);
            }
            status = "The model activation started.";
        } else {
            status = "The activation did not complete.";
            if (selected != nullptr && ImGui::Button("Activate again")) view.acted = false;
        }
    } else if (view.page == 4u) {
        ImGui::InputText("Instance name", view.instance_name.data(), view.instance_name.size());
        if (ImGui::TreeNode("Locations")) {
            ImGui::InputText("Journal directory", view.journal_path.data(),
                             view.journal_path.size());
            ImGui::InputText("Settings file", view.settings_path.data(),
                             view.settings_path.size());
            ImGui::InputText("Model directory", view.models_path.data(),
                             view.models_path.size());
            ImGui::TreePop();
        }
        if (!view.instance_created) {
            if (ImGui::Button("Create and start")) {
                instances::Definition definition;
                definition.name = view.instance_name.data();
                definition.journal = view.journal_path.data();
                definition.settings = view.settings_path.data();
                definition.build = view.build_path.data();
                definition.models = view.models_path.data();
                view.instance_index = lifecycle.instances().size();
                if (lifecycle.create(std::move(definition)) &&
                    lifecycle.start(view.instance_index)) {
                    view.instance_created = true;
                    toasts.add("The first instance starts.", toast::Severity::info, now);
                } else {
                    toasts.add(lifecycle.refusal(), toast::Severity::error, now);
                }
            }
            status = "Create the instance to continue.";
        } else {
            const std::vector<instances::LiveInstance> items = lifecycle.instances();
            if (view.instance_index < items.size()) {
                ready = items[view.instance_index].state == instances::LiveState::running;
                status = ready ? "The first instance is running."
                               : items[view.instance_index].result;
            } else {
                status = "The instance is not in the list.";
            }
        }
    } else {
        ready = true;
        status = "Finish sends the first line to the instance.";
    }

    if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());

    const bool last = view.page + 1u == titles.size();
    ImGui::BeginDisabled(!ready);
    if (ImGui::Button(last ? "Finish" : "Continue")) {
        if (last) {
            if (lifecycle.send(view.instance_index, "say Hello.")) {
                toasts.add("The first line was sent.", toast::Severity::success, now);
                view.page = 0u;
                view.entered = 0xffffffffu;
                *open = false;
                ImGui::CloseCurrentPopup();
            } else {
                toasts.add(lifecycle.refusal(), toast::Severity::error, now);
            }
        } else {
            ++view.page;
        }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
        view.page = 0u;
        view.entered = 0xffffffffu;
        *open = false;
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
    if (!popup_open) *open = false;
}

} // namespace aotx::ctrl::wizard
