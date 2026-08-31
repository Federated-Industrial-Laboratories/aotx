// Purpose: Guide simulated detect, build, fetch, activate, and start actions.
// Owns: Modal sequence controls and result notifications.
// Launch shape: One modal advances through five ordered actions.
// Lifetime: Completion closes the current guide sequence.
#include "wizard/wizard.hpp"

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

void draw(LiveState &view, DetectAction &detect, model::StoreAction &model_action,
          instances::Lifecycle &lifecycle, replica::State &state, toast::Lane &toasts,
          double now, bool *open)
{
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
    ImGui::Text("%u of %zu", view.page + 1u, titles.size());
    ImGui::SeparatorText(titles[view.page]);
    if (!view.result.empty()) ImGui::TextWrapped("%s", view.result.c_str());

    if (view.page == 1u) {
        ImGui::InputText("Build directory", view.build_path.data(), view.build_path.size());
    } else if (view.page == 2u && !state.models().empty()) {
        if (view.model_index >= state.models().size()) view.model_index = 0u;
        if (state.models()[view.model_index].role != "language" &&
            state.models()[view.model_index].role != "language-q4") {
            for (std::size_t index = 0u; index < state.models().size(); ++index) {
                if (state.models()[index].role == "language" ||
                    state.models()[index].role == "language-q4") {
                    view.model_index = index;
                    break;
                }
            }
        }
        if (ImGui::BeginCombo("Language model", state.models()[view.model_index].name.c_str())) {
            for (std::size_t index = 0u; index < state.models().size(); ++index) {
                if (state.models()[index].role != "language" &&
                    state.models()[index].role != "language-q4") continue;
                if (ImGui::Selectable(state.models()[index].name.c_str(),
                                      index == view.model_index)) view.model_index = index;
            }
            ImGui::EndCombo();
        }
    } else if (view.page == 4u) {
        ImGui::InputText("Instance name", view.instance_name.data(), view.instance_name.size());
        ImGui::InputText("Journal directory", view.journal_path.data(), view.journal_path.size());
        ImGui::InputText("Settings file", view.settings_path.data(), view.settings_path.size());
        ImGui::InputText("Model directory", view.models_path.data(), view.models_path.size());
    }

    if (ImGui::Button(view.page + 1u == titles.size() ? "Finish" : "Continue")) {
        bool accepted = false;
        if (view.page == 0u) {
            if (!detect.ready() && !detect.running()) {
                const std::filesystem::path script =
                    std::filesystem::path(view.build_path.data()).parent_path() /
                    "tools/profile-detect.sh";
                detect.start(script);
            }
            detect.tick();
            view.result = detect.result();
            accepted = detect.ready();
        } else if (view.page == 1u) {
            const std::filesystem::path build = view.build_path.data();
            accepted = std::filesystem::is_regular_file(build / "aotx_boot") &&
                       std::filesystem::is_regular_file(build / "aotx_models");
            view.result = accepted ? "The build directory is ready."
                                   : "The build directory was refused because a program is absent.";
        } else if (view.page == 2u) {
            if (view.model_index >= state.models().size()) {
                view.result = "The model choice was refused because no model is selected.";
            } else {
                const replica::Model &selected = state.models()[view.model_index];
                if (selected.on_disk) {
                    accepted = true;
                    view.result = selected.name + " is present on disk.";
                } else if (!model_action.running() &&
                           model_action.fetch(view.build_path.data(), view.models_path.data(),
                                              selected.name)) {
                    view.result = "The model fetch started.";
                } else {
                    view.result = model_action.running() ? "The model fetch is active."
                                                         : model_action.refusal();
                }
            }
        } else if (view.page == 3u) {
            if (view.model_index < state.models().size()) {
                const replica::Model &selected = state.models()[view.model_index];
                if (selected.active) {
                    accepted = true;
                    view.result = selected.name + " is active for " + selected.role + ".";
                } else if (!model_action.running() &&
                           model_action.activate(view.build_path.data(), view.models_path.data(),
                                                 selected.role, selected.name)) {
                    view.result = "The model activation started.";
                } else {
                    view.result = model_action.running() ? "The model activation is active."
                                                         : model_action.refusal();
                }
            }
        } else if (view.page == 4u) {
            if (!view.instance_created) {
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
                    view.result = "The first instance child started.";
                } else {
                    view.result = lifecycle.refusal();
                }
            } else {
                const std::vector<instances::LiveInstance> items = lifecycle.instances();
                accepted = view.instance_index < items.size() &&
                           items[view.instance_index].state == instances::LiveState::running;
                view.result = accepted ? "The first instance is running."
                                       : "The first instance is attaching.";
            }
        } else {
            accepted = lifecycle.send(view.instance_index, "say Hello.");
            view.result = accepted ? "The first say line was sent." : lifecycle.refusal();
        }
        toasts.add(view.result, accepted ? toast::Severity::success : toast::Severity::info, now);
        if (accepted) {
            if (view.page + 1u == titles.size()) {
                view.page = 0u;
                *open = false;
                ImGui::CloseCurrentPopup();
            } else {
                ++view.page;
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
        view.page = 0u;
        *open = false;
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
    if (!popup_open) *open = false;
}

} // namespace aotx::ctrl::wizard
