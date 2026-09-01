// Purpose: Guide detect, build, model, activation, start, and first-say actions.
// Owns: Modal sequence controls and result notifications.
// Launch shape: One modal advances through six ordered actions.
// Lifetime: Completion closes the current first-run sequence.
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

namespace {

std::size_t reply_count(const replica::State *state)
{
    if (state == nullptr) return 0u;
    std::size_t count = 0u;
    for (const replica::Agent &agent : state->agents()) {
        for (const replica::TranscriptEvent &event : agent.transcript) {
            if (event.kind == "reply") ++count;
        }
    }
    return count;
}

void add_phase(LiveState &view, const std::string &phase)
{
    if ((phase != "placing" && phase != "replaying" && phase != "running") ||
        phase == view.last_phase) return;
    view.last_phase = phase;
    if (!view.phase_trace.empty()) view.phase_trace += " -> ";
    view.phase_trace += phase;
}

} // namespace

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
        view.store_read = true;
    } else {
        view.store_reason = reason;
        view.store_read = false;
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
        view.action.reset();
        if (view.page == static_cast<unsigned>(Page::start)) {
            view.last_phase.clear();
            view.phase_trace.clear();
        }
        if (view.page == static_cast<unsigned>(Page::first_say)) {
            view.reply_count = reply_count(lifecycle.replica(view.instance_index));
        }
    }
    detect.tick();
    model_action.tick();
    ImGui::Text("%u of %zu", view.page + 1u, titles.size());
    ImGui::SeparatorText(titles[view.page]);
    bool ready = false;
    std::string status;

    if (view.page == 0u) {
        if (!detect.ready() && !detect.running() && view.action.begin()) {
            const std::filesystem::path script =
                std::filesystem::path(view.build_path.data()).parent_path() /
                "tools/profile-detect.sh";
            if (!detect.start(script)) view.action.complete(false);
        }
        if (view.action.active() && !detect.running()) view.action.complete(detect.ready());
        GateFacts facts;
        facts.detected = detect.ready();
        ready = gate_open(Page::detect, facts);
        status = detect.running() ? "The card detection runs." : detect.result();
        if (!ready && !detect.running() && ImGui::Button("Retry")) view.action.reset();
        if (!ready && !detect.running()) {
            ImGui::SameLine();
            ImGui::TextDisabled("Run the card detection again.");
        }
    } else if (view.page == 1u) {
        ImGui::InputText("Build directory", view.build_path.data(), view.build_path.size());
        const std::filesystem::path build = view.build_path.data();
        GateFacts facts;
        facts.build_ready = std::filesystem::is_regular_file(build / "aotx_boot") &&
                            std::filesystem::is_regular_file(build / "aotx_models");
        ready = gate_open(Page::build, facts);
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
        ImGui::BeginDisabled(view.action.active());
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
        ImGui::EndDisabled();
        const replica::Model *selected = view.model_index < view.store_models.size()
            ? &view.store_models[view.model_index] : nullptr;
        if (view.action.active() && selected != nullptr && selected->on_disk) {
            view.action.complete(true);
        } else if (view.action.active() && model_action.finished() &&
                   !model_action.succeeded()) {
            view.action.complete(false);
        }
        if (selected != nullptr && selected->on_disk) {
            status = selected->name + " is on disk.";
        } else if (model_action.running()) {
            status = model_action.progress().empty() ? "The model fetch runs."
                                                     : model_action.progress();
        } else if (view.action.active() && model_action.succeeded()) {
            status = "The model fetch completed. The catalog refresh is pending.";
        } else if (selected != nullptr && view.action.failed()) {
            if (ImGui::Button("Retry")) view.action.reset();
            ImGui::SameLine();
            ImGui::TextDisabled("Run the model fetch again.");
            status = model_action.progress().empty() ? "The model fetch did not complete."
                                                     : model_action.progress();
        } else if (selected != nullptr) {
            if (ImGui::Button("Fetch") && view.action.begin()) {
                if (!model_action.fetch(view.build_path.data(), view.models_path.data(),
                                        selected->name)) {
                    view.action.complete(false);
                    toasts.add(model_action.refusal(), toast::Severity::error, now);
                }
            }
            ImGui::SameLine();
            ImGui::TextDisabled("Fetch the selected model to disk.");
            status = "The model is not on disk. Fetch it to continue.";
        } else {
            status = view.store_reason.empty() ? "The store lists no language model."
                                               : view.store_reason;
        }
        GateFacts facts;
        facts.catalog_read = view.store_read;
        facts.model_on_disk = selected != nullptr && selected->on_disk;
        ready = gate_open(Page::model, facts);
    } else if (view.page == 3u) {
        refresh_store(view, now);
        const replica::Model *selected = view.model_index < view.store_models.size()
            ? &view.store_models[view.model_index] : nullptr;
        if (selected != nullptr && selected->active) {
            if (view.action.active()) view.action.complete(true);
            status = selected->name + " is active for " + selected->role + ".";
        } else if (model_action.running()) {
            status = "The model activation runs.";
        } else if (view.action.active() && model_action.finished() &&
                   !model_action.succeeded()) {
            view.action.complete(false);
            status = "The model activation did not complete.";
        } else if (view.action.active() && model_action.succeeded()) {
            status = "The model activation completed. The catalog refresh is pending.";
        } else if (selected != nullptr && view.action.begin()) {
            if (!model_action.activate(view.build_path.data(), view.models_path.data(),
                                       selected->role, selected->name)) {
                view.action.complete(false);
                toasts.add(model_action.refusal(), toast::Severity::error, now);
            }
            status = "The model activation started.";
        } else {
            status = "The activation did not complete.";
            if (selected != nullptr && view.action.failed() && ImGui::Button("Retry")) {
                view.action.reset();
            }
            if (selected != nullptr) {
                ImGui::SameLine();
                ImGui::TextDisabled("Run the model activation again.");
            }
        }
        GateFacts facts;
        facts.catalog_read = view.store_read;
        facts.model_active = selected != nullptr && selected->active;
        ready = gate_open(Page::activate, facts);
    } else if (view.page == 4u) {
        ImGui::BeginDisabled(view.instance_created || view.action.active());
        ImGui::InputText("Instance name", view.instance_name.data(), view.instance_name.size());
        if (ImGui::TreeNode("Locations")) {
            ImGui::InputText("Journal directory", view.journal_path.data(),
                             view.journal_path.size());
            ImGui::InputText("Settings file", view.settings_path.data(),
                             view.settings_path.size());
            ImGui::InputText("Model directory", view.models_path.data(),
                             view.models_path.size());
            ImGui::InputText("Tool root", view.tools_path.data(), view.tools_path.size());
            ImGui::TreePop();
        }
        ImGui::EndDisabled();
        if (!view.instance_created && view.action.begin()) {
            instances::Definition definition;
            definition.name = view.instance_name.data();
            definition.journal = view.journal_path.data();
            definition.settings = view.settings_path.data();
            definition.build = view.build_path.data();
            definition.models = view.models_path.data();
            definition.tools = view.tools_path.data();
            view.instance_index = lifecycle.instances().size();
            if (lifecycle.create(std::move(definition))) {
                view.instance_created = true;
            }
            if (view.instance_created && lifecycle.start(view.instance_index)) {
                toasts.add("The first instance starts.", toast::Severity::info, now);
            } else {
                view.action.complete(false);
                view.result = lifecycle.refusal();
                toasts.add(view.result, toast::Severity::error, now);
            }
        }
        if (view.instance_created && !view.action.active() && !view.action.completed() &&
            !view.action.failed() &&
            view.action.begin() && lifecycle.start(view.instance_index)) {
            view.result.clear();
            toasts.add("The first instance starts.", toast::Severity::info, now);
        } else if (view.instance_created && view.action.active()) {
            const std::vector<instances::LiveInstance> started = lifecycle.instances();
            if (view.instance_index < started.size() && started[view.instance_index].process < 0 &&
                !start_gate_open(started[view.instance_index])) {
                view.action.complete(false);
                view.result = lifecycle.refusal().empty()
                    ? started[view.instance_index].result : lifecycle.refusal();
            }
        }
        const std::vector<instances::LiveInstance> items = lifecycle.instances();
        if (view.instance_created && view.instance_index < items.size()) {
            const instances::LiveInstance &item = items[view.instance_index];
            add_phase(view, item.phase);
            ready = start_gate_open(item);
            if (ready && view.action.active()) view.action.complete(true);
            status = ready ? "The first instance is running and its socket answers."
                           : item.result;
            if (item.process < 0 && !ready && view.action.active()) {
                view.action.complete(false);
            }
        } else if (view.result.empty()) {
            status = "The first instance starts on this page.";
        } else {
            status = view.result;
        }
        if (!view.phase_trace.empty()) {
            ImGui::Text("Phases: %s", view.phase_trace.c_str());
        }
        if (!ready && view.action.failed() && ImGui::Button("Retry")) {
            view.result.clear();
            view.action.reset();
        }
    } else {
        replica::State *started = lifecycle.replica(view.instance_index);
        const std::size_t replies = reply_count(started);
        if (replies > view.reply_count && view.action.active()) view.action.complete(true);
        if (!view.action.completed() && !view.action.active() && view.action.begin()) {
            if (!lifecycle.send(view.instance_index, "say Hello.")) {
                view.action.complete(false);
                view.result = lifecycle.refusal();
            }
        }
        GateFacts facts;
        facts.reply_received = replies > view.reply_count;
        ready = gate_open(Page::first_say, facts);
        status = ready ? "The first reply arrived."
                       : view.action.active() ? "The first reply is pending."
                                              : view.result;
        if (!ready && view.action.failed() && ImGui::Button("Retry")) {
            view.reply_count = replies;
            view.result.clear();
            view.action.reset();
        }
    }

    if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());

    const bool last = view.page + 1u == titles.size();
    ImGui::BeginDisabled(!ready);
    if (ImGui::Button(last ? "Finish" : "Continue")) {
        if (last) {
            toasts.add("The first reply arrived.", toast::Severity::success, now);
            view.page = 0u;
            view.entered = 0xffffffffu;
            *open = false;
            ImGui::CloseCurrentPopup();
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
