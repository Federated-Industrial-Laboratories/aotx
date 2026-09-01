// Purpose: List, fetch, and activate simulated model entries.
// Owns: Model action buttons and their result notifications.
// Launch shape: One panel draws all catalog entries each frame.
// Lifetime: Fetch progress advances in the simulated state.
#include "model/model.hpp"

#include "imgui.h"
#include "process/child.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

#include <fcntl.h>

#include <array>
#include <cerrno>
#include <utility>

namespace aotx::ctrl::model {
namespace {

void activate(sim::State &state, toast::Lane &toasts, std::size_t index,
              const char *role, double now)
{
    if (state.activate_model(index, role)) {
        toasts.add(state.models[index].name + " is active for " + role + ".",
                   toast::Severity::success, now);
    } else {
        toasts.add(state.refusal(), toast::Severity::error, now);
    }
}

} // namespace

struct StoreAction::Impl {
    pid_t child = -1;
    int out = -1;
    std::string action;
    std::string result;
    std::string refusal;
    std::string progress;
    std::string partial;
    bool finished = false;
    bool succeeded = false;

    ~Impl()
    {
        if (out >= 0) ::close(out);
        if (child < 0) return;
        const process::End ended = process::end_child(child);
        result = ended == process::End::kill
            ? "The model child received SIGKILL after the bounded wait."
            : "The model child ended after SIGTERM.";
    }

    bool start(const std::filesystem::path &build, const std::filesystem::path &models,
               const std::string &command, const std::string &first,
               const std::string &second)
    {
        refusal.clear();
        if (child >= 0) {
            refusal = "The model action was refused because another action is active.";
            return false;
        }
        const std::filesystem::path program = build / "aotx_models";
        if (!std::filesystem::is_regular_file(program)) {
            refusal = "The model action was refused because aotx_models is not in the build.";
            return false;
        }
        int lines[2] = {-1, -1};
        if (::pipe(lines) != 0) {
            refusal = "The model action was refused because the pipe does not open.";
            return false;
        }
        child = fork();
        if (child < 0) {
            ::close(lines[0]);
            ::close(lines[1]);
            refusal = "The model action was refused because the child does not start.";
            return false;
        }
        if (child == 0) {
            ::dup2(lines[1], 1);
            ::dup2(lines[1], 2);
            ::close(lines[0]);
            ::close(lines[1]);
            if (second.empty()) {
                execl(program.c_str(), program.c_str(), "--dir", models.c_str(),
                      command.c_str(), first.c_str(), static_cast<char *>(nullptr));
            } else {
                execl(program.c_str(), program.c_str(), "--dir", models.c_str(),
                      command.c_str(), first.c_str(), second.c_str(),
                      static_cast<char *>(nullptr));
            }
            _exit(127);
        }
        ::close(lines[1]);
        out = lines[0];
        ::fcntl(out, F_SETFL, O_NONBLOCK);
        progress.clear();
        partial.clear();
        finished = false;
        succeeded = false;
        action = command + " " + (second.empty() ? first : second);
        result = "The model " + action + " started.";
        return true;
    }
};

StoreAction::StoreAction() : impl_(std::make_unique<Impl>()) {}
StoreAction::~StoreAction() = default;
bool StoreAction::fetch(const std::filesystem::path &build,
                        const std::filesystem::path &models, const std::string &name)
{
    return impl_->start(build, models, "fetch", name, "");
}
bool StoreAction::activate(const std::filesystem::path &build,
                           const std::filesystem::path &models, const std::string &role,
                           const std::string &name)
{
    return impl_->start(build, models, "activate", role, name);
}
void StoreAction::tick()
{
    if (impl_->out >= 0) {
        std::array<char, 512> bytes{};
        ssize_t got = 0;
        while ((got = ::read(impl_->out, bytes.data(), bytes.size())) > 0) {
            impl_->partial.append(bytes.data(), static_cast<std::size_t>(got));
        }
        std::size_t mark = 0u;
        while ((mark = impl_->partial.find('\n')) != std::string::npos) {
            if (mark > 0u) impl_->progress = impl_->partial.substr(0u, mark);
            impl_->partial.erase(0u, mark + 1u);
        }
    }
    if (impl_->child < 0) return;
    int status = 0;
    const pid_t ended = waitpid(impl_->child, &status, WNOHANG);
    if (ended <= 0) return;
    impl_->child = -1;
    impl_->finished = true;
    if (impl_->out >= 0) {
        ::close(impl_->out);
        impl_->out = -1;
    }
    const std::string tail = impl_->progress.empty() ? "" : ": " + impl_->progress;
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        impl_->succeeded = true;
        impl_->result = "The model " + impl_->action + " completed" + tail + ".";
    } else {
        impl_->result = "The model " + impl_->action + " failed" + tail + ".";
    }
}
bool StoreAction::running() const { return impl_->child >= 0; }
bool StoreAction::finished() const { return impl_->finished; }
bool StoreAction::succeeded() const { return impl_->succeeded; }
const std::string &StoreAction::progress() const { return impl_->progress; }
std::string StoreAction::take_result()
{
    std::string out;
    out.swap(impl_->result);
    return out;
}
const std::string &StoreAction::refusal() const { return impl_->refusal; }

void draw(sim::State &state, toast::Lane &toasts, double now, bool *open)
{
    if (!ImGui::Begin("Models", open)) {
        ImGui::End();
        return;
    }
    ImGui::SeparatorText("Role assignments");
    static const char *roles[] = {"language", "embedding", "reranker"};
    for (const char *role : roles) {
        const sim::Model *assigned = nullptr;
        for (const sim::Model &item : state.models) {
            const std::string &value = role == roles[0] ? item.language_role
                                      : role == roles[1] ? item.embedding_role
                                                         : item.reranker_role;
            if (!value.empty()) assigned = &item;
        }
        ImGui::Text("%s: %s", role, assigned == nullptr ? "not assigned" : assigned->name.c_str());
    }
    ImGui::SeparatorText("Catalog");
    for (std::size_t index = 0; index < state.models.size(); ++index) {
        sim::Model &item = state.models[index];
        ImGui::PushID(static_cast<int>(index));
        ImGui::TextUnformatted(item.name.c_str());
        ImGui::TextDisabled("%s", item.state.c_str());
        if (item.state == "fetching") {
            ImGui::ProgressBar(item.fetch_progress, ImVec2(-1.0f, 0.0f));
        } else if (item.state == "catalog") {
            if (ImGui::Button("Fetch")) {
                if (state.fetch_model(index)) {
                    toasts.add(item.name + " fetch started.", toast::Severity::info, now);
                } else {
                    toasts.add(state.refusal(), toast::Severity::error, now);
                }
            }
        }
        if (item.state == "on disk" || item.state == "active") {
            if (ImGui::Button("Use for language")) activate(state, toasts, index, "language", now);
            ImGui::SameLine();
            if (ImGui::Button("Use for embedding")) activate(state, toasts, index, "embedding", now);
            ImGui::SameLine();
            if (ImGui::Button("Use for reranker")) {
                activate(state, toasts, index, "reranker", now);
            }
        }
        if (!item.language_role.empty()) ImGui::Text("Language: %s", item.language_role.c_str());
        if (!item.embedding_role.empty()) ImGui::Text("Embedding: %s", item.embedding_role.c_str());
        if (!item.reranker_role.empty()) {
            ImGui::Text("Reranker: %s", item.reranker_role.c_str());
        }
        ImGui::Separator();
        ImGui::PopID();
    }
    ImGui::End();
}

void draw(StoreAction &action, const std::filesystem::path &build, replica::State &state,
          client::Client &client, toast::Lane &toasts, double now, bool *open)
{
    if (!ImGui::Begin("Models", open)) {
        ImGui::End();
        return;
    }
    if (action.running() && !action.progress().empty()) {
        ImGui::TextUnformatted(action.progress().c_str());
        ImGui::Separator();
    }
    ImGui::Text("Store: %s", state.models_directory().string().c_str());
    ImGui::Text("Resident language: %s", state.language_model().c_str());
    ImGui::SeparatorText("Active manifest roles");
    bool assigned = false;
    for (const replica::Model &item : state.models()) {
        if (item.active) {
            ImGui::Text("%s: %s", item.role.c_str(), item.name.c_str());
            assigned = true;
        }
    }
    if (!assigned) ImGui::TextDisabled("The manifest has no active roles.");
    ImGui::SeparatorText("Catalog");
    for (const replica::Model &item : state.models()) {
        ImGui::PushID(item.name.c_str());
        ImGui::TextUnformatted(item.name.c_str());
        ImGui::TextDisabled("%s  %s  %llu bytes", item.role.c_str(), item.quant.c_str(),
                            static_cast<unsigned long long>(item.bytes));
        if (item.fetching) {
            const float progress = item.fetch_total == 0u ? 0.0f :
                static_cast<float>(static_cast<double>(item.fetched) /
                                   static_cast<double>(item.fetch_total));
            ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), item.fetch_result.c_str());
        } else if (!item.on_disk) {
            if (ImGui::Button("Fetch") &&
                !action.fetch(build, state.models_directory(), item.name)) {
                toasts.add(action.refusal(), toast::Severity::error, now);
            }
        } else if (!item.active) {
            ImGui::TextDisabled("The model is on disk but is not active in the manifest.");
            if (ImGui::Button("Activate") &&
                !action.activate(build, state.models_directory(), item.role, item.name)) {
                toasts.add(action.refusal(), toast::Severity::error, now);
            }
        }
        if (item.active) {
            if (ImGui::Button("Load")) {
                client.send_line("model load " + item.role + " " + item.name);
            }
        }
        if (!item.fetch_result.empty() && !item.fetching) {
            ImGui::TextDisabled("Fetch: %s", item.fetch_result.c_str());
        }
        ImGui::Separator();
        ImGui::PopID();
    }
    if (state.models().empty()) {
        ImGui::TextDisabled("The catalog has no readable entries.");
    }
    ImGui::End();
}

} // namespace aotx::ctrl::model
