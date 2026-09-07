// Purpose: Run model fetch and activation through the model store program.
// Owns: One child process and its progress output.
// Threading: One interface thread starts and polls each action.
// Lifetime: The action ends its child at destruction.
#include "model/model.hpp"
#include "process/child.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <array>

namespace aotx::ctrl::model {
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
        if (::pipe2(lines, O_CLOEXEC) != 0) {
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

} // namespace aotx::ctrl::model
