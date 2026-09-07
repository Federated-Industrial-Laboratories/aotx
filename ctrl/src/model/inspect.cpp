// Purpose: Run bounded model header inspection without an interface wait.
// Owns: One child process group, one output pipe, and one pending request.
// Threading: One interface thread starts, cancels, and polls the action.
// Lifetime: Destruction stops the process group and reaps its child.
#include "model/inspect.hpp"
#include "model/inspect_parse.hpp"

#include <array>
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace aotx::ctrl::model {

struct InspectAction::Impl {
    InspectLimits limits;
    InspectStatus state = InspectStatus::idle;
    std::filesystem::path build;
    std::string input, output, message;
    std::optional<InspectHeader> header;
    std::chrono::steady_clock::time_point started;
    pid_t child = -1;
    int pipe = -1;
    bool stopping = false, pending = false;

    explicit Impl(InspectLimits value) : limits(value) {}

    ~Impl()
    {
        stop();
        if (child > 0) while (::waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
    }

    void close_pipe()
    {
        if (pipe >= 0) ::close(pipe);
        pipe = -1;
    }

    void stop()
    {
        close_pipe();
        if (child > 0) {
            // The unreaped child keeps its process group identity reserved.
            (void)::kill(-child, SIGKILL);
            stopping = true;
        }
    }

    void fail(const char *reason)
    {
        stop();
        pending = false;
        header.reset();
        state = InspectStatus::failed;
        message = reason;
    }

    bool launch()
    {
        pending = false;
        const std::string program = (build / "aotx_models").string();
        int ends[2];
        if (::pipe2(ends, O_CLOEXEC) != 0) {
            fail("The inspection pipe does not open.");
            return false;
        }
        // Child redirection cannot alias a pipe descriptor with a standard descriptor.
        bool good = true;
        for (int &end : ends) if (end < 3) {
            const int copy = ::fcntl(end, F_DUPFD_CLOEXEC, 3);
            ::close(end);
            end = copy;
            if (copy < 0) good = false;
        }
        if (good) good = ::fcntl(ends[0], F_SETFL, O_NONBLOCK) == 0;
        posix_spawn_file_actions_t actions;
        posix_spawnattr_t attributes;
        const bool have_actions = ::posix_spawn_file_actions_init(&actions) == 0;
        const bool have_attributes = ::posix_spawnattr_init(&attributes) == 0;
        good = good && have_actions && have_attributes;
        if (good) good =
            ::posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0) == 0 &&
            ::posix_spawn_file_actions_adddup2(&actions, ends[1], 1) == 0 &&
            ::posix_spawn_file_actions_adddup2(&actions, ends[1], 2) == 0 &&
            ::posix_spawn_file_actions_addclose(&actions, ends[0]) == 0 &&
            ::posix_spawn_file_actions_addclose(&actions, ends[1]) == 0 &&
            ::posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP) == 0 &&
            ::posix_spawnattr_setpgroup(&attributes, 0) == 0;
        char *arguments[] = {const_cast<char *>(program.c_str()),
                             const_cast<char *>("inspect"),
                             const_cast<char *>(input.c_str()), nullptr};
        pid_t spawned = -1;
        if (good) good = ::posix_spawn(&spawned, program.c_str(), &actions, &attributes,
                                       arguments, environ) == 0;
        if (have_actions) ::posix_spawn_file_actions_destroy(&actions);
        if (have_attributes) ::posix_spawnattr_destroy(&attributes);
        if (ends[1] >= 0) ::close(ends[1]);
        if (!good) {
            if (ends[0] >= 0) ::close(ends[0]);
            fail("The inspection child does not start. Check the build path and file access.");
            return false;
        }
        child = spawned;
        pipe = ends[0];
        return true;
    }

    void drain()
    {
        std::array<char, 4096> bytes{};
        // A frame reads at most the output bound plus one refusal byte.
        for (std::size_t read_count = 0; pipe >= 0 && read_count <= limits.output_bytes / bytes.size();
             ++read_count) {
            const ssize_t got = ::read(pipe, bytes.data(), bytes.size());
            if (got > 0) {
                const auto count = static_cast<std::size_t>(got);
                if (count > limits.output_bytes - output.size()) {
                    fail("The inspection output exceeds the byte limit.");
                    return;
                }
                output.append(bytes.data(), count);
            } else if (got == 0) close_pipe();
            else if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            else if (errno != EINTR) {
                fail("The inspection output does not read.");
                return;
            }
        }
    }

    void poll()
    {
        if (stopping) {
            const pid_t ended = ::waitpid(child, nullptr, WNOHANG);
            if (ended == child || (ended < 0 && errno == ECHILD)) {
                child = -1;
                stopping = false;
            } else if (ended < 0 && errno != EINTR) {
                pending = false;
                state = InspectStatus::failed;
                message = "The inspection child status does not read.";
            }
        }
        if (state != InspectStatus::running) return;
        if (std::chrono::steady_clock::now() - started >= limits.duration) {
            fail("The inspection exceeds the time limit.");
            return;
        }
        if (stopping) return;
        if (pending && !launch()) return;
        drain();
        if (state != InspectStatus::running || child < 0) return;
        siginfo_t info{};
        // Keep the child unreaped until all pipe writers close or the deadline expires.
        if (::waitid(P_PID, static_cast<id_t>(child), &info, WEXITED | WNOHANG | WNOWAIT) != 0) {
            if (errno != EINTR) fail("The inspection child status does not read.");
            return;
        }
        if (info.si_pid == 0) return;
        drain();
        if (state != InspectStatus::running || pipe >= 0) return;
        int status = 0;
        const pid_t ended = ::waitpid(child, &status, WNOHANG);
        if (ended < 0 && errno == EINTR) return;
        if (ended != child) {
            fail("The inspection child status does not read.");
            return;
        }
        child = -1;
        auto parsed = aotx_parse_inspection(output, input);
        if (!WIFEXITED(status) || !parsed ||
            (WEXITSTATUS(status) != 0 &&
             !(WEXITSTATUS(status) == 1 && !parsed->build_support))) {
            fail("The inspection failed or its report is incomplete or invalid.");
            return;
        }
        header = std::move(parsed);
        state = header->build_support ? InspectStatus::complete : InspectStatus::unsupported;
        message = header->build_support
            ? "The listed header fields are supported. Runtime use is not verified."
            : "The listed header fields are not supported. Runtime use is not verified.";
    }
};

InspectAction::InspectAction(InspectLimits limits) : impl_(std::make_unique<Impl>(limits)) {}
InspectAction::~InspectAction() = default;

bool InspectAction::start(const std::filesystem::path &build, const std::string &input)
{
    impl_->stop();
    impl_->pending = false;
    impl_->build = build;
    impl_->input = input;
    impl_->header.reset();
    impl_->output.clear();
    impl_->message = "The model header inspection is active.";
    impl_->state = InspectStatus::running;
    impl_->started = std::chrono::steady_clock::now();
    if (build.empty() || input.empty() || input.find('\0') != std::string::npos ||
        build.string().find('\0') != std::string::npos || impl_->limits.output_bytes == 0 ||
        impl_->limits.duration.count() <= 0) {
        impl_->fail("The inspection needs a build path, a source, and positive limits.");
        return false;
    }
    impl_->pending = true;
    return impl_->stopping || impl_->launch();
}

void InspectAction::tick() { impl_->poll(); }
void InspectAction::cancel()
{
    impl_->stop();
    impl_->pending = false;
    impl_->header.reset();
    impl_->output.clear();
    impl_->state = InspectStatus::cancelled;
    impl_->message = "The model header inspection was canceled.";
}
bool InspectAction::running() const { return impl_->state == InspectStatus::running; }
InspectStatus InspectAction::status() const { return impl_->state; }
const std::filesystem::path &InspectAction::build() const { return impl_->build; }
const std::string &InspectAction::input() const { return impl_->input; }
const std::string &InspectAction::message() const { return impl_->message; }
const std::string &InspectAction::output() const { return impl_->output; }
const std::optional<InspectHeader> &InspectAction::header() const { return impl_->header; }

} // namespace aotx::ctrl::model
