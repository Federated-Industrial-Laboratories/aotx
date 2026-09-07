// Purpose: Read the card profile from the profile detection child.
// Owns: One detection child and its reported profile and architecture.
// Threading: One interface thread starts and polls the child.
// Lifetime: The action reaps its child at destruction.
#include "wizard/wizard.hpp"
#include "process/child.hpp"
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <cerrno>
#include <charconv>
#include <sstream>
#include <system_error>

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

} // namespace aotx::ctrl::wizard
