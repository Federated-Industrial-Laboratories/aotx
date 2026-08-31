// Purpose: Speak queued lines through local piper and pw-play children.
// Owns: Child process pipes, temporary wave files, and queue service.
// Launch shape: One worker serializes all synthesis and playback children.
// Lifetime: Each temporary file exists for one queued line.
#include "voice/voice.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <vector>

#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>

namespace aotx::ctrl::voice {
namespace {

std::filesystem::path find_program(const char *name)
{
    const char *path_value = std::getenv("PATH");
    if (path_value == nullptr) return {};
    std::string paths = path_value;
    std::size_t first = 0;
    while (first <= paths.size()) {
        const std::size_t last = paths.find(':', first);
        const std::filesystem::path candidate =
            std::filesystem::path(paths.substr(first, last - first)) / name;
        if (::access(candidate.c_str(), X_OK) == 0) return candidate;
        if (last == std::string::npos) break;
        first = last + 1;
    }
    return {};
}

std::vector<std::filesystem::path> find_voices()
{
    std::vector<std::filesystem::path> result;
    const char *home = std::getenv("HOME");
    if (home == nullptr) return result;
    const std::filesystem::path directory =
        std::filesystem::path(home) / ".local/share/piper-voices";
    std::error_code error;
    for (std::filesystem::directory_iterator item(directory, error), end;
         !error && item != end; item.increment(error)) {
        if (item->is_regular_file(error) && item->path().extension() == ".onnx") {
            result.push_back(item->path());
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

bool child_result(pid_t child)
{
    int status = 0;
    while (::waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) return false;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

const std::filesystem::path &select_agent_voice(
    const std::vector<std::filesystem::path> &voices, std::size_t index)
{
    const std::size_t first_agent = voices.size() > 1 ? 1 : 0;
    const std::size_t count = voices.size() - first_agent;
    return voices[first_agent + index % count];
}

} // namespace

Queue::Queue()
{
    piper_ = find_program("piper");
    player_ = find_program("pw-play");
    const std::vector<std::filesystem::path> voices = find_voices();
    if (!voices.empty()) {
        system_voice_ = voices.front();
        const auto first_agent = voices.size() > 1 ? voices.begin() + 1 : voices.begin();
        agent_voices_.assign(first_agent, voices.end());
    }
    if (piper_.empty() || player_.empty() || system_voice_.empty()) {
        refusal_ = "Voice is unavailable because a local speech component is not available.";
        return;
    }
    worker_ = std::thread(&Queue::run, this);
}

Queue::~Queue()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    ready_.notify_one();
    if (worker_.joinable()) worker_.join();
}

bool Queue::enabled() const { return refusal_.empty(); }
const std::string &Queue::refusal() const { return refusal_; }
const std::filesystem::path &Queue::system_voice() const { return system_voice_; }
const std::filesystem::path &Queue::agent_voice(std::size_t index) const
{
    if (agent_voices_.empty()) return system_voice_;
    return agent_voices_[index % agent_voices_.size()];
}

Source Source::system() { return {Kind::system, 0}; }
Source Source::agent(std::size_t index) { return {Kind::agent, index}; }

void Queue::speak(Source source, std::string line)
{
    if (!enabled() || line.empty()) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (lines_.size() >= 6u) lines_.pop_front();
        lines_.push_back({source, std::move(line)});
    }
    ready_.notify_one();
}

void Queue::run()
{
    sigset_t blocked;
    ::sigemptyset(&blocked);
    ::sigaddset(&blocked, SIGPIPE);
    ::pthread_sigmask(SIG_BLOCK, &blocked, nullptr);
    for (;;) {
        Line line;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this] { return stop_ || !lines_.empty(); });
            if (lines_.empty() && stop_) return;
            line = std::move(lines_.front());
            lines_.pop_front();
        }
        play(line);
    }
}

void Queue::play(const Line &line) const
{
    std::array<char, 32> pattern{};
    std::copy_n("/tmp/aotx_ctrl_voice_XXXXXX", 28, pattern.begin());
    const int file = ::mkstemp(pattern.data());
    if (file < 0) return;
    ::close(file);
    const std::filesystem::path wave = pattern.data();
    const std::filesystem::path &voice =
        line.source.kind == Source::Kind::system ? system_voice_
                                                : agent_voice(line.source.agent_index);
    int input[2] = {-1, -1};
    if (::pipe(input) != 0) {
        std::filesystem::remove(wave);
        return;
    }
    const pid_t synth = ::fork();
    if (synth == 0) {
        ::dup2(input[0], STDIN_FILENO);
        ::close(input[0]);
        ::close(input[1]);
        ::execl(piper_.c_str(), piper_.c_str(), "-m", voice.c_str(), "-f", wave.c_str(),
                static_cast<char *>(nullptr));
        ::_exit(127);
    }
    ::close(input[0]);
    if (synth < 0) {
        ::close(input[1]);
        std::filesystem::remove(wave);
        return;
    }
    const std::string text = line.text + "\n";
    std::size_t offset = 0;
    while (offset < text.size()) {
        const ssize_t count = ::write(input[1], text.data() + offset, text.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) break;
        offset += static_cast<std::size_t>(count);
    }
    ::close(input[1]);
    if (child_result(synth)) {
        const pid_t player = ::fork();
        if (player == 0) {
            ::execl(player_.c_str(), player_.c_str(), wave.c_str(),
                    static_cast<char *>(nullptr));
            ::_exit(127);
        }
        if (player > 0) child_result(player);
    }
    std::filesystem::remove(wave);
}

bool verify_source_paths()
{
    const std::vector<std::filesystem::path> voices = {"system", "agent-a", "agent-b"};
    const Source source = Source::agent(3);
    return Source::system().kind == Source::Kind::system &&
           source.kind == Source::Kind::agent && source.agent_index == 3 &&
           select_agent_voice(voices, 0) == voices[1] &&
           select_agent_voice(voices, 1) == voices[2] &&
           select_agent_voice(voices, 2) == voices[1] &&
           select_agent_voice(voices, 3) == voices[2] && voices[0] != voices[1];
}

} // namespace aotx::ctrl::voice
