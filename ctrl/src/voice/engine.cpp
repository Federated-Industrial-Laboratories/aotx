// Purpose: Run speech synthesis and stream its samples to one player.
// Owns: Speech children, the playback child, and their pipes.
// Launch shape: One worker runs one utterance at a time.
// Lifetime: The playback child remains until queue shutdown.
#include "voice/voice.hpp"

#include <cerrno>
#include <csignal>
#include <cstdio>

#include <fcntl.h>
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>

namespace aotx::ctrl::voice {
namespace {

bool child_result(pid_t child)
{
    int status = 0;
    while (::waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) return false;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

} // namespace

void Queue::run()
{
    sigset_t blocked;
    ::sigemptyset(&blocked);
    ::sigaddset(&blocked, SIGPIPE);
    ::pthread_sigmask(SIG_BLOCK, &blocked, nullptr);
    for (;;) {
        Line line;
        float rate = 1.0f;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this] { return stop_ || !lines_.empty(); });
            if (stop_) break;
            line = std::move(lines_.front());
            lines_.pop_front();
            rate = controls_.rate;
        }
        play(line, rate);
    }
    stop_player();
}

bool Queue::start_player()
{
    // A player that ended on its own is reaped here, and the next line starts a new one.
    if (player_child_ >= 0 && player_input_ >= 0) {
        int status = 0;
        pid_t ended = -1;
        while ((ended = ::waitpid(player_child_, &status, WNOHANG)) < 0 && errno == EINTR) {}
        if (ended == 0) return true;
        ::close(player_input_);
        player_input_ = -1;
        std::lock_guard<std::mutex> lock(mutex_);
        player_child_ = -1;
    }
    int audio[2] = {-1, -1};
    if (::pipe2(audio, O_CLOEXEC) != 0) return false;
    const pid_t child = ::fork();
    if (child == 0) {
        ::dup2(audio[0], STDIN_FILENO);
        ::close(audio[0]);
        ::close(audio[1]);
        ::execl(player_.c_str(), player_.c_str(), "--rate", "22050", "--channels", "1",
                "--format", "s16", "-", static_cast<char *>(nullptr));
        ::_exit(127);
    }
    ::close(audio[0]);
    if (child < 0) {
        ::close(audio[1]);
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        player_child_ = child;
    }
    player_input_ = audio[1];
    return true;
}

void Queue::stop_player()
{
    if (player_input_ >= 0) {
        ::close(player_input_);
        player_input_ = -1;
    }
    if (player_child_ >= 0) {
        child_result(player_child_);
        std::lock_guard<std::mutex> lock(mutex_);
        player_child_ = -1;
    }
}

void Queue::play(const Line &line, float rate)
{
    if (!start_player()) return;
    int input[2] = {-1, -1};
    if (::pipe2(input, O_CLOEXEC) != 0) return;
    const std::string length = std::to_string(
#ifdef AOTX_AFFECT
        line.coupled ? line.values.length :
#endif
        1.0f / rate);
#ifdef AOTX_AFFECT
    const std::string noise = std::to_string(line.values.noise);
    const std::string width = std::to_string(line.values.width);
    if (line.coupled) {
        std::fprintf(stderr,
            "Spoken voice coupling: agent %zu; length %.4f; noise %.4f; width %.4f; valence %.4f; arousal %.4f.\n",
            line.agent, line.values.length, line.values.noise, line.values.width,
            line.valence, line.arousal);
    }
#endif
    const pid_t synth = ::fork();
    if (synth == 0) {
        ::dup2(input[0], STDIN_FILENO);
        ::dup2(player_input_, STDOUT_FILENO);
        ::close(input[0]);
        ::close(input[1]);
        ::close(player_input_);
#ifdef AOTX_AFFECT
        if (line.coupled) {
            ::execl(piper_.c_str(), piper_.c_str(), "-m", line.voice.c_str(), "--output-raw",
                    "--length-scale", length.c_str(), "--noise-scale", noise.c_str(),
                    "--noise-w-scale", width.c_str(), static_cast<char *>(nullptr));
        }
#endif
        ::execl(piper_.c_str(), piper_.c_str(), "-m", line.voice.c_str(), "--output-raw",
                "--length-scale", length.c_str(), static_cast<char *>(nullptr));
        ::_exit(127);
    }
    ::close(input[0]);
    if (synth < 0) {
        ::close(input[1]);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        synth_child_ = synth;
    }
    const std::string text = line.text + "\n";
    std::size_t offset = 0u;
    while (offset < text.size()) {
        const ssize_t count = ::write(input[1], text.data() + offset, text.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) break;
        offset += static_cast<std::size_t>(count);
    }
    ::close(input[1]);
    child_result(synth);
    std::lock_guard<std::mutex> lock(mutex_);
    synth_child_ = -1;
}

} // namespace aotx::ctrl::voice
