// Purpose: Queue speech lines for local synthesis and playback.
// Owns: Voice discovery, one worker thread, and queued lines.
// Launch shape: One worker starts one synthesizer and one player per line.
// Lifetime: The queue drains before the worker stops.
#ifndef AOTX_CTRL_VOICE_HPP
#define AOTX_CTRL_VOICE_HPP

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>

namespace aotx::ctrl::voice {

enum class Source { system, agent };

class Queue {
  public:
    Queue();
    ~Queue();
    Queue(const Queue &) = delete;
    Queue &operator=(const Queue &) = delete;

    bool enabled() const;
    const std::string &refusal() const;
    const std::filesystem::path &system_voice() const;
    const std::filesystem::path &agent_voice() const;
    void speak(Source source, std::string line);

  private:
    struct Line {
        Source source;
        std::string text;
    };

    void run();
    void play(const Line &line) const;

    std::filesystem::path piper_;
    std::filesystem::path player_;
    std::filesystem::path system_voice_;
    std::filesystem::path agent_voice_;
    std::string refusal_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Line> lines_;
    bool stop_ = false;
    std::thread worker_;
};

} // namespace aotx::ctrl::voice

#endif
