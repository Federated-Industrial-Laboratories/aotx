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
#include <vector>

namespace aotx::ctrl::voice {

struct Source {
    enum class Kind { system, agent };
    static Source system();
    static Source agent(std::size_t index);

    Kind kind;
    std::size_t agent_index;
};

class Queue {
  public:
    Queue();
    ~Queue();
    Queue(const Queue &) = delete;
    Queue &operator=(const Queue &) = delete;

    bool enabled() const;
    const std::string &refusal() const;
    const std::filesystem::path &system_voice() const;
    const std::filesystem::path &agent_voice(std::size_t index) const;
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
    std::vector<std::filesystem::path> agent_voices_;
    std::string refusal_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Line> lines_;
    bool stop_ = false;
    std::thread worker_;
};

bool verify_source_paths();

} // namespace aotx::ctrl::voice

#endif
