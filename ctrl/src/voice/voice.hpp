// Purpose: Queue prioritized speech for local synthesis and persistent playback.
// Owns: Voice discovery, category controls, assignments, and one playback child.
// Launch shape: One worker serializes synthesis into one playback stream.
// Lifetime: The playback child exists from the first line until queue shutdown.
#ifndef AOTX_CTRL_VOICE_HPP
#define AOTX_CTRL_VOICE_HPP

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef AOTX_AFFECT
#include "voice/coupling.hpp"
#include <array>
#endif

#include <sys/types.h>

namespace aotx::ctrl::voice {

enum class Category { reply, toast, tool, lifecycle };

struct Source {
    enum class Kind { system, agent };
    static Source system();
    static Source agent(std::size_t index);

    Kind kind;
    std::size_t agent_index;
};

struct Controls {
    bool master = true;
    bool replies = true;
    bool toasts = true;
    bool tools = false;
    bool lifecycle = true;
#ifdef AOTX_AFFECT
    bool coupling = false;
#endif
    float rate = 1.0f;
    std::size_t depth = 6u;
};

class Queue {
  public:
    Queue();
    ~Queue();
    Queue(const Queue &) = delete;
    Queue &operator=(const Queue &) = delete;

    bool enabled() const;
    const std::string &refusal() const;
    Controls controls() const;
    std::vector<std::filesystem::path> voices() const;
    std::size_t agent_count() const;
    std::size_t agent_assignment(std::size_t index) const;
    void set_master(bool enabled);
    void set_category(Category category, bool enabled);
    void set_rate(float rate);
    void set_depth(std::size_t depth);
    void set_agent_count(std::size_t count);
    void set_agent_assignment(std::size_t agent, std::size_t voice);
#ifdef AOTX_AFFECT
    void set_coupling(bool enabled);
    void clear_affect_states();
    void set_affect_state(std::size_t agent, double valence, double arousal);
#endif
    void speak(Category category, Source source, std::string line);
    void test();

  private:
    struct Line {
        Category category;
        std::filesystem::path voice;
        std::string text;
#ifdef AOTX_AFFECT
        CouplingValues values;
        double valence = 0.0;
        double arousal = 0.0;
        std::size_t agent = 0u;
        bool coupled = false;
#endif
    };

    bool category_on(Category category) const;
    void trim_locked();
    void run();
    bool start_player();
    void stop_player();
    void play(const Line &line, float rate);

    std::filesystem::path piper_;
    std::filesystem::path player_;
    std::vector<std::filesystem::path> voices_;
    std::vector<std::size_t> assignments_;
#ifdef AOTX_AFFECT
    std::vector<std::array<double, 2>> affect_states_;
#endif
    std::string refusal_;
    mutable std::mutex mutex_;
    Controls controls_;
    std::condition_variable ready_;
    std::deque<Line> lines_;
    bool stop_ = false;
    std::thread worker_;
    pid_t player_child_ = -1;
    pid_t synth_child_ = -1;
    int player_input_ = -1;
};

std::string tool_line(std::size_t agent, const std::string &tool);
bool verify_source_paths();
bool verify_queue_rules();

} // namespace aotx::ctrl::voice

#endif
