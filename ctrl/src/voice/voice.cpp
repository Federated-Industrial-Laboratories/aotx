// Purpose: Prioritize speech and stream synthesized audio to one playback child.
// Owns: Queue rules, speech controls, synthesis children, and playback pipes.
// Launch shape: One worker starts serial synthesis for one persistent player.
// Lifetime: Queue shutdown drains speech and then ends the playback child.
#include "voice/voice.hpp"

#include <algorithm>
#include <csignal>
#include <cstdlib>

#include <unistd.h>

namespace aotx::ctrl::voice {
namespace {

std::filesystem::path find_program(const char *name)
{
    const char *path_value = std::getenv("PATH");
    if (path_value == nullptr) return {};
    std::string paths = path_value;
    std::size_t first = 0u;
    while (first <= paths.size()) {
        const std::size_t last = paths.find(':', first);
        const std::filesystem::path candidate =
            std::filesystem::path(paths.substr(first, last - first)) / name;
        if (::access(candidate.c_str(), X_OK) == 0) return candidate;
        if (last == std::string::npos) break;
        first = last + 1u;
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

bool is_reply(Category category) { return category == Category::reply; }

struct RuleLine {
    Category category;
    std::string text;
};

template <typename Line>
void apply_rule(std::deque<Line> &lines, Line line, std::size_t depth)
{
    if (is_reply(line.category)) {
        lines.erase(std::remove_if(lines.begin(), lines.end(),
                                   [](const Line &held) {
                                       return !is_reply(held.category);
                                   }),
                    lines.end());
        const auto first_status = std::find_if(lines.begin(), lines.end(),
                                               [](const Line &held) {
                                                   return !is_reply(held.category);
                                               });
        lines.insert(first_status, std::move(line));
    } else {
        lines.push_back(std::move(line));
    }
    while (lines.size() > depth) {
        const auto status = std::find_if(lines.begin(), lines.end(),
                                         [](const Line &held) {
                                             return !is_reply(held.category);
                                         });
        if (status != lines.end()) lines.erase(status);
        else lines.pop_front();
    }
}

std::string class_name(std::string tool)
{
    const std::size_t argument = tool.find_first_of(" \t\r\n");
    if (argument != std::string::npos) tool.resize(argument);
    const std::size_t slash = tool.find_last_of('/');
    if (slash != std::string::npos) tool.erase(0u, slash + 1u);
    const std::size_t end = tool.find_first_of("_-.:");
    if (end != std::string::npos) tool.resize(end);
    return tool.empty() ? "tool" : tool;
}

} // namespace

Queue::Queue()
{
    piper_ = find_program("piper");
    player_ = find_program("pw-play");
    voices_ = find_voices();
    if (piper_.empty() || player_.empty() || voices_.empty()) {
        refusal_ = "Voice is unavailable because a local speech component is not available.";
        return;
    }
    assignments_.push_back(voices_.size() > 1u ? 1u : 0u);
    worker_ = std::thread(&Queue::run, this);
}

// A close does not speak the queued lines. The queue empties, the line in synthesis and
// the player receive SIGTERM, and the worker ends after it reaps them.
Queue::~Queue()
{
    pid_t synth = -1, player = -1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        lines_.clear();
        synth = synth_child_;
        player = player_child_;
    }
    if (synth >= 0) ::kill(synth, SIGTERM);
    if (player >= 0) ::kill(player, SIGTERM);
    ready_.notify_one();
    if (worker_.joinable()) worker_.join();
}

bool Queue::enabled() const { return refusal_.empty(); }
const std::string &Queue::refusal() const { return refusal_; }

Controls Queue::controls() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return controls_;
}

std::vector<std::filesystem::path> Queue::voices() const { return voices_; }

std::size_t Queue::agent_count() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return assignments_.size();
}

std::size_t Queue::agent_assignment(std::size_t index) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return index < assignments_.size() ? assignments_[index] : 0u;
}

void Queue::set_master(bool enabled_value)
{
    std::lock_guard<std::mutex> lock(mutex_);
    controls_.master = enabled_value;
    if (!enabled_value) lines_.clear();
}

void Queue::set_category(Category category, bool enabled_value)
{
    std::lock_guard<std::mutex> lock(mutex_);
    switch (category) {
    case Category::reply: controls_.replies = enabled_value; break;
    case Category::toast: controls_.toasts = enabled_value; break;
    case Category::tool: controls_.tools = enabled_value; break;
    case Category::lifecycle: controls_.lifecycle = enabled_value; break;
    }
    if (!enabled_value) {
        lines_.erase(std::remove_if(lines_.begin(), lines_.end(),
                                    [category](const Line &line) {
                                        return line.category == category;
                                    }),
                     lines_.end());
    }
}

void Queue::set_rate(float rate)
{
    std::lock_guard<std::mutex> lock(mutex_);
    controls_.rate = std::clamp(rate, 0.5f, 2.0f);
}

void Queue::set_depth(std::size_t depth)
{
    std::lock_guard<std::mutex> lock(mutex_);
    controls_.depth = std::clamp<std::size_t>(depth, 1u, 32u);
    trim_locked();
}

void Queue::set_agent_count(std::size_t count)
{
    std::lock_guard<std::mutex> lock(mutex_);
    count = std::max<std::size_t>(count, 1u);
    const std::size_t old = assignments_.size();
    assignments_.resize(count);
    for (std::size_t index = old; index < count; ++index) {
        assignments_[index] = voices_.empty() ? 0u : (index + 1u) % voices_.size();
    }
}

void Queue::set_agent_assignment(std::size_t agent, std::size_t voice)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (agent < assignments_.size() && voice < voices_.size()) assignments_[agent] = voice;
}

#ifdef AOTX_AFFECT
void Queue::set_coupling(bool enabled_value)
{
    std::lock_guard<std::mutex> lock(mutex_);
    controls_.coupling = enabled_value;
}

void Queue::clear_affect_states()
{
    std::lock_guard<std::mutex> lock(mutex_);
    affect_states_.clear();
}

void Queue::set_affect_state(std::size_t agent, double valence, double arousal)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (affect_states_.size() <= agent) affect_states_.resize(agent + 1u);
    affect_states_[agent] = {valence, arousal};
}
#endif

bool Queue::category_on(Category category) const
{
    switch (category) {
    case Category::reply: return controls_.replies;
    case Category::toast: return controls_.toasts;
    case Category::tool: return controls_.tools;
    case Category::lifecycle: return controls_.lifecycle;
    }
    return false;
}

void Queue::trim_locked()
{
    while (lines_.size() > controls_.depth) {
        const auto status = std::find_if(lines_.begin(), lines_.end(),
                                         [](const Line &line) {
                                             return !is_reply(line.category);
                                         });
        if (status != lines_.end()) lines_.erase(status);
        else lines_.pop_front();
    }
}

void Queue::speak(Category category, Source source, std::string line)
{
    if (!enabled() || line.empty()) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!controls_.master || !category_on(category)) return;
        std::size_t voice = 0u;
        if (source.kind == Source::Kind::agent && !assignments_.empty()) {
            voice = assignments_[source.agent_index % assignments_.size()];
        }
        Line made{category, voices_[voice % voices_.size()], std::move(line)};
#ifdef AOTX_AFFECT
        if (controls_.coupling && source.kind == Source::Kind::agent) {
            if (source.agent_index < affect_states_.size()) {
                made.valence = affect_states_[source.agent_index][0];
                made.arousal = affect_states_[source.agent_index][1];
            }
            made.values = spoken_voice_values(made.valence, made.arousal);
            made.agent = source.agent_index;
            made.coupled = true;
        }
#endif
        apply_rule(lines_, std::move(made), controls_.depth);
    }
    ready_.notify_one();
}

void Queue::test()
{
    if (!enabled()) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!controls_.master) return;
        lines_.erase(std::remove_if(lines_.begin(), lines_.end(),
                                    [](const Line &held) {
                                        return !is_reply(held.category);
                                    }),
                     lines_.end());
        Line made{Category::reply, voices_.front(), "The voice test is ready."};
#ifdef AOTX_AFFECT
        if (controls_.coupling) {
            if (!affect_states_.empty()) {
                made.valence = affect_states_[0][0];
                made.arousal = affect_states_[0][1];
            }
            made.values = spoken_voice_values(made.valence, made.arousal);
            made.coupled = true;
        }
#endif
        lines_.push_back(std::move(made));
        trim_locked();
    }
    ready_.notify_one();
}

Source Source::system() { return {Kind::system, 0u}; }
Source Source::agent(std::size_t index) { return {Kind::agent, index}; }

std::string tool_line(std::size_t agent, const std::string &tool)
{
    return "Agent " + std::to_string(agent) + " calls the " + class_name(tool) + " tool.";
}

bool verify_source_paths()
{
    const Source source = Source::agent(3u);
    return Source::system().kind == Source::Kind::system &&
           source.kind == Source::Kind::agent && source.agent_index == 3u;
}

bool verify_queue_rules()
{
    Controls controls;
    if (controls.tools || !controls.replies || !controls.toasts || !controls.lifecycle) {
        return false;
    }
    std::deque<RuleLine> lines;
    apply_rule(lines, {Category::toast, "warning"}, 6u);
    apply_rule(lines, {Category::lifecycle, "placing"}, 6u);
    apply_rule(lines, {Category::reply, "answer"}, 6u);
    if (lines.size() != 1u || lines.front().category != Category::reply) return false;
    apply_rule(lines, {Category::reply, "second answer"}, 6u);
    apply_rule(lines, {Category::toast, "error"}, 6u);
    if (lines.size() != 3u || lines[0].text != "answer" ||
        lines[1].text != "second answer" || lines[2].text != "error") return false;
    apply_rule(lines, {Category::tool, "tool"}, 2u);
    if (lines.size() != 2u || lines[0].category != Category::reply ||
        lines[1].category != Category::reply) return false;
    const std::string compressed = tool_line(4u, "fs_read secret argument");
    return compressed == "Agent 4 calls the fs tool." &&
           compressed.find("secret argument") == std::string::npos;
}

} // namespace aotx::ctrl::voice
