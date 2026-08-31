// Purpose: Define the simulated state that supplies the control shell.
// Owns: Instances, catalogs, settings, transcripts, and timed events.
// Launch shape: One user interface thread advances one simulated state.
// Lifetime: State exists from program start until program exit.
#ifndef AOTX_CTRL_SIM_HPP
#define AOTX_CTRL_SIM_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace aotx::ctrl::sim {

enum class InstanceState { running, attaching, stopped };
enum class Role { user, system, agent };
enum class EventKind { message, tool_call, reply_bound };

struct Card {
    std::string name;
    unsigned memory_used_mib;
    unsigned memory_total_mib;
};

struct Instance {
    std::string name;
    InstanceState state;
    std::vector<Card> cards;
};

struct Model {
    std::string name;
    std::string state;
    float fetch_progress;
};

struct Module {
    std::string name;
    std::string kind;
};

struct Setting {
    std::string key;
    std::string value;
};

struct TranscriptEvent {
    EventKind kind;
    Role role;
    std::string stated;
    std::string detail;
    bool streaming;
};

class State {
  public:
    State();
    void tick(double now);
    void send(std::string text, double now);
    void continue_reply(double now);
    void set_instance_state(std::size_t index, InstanceState state);

    std::vector<Instance> instances;
    std::vector<Model> models;
    std::vector<Module> modules;
    std::vector<Setting> settings;
    std::vector<TranscriptEvent> transcript;
    std::size_t selected_instance = 0;
    bool instance_selection_requested = false;

  private:
    void start_reply(double now, std::string reply);
    std::string reply_source_;
    std::size_t reply_offset_ = 0;
    double next_reply_tick_ = 0.0;
    double next_fetch_tick_ = 0.0;
};

const char *state_name(InstanceState state);

} // namespace aotx::ctrl::sim

#endif
