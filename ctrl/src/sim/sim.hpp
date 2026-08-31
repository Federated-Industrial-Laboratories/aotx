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
enum class AuthorizationState { pending, granted, refused };

struct Card {
    std::string name;
    unsigned memory_used_mib;
    unsigned memory_total_mib;
};

struct Instance {
    std::string name;
    InstanceState state;
    std::vector<Card> cards;
    // The time when the state last changed; zero at the start.
    double state_since = 0.0;
};

struct Model {
    std::string name;
    std::string state;
    float fetch_progress;
    std::string language_role;
    std::string embedding_role;
    std::string reranker_role;
};

struct Module {
    std::string name;
    std::string kind;
};

struct ModuleDirectory {
    std::string path;
    std::string name;
    std::string kind;
};

struct Setting {
    std::string key;
    std::string value;
    std::string default_value;
    std::string valid_values;
    bool live;
};

struct Authorization {
    unsigned id;
    std::string agent;
    std::string tool;
    std::string argument;
    AuthorizationState state;
};

struct Agent {
    std::string name;
    std::string role;
    std::string state;
    unsigned pages;
};

struct TranscriptEvent {
    EventKind kind;
    Role role;
    std::string stated;
    std::string detail;
    bool streaming;
    unsigned agent_index = 0;
};

struct PastRun {
    std::string name;
    std::string result;
    std::vector<TranscriptEvent> transcript;
};

class State {
  public:
    State();
    void tick(double now);
    void send(std::string text, double now);
    void continue_reply(double now);
    void set_instance_state(std::size_t index, InstanceState state, double now);
    bool answer_authorization(std::size_t index, AuthorizationState answer);
    bool fetch_model(std::size_t index);
    bool activate_model(std::size_t index, const std::string &role);
    bool import_module(const std::string &directory);
    bool set_value(std::size_t index, const std::string &value);
    std::vector<std::string> take_results();
    const std::string &refusal() const;

    std::vector<Instance> instances;
    std::vector<Model> models;
    std::vector<Module> modules;
    std::vector<ModuleDirectory> module_directories;
    std::vector<Setting> settings;
    std::vector<Authorization> authorizations;
    std::vector<Agent> agents;
    std::vector<PastRun> past_runs;
    std::vector<TranscriptEvent> transcript;
    std::size_t selected_instance = 0;
    bool instance_selection_requested = false;
    unsigned reply_bound = 256;
    bool auto_continue = false;
    unsigned page_limit = 160;
    float tick_rate_hz = 100.0f;
    float ring_occupancy = 0.28f;

  private:
    void start_reply(double now, std::string reply);
    void finish_reply();
    std::string reply_source_;
    std::size_t reply_offset_ = 0;
    double next_reply_tick_ = 0.0;
    double next_fetch_tick_ = 0.0;
    std::vector<std::string> results_;
    std::string refusal_;
};

const char *state_name(InstanceState state);
bool verify_paths();

} // namespace aotx::ctrl::sim

#endif
