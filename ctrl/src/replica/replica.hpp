// Purpose: Define the live disk replica state for the control program.
// Owns: Boot discovery, parsed derived records, and directory watches.
// Launch shape: One interface thread refreshes all changed files.
// Lifetime: State exists from program start until program exit.
#ifndef AOTX_CTRL_REPLICA_HPP
#define AOTX_CTRL_REPLICA_HPP

#include <cstdint>
#include <filesystem>
#include <memory>
#ifdef AOTX_AFFECT
#include <array>
#include <optional>
#endif
#include <string>
#include <vector>

namespace aotx::ctrl::replica {

struct TranscriptEvent {
    std::uint64_t tick = 0u;
    std::string kind;
    std::string text;
    std::string tool;
    std::uint64_t request = 0u;
    std::string status;
    std::uint64_t turn = 0u;
    std::vector<std::string> token_text;
};

struct TokenStat {
    std::uint64_t tick = 0u;
    unsigned agent = 0u;
    unsigned turn = 0u;
    unsigned index = 0u;
    unsigned token = 0u;
    double logprob = 0.0;
    double entropy = 0.0;
    bool think = false;
};

struct PageStat {
    std::uint64_t tick = 0u;
    unsigned agent = 0u;
    unsigned page = 0u;
    unsigned residency = 0u;
    double mass = 0.0;
};

#ifdef AOTX_AFFECT
struct AffectTrace {
    std::uint64_t tick = 0u;
    unsigned agent = 0u;
    std::uint64_t turn = 0u;
    std::array<double, 4> prompt{};
    std::array<double, 4> reply{};
    std::array<double, 2> guard{};
    double logprob = 0.0;
    double entropy = 0.0;
    unsigned rows = 0u;
    unsigned think = 0u;
    std::vector<std::string> reason;
    std::array<double, 4> effective{};
    unsigned flags = 0u;
    bool trace = false;
};

struct QualityLine {
    std::uint64_t tick = 0u;
    unsigned agent = 0u;
    std::uint64_t turn = 0u;
    std::optional<double> coherence_prompt;
    std::optional<double> coherence_turn;
    double repetition = 0.0;
    unsigned tokens = 0u;
    unsigned limit = 0u;
    bool limit_hit = false;
    bool refusal = false;
    std::array<double, 2> guard{};
    unsigned flags = 0u;
};
#endif

struct ModelParameter {
    std::string name;
    double initial = 0.0;
    double least = 0.0;
    double most = 0.0;
};

struct ModelParameters {
    std::string name;
    std::vector<ModelParameter> values;
};

struct SteerVector {
    std::string name;
    std::string file;
    double potency_nats = 0.0;
};

struct VoiceProfile {
    std::string name;
    std::string file;
    unsigned entries = 0u;
};

struct Agent {
    unsigned id = 0u;
    std::string conversation;
    std::vector<TranscriptEvent> transcript;
    std::string folded_reply;
    std::uint64_t part_lines = 0u;
    bool fold_replaced = false;
    bool reply_bound = false;
    bool reply_in_flight = false;
    bool window_open = true;
    std::size_t open_part = static_cast<std::size_t>(-1);
    std::uint64_t open_part_turn = 0u;
};

struct Note {
    std::string agent;
    std::uint64_t sequence = 0u;
    std::string text;
    std::uint64_t tick = 0u;
    std::string boot;
};

struct Request {
    std::uint64_t request = 0u;
    std::uint64_t agent = 0u;
    std::uint64_t turn = 0u;
    std::string tool;
    std::string side;
    std::uint64_t number = 0u;
    std::string argument;
    std::uint64_t deadline = 0u;
    std::string authorization;
    std::uint64_t tick = 0u;
};

struct PendingRequest {
    std::uint64_t request = 0u;
    std::uint64_t agent = 0u;
    std::uint64_t turn = 0u;
    std::string tool;
    std::string path;
};

struct AgentState {
    std::uint64_t agent = 0u;
    std::string event;
    std::uint64_t role = 0u;
    std::uint64_t parent = 0u;
    std::uint64_t state = 0u;
    std::uint64_t turn = 0u;
    std::uint64_t ticks = 0u;
};

struct Module {
    std::string name;
    std::string kind;
    std::string side;
    std::string directory;
    std::string program;
    std::uint64_t timeout = 0u;
    std::string authorization;
    std::uint64_t import = 0u;
    std::uint64_t number = 0u;
};

struct Model {
    std::string name;
    std::string role;
    std::string file;
    std::string quant;
    std::string source;
    std::string digest;
    std::uint64_t bytes = 0u;
    bool verified = false;
    bool on_disk = false;
    bool active = false;
    bool fetching = false;
    std::uint64_t fetched = 0u;
    std::uint64_t fetch_total = 0u;
    std::string fetch_result;
};

struct Boot {
    std::string name;
    std::filesystem::path directory;
};

class State {
  public:
    State(std::filesystem::path journal, std::filesystem::path settings);
    ~State();
    State(const State &) = delete;
    State &operator=(const State &) = delete;

    bool open();
    void tick(double now);
    std::vector<std::string> take_results();
    bool create_conversation(std::string &command);
    void select_agent(std::size_t index);
    std::size_t selected_agent() const;

    const std::filesystem::path &journal() const;
    const std::filesystem::path &settings() const;
    const std::string &phase() const;
    const std::string &language_model() const;
    const std::vector<Boot> &boots() const;
    std::vector<Agent> &agents();
    const std::vector<Agent> &agents() const;
    const std::vector<Note> &notes() const;
    const std::vector<Request> &requests() const;
    const std::vector<PendingRequest> &pending_requests() const;
    const std::vector<AgentState> &agent_states() const;
    const std::vector<Module> &modules() const;
    const std::vector<Model> &models() const;
    const std::vector<ModelParameters> &model_parameters() const;
    const std::vector<SteerVector> &steer_vectors() const;
    const std::vector<VoiceProfile> &voice_profiles() const;
    const std::vector<TokenStat> &tokens() const;
    const std::vector<PageStat> &pages() const;
#ifdef AOTX_AFFECT
    const std::vector<AffectTrace> &affect_traces() const;
    const std::vector<QualityLine> &quality_lines() const;
#endif
    double token_rate() const;
    const std::filesystem::path &models_directory() const;
    const std::vector<std::string> &console() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool setting_value(const std::filesystem::path &path, const std::string &key,
                   std::string &value);
bool read_boot_transcripts(const std::filesystem::path &boot, std::vector<Agent> &agents,
                           std::string &reason);
bool verify_fixtures();

} // namespace aotx::ctrl::replica

#endif
