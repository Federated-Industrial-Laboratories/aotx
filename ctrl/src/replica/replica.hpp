// Purpose: Define the live disk replica state for the control program.
// Owns: Boot discovery, parsed derived records, and directory watches.
// Launch shape: One interface thread refreshes all changed files.
// Lifetime: State exists from program start until program exit.
#ifndef AOTX_CTRL_REPLICA_HPP
#define AOTX_CTRL_REPLICA_HPP

#include <cstdint>
#include <filesystem>
#include <memory>
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
};

struct Agent {
    unsigned id = 0u;
    std::string conversation;
    std::vector<TranscriptEvent> transcript;
    std::string folded_reply;
    std::uint64_t part_lines = 0u;
    bool fold_replaced = false;
    bool reply_bound = false;
    bool window_open = true;
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
    std::size_t create_conversation();

    const std::filesystem::path &journal() const;
    const std::filesystem::path &settings() const;
    const std::string &phase() const;
    const std::string &language_model() const;
    const std::vector<Boot> &boots() const;
    std::vector<Agent> &agents();
    const std::vector<Agent> &agents() const;
    const std::vector<Note> &notes() const;
    const std::vector<Request> &requests() const;
    const std::vector<Module> &modules() const;
    const std::vector<Model> &models() const;
    const std::filesystem::path &models_directory() const;
    const std::vector<std::string> &console() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool setting_value(const std::filesystem::path &path, const std::string &key,
                   std::string &value);
bool verify_fixtures();

} // namespace aotx::ctrl::replica

#endif
