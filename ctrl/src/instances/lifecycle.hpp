// Purpose: Define headless instance creation, start, state, and stop operations.
// Owns: Instance settings files, child processes, and socket clients.
// Launch shape: One interface thread manages one child for each instance.
// Lifetime: The manager stops all owned children before it is destroyed.
#ifndef AOTX_CTRL_LIFECYCLE_HPP
#define AOTX_CTRL_LIFECYCLE_HPP

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace aotx::ctrl::instances {

#ifndef AOTX_CTRL_LANGUAGE_ROLE
#define AOTX_CTRL_LANGUAGE_ROLE "language"
#endif

enum class LiveState { attaching, running, stopped };

struct Definition {
    std::string name;
    std::filesystem::path journal;
    std::filesystem::path settings;
    std::filesystem::path build;
    std::filesystem::path models;
    std::string roles = "embedding,reranker," AOTX_CTRL_LANGUAGE_ROLE;
    unsigned card = 0u;
};

struct LiveInstance {
    Definition definition;
    LiveState state = LiveState::stopped;
    std::string phase = "unknown";
    std::string connection = "not connected";
    std::string result;
    int process = -1;
    bool owned = false;
};

class Lifecycle {
  public:
    Lifecycle();
    ~Lifecycle();
    Lifecycle(const Lifecycle &) = delete;
    Lifecycle &operator=(const Lifecycle &) = delete;

    void seed(Definition definition);
    bool create(Definition definition);
    bool start(std::size_t index);
    bool stop(std::size_t index);
    bool send(std::size_t index, const std::string &line);
    int mirror_descriptor(std::size_t index) const;
    void tick(double now);
    std::vector<std::string> take_results();
    std::vector<LiveInstance> instances() const;
    const std::string &refusal() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char *state_name(LiveState state);

} // namespace aotx::ctrl::instances

#endif
