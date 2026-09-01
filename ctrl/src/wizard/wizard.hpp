// Purpose: Define the simulated and live first-run sequence.
// Owns: Six page states, paths, choices, action latches, and completion state.
// Launch shape: One modal shows one sequence page at a time.
// Lifetime: State remains until the first-run sequence closes.
#ifndef AOTX_CTRL_WIZARD_HPP
#define AOTX_CTRL_WIZARD_HPP

#include "sim/sim.hpp"
#include "toast/toast.hpp"
#include "instances/lifecycle.hpp"
#include "model/model.hpp"
#include "replica/replica.hpp"
#include "wizard/gates.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

namespace aotx::ctrl::wizard {

struct State {
    unsigned page = 0;
    std::array<char, 256> build_path{};
    std::size_t model_index = 3;
};

class DetectAction {
  public:
    DetectAction();
    ~DetectAction();
    DetectAction(const DetectAction &) = delete;
    DetectAction &operator=(const DetectAction &) = delete;
    bool start(const std::filesystem::path &script);
    void tick();
    bool running() const;
    bool ready() const;
    const std::string &profile() const;
    const std::string &architecture() const;
    const std::string &result() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct LiveState {
    unsigned page = 0u;
    std::string model_name;
    std::vector<replica::Model> store_models;
    double store_read_at = 0.0;
    std::string store_reason;
    bool store_read = false;
    std::array<char, 512> build_path{};
    std::array<char, 512> journal_path{};
    std::array<char, 512> settings_path{};
    std::array<char, 512> models_path{};
    std::array<char, 512> tools_path{};
    std::array<char, 81> instance_name{};
    std::size_t model_index = 0u;
    std::size_t instance_index = 0u;
    bool instance_created = false;
    unsigned entered = 0xffffffffu;
    ActionLatch action;
    std::size_t reply_count = 0u;
    std::string last_phase;
    std::string phase_trace;
    std::string result;
};

void draw(State &view, sim::State &state, toast::Lane &toasts, double now, bool *open);
void draw(LiveState &view, DetectAction &detect, model::StoreAction &model_action,
          instances::Lifecycle &lifecycle, replica::State &state, toast::Lane &toasts,
          double now, bool *open);

} // namespace aotx::ctrl::wizard

#endif
