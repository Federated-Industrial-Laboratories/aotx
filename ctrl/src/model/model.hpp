// Purpose: Define the live and simulated model panels.
// Owns: Catalog presentation and model action controls.
// Launch shape: One user interface thread draws one model panel.
// Lifetime: Model rows remain in the replica or simulated state.
#ifndef AOTX_CTRL_MODEL_HPP
#define AOTX_CTRL_MODEL_HPP

#include "sim/sim.hpp"
#include "toast/toast.hpp"

#include "client/client.hpp"
#include "replica/replica.hpp"
#include "model/controls.hpp"
#include "model/details.hpp"

#include <filesystem>
#include <array>
#include <map>
#include <memory>
#include <string>

namespace aotx::ctrl::model {

class StoreAction {
  public:
    StoreAction();
    ~StoreAction();
    StoreAction(const StoreAction &) = delete;
    StoreAction &operator=(const StoreAction &) = delete;

    bool fetch(const std::filesystem::path &build, const std::filesystem::path &models,
               const std::string &name);
    bool activate(const std::filesystem::path &build, const std::filesystem::path &models,
                  const std::string &role, const std::string &name);
    void tick();
    bool running() const;
    bool finished() const;
    bool succeeded() const;
    const std::string &progress() const;
    std::string take_result();
    const std::string &refusal() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct LivePanelState {
    std::string binding;
    std::map<std::string, double> values;
    std::vector<Preset> presets;
    std::array<int, 2> steer{{0, 0}};
    std::array<float, 2> strength{{0.0f, 0.0f}};
    int voice = 0;
    std::string result;
};

void draw(sim::State &state, toast::Lane &toasts, double now, bool *open);
void draw(StoreAction &action, LivePanelState &panel, DetailsState &details,
          const std::filesystem::path &build,
          replica::State &state, client::Client &client, toast::Lane &toasts, double now,
          bool *open);

} // namespace aotx::ctrl::model

#endif
