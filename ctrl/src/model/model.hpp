// Purpose: Define the simulated model catalog panel.
// Owns: Catalog presentation and model action controls.
// Launch shape: One user interface thread draws one model panel.
// Lifetime: Model state remains in the simulated state.
#ifndef AOTX_CTRL_MODEL_HPP
#define AOTX_CTRL_MODEL_HPP

#include "sim/sim.hpp"
#include "toast/toast.hpp"

#include "client/client.hpp"
#include "replica/replica.hpp"

#include <filesystem>
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

void draw(sim::State &state, toast::Lane &toasts, double now, bool *open);
void draw(StoreAction &action, const std::filesystem::path &build, replica::State &state,
          client::Client &client, toast::Lane &toasts, double now, bool *open);

} // namespace aotx::ctrl::model

#endif
