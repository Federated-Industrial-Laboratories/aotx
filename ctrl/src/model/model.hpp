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

namespace aotx::ctrl::model {

void draw(sim::State &state, toast::Lane &toasts, double now, bool *open);
void draw(replica::State &state, client::Client &client, toast::Lane &toasts,
          double now, bool *open);

} // namespace aotx::ctrl::model

#endif
