// Purpose: Define controls for one simulated system instance.
// Owns: Control window presentation state.
// Launch shape: One user interface thread draws one control window.
// Lifetime: Values remain in the simulated state.
#ifndef AOTX_CTRL_CONTROL_HPP
#define AOTX_CTRL_CONTROL_HPP

#include "sim/sim.hpp"
#include "toast/toast.hpp"

#include "client/client.hpp"
#include "instances/lifecycle.hpp"
#include "replica/replica.hpp"

namespace aotx::ctrl::control {

void draw(sim::State &state, toast::Lane &toasts, double now, bool *open);

struct LiveState {
    int reply_bound = 256;
    bool auto_continue = false;
    int pages = 0;
    bool initialized = false;
};

void draw(LiveState &view, instances::Lifecycle &lifecycle, replica::State &state,
          client::Client &client,
          toast::Lane &toasts, double now, bool *open);

} // namespace aotx::ctrl::control

#endif
