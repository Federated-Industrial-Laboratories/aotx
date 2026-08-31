// Purpose: Define the simulated first-run guide.
// Owns: Guide step and completion state.
// Launch shape: One modal shows one guide step at a time.
// Lifetime: State remains until the guide closes.
#ifndef AOTX_CTRL_WIZARD_HPP
#define AOTX_CTRL_WIZARD_HPP

#include "sim/sim.hpp"
#include "toast/toast.hpp"

namespace aotx::ctrl::wizard {

struct State {
    unsigned step = 0;
};

void draw(State &view, sim::State &state, toast::Lane &toasts, double now, bool *open);

} // namespace aotx::ctrl::wizard

#endif
