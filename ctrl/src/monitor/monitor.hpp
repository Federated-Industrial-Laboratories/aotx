// Purpose: Define the simulated system monitor panel.
// Owns: Monitor presentation only.
// Launch shape: One user interface thread draws one monitor panel.
// Lifetime: Metrics remain in the simulated state.
#ifndef AOTX_CTRL_MONITOR_HPP
#define AOTX_CTRL_MONITOR_HPP

#include "sim/sim.hpp"

namespace aotx::ctrl::monitor {

void draw(const sim::State &state, bool *open);

} // namespace aotx::ctrl::monitor

#endif
