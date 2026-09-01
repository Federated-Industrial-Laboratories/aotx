// Purpose: Define the simulated system monitor panel.
// Owns: Monitor presentation only.
// Launch shape: One user interface thread draws one monitor panel.
// Lifetime: Metrics remain in the simulated state.
#ifndef AOTX_CTRL_MONITOR_HPP
#define AOTX_CTRL_MONITOR_HPP

#include "sim/sim.hpp"
#include "monitor/telemetry.hpp"

namespace aotx::ctrl::client { class Client; }
namespace aotx::ctrl::replica { class State; }

namespace aotx::ctrl::monitor {

void draw(const sim::State &state, bool *open);
void draw(Telemetry &telemetry, const replica::State &state, const client::Client &client,
          double now, bool *open);

} // namespace aotx::ctrl::monitor

#endif
