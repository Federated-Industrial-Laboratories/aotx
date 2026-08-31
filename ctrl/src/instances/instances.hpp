// Purpose: Draw tabs and controls for simulated instances.
// Owns: Instance selection and state action presentation.
// Launch shape: One user interface thread draws one instance window.
// Lifetime: Instance data remains owned by the simulated state.
#ifndef AOTX_CTRL_INSTANCES_HPP
#define AOTX_CTRL_INSTANCES_HPP

#include "sim/sim.hpp"
#include "toast/toast.hpp"

namespace aotx::ctrl::instances {

void draw(sim::State &state, toast::Lane &toasts, double now, bool *open);

} // namespace aotx::ctrl::instances

#endif
