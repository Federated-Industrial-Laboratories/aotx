// Purpose: Define the dock space, menus, and default layout.
// Owns: Window visibility and layout rebuild requests.
// Launch shape: One user interface thread draws one main dock space.
// Lifetime: Shell state exists for the program lifetime.
#ifndef AOTX_CTRL_SHELL_HPP
#define AOTX_CTRL_SHELL_HPP

#include "sim/sim.hpp"

namespace aotx::ctrl::shell {

struct State {
    bool show_chat = true;
    bool show_instances = true;
    bool show_about = false;
    bool rebuild_layout = false;
};

void draw_dock_space(State &shell, sim::State &simulated);

} // namespace aotx::ctrl::shell

#endif
