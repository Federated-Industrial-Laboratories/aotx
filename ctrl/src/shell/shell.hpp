// Purpose: Define the dock space, menus, and default layout.
// Owns: Window visibility and layout rebuild requests.
// Launch shape: One user interface thread draws one main dock space.
// Lifetime: Shell state exists for the program lifetime.
#ifndef AOTX_CTRL_SHELL_HPP
#define AOTX_CTRL_SHELL_HPP

#include "sim/sim.hpp"

namespace aotx::ctrl::client { class Client; }
namespace aotx::ctrl::replica { class State; }

namespace aotx::ctrl::shell {

struct State {
    bool show_instances = true;
    bool show_control = true;
    bool show_models = true;
    bool show_modules = true;
    bool show_settings = true;
    bool show_monitor = true;
    bool show_browser = true;
    bool show_wizard = true;
    bool show_about = false;
    bool rebuild_layout = false;
};

void draw_dock_space(State &shell, sim::State &simulated);
void draw_dock_space(State &shell, replica::State &replica, const client::Client &client);

} // namespace aotx::ctrl::shell

#endif
