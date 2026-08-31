// Purpose: Define the simulated settings editor.
// Owns: Editable setting buffers and initialization state.
// Launch shape: One user interface thread draws one settings panel.
// Lifetime: Buffers exist for the program lifetime.
#ifndef AOTX_CTRL_SETTINGS_HPP
#define AOTX_CTRL_SETTINGS_HPP

#include "sim/sim.hpp"
#include "toast/toast.hpp"

#include <array>
#include <vector>

namespace aotx::ctrl::settings {

struct State {
    std::vector<std::array<char, 32>> values;
};

void draw(State &view, sim::State &state, toast::Lane &toasts, double now, bool *open);

} // namespace aotx::ctrl::settings

#endif
