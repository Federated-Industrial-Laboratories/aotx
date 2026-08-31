// Purpose: Define the simulated transcript browser.
// Owns: Selected stored run.
// Launch shape: One user interface thread draws one browser panel.
// Lifetime: Selection exists for the program lifetime.
#ifndef AOTX_CTRL_BROWSER_HPP
#define AOTX_CTRL_BROWSER_HPP

#include "sim/sim.hpp"

#include <cstddef>

namespace aotx::ctrl::browser {

struct State {
    std::size_t selected = 0;
};

void draw(State &view, const sim::State &state, bool *open);

} // namespace aotx::ctrl::browser

#endif
