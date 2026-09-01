// Purpose: Define the simulated transcript browser.
// Owns: Selected stored run.
// Launch shape: One user interface thread draws one browser panel.
// Lifetime: Selection exists for the program lifetime.
#ifndef AOTX_CTRL_BROWSER_HPP
#define AOTX_CTRL_BROWSER_HPP

#include "sim/sim.hpp"
#include "replica/replica.hpp"

#include <cstddef>

namespace aotx::ctrl::browser {

struct State {
    std::size_t selected = 0;
    std::string loaded_boot;
    std::string result;
    std::vector<replica::Agent> agents;
};

void draw(State &view, const sim::State &state, bool *open);
void draw(State &view, const replica::State &state, bool *open);

} // namespace aotx::ctrl::browser

#endif
