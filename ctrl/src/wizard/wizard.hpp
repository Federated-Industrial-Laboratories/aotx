// Purpose: Define the simulated first-run sequence.
// Owns: Sequence page, build path, model choice, and completion state.
// Launch shape: One modal shows one sequence page at a time.
// Lifetime: State remains until the first-run sequence closes.
#ifndef AOTX_CTRL_WIZARD_HPP
#define AOTX_CTRL_WIZARD_HPP

#include "sim/sim.hpp"
#include "toast/toast.hpp"

#include <array>
#include <cstddef>

namespace aotx::ctrl::wizard {

struct State {
    unsigned page = 0;
    std::array<char, 256> build_path{};
    std::size_t model_index = 3;
};

void draw(State &view, sim::State &state, toast::Lane &toasts, double now, bool *open);

} // namespace aotx::ctrl::wizard

#endif
