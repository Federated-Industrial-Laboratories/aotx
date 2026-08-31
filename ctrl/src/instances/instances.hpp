// Purpose: Draw tabs and controls for simulated instances.
// Owns: Instance selection and state action presentation.
// Launch shape: One user interface thread draws one instance window.
// Lifetime: Instance data remains owned by the simulated state.
#ifndef AOTX_CTRL_INSTANCES_HPP
#define AOTX_CTRL_INSTANCES_HPP

#include "sim/sim.hpp"
#include "toast/toast.hpp"
#include "instances/lifecycle.hpp"

#include <array>

namespace aotx::ctrl::instances {

struct View {
    View();
    std::array<char, 81> new_name{};
    std::array<char, 512> journal{};
    std::array<char, 512> settings{};
    std::array<char, 512> build{};
    std::array<char, 512> models{};
    int card = 0;
    bool create_visible = false;
};

void draw(View &view, sim::State &state, toast::Lane &toasts, double now, bool *open);
void draw(View &view, Lifecycle &lifecycle, toast::Lane &toasts, double now, bool *open);

} // namespace aotx::ctrl::instances

#endif
