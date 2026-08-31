// Purpose: Define the simulated module catalog panel.
// Owns: Directory import text and catalog filters.
// Launch shape: One user interface thread draws one module panel.
// Lifetime: Import text exists for the program lifetime.
#ifndef AOTX_CTRL_MODULE_HPP
#define AOTX_CTRL_MODULE_HPP

#include "sim/sim.hpp"
#include "toast/toast.hpp"

#include "client/client.hpp"
#include "replica/replica.hpp"

#include <array>
#include <filesystem>

namespace aotx::ctrl::module {

struct State {
    std::array<char, 256> directory{};
    std::filesystem::path browser;
    bool show_picker = false;
};

void draw(State &view, sim::State &state, toast::Lane &toasts, double now, bool *open);
void draw(State &view, replica::State &state, client::Client &client,
          toast::Lane &toasts, double now, bool *open);

} // namespace aotx::ctrl::module

#endif
