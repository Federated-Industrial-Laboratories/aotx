// Purpose: Define the simulated settings editor.
// Owns: Editable setting buffers and initialization state.
// Launch shape: One user interface thread draws one settings panel.
// Lifetime: Buffers exist for the program lifetime.
#ifndef AOTX_CTRL_SETTINGS_HPP
#define AOTX_CTRL_SETTINGS_HPP

#include "sim/sim.hpp"
#include "toast/toast.hpp"

#include "client/client.hpp"
#include "replica/replica.hpp"

#include <array>
#include <string>
#include <vector>

namespace aotx::ctrl::settings {

struct State {
    std::vector<std::array<char, 256>> values;
    bool live_initialized = false;
};

void draw(State &view, sim::State &state, toast::Lane &toasts, double now, bool *open);
void draw(State &view, replica::State &state, client::Client &client,
          toast::Lane &toasts, double now, bool *open);
bool apply_value(replica::State &state, client::Client &client, toast::Lane &toasts,
                 double now, const std::string &key, const std::string &value);

} // namespace aotx::ctrl::settings

#endif
