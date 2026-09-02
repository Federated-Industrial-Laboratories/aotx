// Purpose: Define the affect trace panel state and draw call.
// Owns: The selected stream agent.
// Launch shape: One interface thread draws one trace window.
// Lifetime: Selection stays until the selected instance changes.
#ifndef AOTX_CTRL_AFFECT_PANEL_HPP
#define AOTX_CTRL_AFFECT_PANEL_HPP

namespace aotx::ctrl::replica { class State; }

namespace aotx::ctrl::affect {

struct State {
    unsigned agent = 0u;
};

void draw(State &view, const replica::State &state, bool *open);

} // namespace aotx::ctrl::affect

#endif
