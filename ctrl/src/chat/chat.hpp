// Purpose: Draw a transcript and a multiline text editor.
// Owns: Editor text and transcript presentation state.
// Launch shape: One user interface thread draws one chat window.
// Lifetime: Editor state exists for the program lifetime.
#ifndef AOTX_CTRL_CHAT_HPP
#define AOTX_CTRL_CHAT_HPP

#include "sim/sim.hpp"

#include <array>

namespace aotx::ctrl::chat {

struct View {
    std::array<char, 8192> editor{};
    bool follow = true;
};

void draw(View &view, sim::State &state, double now, bool *open);

} // namespace aotx::ctrl::chat

#endif
