// Purpose: Draw a transcript and a multiline text editor.
// Owns: Editor text and transcript presentation state.
// Launch shape: One user interface thread draws one chat window.
// Lifetime: Editor state exists for the program lifetime.
#ifndef AOTX_CTRL_CHAT_HPP
#define AOTX_CTRL_CHAT_HPP

#include "sim/sim.hpp"

#include <array>
#include <string>
#include <vector>

namespace aotx::ctrl::voice { class Queue; }

namespace aotx::ctrl::chat {

struct View {
    std::array<char, 4001> editor{};
    std::vector<bool> spoken;
    bool follow = true;
};

std::string window_name(const sim::Conversation &conversation, std::size_t index);
void draw(View &view, sim::State &state, std::size_t conversation, voice::Queue &speech,
          double now);
bool verify_key_paths();

} // namespace aotx::ctrl::chat

#endif
