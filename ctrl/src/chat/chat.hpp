// Purpose: Draw a transcript and a multiline text editor.
// Owns: Editor text and transcript presentation state.
// Launch shape: One user interface thread draws one chat window.
// Lifetime: Editor state exists for the program lifetime.
#ifndef AOTX_CTRL_CHAT_HPP
#define AOTX_CTRL_CHAT_HPP

#include "sim/sim.hpp"
#include "chat/persona.hpp"
#include "tools/panel.hpp"

#include <array>
#include <string>
#include <vector>

namespace aotx::ctrl::voice { class Queue; }
namespace aotx::ctrl::client { class Client; }
namespace aotx::ctrl::replica { class State; struct Agent; }
namespace aotx::ctrl::instances { class Lifecycle; }

namespace aotx::ctrl::chat {

struct View {
    tools::View tool_view;
    std::array<char, 4001> editor{};
    std::vector<bool> spoken;
    unsigned live_agent = ~0u;
    bool bound = false;
    bool follow = true;
    bool persona_loaded = false;
    bool override_on = false;
    std::array<char, persona::voice_bytes + 1u> default_voice{};
    std::array<char, persona::voice_bytes + 1u> override_voice{};
    std::array<char, 81> conversation_name{};
    std::string persona_key;
    std::string action_result;
    std::string pending_role;
    bool persona_importing = false;
    bool persona_spawning = false;
};

std::string window_name(const sim::Conversation &conversation, std::size_t index);
std::string window_name(const replica::Agent &agent);
void draw(View &view, sim::State &state, std::size_t conversation, voice::Queue &speech,
          double now);
void draw(View &view, replica::State &state, std::size_t conversation, client::Client &client,
          voice::Queue &speech, instances::Lifecycle &lifecycle, std::size_t instance,
          persona::Store &personas, bool confidence);
bool verify_key_paths();

} // namespace aotx::ctrl::chat

#endif
