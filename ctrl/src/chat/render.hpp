// Purpose: Draw one pure formatting document with ImGui.
// Owns: No document or graphics state after the call returns.
// Launch shape: One interface thread draws one reply block at a time.
// Lifetime: Temporary style changes end before the call returns.
#ifndef AOTX_CTRL_CHAT_RENDER_HPP
#define AOTX_CTRL_CHAT_RENDER_HPP

#include <string>

namespace aotx::ctrl::chat {

void draw_formatted(const std::string &reply);

} // namespace aotx::ctrl::chat

#endif
