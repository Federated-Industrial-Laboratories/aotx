// Purpose: Define the live voice settings panel.
// Owns: Voice control presentation and immediate control updates.
// Launch shape: One interface thread draws one settings panel.
// Lifetime: The panel changes the process voice queue directly.
#ifndef AOTX_CTRL_VOICE_PANEL_HPP
#define AOTX_CTRL_VOICE_PANEL_HPP

namespace aotx::ctrl::voice {

class Queue;
void draw(Queue &queue, bool *open);

} // namespace aotx::ctrl::voice

#endif
