// Purpose: Declare live controls for instance and conversation tool choices.
// Owns: Per-panel command status and initial query state.
// Launch shape: One interface thread draws each open panel.
// Lifetime: The panel view lasts as long as its conversation.
#ifndef AOTX_CTRL_TOOLS_PANEL_HPP
#define AOTX_CTRL_TOOLS_PANEL_HPP

#include "replica/replica.hpp"
#include "client/client.hpp"
#include "tools/query.hpp"
#include <string>

namespace aotx::ctrl::tools {
struct View {
    Query query;
    std::string result;
};
void draw(View &view, const replica::State &state, client::Client &client, unsigned agent);
}
#endif
